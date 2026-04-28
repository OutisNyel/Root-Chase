#!/usr/bin/env python3
"""
Convert latest aisrv helper and learner training logs into TensorBoard events.

The script is intentionally host-side/offline:
- It reads Kaiwu JSON log lines from the aisrv and learner log directories.
- It selects the latest active aisrv_kaiwu_rl_helper and learner_train logs by default.
- It merges records from all selected workers by log timestamp.
- It writes TensorBoard scalars under train_monitor/tensorboard by default.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shutil
import socket
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
HELPER_LOG_RE = re.compile(
    r"^aisrv_kaiwu_rl_helper_pid(?P<pid>\d+)_log_(?P<hour>\d{4}-\d{2}-\d{2}-\d{2})\.log$"
)
HELPER_LOG_NAME_RE = re.compile(
    r"aisrv_kaiwu_rl_helper_pid(?P<pid>\d+)_log_(?P<hour>\d{4}-\d{2}-\d{2}-\d{2})"
)
LEARNER_LOG_RE = re.compile(
    r"^learner_train_pid(?P<pid>\d+)_log_(?P<hour>\d{4}-\d{2}-\d{2}-\d{2})\.log$"
)
LEARNER_LOG_NAME_RE = re.compile(
    r"learner_train_pid(?P<pid>\d+)_log_(?P<hour>\d{4}-\d{2}-\d{2}-\d{2})"
)
GAMEOVER_RE = re.compile(
    rf"\[GAMEOVER\]\s+episode:(?P<episode>\d+)\s+steps:(?P<steps>\d+)\s+"
    rf"result:(?P<result>\w+)\s+sim_score:(?P<sim_score>{FLOAT_PATTERN})\s+"
    rf"total_reward:(?P<total_reward>{FLOAT_PATTERN})(?P<extra>.*)$"
)
LOSS_RE = re.compile(
    rf"\[train\]\s+total_loss:(?P<total>{FLOAT_PATTERN})\s+"
    rf"policy_loss:(?P<policy>{FLOAT_PATTERN})\s+"
    rf"value_loss:(?P<value>{FLOAT_PATTERN})\s+"
    rf"entropy:(?P<entropy>{FLOAT_PATTERN})"
    rf"(?:\s+aux_pos:(?P<aux_pos>{FLOAT_PATTERN}))?"
    rf"(?:\s+aux_dist:(?P<aux_dist>{FLOAT_PATTERN}))?"
    rf"(?:\s+kl:(?P<kl>{FLOAT_PATTERN}))?"
    rf"(?:\s+clip_frac:(?P<clip_frac>{FLOAT_PATTERN}))?"
    rf"(?:\s+grad_norm:(?P<grad_norm>{FLOAT_PATTERN}))?"
)
EXTRA_METRIC_RE = re.compile(
    rf"(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<value>{FLOAT_PATTERN})"
)
TRAINING_METRICS_PREFIX = "training_metrics is "


@dataclass(frozen=True)
class LogFileInfo:
    path: Path
    source: str
    pid: str
    hour: str
    hour_dt: datetime
    size: int
    mtime: float


@dataclass(frozen=True)
class MetricRecord:
    timestamp: Optional[datetime]
    wall_time: float
    file_path: Path
    pid: str
    kind: str
    metrics: Dict[str, float]


class TensorBoardScalarWriter:
    def __init__(self, log_dir: Path) -> None:
        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError:
            self._event_cls = None
            self._summary_cls = None
            self._writer = MinimalEventFileWriter(log_dir)
            return

        self._event_cls = Event
        self._summary_cls = Summary
        self._writer = EventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        if self._event_cls is None or self._summary_cls is None:
            self._writer.add_scalar(tag, value, step, wall_time)
            return

        event = self._event_cls(
            wall_time=wall_time,
            step=int(step),
            summary=self._summary_cls(
                value=[self._summary_cls.Value(tag=tag, simple_value=float(value))]
            ),
        )
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class MinimalEventFileWriter:
    """Small TensorBoard v1 scalar event writer without external packages."""

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        host = sanitize_tag_part(socket.gethostname())
        filename = f"events.out.tfevents.{now}.{host}.{os.getpid()}.0"
        self._handle = (log_dir / filename).open("wb")
        self._write_record(_event_file_version(time.time()))

    def add_scalar(self, tag: str, value: float, step: int, wall_time: float) -> None:
        self._write_record(_event_scalar(wall_time, step, tag, value))

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def _write_record(self, payload: bytes) -> None:
        length = struct.pack("<Q", len(payload))
        self._handle.write(length)
        self._handle.write(struct.pack("<I", _masked_crc32c(length)))
        self._handle.write(payload)
        self._handle.write(struct.pack("<I", _masked_crc32c(payload)))


def _make_crc32c_table() -> List[int]:
    table: List[int] = []
    polynomial = 0x82F63B78
    for value in range(256):
        crc = value
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return table


CRC32C_TABLE = _make_crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    rotated = ((crc >> 15) | ((crc << 17) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return (rotated + 0xA282EAD8) & 0xFFFFFFFF


def _varint(value: int) -> bytes:
    if value < 0:
        value += 1 << 64
    chunks = bytearray()
    while value >= 0x80:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    chunks.append(value)
    return bytes(chunks)


def _field_key(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _field_varint(field_number: int, value: int) -> bytes:
    return _field_key(field_number, 0) + _varint(value)


def _field_fixed64(field_number: int, value: float) -> bytes:
    return _field_key(field_number, 1) + struct.pack("<d", float(value))


def _field_bytes(field_number: int, value: bytes) -> bytes:
    return _field_key(field_number, 2) + _varint(len(value)) + value


def _field_string(field_number: int, value: str) -> bytes:
    return _field_bytes(field_number, value.encode("utf-8"))


def _field_fixed32(field_number: int, value: float) -> bytes:
    return _field_key(field_number, 5) + struct.pack("<f", float(value))


def _event_file_version(wall_time: float) -> bytes:
    return _field_fixed64(1, wall_time) + _field_string(3, "brain.Event:2")


def _event_scalar(wall_time: float, step: int, tag: str, value: float) -> bytes:
    summary_value = _field_string(1, tag) + _field_fixed32(2, value)
    summary = _field_bytes(1, summary_value)
    return (
        _field_fixed64(1, wall_time)
        + _field_varint(2, int(step))
        + _field_bytes(5, summary)
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_train_log_dir() -> Path:
    root = repo_root()
    sibling_data_dir = root.with_name(f"{root.name}.d")
    candidates = [
        sibling_data_dir / "train" / "log",
        root / "train" / "log",
        root / "log",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_tensorboard_dir() -> Path:
    return repo_root() / "train_monitor" / "tensorboard"


def parse_hour(hour: str) -> datetime:
    return datetime.strptime(hour, "%Y-%m-%d-%H")


def component_search_dirs(log_dir: Path, component_name: str) -> List[Path]:
    candidates: List[Path] = []
    if log_dir.name.lower() == component_name:
        candidates.append(log_dir)
    child = log_dir / component_name
    if child.exists():
        candidates.append(child)
    sibling = log_dir.parent / component_name
    if sibling.exists():
        candidates.append(sibling)
    if not candidates:
        candidates.append(log_dir)

    unique: List[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def discover_logs_by_pattern(
    log_dir: Path,
    component_name: str,
    glob_pattern: str,
    name_re: re.Pattern[str],
    source: str,
) -> List[LogFileInfo]:
    infos: List[LogFileInfo] = []
    for search_dir in component_search_dirs(log_dir, component_name):
        for path in search_dir.glob(glob_pattern):
            match = name_re.match(path.name)
            if not match:
                continue
            try:
                stat = path.stat()
            except OSError as exc:
                sys.stderr.write(f"[WARN] failed to stat {path}: {exc}\n")
                continue
            hour = match.group("hour")
            infos.append(
                LogFileInfo(
                    path=path,
                    source=source,
                    pid=match.group("pid"),
                    hour=hour,
                    hour_dt=parse_hour(hour),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )
    infos.sort(key=lambda item: (item.mtime, item.hour_dt, item.pid, item.path.name))
    return infos


def discover_metric_logs(log_dir: Path) -> List[LogFileInfo]:
    infos = []
    infos.extend(
        discover_logs_by_pattern(
            log_dir=log_dir,
            component_name="aisrv",
            glob_pattern="aisrv_kaiwu_rl_helper_pid*_log_*.log",
            name_re=HELPER_LOG_RE,
            source="aisrv_helper",
        )
    )
    infos.extend(
        discover_logs_by_pattern(
            log_dir=log_dir,
            component_name="learner",
            glob_pattern="learner_train_pid*_log_*.log",
            name_re=LEARNER_LOG_RE,
            source="learner_train",
        )
    )
    infos.sort(key=lambda item: (item.mtime, item.hour_dt, item.source, item.pid, item.path.name))
    return infos


def build_log_info(path: Path) -> Optional[LogFileInfo]:
    match = HELPER_LOG_NAME_RE.search(path.name)
    source = "aisrv_helper"
    if not match:
        match = LEARNER_LOG_NAME_RE.search(path.name)
        source = "learner_train"
    if not match:
        sys.stderr.write(f"[WARN] skip unsupported metric log file: {path}\n")
        return None
    try:
        stat = path.stat()
    except OSError as exc:
        sys.stderr.write(f"[WARN] failed to stat {path}: {exc}\n")
        return None

    hour = match.group("hour")
    return LogFileInfo(
        path=path,
        source=source,
        pid=match.group("pid"),
        hour=hour,
        hour_dt=parse_hour(hour),
        size=stat.st_size,
        mtime=stat.st_mtime,
    )


def collect_explicit_logs(files: Sequence[str]) -> List[LogFileInfo]:
    infos: List[LogFileInfo] = []
    seen: set[Path] = set()
    for value in files:
        path = Path(value).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        info = build_log_info(path)
        if info is not None:
            infos.append(info)
    infos.sort(key=lambda item: (item.hour_dt, item.pid, item.path.name))
    return infos


def select_logs(infos: Sequence[LogFileInfo], include_all: bool, active_window_seconds: float) -> List[LogFileInfo]:
    if include_all or not infos:
        return list(infos)

    non_empty = [info for info in infos if info.size > 0]
    if not non_empty:
        return []

    latest_mtime = max(info.mtime for info in non_empty)
    selected = [
        info for info in non_empty if latest_mtime - info.mtime <= active_window_seconds
    ]
    selected.sort(key=lambda item: (item.hour_dt, item.pid, item.path.name))
    return selected


def parse_log_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def to_wall_time(timestamp: Optional[datetime], fallback_path: Path) -> float:
    if timestamp is not None:
        return timestamp.timestamp()
    try:
        return fallback_path.stat().st_mtime
    except OSError:
        return time.time()


def safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value))
        except (TypeError, ValueError):
            return None
    if not math.isfinite(number):
        return None
    return number


def flatten_numeric_metrics(prefix: str, value: Any) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, nested_value in value.items():
            safe_key = sanitize_tag_part(str(key))
            nested_prefix = f"{prefix}/{safe_key}" if prefix else safe_key
            flattened.update(flatten_numeric_metrics(nested_prefix, nested_value))
        return flattened

    number = safe_float(value)
    if number is not None and prefix:
        flattened[prefix] = number
    return flattened


def sanitize_tag_part(value: str) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9_.\-/]", "_", value)
    value = value.strip("/_")
    return value or "unknown"


def parse_training_metrics(message: str) -> Optional[Dict[str, float]]:
    marker_index = message.find(TRAINING_METRICS_PREFIX)
    if marker_index < 0:
        return None
    payload = message[marker_index + len(TRAINING_METRICS_PREFIX) :].strip()
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return flatten_numeric_metrics("training", parsed)


def parse_learner_train(message: str) -> Optional[Dict[str, float]]:
    match = LOSS_RE.search(message)
    if not match:
        return None

    metrics = {
        "training/algorithm/total_loss": float(match.group("total")),
        "training/algorithm/policy_loss": float(match.group("policy")),
        "training/algorithm/value_loss": float(match.group("value")),
        "training/algorithm/entropy_loss": float(match.group("entropy")),
    }
    optional_fields = {
        "aux_pos": "aux_pos_loss",
        "aux_dist": "aux_dist_loss",
        "kl": "approx_kl",
        "clip_frac": "clip_frac",
        "grad_norm": "grad_norm",
    }
    for group_name, metric_name in optional_fields.items():
        value = match.group(group_name)
        if value is not None:
            metrics[f"training/algorithm/{metric_name}"] = float(value)
    return metrics


def parse_gameover(message: str) -> Optional[Dict[str, float]]:
    match = GAMEOVER_RE.search(message)
    if not match:
        return None

    result = match.group("result").upper()
    metrics: Dict[str, float] = {
        "gameover/episode": float(match.group("episode")),
        "gameover/episode_steps": float(match.group("steps")),
        "gameover/sim_score": float(match.group("sim_score")),
        "gameover/total_reward": float(match.group("total_reward")),
        "gameover/success": 1.0 if result in {"SUCCESS", "WIN"} else 0.0,
        "gameover/fail": 0.0 if result in {"SUCCESS", "WIN"} else 1.0,
    }
    for extra_match in EXTRA_METRIC_RE.finditer(match.group("extra") or ""):
        key = sanitize_tag_part(extra_match.group("key"))
        metrics[f"gameover/{key}"] = float(extra_match.group("value"))
    return metrics


def iter_metric_records(logs: Iterable[LogFileInfo]) -> Iterator[MetricRecord]:
    for info in logs:
        try:
            with info.path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, dict):
                        continue
                    message = str(entry.get("message", "")).strip()
                    timestamp = parse_log_time(entry.get("time"))
                    wall_time = to_wall_time(timestamp, info.path)

                    gameover_metrics = parse_gameover(message)
                    if gameover_metrics:
                        yield MetricRecord(
                            timestamp=timestamp,
                            wall_time=wall_time,
                            file_path=info.path,
                            pid=info.pid,
                            kind="gameover",
                            metrics=gameover_metrics,
                        )
                        continue

                    training_metrics = parse_training_metrics(message)
                    if training_metrics:
                        yield MetricRecord(
                            timestamp=timestamp,
                            wall_time=wall_time,
                            file_path=info.path,
                            pid=info.pid,
                            kind="training",
                            metrics=training_metrics,
                        )
                        continue

                    learner_train_metrics = parse_learner_train(message)
                    if learner_train_metrics:
                        yield MetricRecord(
                            timestamp=timestamp,
                            wall_time=wall_time,
                            file_path=info.path,
                            pid=info.pid,
                            kind="learner_train",
                            metrics=learner_train_metrics,
                        )
        except OSError as exc:
            sys.stderr.write(f"[WARN] failed to read {info.path}: {exc}\n")


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        try:
            if path.is_file() and (
                path.name.startswith("events.out.tfevents")
                or path.name == "aisrv_rl_helper_manifest.json"
            ):
                path.unlink()
            elif path.is_dir() and path.name.startswith(".tmp"):
                shutil.rmtree(path)
        except OSError as exc:
            sys.stderr.write(f"[WARN] failed to clean {path}: {exc}\n")


def write_manifest(
    output_dir: Path,
    selected_logs: Sequence[LogFileInfo],
    records: Sequence[MetricRecord],
    tags_written: int,
) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "selected_files": [str(info.path) for info in selected_logs],
        "selected_files_by_source": {
            source: [str(info.path) for info in selected_logs if info.source == source]
            for source in sorted({info.source for info in selected_logs})
        },
        "record_count": len(records),
        "scalar_count": tags_written,
        "records_by_kind": {
            kind: sum(1 for record in records if record.kind == kind)
            for kind in sorted({record.kind for record in records})
        },
    }
    manifest_path = output_dir / "aisrv_rl_helper_manifest.json"
    try:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        sys.stderr.write(f"[WARN] failed to write manifest {manifest_path}: {exc}\n")


def convert(args: argparse.Namespace) -> int:
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.file:
        selected_logs = collect_explicit_logs(args.file)
    else:
        infos = discover_metric_logs(log_dir)
        if not infos:
            sys.stderr.write(
                f"[ERROR] no aisrv helper or learner train logs found under {log_dir}\n"
            )
            return 2
        selected_logs = select_logs(
            infos,
            include_all=args.all,
            active_window_seconds=args.active_window_seconds,
        )

    if not selected_logs:
        sys.stderr.write(
            f"[ERROR] no non-empty aisrv helper or learner train logs selected under {log_dir}\n"
        )
        return 2
    records = list(iter_metric_records(selected_logs))
    records.sort(
        key=lambda record: (
            record.timestamp or datetime.min,
            record.pid,
            str(record.file_path),
        )
    )

    if args.dry_run:
        print_summary(output_dir, selected_logs, records, tags_written=0)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_output_dir(output_dir)

    writer = TensorBoardScalarWriter(output_dir)
    tags_written = 0
    try:
        for step, record in enumerate(records, start=1):
            for tag, value in sorted(record.metrics.items()):
                writer.add_scalar(tag, value, step, record.wall_time)
                tags_written += 1
                if args.include_pid_tags:
                    writer.add_scalar(f"pid/{record.pid}/{tag}", value, step, record.wall_time)
                    tags_written += 1
        writer.flush()
    finally:
        writer.close()

    write_manifest(output_dir, selected_logs, records, tags_written)
    print_summary(output_dir, selected_logs, records, tags_written)
    return 0


def print_summary(
    output_dir: Path,
    selected_logs: Sequence[LogFileInfo],
    records: Sequence[MetricRecord],
    tags_written: int,
) -> None:
    latest_hour = max((info.hour for info in selected_logs), default="")
    by_kind: Dict[str, int] = {}
    for record in records:
        by_kind[record.kind] = by_kind.get(record.kind, 0) + 1

    print(f"selected_hour: {latest_hour}")
    print(f"selected_files: {len(selected_logs)}")
    for info in selected_logs:
        print(f"  - {info.source} pid{info.pid}: {info.path}")
    print(f"records: {len(records)} {by_kind}")
    print(f"scalars_written: {tags_written}")
    print(f"tensorboard_dir: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge latest aisrv helper and learner train logs into TensorBoard scalars."
    )
    parser.add_argument(
        "--log-dir",
        default=str(default_train_log_dir()),
        help="Training log root. It may point to train/log or train/log/aisrv.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help=(
            "Explicit aisrv_kaiwu_rl_helper or learner_train log file to convert. Repeat this "
            "option to merge multiple files. When provided, --log-dir, --all, "
            "and --active-window-seconds are not used for selecting input."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_tensorboard_dir()),
        help="TensorBoard output directory.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all discovered metric logs instead of only the latest active logs.",
    )
    parser.add_argument(
        "--active-window-seconds",
        type=float,
        default=120.0,
        help=(
            "Default selection keeps non-empty helper logs whose modified time is "
            "within this many seconds of the newest helper log."
        ),
    )
    parser.add_argument(
        "--include-pid-tags",
        action="store_true",
        help="Also write per-PID tags such as pid/307/gameover/total_reward.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous event files and manifest from the output directory first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print the selected records without writing TensorBoard files.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
