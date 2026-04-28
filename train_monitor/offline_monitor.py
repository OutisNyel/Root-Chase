#!/usr/bin/env python3
"""
Host-side offline monitor for Kaiwu training logs.

Use this script to inspect logs without container monitor services:
1) summary: aggregate health + training metrics
2) tail: inspect recent lines with filters
3) watch: refresh summary periodically
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
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
STEP_RE = re.compile(
    r"train count is (?P<train>\d+), global step is (?P<global>\d+)"
)
SAVE_RE = re.compile(r"save model .*model\.ckpt-(?P<step>\d+)\.pkl")
GAMEOVER_RE = re.compile(
    rf"\[GAMEOVER\]\s+episode:(?P<episode>\d+)\s+steps:(?P<steps>\d+)\s+"
    rf"result:(?P<result>\w+)\s+sim_score:(?P<sim_score>{FLOAT_PATTERN})\s+"
    rf"total_reward:(?P<reward>{FLOAT_PATTERN})(?P<extra>.*)$"
)
EXTRA_METRIC_RE = re.compile(rf"(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<value>{FLOAT_PATTERN})")


@dataclass
class ParsedLine:
    timestamp: Optional[datetime]
    time_text: str
    level: str
    module: str
    message: str
    file_path: str
    raw: str


def parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    value = value.strip()
    fmts = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
    for fmt in fmts:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def normalize_level(level: Any) -> str:
    if not isinstance(level, str) or not level:
        return "UNKNOWN"
    return level.strip().upper()


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None
    return None


def parse_line(line: str, module: str, file_path: Path) -> ParsedLine:
    raw = line.rstrip("\n")
    obj = parse_json_line(raw)
    if obj is None:
        return ParsedLine(
            timestamp=None,
            time_text="",
            level="RAW",
            module=module,
            message=raw,
            file_path=str(file_path),
            raw=raw,
        )

    time_text = str(obj.get("time", "")).strip()
    level = normalize_level(obj.get("level"))
    msg = str(obj.get("message", "")).strip()
    return ParsedLine(
        timestamp=parse_time(time_text),
        time_text=time_text,
        level=level,
        module=str(obj.get("module", module)).strip() or module,
        message=msg,
        file_path=str(file_path),
        raw=raw,
    )


def discover_log_files(log_dir: Path, module_filter: Optional[str] = None) -> List[Path]:
    files = [p for p in log_dir.rglob("*.log") if p.is_file()]
    if module_filter:
        mf = module_filter.lower()
        files = [p for p in files if p.parent.name.lower() == mf or mf in p.name.lower()]
    files.sort(key=lambda p: (p.parent.name.lower(), p.name.lower()))
    return files


def iter_lines(files: Iterable[Path]) -> Iterable[Tuple[Path, ParsedLine]]:
    for fp in files:
        module = fp.parent.name
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip("\r\n")
                    if not line:
                        continue
                    yield fp, parse_line(line, module=module, file_path=fp)
        except OSError as exc:
            sys.stderr.write(f"[WARN] failed to read {fp}: {exc}\n")


def build_summary(log_dir: Path, max_alerts: int = 30) -> Dict[str, Any]:
    files = discover_log_files(log_dir)
    levels = Counter()
    modules: Dict[str, Dict[str, Any]] = {}

    learner_latest_loss: Optional[Dict[str, Any]] = None
    learner_latest_step: Optional[Dict[str, Any]] = None
    learner_latest_ckpt_step: Optional[int] = None

    gameover_total = 0
    gameover_fail = 0
    gameover_reward_sum = 0.0
    gameover_metric_sums: Counter[str] = Counter()
    gameover_metric_counts: Counter[str] = Counter()
    gameover_latest: Optional[Dict[str, Any]] = None

    alerts: deque[Dict[str, Any]] = deque(maxlen=max_alerts)
    total_lines = 0

    for fp, entry in iter_lines(files):
        total_lines += 1
        levels[entry.level] += 1

        m = modules.setdefault(
            entry.module,
            {
                "lines": 0,
                "levels": Counter(),
                "latest_time": "",
                "latest_file": "",
            },
        )
        m["lines"] += 1
        m["levels"][entry.level] += 1
        if entry.timestamp and (
            not m["latest_time"] or entry.time_text > m["latest_time"]
        ):
            m["latest_time"] = entry.time_text
            m["latest_file"] = str(fp)

        if entry.level in {"ERROR", "WARNING"}:
            alerts.append(
                {
                    "time": entry.time_text,
                    "level": entry.level,
                    "module": entry.module,
                    "file": str(fp),
                    "message": entry.message,
                }
            )

        if entry.module == "learner":
            loss_match = LOSS_RE.search(entry.message)
            if loss_match:
                learner_latest_loss = {
                    "time": entry.time_text,
                    "total_loss": float(loss_match.group("total")),
                    "policy_loss": float(loss_match.group("policy")),
                    "value_loss": float(loss_match.group("value")),
                    "entropy": float(loss_match.group("entropy")),
                }
                optional_loss_fields = {
                    "aux_pos": "aux_pos_loss",
                    "aux_dist": "aux_dist_loss",
                    "kl": "approx_kl",
                    "clip_frac": "clip_frac",
                    "grad_norm": "grad_norm",
                }
                for group_name, output_name in optional_loss_fields.items():
                    value = loss_match.group(group_name)
                    if value is not None:
                        learner_latest_loss[output_name] = float(value)

            step_match = STEP_RE.search(entry.message)
            if step_match:
                learner_latest_step = {
                    "time": entry.time_text,
                    "train_count": int(step_match.group("train")),
                    "global_step": int(step_match.group("global")),
                }

            save_match = SAVE_RE.search(entry.message)
            if save_match:
                learner_latest_ckpt_step = int(save_match.group("step"))

        if entry.module == "aisrv":
            go_match = GAMEOVER_RE.search(entry.message)
            if go_match:
                gameover_total += 1
                result = go_match.group("result").upper()
                if result not in {"SUCCESS", "WIN"}:
                    gameover_fail += 1
                reward = float(go_match.group("reward"))
                gameover_reward_sum += reward
                extra_metrics = {
                    m.group("key"): float(m.group("value"))
                    for m in EXTRA_METRIC_RE.finditer(go_match.group("extra") or "")
                }
                for key, value in extra_metrics.items():
                    gameover_metric_sums[key] += value
                    gameover_metric_counts[key] += 1
                gameover_latest = {
                    "time": entry.time_text,
                    "episode": int(go_match.group("episode")),
                    "steps": int(go_match.group("steps")),
                    "result": result,
                    "sim_score": float(go_match.group("sim_score")),
                    "total_reward": reward,
                }
                gameover_latest.update(extra_metrics)

    module_summary: Dict[str, Any] = {}
    for mod, info in sorted(modules.items(), key=lambda x: x[0]):
        module_summary[mod] = {
            "lines": info["lines"],
            "latest_time": info["latest_time"],
            "latest_file": info["latest_file"],
            "levels": dict(sorted(info["levels"].items())),
        }

    gameover_avg_reward = (
        round(gameover_reward_sum / gameover_total, 6) if gameover_total else None
    )
    gameover_avg_behavior = {
        key: round(gameover_metric_sums[key] / gameover_metric_counts[key], 6)
        for key in sorted(gameover_metric_sums)
        if gameover_metric_counts[key] > 0
    }

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_dir": str(log_dir.resolve()),
        "files": len(files),
        "lines": total_lines,
        "levels": dict(sorted(levels.items())),
        "modules": module_summary,
        "learner": {
            "latest_step": learner_latest_step,
            "latest_loss": learner_latest_loss,
            "latest_checkpoint_step": learner_latest_ckpt_step,
        },
        "aisrv_gameover": {
            "total": gameover_total,
            "failed": gameover_fail,
            "failed_ratio": round(gameover_fail / gameover_total, 6)
            if gameover_total
            else None,
            "avg_total_reward": gameover_avg_reward,
            "avg_behavior_metrics": gameover_avg_behavior,
            "latest": gameover_latest,
        },
        "recent_alerts": list(alerts),
    }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"Generated At: {summary['generated_at']}")
    print(f"Log Dir     : {summary['log_dir']}")
    print(f"Files/Lines : {summary['files']} / {summary['lines']}")
    print(f"Levels      : {summary['levels']}")
    print("")
    print("Modules:")
    for module, item in summary["modules"].items():
        print(
            f"  - {module:<10} lines={item['lines']:<8} "
            f"latest={item['latest_time']} levels={item['levels']}"
        )

    learner = summary["learner"]
    print("")
    print("Learner:")
    print(f"  - latest_step      : {learner['latest_step']}")
    print(f"  - latest_loss      : {learner['latest_loss']}")
    print(f"  - latest_ckpt_step : {learner['latest_checkpoint_step']}")

    aisrv = summary["aisrv_gameover"]
    print("")
    print("Aisrv Gameover:")
    print(
        f"  - total={aisrv['total']} failed={aisrv['failed']} "
        f"failed_ratio={aisrv['failed_ratio']} avg_reward={aisrv['avg_total_reward']}"
    )
    print(f"  - avg_behavior={aisrv['avg_behavior_metrics']}")
    print(f"  - latest={aisrv['latest']}")

    alerts = summary["recent_alerts"]
    print("")
    print(f"Recent Alerts ({len(alerts)}):")
    for item in alerts[-10:]:
        msg = item["message"]
        if len(msg) > 140:
            msg = msg[:137] + "..."
        print(
            f"  - {item['time']} [{item['level']}] {item['module']} "
            f"{Path(item['file']).name}: {msg}"
        )


def command_summary(args: argparse.Namespace) -> int:
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        sys.stderr.write(f"[ERROR] log dir not found: {log_dir}\n")
        return 2

    summary = build_summary(log_dir=log_dir, max_alerts=args.max_alerts)
    print_summary(summary)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("")
        print(f"JSON saved: {out.resolve()}")
    return 0


def command_tail(args: argparse.Namespace) -> int:
    log_dir = Path(args.log_dir)
    files = discover_log_files(log_dir, module_filter=args.module)
    if not files:
        sys.stderr.write(f"[ERROR] no log files found under: {log_dir}\n")
        return 2

    grep_text = args.grep.lower() if args.grep else ""
    level_filter = normalize_level(args.level) if args.level else ""
    buf: deque[ParsedLine] = deque(maxlen=args.lines)

    for _, entry in iter_lines(files):
        if grep_text and grep_text not in entry.message.lower():
            continue
        if level_filter and entry.level != level_filter:
            continue
        buf.append(entry)

    for entry in buf:
        t = entry.time_text or "-"
        msg = entry.message if entry.message else entry.raw
        print(f"{t} [{entry.level:<7}] {entry.module:<10} {msg}")
    return 0


def command_watch(args: argparse.Namespace) -> int:
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        sys.stderr.write(f"[ERROR] log dir not found: {log_dir}\n")
        return 2

    try:
        while True:
            summary = build_summary(log_dir=log_dir, max_alerts=args.max_alerts)
            # ANSI clear-screen
            print("\033[2J\033[H", end="")
            print_summary(summary)
            if args.json_out:
                out = Path(args.json_out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline monitor for Kaiwu training logs."
    )
    parser.add_argument(
        "--log-dir",
        default="train/log",
        help="log directory (default: train/log)",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="optional path to write structured summary json",
    )
    parser.add_argument(
        "--max-alerts",
        type=int,
        default=30,
        help="max warning/error records kept in summary",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_summary = sub.add_parser("summary", help="print aggregated summary")
    p_summary.set_defaults(func=command_summary)

    p_tail = sub.add_parser("tail", help="print recent log lines")
    p_tail.add_argument("--module", default="", help="module filter, e.g. learner")
    p_tail.add_argument("--grep", default="", help="message contains text")
    p_tail.add_argument("--level", default="", help="level filter, e.g. ERROR")
    p_tail.add_argument("--lines", type=int, default=50, help="number of lines")
    p_tail.set_defaults(func=command_tail)

    p_watch = sub.add_parser("watch", help="refresh summary periodically")
    p_watch.add_argument(
        "--interval", type=int, default=5, help="refresh interval seconds"
    )
    p_watch.set_defaults(func=command_watch)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
