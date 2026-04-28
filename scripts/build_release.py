#!/usr/bin/env python3
"""Build a GorgeChase release package."""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".pt",
    ".pth",
    ".onnx",
    ".npy",
    ".npz",
}

ITEMS_TO_COPY = [
    "conf",
    "agent_ppo",
    ".vscode",
    "kaiwu.json",
    "train_test.py",
]


def should_exclude(path: Path, include_checkpoints: bool) -> bool:
    """Return True for local artifacts that should not ship by default."""
    if set(path.parts) & EXCLUDED_DIRS:
        return True
    if not include_checkpoints and ("ckpt" in path.parts or path.suffix.lower() == ".pkl"):
        return True
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def copy_release_item(source_path: Path, dest_path: Path, include_checkpoints: bool) -> None:
    if source_path.is_dir():
        def ignore(directory, names):
            directory_path = Path(directory)
            return [
                name
                for name in names
                if should_exclude(directory_path / name, include_checkpoints)
            ]

        shutil.copytree(source_path, dest_path, dirs_exist_ok=True, ignore=ignore)
        return

    if should_exclude(source_path, include_checkpoints):
        print(f"Skipped local artifact: {source_path}")
        return
    shutil.copy2(source_path, dest_path)


def copy_release_tree(code_dir: Path, release_code_dir: Path, include_checkpoints: bool) -> bool:
    release_code_dir.mkdir(parents=True, exist_ok=True)

    for item in ITEMS_TO_COPY:
        source_path = code_dir / item
        dest_path = release_code_dir / item

        if not source_path.exists():
            print(f"Warning: item not found: {item}")
            continue

        try:
            copy_release_item(source_path, dest_path, include_checkpoints)
        except Exception as exc:
            print(f"Error copying {item}: {exc}")
            return False

        item_type = "directory" if source_path.is_dir() else "file"
        print(f"Copied {item_type}: {item}")

    return True


def create_release(
    output_dir: str = "release",
    archive_name: str = "gorgechase-release",
    use_zip: bool = False,
    include_checkpoints: bool = False,
) -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    code_dir = repo_root / "code"
    if not code_dir.exists():
        print(f"Code directory not found: {code_dir}")
        return False

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_name = f"{archive_name}-{timestamp}"
    release_path = output_path / release_name

    if use_zip:
        staging_dir = output_path / "staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

        if not copy_release_tree(code_dir, staging_dir / "code", include_checkpoints):
            return False

        archive_path = f"{release_path}.zip"
        try:
            shutil.make_archive(str(release_path), "zip", staging_dir)
            archive_size_mb = Path(archive_path).stat().st_size / (1024 * 1024)
        except Exception as exc:
            print(f"Error creating archive: {exc}")
            return False
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

        latest_path = output_path / "latest.zip"
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(archive_path, latest_path)

        print("\nRelease package created successfully")
        print(f"  Archive: {archive_path}")
        print(f"  Latest:  {latest_path}")
        print(f"  Size:    {archive_size_mb:.2f} MB")
        return True

    release_code_dir = release_path / "code"
    if not copy_release_tree(code_dir, release_code_dir, include_checkpoints):
        return False

    total_size_mb = sum(
        file.stat().st_size for file in release_path.rglob("*") if file.is_file()
    ) / (1024 * 1024)

    latest_path = output_path / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        if latest_path.is_symlink() or latest_path.is_file():
            latest_path.unlink()
        else:
            shutil.rmtree(latest_path)
    shutil.copytree(release_path, latest_path)

    print("\nRelease directory created successfully")
    print(f"  Directory: {release_path}")
    print(f"  Latest:    {latest_path}")
    print(f"  Size:      {total_size_mb:.2f} MB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="GorgeChase release packager")
    parser.add_argument(
        "--output",
        "-o",
        default="release",
        help="Output directory for release artifacts. Defaults to repo-root release/.",
    )
    parser.add_argument(
        "--name",
        "-n",
        default="gorgechase-release",
        help="Archive or directory name prefix.",
    )
    parser.add_argument(
        "--zip",
        "-z",
        action="store_true",
        help="Create a zip archive instead of a directory.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include agent_ppo/ckpt and .pkl files for a private preload package.",
    )

    args = parser.parse_args()
    success = create_release(
        output_dir=args.output,
        archive_name=args.name,
        use_zip=args.zip,
        include_checkpoints=args.include_checkpoints,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
