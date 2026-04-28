#!/usr/bin/env python3
"""
GorgeChase Release Packaging Script
Packages code and necessary components into a release archive
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import argparse
import sys


def create_release(output_dir="release", archive_name="gorgechase-release", use_zip=False):
    """Create release package with specified components.
    
    Args:
        output_dir: Output directory for release
        archive_name: Archive name prefix
        use_zip: If True, create zip archive; if False, create directory structure
    """
    
    # Define paths
    code_dir = Path(__file__).parent.parent / "code"
    if not code_dir.exists():
        print(f"❌ Code directory not found: {code_dir}")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped release name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_name = f"{archive_name}-{timestamp}"
    release_path = output_path / release_name
    
    # Items to copy
    items_to_copy = [
        "conf",
        "agent_ppo",
        ".vscode",
        "kaiwu.json",
        "train_test.py"
    ]
    
    if use_zip:
        # Zip mode: use staging directory
        staging_dir = output_path / "staging"
        
        # Clean up previous staging if exists
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        
        # Create staging structure
        staging_code_dir = staging_dir / "code"
        staging_code_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy items
        for item in items_to_copy:
            source_path = code_dir / item
            dest_path = staging_code_dir / item
            
            if not source_path.exists():
                print(f"⚠  Warning: Item not found - {item}")
                continue
            
            try:
                if source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    print(f"✓ Copied directory: {item}")
                else:
                    shutil.copy2(source_path, dest_path)
                    print(f"✓ Copied file: {item}")
            except Exception as e:
                print(f"❌ Error copying {item}: {e}")
                return False
        
        # Create zip archive
        archive_path = f"{release_path}.zip"
        try:
            shutil.make_archive(
                str(release_path),
                'zip',
                staging_dir
            )
            
            # Get archive size
            archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
            
            print(f"\n✅ Release package created successfully!")
            print(f"   Archive: {archive_path}")
            print(f"   Size: {archive_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error creating archive: {e}")
            return False
        
        # Clean up staging directory
        try:
            shutil.rmtree(staging_dir)
            print(f"✓ Cleaned up temporary files")
        except Exception as e:
            print(f"⚠  Warning: Could not clean staging directory: {e}")
        
        # Create latest.zip reference
        latest_path = output_path / "latest.zip"
        try:
            if latest_path.exists():
                latest_path.unlink()
            shutil.copy2(archive_path, latest_path)
            print(f"✓ Created latest.zip reference")
        except Exception as e:
            print(f"⚠  Warning: Could not create latest.zip: {e}")
        
        print(f"\n📦 Release Info:")
        print(f"   Name: {release_name}")
        print(f"   Location: {archive_path}")
        print(f"   Latest: {latest_path}")
        
    else:
        # Directory mode: create release directory structure directly
        release_code_dir = release_path / "code"
        release_code_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy items
        for item in items_to_copy:
            source_path = code_dir / item
            dest_path = release_code_dir / item
            
            if not source_path.exists():
                print(f"⚠  Warning: Item not found - {item}")
                continue
            
            try:
                if source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    print(f"✓ Copied directory: {item}")
                else:
                    shutil.copy2(source_path, dest_path)
                    print(f"✓ Copied file: {item}")
            except Exception as e:
                print(f"❌ Error copying {item}: {e}")
                return False
        
        # Get directory size
        total_size = sum(
            f.stat().st_size for f in release_path.rglob('*') if f.is_file()
        ) / (1024 * 1024)
        
        print(f"\n✅ Release directory created successfully!")
        print(f"   Directory: {release_path}")
        print(f"   Size: {total_size:.2f} MB")
        
        # Create latest symlink reference
        latest_path = output_path / "latest"
        try:
            if latest_path.exists() or latest_path.is_symlink():
                if latest_path.is_symlink():
                    latest_path.unlink()
                elif latest_path.is_dir():
                    shutil.rmtree(latest_path)
            # Create symlink or copy depending on OS
            try:
                latest_path.symlink_to(release_path, target_is_directory=True)
                print(f"✓ Created latest symlink")
            except (OSError, NotImplementedError):
                # Fallback to copying if symlink not supported
                shutil.copytree(release_path, latest_path, dirs_exist_ok=True)
                print(f"✓ Created latest directory (fallback)")
        except Exception as e:
            print(f"⚠  Warning: Could not create latest reference: {e}")
        
        print(f"\n📦 Release Info:")
        print(f"   Name: {release_name}")
        print(f"   Location: {release_path}")
        print(f"   Latest: {latest_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="GorgeChase Release Packager")
    parser.add_argument(
        "--output", "-o",
        default="release",
        help="Output directory for release (default: release)"
    )
    parser.add_argument(
        "--name", "-n",
        default="gorgechase-release",
        help="Archive name prefix (default: gorgechase-release)"
    )
    parser.add_argument(
        "--zip", "-z",
        action="store_true",
        help="Create zip archive (default: create directory)"
    )
    
    args = parser.parse_args()
    
    success = create_release(args.output, args.name, args.zip)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
