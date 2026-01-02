
import shutil
import os
from pathlib import Path

def cleanup():
    # 1. Output directory
    output_dir = Path("output")
    if output_dir.exists():
        print(f"Removing {output_dir}...")
        try:
            shutil.rmtree(output_dir)
            print("  Deleted output/.")
        except Exception as e:
            print(f"  Failed to delete output/: {e}")
    else:
        print("output/ does not exist.")

    # 2. Data Raw directory (User Data/Cache)
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        print(f"Removing {raw_dir}...")
        try:
            shutil.rmtree(raw_dir)
            print("  Deleted data/raw/.")
        except Exception as e:
            print(f"  Failed to delete data/raw/: {e}")
    else:
        print("data/raw/ does not exist.")
        
    # Recreate empty directories to maintain structure
    output_dir.mkdir(exist_ok=True)
    (output_dir / "backtests").mkdir(exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("Recreated empty output/backtests and data/raw directories.")

if __name__ == "__main__":
    cleanup()
