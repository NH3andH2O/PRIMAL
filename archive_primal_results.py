#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


def move_directory_contents(src_dir: Path, dst_dir: Path) -> None:
    """Move all items inside src_dir into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        shutil.move(str(item), str(dst_dir / item.name))


def main() -> None:
    root = Path(__file__).resolve().parent
    primal_results = root / "primal_results"
    train_primal = root / "train_primal"
    model_primal = root / "model_primal"

    if not primal_results.exists():
        raise FileNotFoundError(f"找不到資料夾: {primal_results}")
    if not train_primal.exists():
        raise FileNotFoundError(f"找不到資料夾: {train_primal}")
    if not model_primal.exists():
        raise FileNotFoundError(f"找不到資料夾: {model_primal}")

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    target_root = primal_results / timestamp
    target_root.mkdir(parents=True, exist_ok=False)

    target_train = target_root / "train_primal"
    target_model = target_root / "model_primal"

    move_directory_contents(train_primal, target_train)
    move_directory_contents(model_primal, target_model)

    print(f"已建立: {target_root}")
    print(f"已移動 train_primal 內容到: {target_train}")
    print(f"已移動 model_primal 內容到: {target_model}")


if __name__ == "__main__":
    main()
