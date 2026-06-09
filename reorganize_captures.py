"""
reorganize_captures.py — 把抓帧输出重组为 COLMAP rig_configurator 期望的目录布局

输入: ./captures/<session>/<gid>_<cam>.jpg
输出: ./colmap_images/<cam>/<gid>.jpg
      (这样 image_prefix="<cam>/" 时, 相同 gid 在 6 个目录下都对应同一帧)
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="rig_capture.py 的输出 session 目录")
    ap.add_argument("--dst", required=True, type=Path,
                    help="重组后的输出根目录")
    ap.add_argument("--ext", default=".jpg")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise FileNotFoundError(args.src)
    args.dst.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in sorted(args.src.glob(f"*{args.ext}")):
        parts = src.stem.split("_", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        gid, cam = parts
        cam_dir = args.dst / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        dst = cam_dir / f"{gid}{args.ext}"
        shutil.copy2(src, dst)
        moved += 1

    print(f"重组 {moved} 张图 → {args.dst}")
    print("结构:")
    for cam_dir in sorted(args.dst.iterdir()):
        n = len(list(cam_dir.glob(f"*{args.ext}")))
        print(f"  {cam_dir.name}/  {n} 张")


if __name__ == "__main__":
    main()
