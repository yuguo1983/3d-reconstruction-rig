"""
make_charuco_board.py — 生成 6 路联合标定用的 ChArUco 板 PNG

默认参数跟 calibrate_rig.py 对齐:
  - 9 x 7 squares
  - 20mm square, 14mm marker
  - DICT_6X6_250

输出: charuco_9x7_20mm.png (默认)
  - 分辨率 20 px/mm, 总尺寸 3600 x 2800 px
  - 适合 A3 喷绘/打印 (300dpi 下 18cm x 14cm, A3 留 12cm 边距)

用法:
  python make_charuco_board.py                            # 默认参数
  python make_charuco_board.py --out my_board.png         # 自定义文件名
  python make_charuco_board.py --px-per-mm 30             # 更高分辨率
  python make_charuco_board.py --squares-x 5 --squares-y 7 # 改尺寸

打印后建议:
  - 用哑光相纸喷绘, 贴在一块平整硬板 (KT板/亚克力) 上
  - 不能弯曲, 不能反光
  - 标定时板要占视野中央 60% 以上
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import cv2.aruco as aruco


def make_board(sx: int, sy: int, square_m: float, marker_m: float,
               dict_name: str):
    d = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    return aruco.CharucoBoard((sx, sy), square_m, marker_m, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares-x", type=int, default=9)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-length", type=float, default=0.020,
                    help="方格边长, 米 (默认 0.020 = 20mm)")
    ap.add_argument("--marker-length", type=float, default=0.014,
                    help="ArUco marker 边长, 米 (默认 0.014 = 14mm)")
    ap.add_argument("--dict", type=str, default="DICT_6X6_250",
                    help="ArUco 字典 (跟 calibrate_rig.py 的 --dict 一致)")
    ap.add_argument("--px-per-mm", type=float, default=20.0,
                    help="每毫米多少像素 (默认 20, 约 508dpi)")
    ap.add_argument("--margin-px", type=int, default=20,
                    help="板四周留白像素, 方便裁切")
    ap.add_argument("--out", type=Path, default=Path("charuco_9x7_20mm.png"))
    ap.add_argument("--show", action="store_true",
                    help="生成后弹窗预览 (会卡住)")
    args = ap.parse_args()

    board = make_board(args.squares_x, args.squares_y,
                       args.square_length, args.marker_length, args.dict)

    # 总物理尺寸 -> 像素
    w_mm = args.squares_x * args.square_length * 1000
    h_mm = args.squares_y * args.square_length * 1000
    w_px = int(round(w_mm * args.px_per_mm))
    h_px = int(round(h_mm * args.px_per_mm))
    print(f"板物理尺寸: {w_mm:.0f}mm x {h_mm:.0f}mm = {w_mm/10:.1f}cm x {h_mm/10:.1f}cm")
    print(f"图像分辨率: {w_px} x {h_px} px ({args.px_per_mm} px/mm)")

    img = board.generateImage((w_px, h_px))

    # 加白边, 方便裁切
    if args.margin_px > 0:
        img = cv2.copyMakeBorder(
            img, args.margin_px, args.margin_px, args.margin_px, args.margin_px,
            cv2.BORDER_CONSTANT, value=255)
        h2, w2 = img.shape[:2]
        print(f"加 {args.margin_px}px 白边后: {w2} x {h2} px")

    cv2.imwrite(str(args.out), img)
    print(f"写入: {args.out}")

    # 打印说明
    print()
    print("=" * 60)
    print(f"打印建议: 用 A3 哑光相纸喷绘, 实际尺寸 {w_mm/10:.1f} x {h_mm/10:.1f} cm")
    print(f"  - 打印机选 [实际尺寸] / [100% 缩放], 不要 [适合纸张]")
    print(f"  - 用尺子量一下打印出来的方格边长, 应 = {args.square_length*1000:.0f}mm")
    print(f"  - 如果差 > 5%, 改 calibrate_rig.py 的 --square-length")
    print(f"  - 然后用 calibrate_rig.py 标定时, 参数跟这里一致")
    print("=" * 60)

    if args.show:
        # 缩到屏幕能放下
        scale = min(1.0, 1200 / max(img.shape[:2]))
        if scale < 1.0:
            preview = cv2.resize(img, None, fx=scale, fy=scale)
        else:
            preview = img
        cv2.imshow("charuco board", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
