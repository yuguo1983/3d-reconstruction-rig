"""
compare_calib.py — 把 calibrate_rig.py 的输出跟 ground truth 对比

输入:
  --gt        synthetic/rig_calib.json
  --computed  calib/rig_calib.json (calibrate_rig.py 实际输出)
  --tolerance 允许的偏差 (默认内参 1%, 外参 2cm, 旋转 1°)

用法:
  python compare_calib.py --gt ./synthetic/rig_calib.json --computed ./calib/rig_calib.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def quat_to_rot(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-9:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ])


def rotation_angle_deg(R1, R2):
    R = R1.T @ R2
    cos = (np.trace(R) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cos, -1, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--computed", required=True, type=Path)
    ap.add_argument("--fx-tol", type=float, default=1.0,
                    help="fx/fy 允许偏差, 百分比")
    ap.add_argument("--cx-tol", type=float, default=5.0,
                    help="cx/cy 允许偏差, 像素")
    ap.add_argument("--rot-tol", type=float, default=1.0,
                    help="旋转角度允许偏差, 度")
    ap.add_argument("--trans-tol", type=float, default=0.02,
                    help="平移允许偏差, 米")
    args = ap.parse_args()

    if not args.gt.exists():
        print(f"[!] 找不到 ground truth: {args.gt}")
        return 1
    if not args.computed.exists():
        print(f"[!] 找不到 compute 结果: {args.computed}")
        return 1

    gt = json.loads(args.gt.read_text())
    cmp_ = json.loads(args.computed.read_text())

    print(f"=== 内参 ===")
    print(f"  {'相机':18s} {'fx Δ%':>8s} {'fy Δ%':>8s} {'cx Δpx':>8s} {'cy Δpx':>8s} {'RMS':>8s}")
    fx_errs, cx_errs = [], []
    for cam_name, gt_int in gt["intrinsics"].items():
        if cam_name not in cmp_["intrinsics"]:
            print(f"  [{cam_name:18s}] 未标定")
            continue
        c_int = cmp_["intrinsics"][cam_name]
        fx_g, fy_g = gt_int["K"][0][0], gt_int["K"][1][1]
        cx_g, cy_g = gt_int["K"][0][2], gt_int["K"][1][2]
        fx_c, fy_c = c_int["K"][0][0], c_int["K"][1][1]
        cx_c, cy_c = c_int["K"][0][2], c_int["K"][1][2]
        fx_e = abs(fx_c - fx_g) / fx_g * 100
        fy_e = abs(fy_c - fy_g) / fy_g * 100
        cx_e = abs(cx_c - cx_g)
        cy_e = abs(cy_c - cy_g)
        rms = c_int.get("rms_px", float("nan"))
        ok = "OK" if (fx_e < args.fx_tol and fy_e < args.fx_tol
                      and cx_e < args.cx_tol and cy_e < args.cx_tol) else "X "
        print(f"  [{cam_name:18s}] {ok} {fx_e:7.2f}% {fy_e:7.2f}% "
              f"{cx_e:7.1f} {cy_e:7.1f} {rms:7.3f}")
        fx_errs.append(max(fx_e, fy_e))
        cx_errs.append(max(cx_e, cy_e))

    print(f"\n=== 外参 (cam_from_ref) ===")
    print(f"  {'相机':18s} {'|T_gt|':>8s} {'|T_cmp|':>8s} {'ΔT':>8s} {'Δ%':>6s} {'ΔR°':>7s}")
    rot_errs, trans_errs = [], []
    for cam_name, gt_ext in gt["extrinsics_ref_to_cam"].items():
        if cam_name == gt["ref_cam"]:
            print(f"  [{cam_name:18s}] REF (跳过)")
            continue
        if cam_name not in cmp_["extrinsics_ref_to_cam"]:
            print(f"  [{cam_name:18s}] 未标定")
            continue
        c_ext = cmp_["extrinsics_ref_to_cam"][cam_name]
        R_g, T_g = np.array(gt_ext["R"]), np.array(gt_ext["T"])
        R_c, T_c = np.array(c_ext["R"]), np.array(c_ext["T"])
        ang = rotation_angle_deg(R_c, R_g)
        tn_g = np.linalg.norm(T_g)
        tn_c = np.linalg.norm(T_c)
        dT = np.linalg.norm(T_c - T_g)
        dT_pct = dT / max(tn_g, 1e-9) * 100
        ok = "OK" if (ang < args.rot_tol and dT < args.trans_tol) else "X "
        print(f"  [{cam_name:18s}] {ok} {tn_g*1000:7.1f}mm {tn_c*1000:7.1f}mm "
              f"{dT*1000:7.1f}mm {dT_pct:5.1f}% {ang:6.2f}")
        rot_errs.append(ang)
        trans_errs.append(dT)

    print(f"\n=== 总结 ===")
    if not fx_errs or not rot_errs:
        print("  没足够数据, 检查 calibrate_rig.py 是否真的解出了内/外参")
        return 1
    max_fx = max(fx_errs)
    max_cx = max(cx_errs)
    max_rot = max(rot_errs)
    max_trans = max(trans_errs)
    print(f"  最大 fx 误差:  {max_fx:.2f}%   (容差 {args.fx_tol}%)")
    print(f"  最大 cx 误差:  {max_cx:.1f}px   (容差 {args.cx_tol}px)")
    print(f"  最大旋转误差:  {max_rot:.2f}°   (容差 {args.rot_tol}°)")
    print(f"  最大平移误差:  {max_trans*1000:.1f}mm   (容差 {args.trans_tol*1000:.0f}mm)")
    passed = (max_fx < args.fx_tol and max_cx < args.cx_tol
              and max_rot < args.rot_tol and max_trans < args.trans_tol)
    if passed:
        print("\nOK 标定精度在容差内, pipeline 标定部分正常")
        return 0
    print("\nX 超过容差, 可能原因:")
    print("   - 标定帧数太少 (建议 >= 20)")
    print("   - ChArUco 检测失败率高 (检查光照、对比度)")
    print("   - 板尺寸 / 字典 / 行列数跟 calibrate_rig.py 参数不一致")
    return 1


if __name__ == "__main__":
    sys.exit(main())
