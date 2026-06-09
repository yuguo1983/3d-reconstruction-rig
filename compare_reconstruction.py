"""
compare_reconstruction.py — 把 COLMAP 的重建结果跟合成 ground truth 对比

读 ./sparse_test/0/{cameras,images}.txt 跟 ./synthetic/rig_calib.json 对比:
  - 内参误差 (fx, fy, cx, cy 偏差 %)
  - 外参误差 (cam_from_ref 旋转角度误差, 平移误差 %)
  - 重投影误差 (从 images.txt 的 points2D 估算)

如果输出为空, 说明 mapper 没注册上, 多半是 rig_config 没接好。
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np


def quat_to_rot(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ])


def parse_cameras_txt(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        # 我们的 rig_config 在末尾用 # 写了相机名; 也可能有注释
        cam_name = None
        for tok in parts[5:]:
            if tok.startswith("#"):
                idx = parts.index(tok)
                if idx + 1 < len(parts):
                    cam_name = parts[idx + 1]
                break
        out[cam_id] = {
            "model": parts[1], "w": int(parts[2]), "h": int(parts[3]),
            "params": [float(x) for x in parts[4:12]],
            "name": cam_name,
        }
    return out


def parse_images_txt(path: Path) -> list[dict]:
    """每张图占两行: 位姿行 + points2D 行"""
    lines = [l for l in path.read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    out = []
    i = 0
    while i < len(lines):
        pose_line = lines[i].split()
        if len(pose_line) >= 10:
            out.append({
                "image_id": int(pose_line[0]),
                "qvec": [float(x) for x in pose_line[1:5]],
                "tvec": [float(x) for x in pose_line[5:8]],
                "camera_id": int(pose_line[8]),
                "name": pose_line[9],
            })
        i += 2  # 跳过 points2D 行
    return out


def rotation_angle_deg(R1, R2):
    """两旋转矩阵之间的轴角"""
    R = R1.T @ R2
    cos = (np.trace(R) - 1) / 2
    cos = np.clip(cos, -1, 1)
    return np.rad2deg(np.arccos(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", type=Path, required=True,
                    help="COLMAP 稀疏重建目录 (含 cameras.txt + images.txt)")
    ap.add_argument("--gt-calib", type=Path, required=True,
                    help="合成 ground truth rig_calib.json")
    ap.add_argument("--cam-prefix", type=str, default="cam",
                    help="相机名前缀 (用于匹配 cam0_north, cam1_xxx 等)")
    args = ap.parse_args()

    sparse = args.sparse
    # COLMAP 通常写为 sparse/0/{cameras,images}.txt
    if (sparse / "cameras.txt").exists():
        cam_dir = sparse
    else:
        subdirs = sorted([d for d in sparse.iterdir() if d.is_dir()])
        if not subdirs:
            print(f"[错误] {sparse} 下没找到 cameras.txt")
            return
        cam_dir = subdirs[-1]

    colmap_cams = parse_cameras_txt(cam_dir / "cameras.txt")
    colmap_imgs = parse_images_txt(cam_dir / "images.txt")
    gt = json.loads(args.gt_calib.read_text())

    print(f"COLMAP 相机: {len(colmap_cams)}, 注册图像: {len(colmap_imgs)}")
    print(f"Ground truth 相机: {len(gt['intrinsics'])}")
    print()

    if not colmap_imgs:
        print("[!] COLMAP 没注册上任何图像. 排查顺序:")
        print("    1. rig_config.json 里 image_prefix 是否对得上 colmap_images/ 下的子目录名")
        print("    2. 6 台相机是否都生成了特征 (看 ./sparse_test/database.db 大小)")
        print("    3. COLMAP mapper 的 log 里是不是有 'no image has any matches'")
        return

    # --- 内参对比 ---
    print("=== 内参误差 ===")
    for cam_id, colmap_cam in colmap_cams.items():
        cam_name = colmap_cam.get("name")
        if cam_name and cam_name in gt["intrinsics"]:
            gt_cam = gt["intrinsics"][cam_name]
            fx_g, fy_g, cx_g, cy_g = gt_cam["K"][0][0], gt_cam["K"][1][1], \
                                       gt_cam["K"][0][2], gt_cam["K"][1][2]
            fx_c, fy_c, cx_c, cy_c = colmap_cam["params"][:4]
            err_fx = abs(fx_c - fx_g) / fx_g * 100
            err_fy = abs(fy_c - fy_g) / fy_g * 100
            err_cx = abs(cx_c - cx_g)
            err_cy = abs(cy_c - cy_g)
            print(f"  [{cam_name:18s}] "
                  f"Δfx={err_fx:5.2f}%  Δfy={err_fy:5.2f}%  "
                  f"Δcx={err_cx:5.1f}px  Δcy={err_cy:5.1f}px")

    # --- 外参对比 (每台相机的第一帧) ---
    print("\n=== 外参误差 (cam_from_ref) ===")
    ref_name = gt["ref_cam"]
    # 找参考相机的位姿
    ref_img = next((im for im in colmap_imgs if ref_name in im["name"]), None)
    if ref_img is None:
        print(f"[!] 找不到参考相机 {ref_name} 的注册结果")
        return
    R_ref = quat_to_rot(*ref_img["qvec"])
    t_ref = np.array(ref_img["tvec"])

    for cam_name in sorted(gt["intrinsics"].keys()):
        if cam_name == ref_name:
            continue
        colmap_img = next((im for im in colmap_imgs if cam_name in im["name"]), None)
        if colmap_img is None:
            print(f"  [{cam_name:18s}] 未注册")
            continue
        R_cam = quat_to_rot(*colmap_img["qvec"])
        t_cam = np.array(colmap_img["tvec"])
        # COLMAP qvec,tvec 是 world->cam. 求 cam_from_ref (用 ref 当世界):
        # R_cam_from_ref = R_cam @ R_ref^T,  T = ...
        R_c2r_actual = R_cam @ R_ref.T
        # 把 GT 也转成 world->cam 才能跟 COLMAP 的 image 对应
        # GT 里 R_cam_from_ref, T_cam_from_ref: x_cam = R x_ref + T
        R_gt = np.array(gt["extrinsics_ref_to_cam"][cam_name]["R"])
        T_gt = np.array(gt["extrinsics_ref_to_cam"][cam_name]["T"])
        # 求 GT 的 world->cam (ref 当世界): x_cam = R_gt x_w + T_gt, 所以 (R, t) = (R_gt, T_gt)
        ang_err = rotation_angle_deg(R_c2r_actual, R_gt)
        t_gt_norm = np.linalg.norm(T_gt)
        # 估计 actual T: 既然 R_c2r_actual = R_cam @ R_ref^T, 我们也能推 T
        # COLMAP tvec 是 cam 在 world(ref) 坐标系的位置
        # T_actual = t_cam - R_cam @ R_ref^T @ t_ref
        T_actual = t_cam - R_cam @ R_ref.T @ t_ref
        T_err = np.linalg.norm(T_actual - T_gt)
        T_err_pct = T_err / max(t_gt_norm, 1e-6) * 100
        print(f"  [{cam_name:18s}] R角误差={ang_err:6.2f}°  "
              f"T误差={T_err*1000:6.1f}mm ({T_err_pct:5.1f}%)  "
              f"|T_gt|={t_gt_norm*1000:.0f}mm")


if __name__ == "__main__":
    main()
