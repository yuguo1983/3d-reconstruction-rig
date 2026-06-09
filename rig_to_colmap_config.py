"""
rig_to_colmap_config.py — 把 calibrate_rig.py 的输出转成 COLMAP rig_configurator 期望的 JSON

COLMAP 期望的 schema (src/colmap/scene/rig.h:42-120):
[
  {
    "cameras": [
      {"image_prefix": "cam0_north/", "ref_sensor": true,
       "camera_model_name": "OPENCV", "camera_params": [fx, fy, cx, cy, k1, k2, p1, p2]},
      {"image_prefix": "cam1_ne/",
       "cam_from_rig_rotation": [w, x, y, z],
       "cam_from_rig_translation": [tx, ty, tz],
       "camera_model_name": "OPENCV",
       "camera_params": [fx, fy, cx, cy, k1, k2, p1, p2]},
      ...
    ]
  }
]

约定:
  - 标定输出的 R, T 是 "cam_from_ref" (x_cam = R x_ref + T)
  - 在 COLMAP 约定中, ref cam 既是 world 又是 rig 坐标系原点
  - 所以 cam_from_rig = cam_from_ref, 直接用 R, T 即可
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def rotation_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """R(3,3) -> [w, x, y, z] 归一化四元数, 与 COLMAP qvec 约定一致"""
    q = np.zeros(4)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    q /= np.linalg.norm(q)
    return [float(x) for x in q]


def intrinsic_params_from_calib(K: list[list[float]], dist: list[float]) -> list[float]:
    """COLMAP OPENCV 模型: [fx, fy, cx, cy, k1, k2, p1, p2]"""
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    k1 = dist[0] if len(dist) > 0 else 0.0
    k2 = dist[1] if len(dist) > 1 else 0.0
    p1 = dist[2] if len(dist) > 2 else 0.0
    p2 = dist[3] if len(dist) > 3 else 0.0
    return [fx, fy, cx, cy, k1, k2, p1, p2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, type=Path,
                    help="calibrate_rig.py 输出的 rig_calib.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="COLMAP rig_configurator 期望的 JSON")
    args = ap.parse_args()

    calib = json.loads(args.calib.read_text())
    ref_cam = calib["ref_cam"]
    intrinsics = calib["intrinsics"]
    extrinsics = calib["extrinsics_ref_to_cam"]  # R, T 是 cam_from_ref

    cameras_cfg = []
    for cam_name in sorted(intrinsics.keys()):
        entry: dict = {"image_prefix": f"{cam_name}/"}

        if cam_name == ref_cam:
            entry["ref_sensor"] = True
        else:
            R = np.array(extrinsics[cam_name]["R"])
            T = np.array(extrinsics[cam_name]["T"])
            entry["cam_from_rig_rotation"] = rotation_to_quat_wxyz(R)
            entry["cam_from_rig_translation"] = [float(x) for x in T]

        entry["camera_model_name"] = "OPENCV"
        entry["camera_params"] = intrinsic_params_from_calib(
            intrinsics[cam_name]["K"], intrinsics[cam_name]["dist"])

        cameras_cfg.append(entry)

    rig_config = [{"cameras": cameras_cfg}]
    args.out.write_text(json.dumps(rig_config, indent=2))
    print(f"写入 {args.out}")
    print(f"包含 {len(cameras_cfg)} 台相机, 参考相机: {ref_cam}")
    for c in cameras_cfg:
        if c.get("ref_sensor"):
            print(f"  [{c['image_prefix']}] REF")
        else:
            t = c["cam_from_rig_translation"]
            print(f"  [{c['image_prefix']}] |T|={sum(x*x for x in t)**0.5:.4f}m")


if __name__ == "__main__":
    main()
