"""
calibrate_rig.py — 6 路固定相机联合标定

输入: rig_capture.py 输出的 ./captures/<session>/<gid>_<cam>.jpg
输出:
  - rig_calib.json     完整标定(内参+外参+对应点)
  - cameras.txt        COLMAP 格式相机内参
  - rig_layout.png     6 台相机的 3D 布局图(便于检视)

依赖: pip install opencv-python numpy matplotlib
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


# ---------- ChArUco 配置 ----------

@dataclass
class CharucoConfig:
    squares_x: int = 9
    squares_y: int = 7
    square_length_m: float = 0.020
    marker_length_m: float = 0.014
    dictionary: str = "DICT_6X6_250"


def make_charuco(cfg: CharucoConfig):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    board = cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y),
        cfg.square_length_m, cfg.marker_length_m, aruco_dict,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def detect_charuco(detector, img) -> tuple[np.ndarray | None, np.ndarray | None]:
    ch_corners, ch_ids, _, _ = detector.detectBoard(img)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None
    return ch_corners, ch_ids.reshape(-1, 1).astype(np.int32)


# ---------- 文件组织 ----------

def discover_sessions(captures_root: Path) -> list[Path]:
    sessions = sorted([p for p in captures_root.iterdir() if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"{captures_root} 下没有 session 子目录")
    return sessions


def group_captures(session_dir: Path) -> list[dict[str, Path]]:
    """把 00001_cam0_north.jpg 这种命名按 gid 聚合。"""
    groups: dict[int, dict[str, Path]] = {}
    for p in session_dir.glob("*.jpg"):
        parts = p.stem.split("_", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        gid = int(parts[0])
        groups.setdefault(gid, {})[parts[1]] = p
    return [groups[k] for k in sorted(groups)]


# ---------- 标定求解 ----------

def collect_observations(groups, cam_names, board, detector, min_corners):
    """per_cam_obs[cam] = list of (objpoints(N,3), imgpoints(N,2), ids(N,))

    保留 charuco id, 这样外参阶段可以按 id 对齐, 而不是按 frame index 对齐
    """
    per_cam: dict[str, list] = {c: [] for c in cam_names}
    all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    used = 0
    for group in groups:
        per_frame: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        ok = True
        for cam, path in group.items():
            img = cv2.imread(str(path))
            if img is None:
                ok = False; break
            ch_corners, ch_ids = detect_charuco(detector, img)
            if ch_corners is None or len(ch_corners) < min_corners:
                ok = False; break
            obj = all_obj[ch_ids.flatten()]
            per_frame[cam] = (obj, ch_corners.reshape(-1, 2).astype(np.float32),
                              ch_ids.flatten())
        if not ok or len(per_frame) != len(cam_names):
            continue
        for cam, (o, i, ids) in per_frame.items():
            per_cam[cam].append((o, i, ids))
        used += 1
    return per_cam, used


def calibrate_intrinsics(per_cam, image_size):
    out = {}
    for cam, obs in per_cam.items():
        if len(obs) < 3:
            print(f"[warn] {cam} 观测帧太少({len(obs)}), 跳过内参")
            continue
        obj = [o for o, _, _ in obs]
        img = [i for _, i, _ in obs]
        w, h = image_size
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (w, h), None, None)
        # RMS 重投影误差
        errs = []
        for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
            errs.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - i) ** 2))))
        out[cam] = {"K": K.tolist(), "dist": dist.flatten().tolist(),
                    "rms_px": float(np.mean(errs)), "n_views": len(obs)}
        print(f"[内参] {cam}: rms={out[cam]['rms_px']:.3f}px, views={len(obs)}")
    return out


def calibrate_extrinsics(per_cam, intrinsics, ref_cam, image_size):
    """对每台相机 vs ref_cam 做 stereoCalibrate(固定内参)
    按 frame index 取共同帧, 再按 charuco id 对齐 object/image 点
    """
    extr = {ref_cam: (np.eye(3), np.zeros((3, 1)))}
    K_ref = np.array(intrinsics[ref_cam]["K"])
    d_ref = np.array(intrinsics[ref_cam]["dist"])
    obs_ref = per_cam[ref_cam]
    for cam, obs in per_cam.items():
        if cam == ref_cam:
            continue
        # 按 frame index 对齐, 再按 charuco id 选公共点
        per_frame_objs = []
        per_frame_i_ref = []
        per_frame_i_cam = []
        for (o_ref, i_ref, ids_ref), (o_cam, i_cam, ids_cam) in zip(obs_ref, obs):
            common_ids, idx_r, idx_c = np.intersect1d(ids_ref, ids_cam, return_indices=True)
            if len(common_ids) < 6:
                continue
            per_frame_objs.append(o_ref[idx_r])
            per_frame_i_ref.append(i_ref[idx_r])
            per_frame_i_cam.append(i_cam[idx_c])
        if len(per_frame_objs) < 5:
            print(f"[warn] {cam} vs {ref_cam} 共同帧仅 {len(per_frame_objs)}, 跳过")
            continue
        K_cam = np.array(intrinsics[cam]["K"])
        d_cam = np.array(intrinsics[cam]["dist"])
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            per_frame_objs, per_frame_i_ref, per_frame_i_cam,
            K_ref, d_ref, K_cam, d_cam, image_size,
            criteria=crit, flags=cv2.CALIB_FIX_INTRINSIC,
        )
        extr[cam] = (R, T)
        print(f"[外参] {ref_cam}->{cam}: stereo rms={ret:.3f}px, "
              f"|T|={np.linalg.norm(T):.4f}m, common_frames={len(per_frame_objs)}")
    return extr


# ---------- COLMAP 工具 ----------

def rotation_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """R(3,3) -> (w,x,y,z) 归一化四元数, 与 COLMAP qvec 约定一致"""
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
    return q / np.linalg.norm(q)


def write_colmap_cameras(path, intrinsics, w, h):
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(intrinsics)}",
    ]
    for i, (name, v) in enumerate(intrinsics.items(), 1):
        K, d = v["K"], v["dist"]
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        k1 = d[0] if len(d) > 0 else 0.0
        k2 = d[1] if len(d) > 1 else 0.0
        p1 = d[2] if len(d) > 2 else 0.0
        p2 = d[3] if len(d) > 3 else 0.0
        lines.append(
            f"{i} OPENCV {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} "
            f"{k1:.6f} {k2:.6f} {p1:.6f} {p2:.6f}  # {name}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def pose_cam_in_ref_world(R_ref_to_cam, T_ref_to_cam):
    """
    OpenCV stereoCalibrate 返回的 R, T 表示:
        x_cam = R * x_ref + T       (即把 ref 系的点变换到 cam 系)
    把它反推为 cam 在 ref(world) 系的相机到世界位姿:
        x_world = R_c2w * x_cam + T_c2w
    """
    R_c2w = R_ref_to_cam.T
    T_c2w = -R_c2w @ T_ref_to_cam.flatten()
    return R_c2w, T_c2w


def write_colmap_images_one_frame(path, cam_names, ref_cam, intrinsics, extr, group):
    """为 calibration 的第一组 capture 写 images.txt, 验证用"""
    name_to_id = {c: i + 1 for i, c in enumerate(cam_names)}
    pose = {}
    for cam, (R, T) in extr.items():
        R_c2w, T_c2w = pose_cam_in_ref_world(R, T)
        pose[cam] = (R_c2w, T_c2w)
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        f"# Number of images: {len(cam_names)}",
    ]
    for i, cam in enumerate(cam_names, 1):
        R, T = pose[cam]
        q = rotation_to_quat_wxyz(R)
        lines.append(
            f"{i} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
            f"{T[0]:.6f} {T[1]:.6f} {T[2]:.6f} {name_to_id[cam]} {group[cam].name}"
        )
        lines.append("")  # 空 points2d
    Path(path).write_text("\n".join(lines) + "\n")


# ---------- 3D 布局可视化 ----------

def plot_rig_layout(intrinsics, extr, ref_cam, out_path):
    """俯视 3D 布局, 便于目视检查各相机的相对位置"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cam, (R, T) in extr.items():
        R_c2w, T_c2w = pose_cam_in_ref_world(R, T)
        # Z 轴方向(相机光轴在世界系的方向): 取 R_c2w 的第三列的相反数
        forward = -R_c2w[:, 2]
        ax.scatter(*T_c2w, s=80, label=cam)
        ax.quiver(*T_c2w, *forward, length=0.05, arrow_length_ratio=0.3)
    ax.scatter(0, 0, 0, c="red", s=120, marker="*", label=ref_cam + " (origin)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Rig 布局 (世界系 = 参考相机)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[viz] 3D 布局写入 {out_path}")


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--session", type=str, default=None)
    ap.add_argument("--ref-cam", type=str, default=None)
    ap.add_argument("--out", type=Path, default=Path("./calib"))
    ap.add_argument("--squares-x", type=int, default=9)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-length", type=float, default=0.020)
    ap.add_argument("--marker-length", type=float, default=0.014)
    ap.add_argument("--dict", type=str, default="DICT_6X6_250")
    ap.add_argument("--min-corners", type=int, default=8)
    args = ap.parse_args()

    charuco_cfg = CharucoConfig(
        args.squares_x, args.squares_y,
        args.square_length, args.marker_length, args.dict,
    )
    board, detector = make_charuco(charuco_cfg)
    print(f"ChArUco: {charuco_cfg.squares_x}x{charuco_cfg.squares_y}, "
          f"square={charuco_cfg.square_length_m*1000:.0f}mm, "
          f"marker={charuco_cfg.marker_length_m*1000:.0f}mm")

    sessions = discover_sessions(args.captures)
    session = sessions[-1] if args.session is None else (args.captures / args.session)
    if not session.exists():
        raise FileNotFoundError(session)
    print(f"使用 session: {session}")

    groups = group_captures(session)
    print(f"发现 {len(groups)} 组 capture")
    cam_names = sorted(groups[0].keys())
    if args.ref_cam is None:
        args.ref_cam = cam_names[0]
    print(f"相机: {cam_names}, 参考: {args.ref_cam}")

    sample = cv2.imread(str(groups[0][args.ref_cam]))
    h, w = sample.shape[:2]
    print(f"图像尺寸: {w}x{h}")

    per_cam, used = collect_observations(groups, cam_names, board, detector, args.min_corners)
    print(f"用于标定的帧: {used}")
    for c, obs in per_cam.items():
        print(f"  {c}: {len(obs)} 帧")

    args.out.mkdir(parents=True, exist_ok=True)
    intrinsics = calibrate_intrinsics(per_cam, (w, h))
    extr = calibrate_extrinsics(per_cam, intrinsics, args.ref_cam, (w, h))

    # 写文件
    with open(args.out / "rig_calib.json", "w") as f:
        json.dump({
            "charuco": asdict(charuco_cfg),
            "ref_cam": args.ref_cam,
            "intrinsics": intrinsics,
            "extrinsics_ref_to_cam": {
                c: {"R": R.tolist(), "T": T.flatten().tolist()}
                for c, (R, T) in extr.items()
            },
            "extrinsics_cam_in_ref_world": {
                c: {"R": pose_cam_in_ref_world(R, T)[0].tolist(),
                    "T": pose_cam_in_ref_world(R, T)[1].tolist()}
                for c, (R, T) in extr.items()
            },
        }, f, indent=2)
    print(f"完整标定写入 {args.out / 'rig_calib.json'}")

    write_colmap_cameras(args.out / "cameras.txt", intrinsics, w, h)
    write_colmap_images_one_frame(
        args.out / "images.txt", cam_names, args.ref_cam, intrinsics, extr, groups[0])
    plot_rig_layout(intrinsics, extr, args.ref_cam, args.out / "rig_layout.png")
    print(f"全部输出: {args.out}")


if __name__ == "__main__":
    main()
