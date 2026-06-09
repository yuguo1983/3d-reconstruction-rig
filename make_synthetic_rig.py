"""
make_synthetic_rig.py — 生成 6 路相机合成数据 (含完美 ground truth)

不需要真实硬件, 全部用 numpy + OpenCV 软件光栅化。
生成 ./synthetic/<gid>_<cam>.jpg + rig_calib.json, 文件名约定和
rig_capture.py 完全一致, 后续的 reorganize/calib/COLMAP 流程都能直接吃。

用法:
    python make_synthetic_rig.py --out ./synthetic --frames 30
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------- 相机布置 ----------

def make_hex_rig(width: int, height: int, radius: float):
    """6 台相机围成六边形, 略加 Y 方向起伏避免共面退化。
    约定: 相机坐标 +X 右, +Y 下, +Z 出 lens (OpenCV 约定),
    场景位于 z_world=0, 相机分布在 -Z 半空间 (z_world < 0),
    这样 OpenCV PnP/三角化能直接解 (z_cam > 0 表示在前方).

    物理上等价于: 把场景放在一个镜头的"前面" (z_cam > 0),
    跟 OpenCV 标定 / COLMAP 的世界坐标正向一致.
    """
    fx = fy = 1500.0
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(4)
    names = ["cam0_north", "cam1_northeast", "cam2_southeast",
             "cam3_south", "cam4_southwest", "cam5_northwest"]
    y_offsets = [0.0, 0.02, -0.02, 0.02, -0.02, 0.0]
    cams = []
    for i, name in enumerate(names):
        yaw = np.deg2rad(i * 60)
        # 相机放在 z_world < 0 半空间, 看向 +Z (即看向原点)
        cam_pos = np.array([radius * np.sin(yaw), y_offsets[i], -radius * np.cos(yaw)])
        target = np.zeros(3)
        world_up = np.array([0.0, 1.0, 0.0])
        # OpenCV look-at: z_cam = (target - cam_pos) normalized
        #   forward = (target - cam_pos).normalized()
        #   right   = forward × world_up
        #   down    = forward × right
        # 这样 (right, down, forward) 是右手系, det=+1
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R_c2w = np.column_stack([right, down, forward])
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ cam_pos
        cams.append({
            "name": name, "K": K, "dist": dist,
            "R_w2c": R_w2c, "t_w2c": t_w2c,
            "R_c2w": R_c2w, "t_c2w": cam_pos,
        })
    return cams


# ---------- 3D 场景 (用 5 个纹理平面, 构成立体感) ----------

def _rotation_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _checker(size: int, cells: int) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    s = size // cells
    for i in range(cells):
        for j in range(cells):
            c = 230 if (i + j) % 2 == 0 else 40
            img[i*s:(i+1)*s, j*s:(j+1)*s] = (c, c, c)
    return img


def _grid(size: int, cells: int) -> np.ndarray:
    img = np.full((size, size, 3), 220, dtype=np.uint8)
    s = size // cells
    for i in range(0, size, s):
        img[i, :] = 30; img[:, i] = 30
    return img


def _dots(size: int, n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 235, dtype=np.uint8)
    for _ in range(n):
        x, y = rng.integers(0, size, 2)
        col = tuple(int(c) for c in rng.integers(40, 230, 3))
        cv2.circle(img, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
    return img


def make_scene() -> list[dict]:
    """5 个纹理平面, 放置在原点附近形成 3D 立体感"""
    return [
        {"pos": [0, 0, 0], "R_local": np.eye(3),
         "size": (0.08, 0.08), "tex": _checker(256, 8)},
        {"pos": [0.025, 0.015, 0.025], "R_local": _rotation_x(np.deg2rad(25)),
         "size": (0.04, 0.04), "tex": _grid(128, 16)},
        {"pos": [-0.025, -0.015, -0.025], "R_local": _rotation_y(np.deg2rad(35)),
         "size": (0.04, 0.04), "tex": _dots(128, 250, seed=1)},
        {"pos": [0, 0.04, 0], "R_local": _rotation_x(np.deg2rad(90)),
         "size": (0.06, 0.06), "tex": _grid(128, 12)},
        {"pos": [0, -0.04, 0], "R_local": _rotation_x(np.deg2rad(-90)),
         "size": (0.06, 0.06), "tex": _dots(128, 250, seed=2)},
    ]


# ---------- ChArUco 标定板 ----------

# 这些参数必须跟 calibrate_rig.py 里的 --squares-x/--square-length 等保持一致
CHARUCO_SX = 9
CHARUCO_SY = 7
CHARUCO_SQUARE_M = 0.020
CHARUCO_MARKER_M = 0.014


def make_charuco_texture() -> np.ndarray:
    """生成 ChArUco 标定板的纹理图 (供 calibrate_rig.py 检测)

    注意: cv2.aruco.CharucoBoard 的 ID 编号约定是
      "ID 0 在 +X 一侧, 沿 +X 递增; ID 0 也在 -Y 一侧, 沿 +Y 递增"
    即世界坐标 (低 X, 低 Y) 处 = ID 0, (高 X, 高 Y) 处 = ID 47.

    而 generateImage() 的图像 y 轴是图像坐标系 (y 向下), ID 0 画在图像的顶 (y=0) 一侧.
    当纹理 (纹理 y 向下) 被透视贴到一块世界 y 向上、沿 +Y 排列的板上时:
      纹理 y=0 -> 投影到世界 y=+max 处, 也就是 (高 Y) 角
      纹理 y=H -> 投影到世界 y=-max 处, 也就是 (低 Y) 角
    但 generateImage 把 ID 0 画在 y=0 (高 Y 角), 而 getChessboardCorners() 期望 ID 0 在 (低 Y).
    这导致 ID 顺序和世界坐标 y 方向相反, PnP 会把板绕 Z 转 180°,
    进一步导致 stereoCalibrate 的基线尺度错乱 (|T| 偏差两个数量级).

    解决方案: 在生成纹理后做一次 warpAffine 让纹理 (image-y-down, ID-0-at-top)
    跟世界 (world-y-up, ID-0-at-bottom) 方向一致.
    一个简单办法: 把 ID 0 放在纹理 y=H 处, 即上下翻转, 但保留 marker 朝向.
    marker 自身是 90° 旋转不变的 (ArUco 检测器会自动处理),
    但 board 整体绕水平轴翻转 (cv2.flip(img, 0)) 会让所有 marker 上下颠倒,
    ArUco 拒绝识别.

    真正能用的办法: 生成纹理时把板绕其水平轴 (X 轴) 翻转一次, 即 4 步:
      1. 拿到 board 角点, Y 取负
      2. 用 board 自定义 generateImage 不可行 (它是 board 内部)
    替代方案: rotate board 90° in world, 改用不同的 ID 约定.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SX, CHARUCO_SY),
        CHARUCO_SQUARE_M, CHARUCO_MARKER_M, aruco_dict,
    )
    px_per_mm = 10
    img = board.generateImage(
        (int(CHARUCO_SX * CHARUCO_SQUARE_M * 1000 * px_per_mm),
         int(CHARUCO_SY * CHARUCO_SQUARE_M * 1000 * px_per_mm)))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
    return img


def make_charuco_poses(n_frames: int, seed: int = 7) -> list[dict]:
    """生成 N 个 ChArUco 板的位姿, 用于标定。
    关键约束: 板始终在 xy 平面, 不做 3D 旋转, 只做 xy 平面内平移。
    这样 warpPerspective 后 marker 仍是方形, ArUco 能稳定检测。
    6 台相机在不同位置看同一块板, 自然得到多视角约束。

    ID 方向约定: cv2.aruco.CharucoBoard.generateImage 把 ID 0 画在纹理 y=0 一侧
    (image-y-down), 而 getChessboardCorners() 期望 ID 0 在世界 (低 X, 低 Y)
    (world-y-up). 这两个 y 方向相反.
    修复: 让 size 的 h 取负, 这样 render_frame 计算 local corners 时 y 取反,
    纹理贴到世界 y=+h/2 一侧 (而不是 -h/2), ID 0 落在世界 (高 Y) 处,
    跟 generateImage 把 ID 0 画在 y=0 处的约定一致.
    """
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n_frames):
        pos = [float(rng.uniform(-0.03, 0.03)),
               float(rng.uniform(-0.03, 0.03)),
               0.0]
        R_local = np.eye(3)
        poses.append({
            "pos": pos,
            "R_local": R_local,
            # h 取负 -> render_frame 里的 [-h/2, +h/2] 变成 [+h/2, -h/2]
            "size": (CHARUCO_SX * CHARUCO_SQUARE_M,
                     -CHARUCO_SY * CHARUCO_SQUARE_M),
            "tex": make_charuco_texture(),
        })
    return poses


# ---------- 渲染 ----------

def render_frame(planes, R_turn, cam, width, height, bg) -> np.ndarray:
    """每帧把所有平面 warpPerspective 到相机平面上, 远到近画"""
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    # 计算每个平面在 cam 坐标的 z 距离
    items = []
    for p in planes:
        center_world = R_turn @ np.array(p["pos"])
        # 把 center 转到 cam 坐标: cam_z = R_w2c @ (world - cam_pos)
        center_cam = cam["R_w2c"] @ (center_world - cam["t_c2w"])
        items.append((center_cam[2], p, center_world))
    items.sort(key=lambda x: x[0])  # z 越大 = 离相机越远 = 先画
    for z_cam, p, center_world in items:
        if z_cam < 0.02:  # 离相机太近或在后
            continue
        w, h = p["size"]
        corners_local = np.array([
            [-w/2, -h/2, 0], [+w/2, -h/2, 0], [+w/2, +h/2, 0], [-w/2, +h/2, 0]
        ], dtype=np.float64)
        corners_world = (R_turn @ p["R_local"] @ corners_local.T).T + center_world
        proj = cam["K"] @ (cam["R_w2c"] @ corners_world.T + cam["t_w2c"].reshape(3, 1))
        z = proj[2]
        if np.any(z <= 0.01):
            continue
        uv = (proj[:2] / z).T
        if uv[:, 0].max() < 0 or uv[:, 1].max() < 0 or \
           uv[:, 0].min() > width or uv[:, 1].min() > height:
            continue
        tex_h, tex_w = p["tex"].shape[:2]
        # src: 纹理坐标 TL→TR→BR→BL (顺时针, y 向下)
        src = np.array([[0, 0], [tex_w, 0], [tex_w, tex_h], [0, tex_h]],
                       dtype=np.float32)
        dst = uv.astype(np.float32)
        # 若 dst 在图像里是 CCW (即"看到了板的背面"), 把 2/4 点对调强制为 CW,
        # 否则 warpPerspective 会把纹理水平翻转, ChArUco marker ID 会乱.
        def _signed_area(pts: np.ndarray) -> float:
            x = pts[:, 0]; y = pts[:, 1]
            return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if _signed_area(dst) < 0:
            dst = dst[[0, 3, 2, 1]]
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(p["tex"], M, (width, height),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        mask = (warped.sum(axis=2) > 30).astype(np.float32)[..., None]
        img = (warped * mask + img * (1 - mask)).astype(np.uint8)
    return img


# ---------- Ground truth 输出 ----------

def save_ground_truth(out_dir: Path, cams: list[dict], n_frames: int):
    intrinsics: dict = {}
    extrinsics: dict = {}
    for i, cam in enumerate(cams):
        intrinsics[cam["name"]] = {
            "K": cam["K"].tolist(),
            "dist": cam["dist"].tolist(),
            "rms_px": 0.0,
            "n_views": n_frames,
        }
        if i == 0:
            extrinsics[cam["name"]] = {"R": np.eye(3).tolist(), "T": np.zeros(3).tolist()}
        else:
            # R_cam_from_ref = R_w2c_cam @ R_ref_c2w
            R_cam_from_ref = cams[i]["R_w2c"] @ cams[0]["R_c2w"]
            T_cam_from_ref = cams[i]["R_w2c"] @ (cams[0]["t_c2w"] - cams[i]["t_c2w"])
            extrinsics[cam["name"]] = {"R": R_cam_from_ref.tolist(),
                                       "T": T_cam_from_ref.tolist()}
    calib = {
        "charuco": {"squares_x": 9, "squares_y": 7,
                    "square_length_m": 0.020, "marker_length_m": 0.014,
                    "dictionary": "DICT_6X6_250"},
        "ref_cam": cams[0]["name"],
        "intrinsics": intrinsics,
        "extrinsics_ref_to_cam": extrinsics,
    }
    (out_dir / "rig_calib.json").write_text(json.dumps(calib, indent=2))
    print(f"  ground truth: {out_dir / 'rig_calib.json'}")


# ---------- 主流程 ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("./synthetic"))
    ap.add_argument("--frames", type=int, default=30,
                    help="scene 帧数 (用于 COLMAP 重建)")
    ap.add_argument("--calib-frames", type=int, default=0,
                    help="ChArUco 标定帧数 (喂给 calibrate_rig.py)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--radius", type=float, default=0.5,
                    help="相机到原点的距离(米)")
    ap.add_argument("--bg", type=int, nargs=3, default=[60, 60, 60])
    args = ap.parse_args()

    cams = make_hex_rig(args.width, args.height, args.radius)
    scene_planes = make_scene()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"6 台相机, 半径 {args.radius}m, 图像 {args.width}x{args.height}")
    for i, cam in enumerate(cams):
        p = cam["t_c2w"]
        print(f"  cam{i} ({cam['name']:20s}) "
              f"pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})m")

    # 1) scene 帧 (用于 COLMAP 重建)
    if args.frames > 0:
        scene_dir = args.out / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[scene] 渲染 {args.frames} 帧 → {scene_dir}")
        for frame_idx in range(args.frames):
            theta = np.deg2rad(frame_idx * (360.0 / args.frames))
            R_turn = _rotation_y(theta)
            for cam in cams:
                img = render_frame(scene_planes, R_turn, cam, args.width,
                                    args.height, tuple(args.bg))
                cv2.imwrite(str(scene_dir / f"{frame_idx:05d}_{cam['name']}.jpg"),
                             img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if (frame_idx + 1) % 10 == 0 or frame_idx == args.frames - 1:
                print(f"  {frame_idx + 1}/{args.frames}")

    # 2) calib 帧 (带 ChArUco 板, 用于 calibrate_rig.py)
    if args.calib_frames > 0:
        calib_dir = args.out / "calib"
        calib_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[calib] 渲染 {args.calib_frames} 帧 ChArUco → {calib_dir}")
        calib_poses = make_charuco_poses(args.calib_frames)
        for frame_idx, board in enumerate(calib_poses):
            planes_for_render = [board]  # 只有 ChArUco 板
            for cam in cams:
                img = render_frame(planes_for_render, np.eye(3), cam,
                                    args.width, args.height, tuple(args.bg))
                cv2.imwrite(str(calib_dir / f"{frame_idx:05d}_{cam['name']}.jpg"),
                             img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if (frame_idx + 1) % 10 == 0 or frame_idx == args.calib_frames - 1:
                print(f"  {frame_idx + 1}/{args.calib_frames}")

    save_ground_truth(args.out, cams, args.frames)
    print(f"\n=== 输出 ===")
    if args.frames > 0:
        print(f"  scene:   {args.out / 'scene'}  ({args.frames}×6 张)")
    if args.calib_frames > 0:
        print(f"  calib:   {args.out / 'calib'}  ({args.calib_frames}×6 张)")
    print(f"  ground truth: {args.out / 'rig_calib.json'}")


if __name__ == "__main__":
    main()
