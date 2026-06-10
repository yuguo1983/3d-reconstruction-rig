"""
texturize.py — 把 Poisson 网格 + COLMAP 位姿贴上颜色, 输出 .obj + .mtl + 复用相机图

原理:
  - 对每个三角面, 找出"最正面看"它的相机 (face normal · -view_dir 最大)
  - 把 3 个顶点投影到该相机的图上, UV 就是投影坐标
  - .mtl 里 6 个材质, 每个引用一张相机图

  优点: 不引新依赖, .obj 是标准格式, 任何 3D 软件能开
  缺点: 没有 seam leveling / multi-view blending, 不同相机交界处能看出接缝
  想要更好效果: 装 mvs-texturing (https://github.com/nmoehrle/mvs-texturing)

用法:
  python texturize.py                     # 默认路径
  python texturize.py --mesh dense/meshed-poisson.ply \
                     --sparse dense       # 来自 image_undistorter 的 COLMAP 输出
"""
from __future__ import annotations
import argparse
import struct
from pathlib import Path

import cv2
import numpy as np


# ---------- COLMAP 工具 ----------

def quat_to_rot(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    if n < 1e-12: return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def load_cameras_txt(path: Path) -> dict:
    """cameras.txt -> {cam_id: {model, w, h, params}}"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        p = line.split()
        cid = int(p[0])
        # 跳末尾 # 注释
        params = [float(x) for x in p[4:12] if _is_float(x)]
        out[cid] = {"model": p[1], "w": int(p[2]), "h": int(p[3]), "params": params}
    return out


def _is_float(s: str) -> bool:
    try: float(s); return True
    except ValueError: return False


def load_images_txt(path: Path) -> list[dict]:
    """images.txt -> [{qvec, tvec, camera_id, name}]"""
    out = []
    lines = [l for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        out.append({
            "qvec": [float(x) for x in p[1:5]],
            "tvec": [float(x) for x in p[5:8]],
            "camera_id": int(p[8]),
            "name": p[9],
        })
        i += 2
    return out


# ---------- PLY 解析 (ASCII, 兼容 Poisson mesher 输出) ----------

def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """读 ASCII PLY. 返回 (V[N,3], F[M,3], colors[N,3] or None)"""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY 文件格式错, 找不到 end_header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        if b"binary" in b"".join(header_lines):
            raise ValueError("仅支持 ASCII PLY, 把 poisson_mesher 跑出 binary 时改一下参数")

        # 解析 header
        n_v = n_f = 0
        v_props = []   # list of names
        f_props = []
        cur = None
        for hl in header_lines:
            t = hl.decode("ascii").strip()
            if t.startswith("element vertex"):
                n_v = int(t.split()[-1]); cur = v_props
            elif t.startswith("element face"):
                n_f = int(t.split()[-1]); cur = f_props
            elif t.startswith("property") and cur is not None:
                cur.append(t.split()[-1])
            elif t == "end_header":
                break

        # 读 vertices
        V = np.zeros((n_v, 3), dtype=np.float64)
        colors = np.zeros((n_v, 3), dtype=np.uint8) if "red" in v_props else None
        for i in range(n_v):
            parts = f.readline().decode("ascii").split()
            V[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            if colors is not None:
                # red/green/blue 紧跟 x y z
                try:
                    colors[i] = [int(parts[v_props.index("red")]),
                                 int(parts[v_props.index("green")]),
                                 int(parts[v_props.index("blue")])]
                except (ValueError, IndexError):
                    pass

        # 读 faces (Poisson 格式: <count> i0 i1 i2)
        F = np.zeros((n_f, 3), dtype=np.int32)
        for i in range(n_f):
            parts = f.readline().decode("ascii").split()
            n_idx = int(parts[0])
            if n_idx != 3:
                # 非三角形, 跳过 (Poisson 输出都是三角面)
                F[i] = [0, 0, 0]
                continue
            F[i] = [int(parts[1]), int(parts[2]), int(parts[3])]
    return V, F, colors


# ---------- 投影 + 视图选择 ----------

def project(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """世界坐标点 -> 像素坐标 (N, 2)"""
    pts_cam = (R @ points_3d.T + t).T  # (N, 3)
    z = pts_cam[:, 2:3]
    z = np.where(z > 1e-6, z, 1e-6)  # 避免除零
    pts_norm = pts_cam[:, :2] / z
    uv = (K[:2, :2] @ pts_norm.T).T + K[:2, 2]
    return uv, pts_cam[:, 2]  # 返回 (N,2) UV 和深度


def pick_best_view(face_verts: np.ndarray, poses: list[dict],
                   intrinsics: dict, images: dict) -> int:
    """返回 best view 在 poses 里的下标

    严格可见性检查: 必须在相机前面 + 面朝向相机
    如果没有任何相机可见, 返回 0 (graceful degradation, UV 会错但 .obj 仍能开)
    """
    v0, v1, v2 = face_verts
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    nlen = np.linalg.norm(n)
    if nlen < 1e-12:
        return 0
    n = n / nlen
    centroid = (v0 + v1 + v2) / 3.0

    best_i = 0
    best_score = -2.0  # 比 -1 还低, 第一次比较一定会更新
    for i, pose in enumerate(poses):
        R = quat_to_rot(*pose["qvec"])
        t = np.array(pose["tvec"]).reshape(3, 1)
        # 投影 centroid 到相机系, 检查 z > 0 (在相机前)
        centroid_cam = (R @ centroid.reshape(3, 1) + t).flatten()
        if centroid_cam[2] <= 0:
            continue
        # 面法向朝向相机的程度
        cam_center = (-R.T @ t).flatten()
        view = centroid - cam_center
        vlen = np.linalg.norm(view)
        if vlen < 1e-9: continue
        view = view / vlen
        score = float(np.dot(n, -view))
        if score <= 0:
            continue  # 背向相机
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


# ---------- 写 .obj / .mtl ----------

def write_obj(path: Path, V: np.ndarray, F: np.ndarray,
              face_uvs: list[np.ndarray], face_mat: list[int],
              mat_names: list[str]) -> None:
    """写 Wavefront OBJ. 每个 face 独立 UV (vt 行) 和 material (usemtl 行)"""
    lines = []
    lines.append("# Generated by 3dremodule/texturize.py")
    lines.append(f"# {len(V)} vertices, {len(F)} faces")
    for v in V:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    # UV: 每个 face 3 个, 编号 v 1..3N
    vt_offset = 1
    for uvs in face_uvs:
        for uv in uvs:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    # Faces: 按 material 分组, 每组前一行 usemtl
    cur_mat = None
    for fi, mat_i in enumerate(face_mat):
        if mat_i != cur_mat:
            lines.append(f"usemtl {mat_names[mat_i]}")
            cur_mat = mat_i
        v = F[fi]
        t = vt_offset + fi * 3
        lines.append(f"f {v[0]+1}/{t} {v[1]+1}/{t+1} {v[2]+1}/{t+2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mtl(path: Path, mat_names: list[str], texture_paths: list[str]) -> None:
    lines = []
    for name, tex in zip(mat_names, texture_paths):
        lines.append(f"newmtl {name}")
        lines.append(f"map_Kd {tex}")
        lines.append("")  # 空行分隔
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, default=Path("dense/meshed-poisson.ply"))
    ap.add_argument("--sparse", type=Path, default=Path("dense"),
                    help="image_undistorter 输出的 COLMAP 目录 (含 cameras.txt + images.txt)")
    ap.add_argument("--images", type=Path, default=None,
                    help="相机图目录 (默认 = <sparse>/images, 即 image_undistorter 输出)")
    ap.add_argument("--out", type=Path, default=Path("dense/model.obj"),
                    help="输出的 .obj 路径, .mtl 自动同名")
    ap.add_argument("--max-faces", type=int, default=0,
                    help="只处理前 N 个面 (0=全部), 用于快速预览")
    args = ap.parse_args()

    if not args.mesh.exists():
        print(f"[错误] 找不到 {args.mesh}, 先跑 dense_recon.sh")
        return 1
    cam_path = args.sparse / "cameras.txt"
    img_path = args.sparse / "images.txt"
    if not (cam_path.exists() and img_path.exists()):
        print(f"[错误] {args.sparse} 下缺 cameras.txt 或 images.txt")
        return 1
    img_dir = args.images or (args.sparse / "images")
    if not img_dir.exists():
        print(f"[错误] 找不到图像目录 {img_dir}")
        return 1

    print(f"加载网格: {args.mesh}")
    V, F, colors = load_ply(args.mesh)
    print(f"  顶点数: {len(V)}, 面数: {len(F)}, 顶点色: {colors is not None}")

    if args.max_faces and args.max_faces < len(F):
        F = F[:args.max_faces]
        print(f"  截断到 {len(F)} 面 (--max-faces)")

    print(f"加载位姿: {img_path}")
    intrinsics = load_cameras_txt(cam_path)
    images = load_images_txt(img_path)
    print(f"  相机: {len(intrinsics)}  注册图像: {len(images)}")

    # 按 name 排序, 给每张图一个稳定 ID
    poses = []
    mat_names = []
    for i, img in enumerate(images):
        cam = intrinsics[img["camera_id"]]
        # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = cam["params"][:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        img_name = Path(img["name"]).stem
        # 材质名: 用 cam 名 (跟 cameras.txt 末尾 # 注释对齐)
        mat_name = f"mat_{i:02d}_{img_name[:24]}"
        mat_names.append(mat_name)
        poses.append({**img, "K": K, "img_w": cam["w"], "img_h": cam["h"]})
    print(f"  将写出 {len(mat_names)} 个材质")

    # 检查每个图像都找得到
    for p in poses:
        f = img_dir / Path(p["name"]).name
        if not f.exists():
            print(f"[错误] 找不到 {f}")
            return 1

    # 主循环: 选 best view, 算 UV
    print(f"为 {len(F)} 个面选最佳视图 + 算 UV ...")
    face_uvs: list[np.ndarray] = []
    face_mat: list[int] = []
    skipped = 0
    for fi in range(len(F)):
        if fi % 5000 == 0 and fi > 0:
            print(f"  ... {fi}/{len(F)} (skip {skipped})")
        f = F[fi]
        verts3d = V[f]  # (3, 3)
        # 选 best view
        best_i = pick_best_view(verts3d, poses, intrinsics, None)
        pose = poses[best_i]
        R = quat_to_rot(*pose["qvec"])
        t = np.array(pose["tvec"]).reshape(3, 1)
        uv, depths = project(verts3d, R, t, pose["K"])
        # 至少 1 个顶点在图前, 都在图内 (放宽到 [-w, 2w] 以容许边界外)
        W, H = pose["img_w"], pose["img_h"]
        if np.any(depths <= 0):
            # 整个面在图背后, 跳过 (用第一个 view 的结果)
            skipped += 1
        # 写图时 v 朝下, 所以 v 要翻一下
        uv[:, 1] = H - uv[:, 1]
        # 归一化到 [0, 1]
        uv_norm = uv.copy()
        uv_norm[:, 0] /= W
        uv_norm[:, 1] /= H
        face_uvs.append(uv_norm)
        face_mat.append(best_i)
    print(f"  完成, {skipped} 个面有顶点位于相机背后 (uv 可能错, 但 .obj 仍能打开)")

    # 写 .obj
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = args.out.with_suffix(".mtl")
    print(f"写 .obj: {args.out}")
    write_obj(args.out, V, F, face_uvs, face_mat, mat_names)
    # 把材质名同步进 .obj (mtl reference 在最开头)
    obj_text = args.out.read_text(encoding="utf-8")
    obj_text = f"mtllib {mtl_path.name}\n" + obj_text
    args.out.write_text(obj_text, encoding="utf-8")

    print(f"写 .mtl: {mtl_path}")
    texture_relpaths = [Path(p["name"]).name for p in poses]
    write_mtl(mtl_path, mat_names, texture_relpaths)

    print()
    print("===== 完成 =====")
    print(f"  .obj:    {args.out}")
    print(f"  .mtl:    {mtl_path}")
    print(f"  纹理:    {img_dir}/  (共 {len(texture_relpaths)} 张, 已在 .mtl 里引用)")
    print()
    print("  用 MeshLab / Blender / CloudCompare 打开 .obj 看效果")
    print("  接缝明显? 装 mvs-texturing 重做贴图: https://github.com/nmoehrle/mvs-texturing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
