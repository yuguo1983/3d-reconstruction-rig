"""
probe_cameras.py — 逐个打开 Windows camera index, 抓帧存出来, 方便标定每个 index 对应哪台相机

用法:
    python probe_cameras.py                          # 扫 index 0-7
    python probe_cameras.py --max-index 15           # 扫 0-15
    python probe_cameras.py --width 1920 --height 1080  # 强制分辨率
    python probe_cameras.py --backend dshow          # Windows 推荐
    python probe_cameras.py --warmup 2.0             # 等相机稳定更久

输出:
    probe_index_N.jpg   每张左上角烧入了 index 编号 + 实际分辨率 + 估算 FPS
    probe_index_N.json  (可选) 详细参数

操作流程:
    1. 跑这个脚本, 6 张图存到 ./probe/
    2. 在每张图前贴个标签 (或用笔在镜头旁写编号), 知道哪个 index = 物理哪个 cam
    3. 把对应关系填进 rig.json 的 cameras 数组 (--index 字段)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import cv2


def _select_backend(name: str) -> int:
    if name == "dshow": return cv2.CAP_DSHOW
    if name == "msmf":  return cv2.CAP_MSMF
    if name == "v4l2":  return cv2.CAP_V4L2
    return 0  # auto


def probe_one(index: int, backend: int, w: int, h: int, fps: int,
              warmup: float, out_dir: Path) -> dict | None:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 等待相机稳定 (自动曝光收敛)
    time.sleep(warmup)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # 测一下实际能达到的帧率
    n_test = 30
    t0 = time.monotonic()
    ok_count = 0
    for _ in range(n_test):
        ok, _ = cap.read()
        if ok: ok_count += 1
    dt = time.monotonic() - t0
    measured_fps = ok_count / dt if dt > 0 else 0.0

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"index": index, "opened": True, "grab_ok": False}

    # 烧入 index 编号 + 实际参数
    label = f"index={index}  {actual_w}x{actual_h}  fps={measured_fps:.1f} (set {actual_fps:.0f})"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    out_path = out_dir / f"probe_index_{index:02d}.jpg"
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "index": index,
        "opened": True,
        "grab_ok": True,
        "set": {"w": w, "h": h, "fps": fps},
        "actual": {"w": actual_w, "h": actual_h, "fps_set": actual_fps,
                   "fps_measured": measured_fps},
        "saved_to": str(out_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=7,
                    help="扫描 index 范围 [0, max_index)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--backend", type=str, default="dshow",
                    choices=["auto", "dshow", "msmf", "v4l2"])
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="每台相机稳定等待秒数")
    ap.add_argument("--out", type=Path, default=Path("./probe"))
    ap.add_argument("--json", type=Path, default=None,
                    help="把结果写成 JSON, 默认 <out>/probe_summary.json")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    backend = _select_backend(args.backend)
    print(f"Backend: {args.backend} ({backend})")
    print(f"扫描 index 0..{args.max_index-1}, 目标 {args.width}x{args.height}@{args.fps}fps")
    print(f"输出目录: {args.out}")
    print(f"暖机 {args.warmup}s/相机\n")

    results: list[dict] = []
    for i in range(args.max_index):
        print(f"[index {i:2d}] 打开中 ...", end=" ", flush=True)
        r = probe_one(i, backend, args.width, args.height, args.fps,
                      args.warmup, args.out)
        if r is None:
            print("无法打开 (没相机? 或被其他程序占用)")
            results.append({"index": i, "opened": False})
        elif not r.get("grab_ok", True):
            print("打开了但抓不到帧")
            results.append(r)
        else:
            a = r["actual"]
            print(f"OK  {a['w']}x{a['h']}  set fps={a['fps_set']:.0f}  实测={a['fps_measured']:.1f}")
            results.append(r)

    json_path = args.json or (args.out / "probe_summary.json")
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n汇总写入: {json_path}")

    # 报告
    opened = [r for r in results if r.get("grab_ok")]
    print(f"\n=== 摘要 ===")
    print(f"  扫了 {args.max_index} 个 index, 成功打开 {len(opened)} 台")
    if opened:
        low_fps = [r for r in opened if r["actual"]["fps_measured"] < args.fps * 0.8]
        if low_fps:
            print(f"  ! 以下相机帧率 < 目标 {args.fps*0.8:.0f} fps, 可能是 USB 带宽撞了:")
            for r in low_fps:
                print(f"     index {r['index']:2d}:  {r['actual']['fps_measured']:.1f} fps")
            print(f"    解决: 换 USB 插口, 或加独立 USB3.0 控制器")
        else:
            print(f"  OK 全部相机帧率达标 (>= {args.fps*0.8:.0f} fps)")

    print(f"\n下一步: 打开 {args.out}/ 下所有 probe_index_N.jpg,")
    print(f"       在每台相机物理位置贴上对应 index 标签,")
    print(f"       然后把 index 填进 rig.json 的 cameras 数组.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
