"""
rig_capture.py — 6 路 UVC 摄像头同步采集（软件时间戳聚类）

用法:
    python rig_capture.py --config rig.json --count 10 --output ./captures

依赖: pip install opencv-python numpy
"""
from __future__ import annotations
import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CamConfig:
    index: int
    name: str
    width: int = 1920
    height: int = 1080
    fps: int = 30
    exposure: Optional[float] = -7.0   # 负值=手动(log2秒); None=保持原状
    focus: Optional[int] = None        # 0-255; None=保持自动


@dataclass
class RigConfig:
    cameras: list[CamConfig]
    sync_window_ms: float = 10.0       # 一组内允许的最大时间戳偏差
    output_dir: str = "./captures"
    backend: str = "auto"              # auto / v4l2 / msmf / dshow


@dataclass
class TimestampedFrame:
    cam_name: str
    timestamp_ns: int
    frame: np.ndarray


class FrameGrabber(threading.Thread):
    """一相机一线程：抓帧、打 monotonic 时间戳、塞入有界队列。"""
    def __init__(self, cam: CamConfig, backend_id: int,
                 out_q: queue.Queue, stop_evt: threading.Event):
        super().__init__(daemon=True, name=f"grab-{cam.name}")
        self.cam = cam
        self.backend_id = backend_id
        self.out_q = out_q
        self.stop_evt = stop_evt
        self.dropped = 0

    def run(self) -> None:
        cap = cv2.VideoCapture(self.cam.index, self.backend_id)
        # 强制 MJPEG：UVC 默认压缩格式，1080p 30fps 单路 ~30Mbps
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam.height)
        cap.set(cv2.CAP_PROP_FPS, self.cam.fps)
        if self.cam.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.cam.exposure)
        if self.cam.focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, self.cam.focus)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if not cap.isOpened():
            sys.stderr.write(f"[{self.cam.name}] 无法打开 index={self.cam.index}\n")
            return
        aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (aw, ah) != (self.cam.width, self.cam.height):
            sys.stderr.write(f"[{self.cam.name}] 实际分辨率 {aw}x{ah} != {self.cam.width}x{self.cam.height}\n")
        # 头 10 帧多半是自动曝光收敛中或编码器未就绪，直接丢
        warmup_left = 10
        while not self.stop_evt.is_set():
            ok, img = cap.read()
            if not ok:
                self.dropped += 1
                continue
            if warmup_left > 0:
                warmup_left -= 1
                continue
            ts = time.monotonic_ns()
            try:
                self.out_q.put_nowait(TimestampedFrame(self.cam.name, ts, img))
            except queue.Full:
                # 消费方慢于生产方，丢老帧保新鲜
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
                self.dropped += 1
        cap.release()


def _select_backend(name: str) -> int:
    if name == "v4l2":  return cv2.CAP_V4L2
    if name == "msmf":  return cv2.CAP_MSMF
    if name == "dshow": return cv2.CAP_DSHOW
    return 0  # auto: 平台默认


class RigCapture:
    def __init__(self, config: RigConfig):
        self.config = config
        # 队列深度 2：保留最新两帧，供同步器二选一
        self.queues: dict[str, queue.Queue] = {
            c.name: queue.Queue(maxsize=2) for c in config.cameras
        }
        self.stop_evt = threading.Event()
        self.backend = _select_backend(config.backend)
        self.grabbers: list[FrameGrabber] = []

    def __enter__(self) -> "RigCapture":
        for cam in self.config.cameras:
            g = FrameGrabber(cam, self.backend, self.queues[cam.name], self.stop_evt)
            g.start()
            self.grabbers.append(g)
        time.sleep(0.5)  # 等 grabber 全部就绪
        return self

    def __exit__(self, *exc) -> None:
        self.stop_evt.set()
        for g in self.grabbers:
            g.join(timeout=2.0)

    def capture_group(self, timeout: float = 2.0) -> Optional[list[TimestampedFrame]]:
        """
        抓一组 6 路帧。
        算法: 每相机各自保留"最新一帧"；只要所有最新帧的时间戳都落在同一
        sync_window_ms 内就接受，否则持续把最老相机的最新帧往下推。
        """
        deadline = time.monotonic() + timeout
        window_ns = int(self.config.sync_window_ms * 1e6)
        latest: dict[str, TimestampedFrame] = {}

        while time.monotonic() < deadline:
            for name, q in self.queues.items():
                while True:
                    try:
                        latest[name] = q.get_nowait()
                    except queue.Empty:
                        break

            if len(latest) < len(self.queues):
                time.sleep(0.005)
                continue

            ts_sorted = sorted(f.timestamp_ns for f in latest.values())
            if ts_sorted[-1] - ts_sorted[0] <= window_ns:
                return [latest[c.name] for c in self.config.cameras]

            # 有人落后太多：等他追上来
            oldest_name = min(latest, key=lambda n: latest[n].timestamp_ns)
            try:
                latest[oldest_name] = self.queues[oldest_name].get(timeout=0.5)
            except queue.Empty:
                return None
        return None


def save_group(group: list[TimestampedFrame], out_dir: Path, gid: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in group:
        path = out_dir / f"{gid:05d}_{f.cam_name}.jpg"
        cv2.imwrite(str(path), f.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def load_config(path: Path) -> RigConfig:
    raw = json.loads(path.read_text())
    raw["cameras"] = [CamConfig(**c) for c in raw["cameras"]]
    return RigConfig(**raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--count", type=int, default=1, help="采集组数")
    ap.add_argument("--output", type=Path, default=Path("./captures"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    session = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output / session
    captured = 0
    fails = 0

    with RigCapture(cfg) as rig:
        while captured < args.count:
            group = rig.capture_group(timeout=2.0)
            if group is None:
                fails += 1
                if fails > args.count * 5:
                    sys.stderr.write("连续同步失败，检查相机/带宽\n")
                    break
                continue
            save_group(group, out_dir, captured)
            captured += 1
            span_ms = (max(f.timestamp_ns for f in group)
                       - min(f.timestamp_ns for f in group)) / 1e6
            print(f"[{captured:04d}/{args.count}] 组内时间跨度 = {span_ms:.2f} ms")

    print(f"输出: {out_dir}")


if __name__ == "__main__":
    main()
