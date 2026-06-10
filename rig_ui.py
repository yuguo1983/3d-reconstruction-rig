"""
rig_ui.py — PyQt5 桌面端, 6 路 UVC 实时预览 + 一键同步抓帧

用法:
    python rig_ui.py [--config rig.json]

依赖: PyQt5, opencv-python (复用 rig_capture.py 的同步抓帧逻辑)

设计:
    - QtFrameGrabber: 继承自 FrameGrabber, 多发一个 frame_ready Qt signal 供预览
    - QtRigCapture: 继承自 RigCapture, __enter__ 用 QtFrameGrabber 替换 grabber
    - CamTile: 一个相机的预览格 (QLabel 子类), 收 signal 画图
    - MainWindow: 2x3 网格 + 工具栏 + 抓帧控制 + 日志
"""
from __future__ import annotations
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from rig_capture import (
    CamConfig, FrameGrabber, RigCapture, TimestampedFrame,
    _select_backend, load_config, save_group,
)


# ---------- Qt 化的 grabber / rig ----------

class QtFrameGrabber(FrameGrabber, QtCore.QObject):
    """在父类行为基础上多发一个 Qt signal, 供 UI 实时预览。

    多继承 threading.Thread + QObject, 这样既能 run() 又能 emit signal。
    注意: emit 时传 img.copy(), 因为 cv2.VideoCapture.read() 会复用 buffer,
    GUI 线程异步消费时如果不 copy 就会读到下一帧的脏数据。
    """
    frame_ready = QtCore.pyqtSignal(str, np.ndarray, int)  # name, bgr, ts_ns

    def __init__(self, cam: CamConfig, backend_id: int,
                 out_q: queue.Queue, stop_evt: threading.Event):
        FrameGrabber.__init__(self, cam, backend_id, out_q, stop_evt)
        QtCore.QObject.__init__(self)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.cam.index, self.backend_id)
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
        warmup_left = 10  # 前 10 帧是自动曝光 / 编码器收敛
        while not self.stop_evt.is_set():
            ok, img = cap.read()
            if not ok:
                self.dropped += 1
                continue
            if warmup_left > 0:
                warmup_left -= 1
                continue
            ts = time.monotonic_ns()
            # 1) 喂给 capture_group 的同步队列
            try:
                self.out_q.put_nowait(TimestampedFrame(self.cam.name, ts, img.copy()))
            except queue.Full:
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
                self.dropped += 1
            # 2) 发给 UI 预览
            self.frame_ready.emit(self.cam.name, img.copy(), ts)
        cap.release()


class QtRigCapture(RigCapture):
    """__enter__ 换成 QtFrameGrabber, capture_group / __exit__ 完全沿用父类。"""

    def __enter__(self) -> "QtRigCapture":
        for cam in self.config.cameras:
            g = QtFrameGrabber(cam, self.backend, self.queues[cam.name], self.stop_evt)
            g.start()
            self.grabbers.append(g)
        time.sleep(0.5)  # 等 grabber 全部就绪
        return self


# ---------- 单个相机的预览格 ----------

class CamTile(QtWidgets.QLabel):
    def __init__(self, cam: CamConfig, parent=None):
        super().__init__(parent)
        self.cam = cam
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background:#000; color:#fff;")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText(f"{cam.name}\n(等待预览)")
        self._fps = 0.0
        self._last_ts = 0
        self._fps_acc_ns = 0  # 累计 dt
        self._fps_count = 0
        self._frame_count = 0
        self._pixmap: QtGui.QPixmap | None = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def reset_stats(self) -> None:
        self._fps = 0.0
        self._last_ts = 0
        self._fps_acc_ns = 0
        self._fps_count = 0
        self._frame_count = 0

    @QtCore.pyqtSlot(str, np.ndarray, int)
    def on_frame(self, name: str, bgr: np.ndarray, ts: int) -> None:
        if name != self.cam.name:
            return
        # FPS: 30 帧滑窗平均
        if self._last_ts:
            dt = ts - self._last_ts
            if dt > 0:
                self._fps_acc_ns += dt
                self._fps_count += 1
                if self._fps_count >= 30:
                    self._fps = self._fps_count * 1e9 / self._fps_acc_ns
                    self._fps_acc_ns = 0
                    self._fps_count = 0
        self._last_ts = ts
        self._frame_count += 1
        # BGR -> RGB, 加文字 overlay
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        painter = QtGui.QPainter(img)
        painter.setPen(QtGui.QColor(0, 255, 0))
        painter.setFont(QtGui.QFont("Consolas", 14, QtGui.QFont.Bold))
        painter.drawText(8, 24, f"{self.cam.name}  {self._fps:4.1f} fps")
        painter.drawText(8, h - 10, f"frame {self._frame_count}")
        painter.end()
        # 缩放到 label 尺寸, 保留长宽比
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pix)


# ---------- 主窗口 ----------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_config: Path | None):
        super().__init__()
        self.setWindowTitle("3dremodule — 6 路相机控制台")
        self.resize(1400, 900)

        self.cfg: RigConfig | None = None
        self.rig: QtRigCapture | None = None
        self.session_dir: Path | None = None
        self.captured_in_session = 0
        self._session_seq = 0  # 防止同秒内点 [新会话] 撞到同一目录

        self._build_ui()
        if initial_config and initial_config.exists():
            self.cfg_edit.setText(str(initial_config))
            self.on_load_config()

    # ----- UI 布局 -----
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Config row
        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.addWidget(QtWidgets.QLabel("Config:"))
        self.cfg_edit = QtWidgets.QLineEdit("rig.json")
        cfg_edit_w = QtWidgets.QWidget(); cfg_edit_w.setMinimumWidth(280)
        cfg_edit_wl = QtWidgets.QHBoxLayout(cfg_edit_w); cfg_edit_wl.setContentsMargins(0,0,0,0)
        cfg_edit_wl.addWidget(self.cfg_edit)
        cfg_row.addWidget(cfg_edit_w)
        self.load_btn = QtWidgets.QPushButton("加载")
        self.load_btn.clicked.connect(self.on_load_config)
        cfg_row.addWidget(self.load_btn)
        cfg_row.addSpacing(20)
        cfg_row.addWidget(QtWidgets.QLabel("同步窗口 (ms):"))
        self.sync_spin = QtWidgets.QDoubleSpinBox()
        self.sync_spin.setRange(0.5, 1000.0)
        self.sync_spin.setValue(10.0)
        self.sync_spin.setSingleStep(1.0)
        cfg_row.addWidget(self.sync_spin)
        cfg_row.addSpacing(20)
        cfg_row.addWidget(QtWidgets.QLabel("输出目录:"))
        self.out_edit = QtWidgets.QLineEdit("./captures")
        cfg_row.addWidget(self.out_edit, 1)
        root.addLayout(cfg_row)

        # 2x3 camera grid
        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(4)
        self.tiles: dict[str, CamTile] = {}
        for i in range(6):
            tile = CamTile(CamConfig(index=i, name=f"cam{i}"))
            self.tiles[f"cam{i}"] = tile
            self.grid.addWidget(tile, i // 3, i % 3)
        grid_box = QtWidgets.QGroupBox("预览")
        grid_box.setLayout(self.grid)
        root.addWidget(grid_box, 1)

        # Capture controls
        cap_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶ 开始预览")
        self.start_btn.clicked.connect(self.on_start)
        cap_row.addWidget(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton("■ 停止")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        cap_row.addWidget(self.stop_btn)
        cap_row.addSpacing(20)
        cap_row.addWidget(QtWidgets.QLabel("目标组数:"))
        self.target_spin = QtWidgets.QSpinBox()
        self.target_spin.setRange(1, 9999)
        self.target_spin.setValue(10)
        cap_row.addWidget(self.target_spin)
        self.capture_btn = QtWidgets.QPushButton("📸 抓 N 组")
        self.capture_btn.clicked.connect(self.on_capture_n)
        self.capture_btn.setEnabled(False)
        cap_row.addWidget(self.capture_btn)
        self.single_btn = QtWidgets.QPushButton("📷 单组抓拍")
        self.single_btn.clicked.connect(self.on_capture_one)
        self.single_btn.setEnabled(False)
        cap_row.addWidget(self.single_btn)
        self.new_session_btn = QtWidgets.QPushButton("↻ 新会话")
        self.new_session_btn.clicked.connect(self.on_new_session)
        cap_row.addWidget(self.new_session_btn)
        cap_row.addStretch(1)
        self.status_label = QtWidgets.QLabel("0/0  |  最近同步: --- ms")
        self.status_label.setStyleSheet("font-family: Consolas;")
        cap_row.addWidget(self.status_label)
        root.addLayout(cap_row)

        # Log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(160)
        self.log.setStyleSheet("font-family: Consolas; font-size: 11px;")
        root.addWidget(self.log)

        self.statusBar().showMessage("就绪. 先点 [加载] 选 rig.json, 再点 [开始预览]")

    # ----- helpers -----
    def log_msg(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def _rebuild_tiles(self, cams: list[CamConfig]) -> None:
        # 清空旧 tile
        for i in reversed(range(self.grid.count())):
            w = self.grid.itemAt(i).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.tiles.clear()
        for i, cam in enumerate(cams):
            tile = CamTile(cam)
            self.tiles[cam.name] = tile
            self.grid.addWidget(tile, i // 3, i % 3)

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.capture_btn.setEnabled(running)
        self.single_btn.setEnabled(running)
        self.load_btn.setEnabled(not running)

    # ----- slots -----
    def on_load_config(self) -> None:
        if self.rig is not None:
            self.log_msg("错误: 先停止预览再加载新 config")
            return
        path = Path(self.cfg_edit.text())
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "错误", f"找不到 {path}")
            return
        try:
            self.cfg = load_config(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "错误", f"解析失败: {e}")
            return
        self.cfg.sync_window_ms = self.sync_spin.value()
        self.cfg.output_dir = self.out_edit.text()
        self._rebuild_tiles(self.cfg.cameras)
        self.log_msg(f"已加载 {path}: {len(self.cfg.cameras)} 台相机")
        for c in self.cfg.cameras:
            self.log_msg(f"  {c.name}: index={c.index} {c.width}x{c.height}@{c.fps}fps"
                         + (f" exp={c.exposure}" if c.exposure is not None else ""))
        self.statusBar().showMessage(f"已加载 {len(self.cfg.cameras)} 台相机, 可以开始预览")

    def on_start(self) -> None:
        if self.cfg is None:
            self.on_load_config()
            if self.cfg is None:
                return
        # 同步 spin/out_edit 的值到 cfg
        self.cfg.sync_window_ms = self.sync_spin.value()
        self.cfg.output_dir = self.out_edit.text()

        self.rig = QtRigCapture(self.cfg)
        self.rig.__enter__()
        # 把每个 grabber 的 signal 接到对应 tile
        for g in self.rig.grabbers:
            g.frame_ready.connect(self._route_frame)
        self._set_running(True)
        for t in self.tiles.values():
            t.reset_stats()
        self.log_msg(f"已启动 {len(self.rig.grabbers)} 个 grabber, 等待 1s 预热...")
        self.statusBar().showMessage("预览中...")
        QtCore.QTimer.singleShot(1000, self._after_warmup)

    def _after_warmup(self) -> None:
        # 1s 预热结束, 报告每路 grabber 状态
        for g in self.rig.grabbers:
            status = "ok" if g.is_alive() else "DEAD (相机没打开?)"
            self.log_msg(f"  [{g.cam.name}] grabber {status}, dropped={g.dropped}")
        self.log_msg("预览就绪, 可以 [单组抓拍] 或 [抓 N 组]")

    def _route_frame(self, name: str, bgr: np.ndarray, ts: int) -> None:
        tile = self.tiles.get(name)
        if tile is not None:
            tile.on_frame(name, bgr, ts)

    def on_stop(self) -> None:
        if self.rig is None:
            return
        self.log_msg("正在停止 grabber...")
        self.rig.__exit__(None, None, None)
        for t in self.tiles.values():
            t.setText(f"{t.cam.name}\n(已停止)")
            t._pixmap = None
        self.rig = None
        self._set_running(False)
        self.log_msg("已停止")
        self.statusBar().showMessage("已停止")

    def on_new_session(self) -> None:
        self.session_dir = None
        self.captured_in_session = 0
        self.status_label.setText("0/0  |  最近同步: --- ms")
        self.log_msg("下次抓拍将创建新会话目录")

    def on_capture_one(self) -> None:
        self._capture_n(1)

    def on_capture_n(self) -> None:
        n = self.target_spin.value()
        self._capture_n(n)

    def _capture_n(self, n: int) -> None:
        if self.rig is None or self.cfg is None:
            self.log_msg("错误: 先开始预览")
            return
        # 懒创建 session dir (加序号防同秒撞名)
        if self.session_dir is None:
            self._session_seq += 1
            session = f"{time.strftime('%Y%m%d_%H%M%S')}_{self._session_seq:03d}"
            self.session_dir = Path(self.out_edit.text()) / session
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.captured_in_session = 0
            self.log_msg(f"新会话: {self.session_dir}")

        window_ns = int(self.sync_spin.value() * 1e6)
        captured = 0
        fails = 0
        max_attempts = n * 10 + 20
        attempts = 0
        self.capture_btn.setEnabled(False)
        self.single_btn.setEnabled(False)
        try:
            while captured < n and attempts < max_attempts:
                attempts += 1
                group = self.rig.capture_group(timeout=2.0)
                if group is None:
                    fails += 1
                    if fails > 5 and fails > captured:
                        self.log_msg(f"连续 {fails} 次同步失败, 检查相机/带宽")
                        break
                    continue
                gid = self.captured_in_session + captured
                save_group(group, self.session_dir, gid)
                captured += 1
                span_ms = (max(f.timestamp_ns for f in group)
                           - min(f.timestamp_ns for f in group)) / 1e6
                total = self.captured_in_session + captured
                self.status_label.setText(f"{total}  |  最近同步: {span_ms:.2f} ms")
                self.log_msg(f"[{total:04d}] span={span_ms:.2f}ms  → {self.session_dir.name}/{gid:05d}_*.jpg")
                # 让 UI 有机会刷新
                QtWidgets.QApplication.processEvents()
        finally:
            self.captured_in_session += captured
            self.capture_btn.setEnabled(True)
            self.single_btn.setEnabled(True)
        if captured < n:
            self.log_msg(f"只抓到 {captured}/{n} 组 (失败 {fails})")
        else:
            self.log_msg(f"完成: {captured} 组, 累计 {self.captured_in_session} 组")


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("rig.json"))
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(args.config)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
