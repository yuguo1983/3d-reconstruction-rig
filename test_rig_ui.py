"""
test_rig_ui.py — rig_ui.py 的单元测试, 无硬件依赖

覆盖:
  - import / 公共类存在
  - rig.json 加载后 MainWindow 创建 6 个 tile
  - CamTile 收帧 -> 累计 FPS, BGR->QImage 转换, 错 cam 名被忽略
  - 按钮初始 / start / stop 状态切换
  - 抓 N 组 (mock capture_group) -> 落盘到 session 目录, 计数器更新
  - 新会话按钮 -> 计数器清零
  - 不存在 / 损坏 config -> 不崩
  - QtFrameGrabber 多继承 QObject + Thread 不冲突

跑法:
    QT_QPA_PLATFORM=offscreen python -m unittest test_rig_ui.py -v
    # 或
    QT_QPA_PLATFORM=offscreen python test_rig_ui.py
"""
from __future__ import annotations
import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 强制 offscreen, 必须在 import PyQt5 之前
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from rig_capture import CamConfig, RigConfig, TimestampedFrame
from rig_ui import (
    CamTile, MainWindow, QtFrameGrabber, QtRigCapture, _select_backend,
)


# ---------- 全局 QApplication (unittest 各测试共享) ----------

_qapp: QtWidgets.QApplication | None = None


def _ensure_app() -> QtWidgets.QApplication:
    global _qapp
    if _qapp is None:
        _qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    return _qapp


def _make_synthetic_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """生成一张带渐变 + 文字的 BGR 假图, 方便视觉确认"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 渐变背景
    img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)  # B
    img[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1)  # G
    img[:, :, 2] = 128  # R
    # 画个白方块
    cv2 = None
    try:
        import cv2 as _cv2
        cv2 = _cv2
        cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 255), 2)
        cv2.putText(img, "TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except ImportError:
        pass
    return img


# ---------- Test: import / 类 ----------

class TestImports(unittest.TestCase):
    def test_classes_exist(self):
        self.assertTrue(callable(QtFrameGrabber))
        self.assertTrue(callable(QtRigCapture))
        self.assertTrue(callable(CamTile))
        self.assertTrue(callable(MainWindow))

    def test_qtframegrabber_is_thread_and_qobject(self):
        from PyQt5.QtCore import QObject
        g = QtFrameGrabber.__mro__
        self.assertIn(QtCore.QObject, g, "QtFrameGrabber 必须多继承 QObject")
        self.assertIn(threading.Thread, g, "QtFrameGrabber 必须多继承 Thread")


# ---------- Test: CamTile ----------

class TestCamTile(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        self.cam = CamConfig(index=0, name="test_cam")
        self.tile = CamTile(self.cam)

    def test_initial_state(self):
        self.assertEqual(self.tile.cam.name, "test_cam")
        self.assertEqual(self.tile._frame_count, 0)
        self.assertEqual(self.tile._fps, 0.0)
        self.assertIsNone(self.tile.pixmap())

    def test_frame_renders_and_sets_pixmap(self):
        img = _make_synthetic_frame()
        # 喂 5 帧
        base_ts = time.monotonic_ns()
        for i in range(5):
            self.tile.on_frame("test_cam", img, base_ts + i * 33_000_000)  # ~30fps
        self.assertEqual(self.tile._frame_count, 5)
        self.assertIsNotNone(self.tile.pixmap(), "收帧后应有 pixmap")
        self.assertFalse(self.tile.pixmap().isNull())

    def test_fps_calculation_after_30_frames(self):
        img = _make_synthetic_frame()
        base_ts = time.monotonic_ns()
        # CamTile 在 fps_count >= 30 时结算, 30 帧产生 29 个间隔
        # 所以喂 31 帧才能触发一次结算
        for i in range(31):
            self.tile.on_frame("test_cam", img, base_ts + i * 33_333_333)  # 30.0 fps
        self.assertGreater(self.tile._fps, 0.0)
        # 应该接近 30
        self.assertAlmostEqual(self.tile._fps, 30.0, delta=2.0,
                               msg=f"FPS 计算偏差: {self.tile._fps}")

    def test_wrong_cam_name_ignored(self):
        img = _make_synthetic_frame()
        self.tile.on_frame("WRONG_CAM", img, time.monotonic_ns())
        self.assertEqual(self.tile._frame_count, 0, "错 cam 名的帧应被忽略")

    def test_reset_stats(self):
        img = _make_synthetic_frame()
        for i in range(5):
            self.tile.on_frame("test_cam", img, time.monotonic_ns() + i * 33_000_000)
        self.tile.reset_stats()
        self.assertEqual(self.tile._frame_count, 0)
        self.assertEqual(self.tile._fps, 0.0)
        self.assertEqual(self.tile._last_ts, 0)


# ---------- Test: MainWindow 构建 / 配置 ----------

class TestMainWindowBuild(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = Path(self.tmpdir) / "test_rig.json"
        self.cfg_path.write_text(json.dumps({
            "sync_window_ms": 10.0,
            "output_dir": str(self.tmpdir),
            "backend": "dshow",
            "cameras": [
                {"index": 0, "name": "camA", "exposure": -7.0, "focus": 80},
                {"index": 1, "name": "camB"},
                {"index": 2, "name": "camC"},
            ],
        }))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_build_no_config(self):
        win = MainWindow(initial_config=None)
        try:
            # 默认有 6 个占位 cam0..cam5
            self.assertEqual(len(win.tiles), 6)
            for i in range(6):
                self.assertIn(f"cam{i}", win.tiles)
            # 按钮初始: start 可用, stop / capture 不可用
            self.assertTrue(win.start_btn.isEnabled())
            self.assertFalse(win.stop_btn.isEnabled())
            self.assertFalse(win.capture_btn.isEnabled())
            self.assertFalse(win.single_btn.isEnabled())
        finally:
            win.close()

    def test_load_config_rebuilds_tiles(self):
        win = MainWindow(initial_config=self.cfg_path)
        try:
            self.assertEqual(len(win.tiles), 3)
            self.assertIn("camA", win.tiles)
            self.assertIn("camB", win.tiles)
            self.assertIn("camC", win.tiles)
            for name, tile in win.tiles.items():
                self.assertEqual(tile.cam.name, name)
        finally:
            win.close()

    def test_load_nonexistent_config_keeps_default(self):
        win = MainWindow(initial_config=None)
        try:
            win.cfg_edit.setText(str(Path(self.tmpdir) / "nope.json"))
            with patch.object(QtWidgets.QMessageBox, "warning") as mb:
                win.on_load_config()
            self.assertIsNone(win.cfg, "不存在的 config 不应让 cfg 被赋值")
            # tile 仍应是默认 6 个
            self.assertEqual(len(win.tiles), 6)
        finally:
            win.close()

    def test_load_corrupt_config_does_not_crash(self):
        bad = Path(self.tmpdir) / "bad.json"
        bad.write_text("{ this is not json }")
        win = MainWindow(initial_config=None)
        try:
            win.cfg_edit.setText(str(bad))
            with patch.object(QtWidgets.QMessageBox, "warning") as mb:
                win.on_load_config()
            self.assertIsNone(win.cfg)
        finally:
            win.close()


# ---------- Test: start / stop 状态机 ----------

class TestStartStop(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = Path(self.tmpdir) / "rig.json"
        self.cfg_path.write_text(json.dumps({
            "sync_window_ms": 10.0,
            "output_dir": str(self.tmpdir),
            "backend": "dshow",
            "cameras": [
                {"index": 0, "name": "camA"},
                {"index": 1, "name": "camB"},
            ],
        }))
        self.win = MainWindow(initial_config=self.cfg_path)

    def tearDown(self):
        if self.win.rig is not None:
            self.win.rig.__exit__(None, None, None)
        self.win.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_capture_button_disabled_before_start(self):
        self.assertFalse(self.win.capture_btn.isEnabled())
        self.assertFalse(self.win.single_btn.isEnabled())

    def test_start_with_nonexistent_cameras_does_not_crash(self):
        # 真的尝试打开不存在的 cam, 应当不崩, 6 个 grabber 全部 DEAD
        self.win.on_start()
        # 等 grabber 启动 + cap 失败
        time.sleep(0.3)
        QtWidgets.QApplication.processEvents()
        # start 之后 capture 按钮应可用
        self.assertTrue(self.win.capture_btn.isEnabled())
        self.assertTrue(self.win.stop_btn.isEnabled())
        self.assertFalse(self.win.start_btn.isEnabled())
        # 停止
        self.win.on_stop()
        self.assertTrue(self.win.start_btn.isEnabled())
        self.assertFalse(self.win.capture_btn.isEnabled())
        self.assertIsNone(self.win.rig)


# ---------- Test: capture 路径 (mock capture_group) ----------

class TestCapturePath(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = Path(self.tmpdir) / "rig.json"
        self.cfg_path.write_text(json.dumps({
            "sync_window_ms": 10.0,
            "output_dir": str(self.tmpdir),
            "backend": "dshow",
            "cameras": [
                {"index": 0, "name": "cam0"},
                {"index": 1, "name": "cam1"},
                {"index": 2, "name": "cam2"},
            ],
        }))
        self.win = MainWindow(initial_config=self.cfg_path)
        # 模拟一个空 rig, 跳过真实相机
        self.win.rig = MagicMock()
        self.win.rig.__exit__ = MagicMock()
        self.win.rig.__enter__ = MagicMock(return_value=self.win.rig)
        # 默认 capture_group 返回 None
        self._mock_frames = []
        self.win.rig.capture_group = MagicMock(side_effect=self._mock_capture_group)

    def tearDown(self):
        self.win.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mock_capture_group(self, timeout=2.0):
        """每次调用返回下一组假帧, 用尽后返回 None"""
        if not self._mock_frames:
            return None
        return self._mock_frames.pop(0)

    def _make_group(self, gid: int, n_cams: int = 3) -> list[TimestampedFrame]:
        ts0 = time.monotonic_ns()
        return [TimestampedFrame(
            cam_name=self.win.cfg.cameras[i].name,
            timestamp_ns=ts0 + i * 1_000_000,  # 1ms 偏差
            frame=_make_synthetic_frame(),
        ) for i in range(n_cams)]

    def test_capture_n_writes_files(self):
        # 准备 5 组帧
        for i in range(5):
            self._mock_frames.append(self._make_group(i))
        # 抓 5 组
        self.win.target_spin.setValue(5)
        # 直接调用, 跳过 on_start
        self.win._capture_n(5)
        # session_dir 应创建
        self.assertIsNotNone(self.win.session_dir)
        self.assertTrue(self.win.session_dir.exists())
        # 应有 5 * 3 = 15 张图
        files = list(self.win.session_dir.glob("*.jpg"))
        self.assertEqual(len(files), 15, f"期望 15 张, 实际 {len(files)}")
        # 计数器
        self.assertEqual(self.win.captured_in_session, 5)

    def test_capture_appends_to_existing_session(self):
        # 第一批 3 组
        for i in range(3):
            self._mock_frames.append(self._make_group(i))
        self.win._capture_n(3)
        first_session = self.win.session_dir
        self.assertEqual(len(list(first_session.glob("*.jpg"))), 9)
        # 第二批 2 组, 不点 [新会话]
        for i in range(3, 5):
            self._mock_frames.append(self._make_group(i))
        self.win._capture_n(2)
        # 应写到同一目录
        self.assertEqual(self.win.session_dir, first_session)
        self.assertEqual(len(list(first_session.glob("*.jpg"))), 15)
        # 累计 5
        self.assertEqual(self.win.captured_in_session, 5)

    def test_new_session_resets_counter(self):
        for i in range(3):
            self._mock_frames.append(self._make_group(i))
        self.win._capture_n(3)
        first_session = self.win.session_dir
        self.win.on_new_session()
        self.assertIsNone(self.win.session_dir)
        self.assertEqual(self.win.captured_in_session, 0)
        # 再抓
        for i in range(3):
            self._mock_frames.append(self._make_group(i))
        self.win._capture_n(3)
        # 新 session 目录
        self.assertNotEqual(self.win.session_dir, first_session)
        self.assertTrue(self.win.session_dir.exists())
        self.assertEqual(len(list(self.win.session_dir.glob("*.jpg"))), 9)

    def test_capture_with_no_rig_logs_error(self):
        self.win.rig = None
        self.win._capture_n(3)
        # 不应崩, 日志应有错误
        log_text = self.win.log.toPlainText()
        self.assertIn("先开始预览", log_text)

    def test_capture_short_when_capture_group_fails(self):
        # 5 组要求, capture_group 只成功 2 次就返回 None
        for i in range(2):
            self._mock_frames.append(self._make_group(i))
        # 改成"不重试"的版本: 失败 1 次就停
        # 由于 _capture_n 内有 fails > captured 才 break, 2 + 几次 None 会停
        self.win._capture_n(5)
        self.assertEqual(self.win.captured_in_session, 2)


# ---------- Test: rig_config 同步 spin ----------

class TestConfigSync(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = Path(self.tmpdir) / "rig.json"
        self.cfg_path.write_text(json.dumps({
            "sync_window_ms": 10.0,
            "output_dir": str(self.tmpdir),
            "backend": "dshow",
            "cameras": [{"index": 0, "name": "camA"}],
        }))
        self.win = MainWindow(initial_config=self.cfg_path)

    def tearDown(self):
        self.win.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_on_start_overrides_sync_window(self):
        self.win.sync_spin.setValue(25.0)
        self.win.out_edit.setText(str(Path(self.tmpdir) / "new_out"))
        # 用 mock 替代真实 grabber
        with patch("rig_ui.QtFrameGrabber") as mock_g:
            mock_inst = MagicMock()
            mock_inst.is_alive.return_value = True
            mock_inst.dropped = 0
            mock_g.return_value = mock_inst
            with patch("rig_ui.threading.Event") as mock_evt:
                mock_evt.return_value = MagicMock()
                self.win.on_start()
        self.assertIsNotNone(self.win.rig)
        self.assertEqual(self.win.cfg.sync_window_ms, 25.0)
        self.assertEqual(self.win.cfg.output_dir, str(Path(self.tmpdir) / "new_out"))
        # 清理
        self.win.rig = None
        self.win._set_running(False)


# ---------- Test: QtRigCapture (无真相机) ----------

class TestQtRigCaptureNoCamera(unittest.TestCase):
    """不实际打开相机, 验证 QtRigCapture 的结构 (queues, grabbers 列表, 生命周期)"""

    def test_queues_created(self):
        cams = [CamConfig(index=i, name=f"c{i}") for i in range(3)]
        cfg = RigConfig(cameras=cams, sync_window_ms=10.0, output_dir="./cap")
        rig = QtRigCapture(cfg)
        self.assertEqual(len(rig.queues), 3)
        self.assertEqual(set(rig.queues.keys()), {"c0", "c1", "c2"})
        for q in rig.queues.values():
            self.assertIsInstance(q, queue.Queue)
            self.assertEqual(q.maxsize, 2)

    def test_grabber_attribute(self):
        cams = [CamConfig(index=0, name="c0")]
        cfg = RigConfig(cameras=cams)
        rig = QtRigCapture(cfg)
        self.assertEqual(rig.grabbers, [])


if __name__ == "__main__":
    # 友好输出
    unittest.main(verbosity=2, argv=sys.argv[:1] + ["-v"])
