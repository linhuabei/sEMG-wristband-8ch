"""
数据采集视图

用途：采集**自采数据集**——8 通道表面肌电 + （将来）视觉方案给出的角度信息。

输出目录（exe 同级）：
    captured_data/capture_20260911_153000/
        emg.csv       ← timestamp,ch1..ch8（单位与设备一致，采样率写进 meta.json）
        angles.csv    ← timestamp,角度1..角度N（角度源接入后才生成）
        meta.json     ← 采样率/通道数/时长/备注/角度源名称

角度源是**预留接口**：将来做视觉角度识别时，实现一个"角度源"子类，
调用 设置角度源(...) 挂上来即可，本页与保存逻辑都不用改。
"""
import csv
import json
import os
import time as tm
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

from core.paths import data_dir as user_data_dir
from ui.themes.pixel_theme import PixelTheme


# ==================== 角度源接口（预留视觉方案）====================
class 角度源:
    """角度数据源基类。接视觉方案时继承它并实现 可用() 与 读取()。"""

    名称 = "未接入"

    def 可用(self) -> bool:
        return False

    def 读取(self):
        """返回 (时间戳数组, 角度数组) 或 None。将来视觉方案在这里实现。"""
        return None

    def 角度名(self):
        return []


class 无角度源(角度源):
    名称 = "未接入（预留：视觉角度识别）"


class 数据采集视图(QWidget):
    """8 通道肌电（+ 未来角度）采集页。"""

    def __init__(self):
        super().__init__()
        self.channels = 8
        self.sampling_rate = 2000        # 由设备上报刷新
        self.角度源 = 无角度源()

        self.采集中 = False
        self.起始时间 = None
        self.样本数 = 0
        self.目录 = None
        self.文件 = None
        self.写入器 = None
        self.角度文件 = None
        self.角度写入器 = None
        self.角度数量 = 0
        self._上次角度轮询 = 0.0

        self.buffer_size = 800
        self.数据缓冲 = np.zeros((self.buffer_size, self.channels))
        self.写入位置 = 0

        self._init_ui()
        self.刷新定时器 = QTimer(self)
        self.刷新定时器.timeout.connect(self._tick_ui)
        self.刷新定时器.start(200)
        self.绘图定时器 = QTimer(self)
        self.绘图定时器.timeout.connect(self._tick_plot)
        self.绘图定时器.start(50)

    # ---------------- 界面 ----------------
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(12)

        header, _ = PixelTheme.page_header(
            "数据采集",
            "采集 8 通道表面肌电（+ 角度）作为自采数据集；文件自动分组保存",
            badges=[PixelTheme.badge("自采数据", PixelTheme.COLORS['primary'])],
        )
        layout.addLayout(header)

        # 控制条：开始/停止与保存路径都由主窗口搬进右侧常驻「硬件控制」栏，
        # 这里只保留控件本体（临时容器，不进页面布局）。
        bar = QFrame(self)
        bar.setObjectName("demoCard")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 10, 16, 10)
        bl.setSpacing(10)
        self.btn_start = QPushButton("开始采集")
        self.btn_start.setObjectName("funcBtn")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._start_capture)
        self.btn_stop = QPushButton("停止并保存")
        self.btn_stop.setObjectName("funcBtn")
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_capture)
        self.备注 = QLabel("备注：")
        self.来源下拉 = QComboBox()
        self.来源下拉.addItem(f"角度源：{self.角度源.名称}")
        bl.addWidget(self.btn_start)
        bl.addWidget(self.btn_stop)
        bl.addStretch()
        bl.addWidget(self.来源下拉)

        # 实时曲线
        card = QFrame()
        card.setObjectName("demoCard")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 12)
        self.曲线图 = pg.PlotWidget()
        self.曲线图.setBackground(None)
        self.曲线图.showGrid(x=False, y=True, alpha=0.2)
        self.曲线图.setLabel("left", "幅值")
        self.曲线图.setLabel("bottom", "最近 0.4 秒")
        self.曲线 = [self.曲线图.plot(pen=pg.mkPen(width=1)) for _ in range(self.channels)]
        self.曲线图.setYRange(-1, 1)
        self.曲线图.setMinimumHeight(200)
        cl.addWidget(self.曲线图)
        layout.addWidget(card)

        # 状态
        self.状态 = QLabel("未开始采集")
        self.状态.setObjectName("statusText")
        self.状态.setWordWrap(True)
        self.路径 = QLabel("")
        self.路径.setObjectName("statusFile")
        self.路径.setWordWrap(True)
        hint = QLabel("开始/停止与保存路径位于右侧「硬件控制 → 自采数据集」栏")
        hint.setObjectName("hintText")
        layout.addWidget(hint)
        layout.addStretch()

    # ---------------- 角度源 ----------------
    def 设置角度源(self, 源: 角度源):
        """将来接入视觉角度识别：实现一个角度源子类，这里挂上来即可。"""
        self.角度源 = 源
        self.来源下拉.clear()
        self.来源下拉.addItem(f"角度源：{源.名称}")

    # ---------------- 数据入口 ----------------
    def set_sampling_rate(self, fs):
        try:
            self.sampling_rate = int(fs)
        except Exception:                       # noqa: BLE001
            pass
        self.buffer_size = max(200, int(self.sampling_rate * 0.4))
        self.数据缓冲 = np.zeros((self.buffer_size, self.channels))
        self.写入位置 = 0

    def update_plot(self, filtered_data):
        """接收上位机滤波后的 8 通道数据（和通道监视页同一份数据）。"""
        d = np.asarray(filtered_data, dtype=float).reshape(-1)
        if d.size < self.channels:
            return
        self.数据缓冲[self.写入位置] = d[:self.channels]
        self.写入位置 = (self.写入位置 + 1) % self.buffer_size
        if self.采集中:
            self.写入器.writerow([f"{tm.time():.3f}"] + [f"{v:.6f}" for v in d[:self.channels]])
            self.样本数 += 1
            self._poll_angles()

    # ---------------- 采集控制 ----------------
    def _start_capture(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.目录 = os.path.join(user_data_dir("captured_data"), f"capture_{ts}")
        os.makedirs(self.目录, exist_ok=True)
        self.文件 = open(os.path.join(self.目录, "emg.csv"), "w",
                         encoding="utf-8", newline="")
        self.写入器 = csv.writer(self.文件)
        self.写入器.writerow(["timestamp"] + [f"ch{i + 1}" for i in range(self.channels)])

        self.角度数量 = 0
        if self.角度源.可用():
            self.角度文件 = open(os.path.join(self.目录, "angles.csv"), "w",
                                 encoding="utf-8", newline="")
            self.角度写入器 = csv.writer(self.角度文件)
            self.角度写入器.writerow(["timestamp"] + list(self.角度源.角度名()))
            self.角度数量 = len(self.角度源.角度名())

        self.采集中 = True
        self.起始时间 = tm.time()
        self.样本数 = 0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.状态.setText(f"采集中（{self.sampling_rate} Hz，"
                           f"角度源：{self.角度源.名称}）")
        self.路径.setText(f"保存路径：{self.目录}")

    def _poll_angles(self):
        """如果接入了角度源，按约 20 Hz 把角度写进 angles.csv（视觉方案将来走这里）。"""
        if self.角度写入器 is None or self.角度源 is None:
            return
        now = tm.time()
        if now - self._上次角度轮询 < 0.05:
            return
        self._上次角度轮询 = now
        try:
            got = self.角度源.读取()
        except Exception:                       # noqa: BLE001
            return
        if not got:
            return
        时间戳, 角度 = got
        self.角度写入器.writerow([f"{时间戳:.3f}"] + [f"{v:.4f}" for v in np.ravel(角度)])

    def _stop_capture(self):
        if self.采集中:
            self.采集中 = False
            for f in (self.文件, self.角度文件):
                if f:
                    f.close()
            self.文件 = self.角度文件 = None
            self.写入器 = self.角度写入器 = None
            时长 = tm.time() - (self.起始时间 or tm.time())
            说明 = {
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "采样率_Hz": self.sampling_rate,
                "通道数": self.channels,
                "通道名": [f"ch{i + 1}" for i in range(self.channels)],
                "样本数": int(self.样本数),
                "时长_秒": round(时长, 2),
                "角度源": self.角度源.名称,
                "角度通道数": int(self.角度数量),
                "备注": "自采数据集：8 通道表面肌电（角度通道为预留）",
            }
            with open(os.path.join(self.目录, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(说明, fh, ensure_ascii=False, indent=2)
            self.状态.setText(f"已保存：{self.样本数} 个采样点，约 {时长:.1f} 秒"
                               f"（{self.样本数 / max(1, self.sampling_rate):.1f} 秒 @ "
                               f"{self.sampling_rate} Hz）")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ---------------- 定时刷新 ----------------
    def _tick_ui(self):
        if self.采集中 and self.起始时间:
            t = tm.time() - self.起始时间
            self.状态.setText(f"采集中：{t:.1f} 秒，已记录 {self.样本数} 个采样点"
                               f"（角度源：{self.角度源.名称}）")

    def _tick_plot(self):
        if self.写入位置 == 0:
            return
        buf = np.roll(self.数据缓冲, -self.写入位置, axis=0)
        x = np.arange(buf.shape[0])
        for i, c in enumerate(self.曲线):
            c.setData(x, buf[:, i])

    def clear_data(self):
        self.数据缓冲[:] = 0.0
        self.写入位置 = 0
