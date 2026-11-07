"""
八通道实时波形监视界面
"""
import csv
import os
import time

from core.paths import data_dir as user_data_dir

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSizePolicy, QMenu, QAction,
                             QPushButton, QButtonGroup)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pyqtgraph as pg

from ui.themes.pixel_theme import PixelTheme


class ChannelMonitorView(QWidget):
    """八通道实时波形监视视图"""


    def __init__(self, signal_processor):
        super().__init__()
        self.signal_processor = signal_processor
        self.channels = 8
        self.buffer_size = 800
        self.data_buffers = [np.zeros(self.buffer_size) for _ in range(8)]
        self.current_index = 0
        self.has_data = False
        self.ch_colors = list(PixelTheme.TRACE_COLORS)
        self._rate_callback = None
        self._start_callback = None
        self._stop_callback = None
        self._rate_buttons = {}
        self.recording = False
        self._csv_file = None
        self._csv_writer = None
        self._rec_count = 0
        self._rec_path = None
        self.sample_rate = 1000

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(12)

        # 页面头
        self.fps_badge = PixelTheme.badge(
            "接收：-- 帧/s", PixelTheme.COLORS['text_secondary']
        )
        self.fs_badge = PixelTheme.badge("1000 Hz", PixelTheme.COLORS['secondary'])
        # 采集工具栏：开始/停止 + 录制 + 采样率切换。
        # 这些控件不再放在本页，而是由主窗口搬进右侧常驻「硬件控制」栏
        # （ui/widgets/control_panel.py），所以这里只是个临时容器。
        self.ctrl_box = QWidget(self)
        ctrl_layout = QHBoxLayout(self.ctrl_box)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(6)
        self.btn_acq_start = QPushButton("开始采集")
        self.btn_acq_start.setObjectName("acqStartBtn")
        self.btn_acq_start.setCursor(Qt.PointingHandCursor)
        self.btn_acq_start.clicked.connect(self._on_acq_start)
        ctrl_layout.addWidget(self.btn_acq_start)
        self.btn_acq_stop = QPushButton("停止采集")
        self.btn_acq_stop.setObjectName("acqStopBtn")
        self.btn_acq_stop.setCursor(Qt.PointingHandCursor)
        self.btn_acq_stop.clicked.connect(self._on_acq_stop)
        ctrl_layout.addWidget(self.btn_acq_stop)

        self.btn_rec = QPushButton("开始录制")
        self.btn_rec.setObjectName("recBtn")
        self.btn_rec.setCursor(Qt.PointingHandCursor)
        self.btn_rec.clicked.connect(self._on_rec_clicked)
        ctrl_layout.addWidget(self.btn_rec)

        self.rate_box = QWidget()
        rate_layout = QHBoxLayout(self.rate_box)
        rate_layout.setContentsMargins(0, 0, 0, 0)
        rate_layout.setSpacing(2)
        self.rate_group = QButtonGroup(self)
        self.rate_group.setExclusive(True)
        for sps in (1000, 2000):
            btn = QPushButton(f"{sps} Hz")
            btn.setObjectName("rateBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, s=sps: self._on_rate_clicked(s))
            self._rate_buttons[sps] = btn
            self.rate_group.addButton(btn)
            rate_layout.addWidget(btn)
        self.set_rate_control_enabled(False)

        # 采样率/帧率徽标也交给右侧控制栏显示，页面只留标题与波形
        header, _ = PixelTheme.page_header(
            "连接与监视",
            "连接设备后显示 8 通道实时波形 · 滤波开关见右侧「硬件控制 → 滤波」",
            badges=[],
        )
        layout.addLayout(header)

        # 波形图区
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setBackground('#ffffff')
        self.plot_widget.setMinimumHeight(320)

        self.plots = []
        self.curves = []
        for i in range(8):
            plot = self.plot_widget.addPlot(row=i // 2, col=i % 2)
            plot.showGrid(x=True, y=True, alpha=0.16)
            plot.setYRange(-1.2, 1.2)
            plot.hideButtons()
            plot.setMenuEnabled(False)
            plot.setTitle(f'CH{i + 1}', color=self.ch_colors[i], size='11pt')
            plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

            left = plot.getAxis('left')
            left.setPen(pg.mkPen('#d3dde7'))
            left.setTextPen('#5f7083')
            left.setWidth(52)
            left.setStyle(tickLength=-4)
            left.setTickFont(QFont("Microsoft YaHei", 10))

            bottom = plot.getAxis('bottom')
            bottom.setPen(pg.mkPen('#d3dde7'))
            bottom.setTextPen('#5f7083')
            bottom.setStyle(tickLength=-4)
            bottom.setTickFont(QFont("Microsoft YaHei", 10))

            curve = plot.plot(pen=pg.mkPen(color=self.ch_colors[i], width=1.4))
            self.plots.append(plot)
            self.curves.append(curve)

        # 波形网格留出呼吸感，避免 8 路挤在一起
        self.plot_widget.ci.layout.setContentsMargins(2, 8, 2, 0)
        self.plot_widget.ci.layout.setSpacing(14)

        layout.addWidget(self.plot_widget, 1)

        # 底栏
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(2, 0, 2, 0)
        hint = QLabel("波形按满量程归一化 · 标题显示实时 RMS · 右键菜单可自动缩放")
        hint.setObjectName("hintText")
        bottom_bar.addWidget(hint)
        bottom_bar.addStretch()
        # 录制状态标签跟着「开始录制」按钮一起放进右侧控制栏
        self.rec_label = QLabel("录制：未开始")
        self.rec_label.setObjectName("recLabel")
        layout.addLayout(bottom_bar)

        self.frame_count = 0
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self._calc_fps)
        self.fps_timer.start(1000)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_plots)
        self.update_timer.start(33)
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_ch_stats)
        self.stats_timer.start(500)

    def _on_mouse_clicked(self, event):
        if event.button() == Qt.RightButton:
            pos = event.screenPos().toPoint()
            for plot in self.plots:
                if plot.isVisible():
                    menu = QMenu(self)
                    menu.setFont(QFont("Microsoft YaHei", 10))
                    a1 = QAction("自动缩放", self)
                    a1.triggered.connect(lambda checked, p=plot: p.autoRange())
                    a2 = QAction("重置视图", self)
                    a2.triggered.connect(
                        lambda checked, p=plot: p.setYRange(-1.2, 1.2)
                    )
                    menu.addAction(a1)
                    menu.addAction(a2)
                    menu.exec_(pos)
                    event.accept()
                    return

    def update_plot(self, filtered_data):
        self.has_data = True
        for i in range(8):
            self.data_buffers[i][self.current_index] = filtered_data[i]
        self.current_index = (self.current_index + 1) % self.buffer_size
        self.frame_count += 1
        # 录制中就把这一拍写进 CSV（和波形看到的是同一份滤波数据）
        if self.recording and self._csv_writer is not None:
            row = [f"{time.time():.6f}"]
            row += [f"{float(v):.6f}" for v in filtered_data]
            self._csv_writer.writerow(row)
            self._rec_count += 1
            if self._rec_count % 1000 == 0:
                self._update_rec_label()

    def _update_plots(self):
        for i in range(8):
            if self.plots[i].isVisible():
                self.curves[i].setData(
                    np.roll(self.data_buffers[i], -self.current_index)
                )

    def _calc_fps(self):
        self.fps_badge.setText(f"接收 {self.frame_count} 帧/s")
        self.frame_count = 0

    def _update_ch_stats(self):
        """每 500ms 把各通道最近窗口 RMS 刷到图标题"""
        if not self.has_data:
            return
        for i in range(self.channels):
            data = np.roll(self.data_buffers[i], -self.current_index)
            rms = float(np.sqrt(np.mean(data * data)))
            if self.plots[i].isVisible():
                self.plots[i].setTitle(
                    f'CH{i + 1} · RMS {rms:.3f}',
                    color=self.ch_colors[i], size='11pt'
                )

    def apply_trace_colors(self, colors):
        """主题切换时同步更新曲线与标题颜色"""
        palette = list(colors)
        while len(palette) < self.channels:
            palette.extend(palette)
        self.ch_colors = palette[:self.channels]
        for i in range(self.channels):
            self.curves[i].setPen(pg.mkPen(color=self.ch_colors[i], width=1.4))
            if not self.has_data:
                self.plots[i].setTitle(
                    f'CH{i + 1}', color=self.ch_colors[i], size='11pt'
                )
        if self.has_data:
            self._update_ch_stats()

    def clear_data(self):
        """模式切换时清空显示缓冲，避免旧状态造成假信号"""
        self.has_data = False
        self.current_index = 0
        self.frame_count = 0
        for i in range(self.channels):
            self.data_buffers[i][:] = 0.0
        for i in range(self.channels):
            self.plots[i].setTitle(
                f'CH{i + 1}', color=self.ch_colors[i], size='11pt'
            )

    def set_sampling_rate(self, rate):
        """跟随固件上报的采样率更新页头标识与按钮选中态"""
        self.sample_rate = int(rate)
        self.fs_badge.setText(f"{int(rate)} Hz")
        self.set_current_rate(int(rate))

    def set_acquisition_control(self, start_cb, stop_cb):
        """由主窗口注入开始/停止采集回调"""
        self._start_callback = start_cb
        self._stop_callback = stop_cb

    def set_rate_control(self, callback):
        """由主窗口注入采样率请求回调"""
        self._rate_callback = callback

    def set_rate_control_enabled(self, enabled):
        for btn in self._rate_buttons.values():
            btn.setEnabled(bool(enabled))
        self.btn_acq_start.setEnabled(bool(enabled))
        self.btn_acq_stop.setEnabled(bool(enabled))
        self.btn_rec.setEnabled(bool(enabled))
        if not enabled and self.recording:
            self._stop_recording()   # 断开连接时别留个半截文件

    def set_current_rate(self, sps):
        btn = self._rate_buttons.get(int(sps))
        if btn is not None:
            btn.setChecked(True)

    def _on_acq_start(self):
        if self._start_callback is not None:
            self._start_callback()

    def _on_acq_stop(self):
        if self._stop_callback is not None:
            self._stop_callback()

    def _on_rec_clicked(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.recording:
            return
        rec_dir = user_data_dir("recordings")
        name = time.strftime("sEMG_%Y%m%d_%H%M%S.csv")
        path = os.path.join(rec_dir, name)

        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        header = ["timestamp"]
        header += [f"CH{i + 1}" for i in range(self.channels)]
        self._csv_writer.writerow(header)

        self.recording = True
        self._rec_path = path
        self._rec_count = 0
        self.btn_rec.setText("停止录制")
        self.btn_rec.setStyleSheet(
            "QPushButton#recBtn {"
            "background-color:#b3261e;color:#ffffff;"
            "border:1px solid #b3261e;border-radius:4px;"
            "padding:6px 14px;font-weight:bold;"
            "}"
        )
        self.rec_label.setStyleSheet("color:#b3261e;font-size:13px;font-weight:bold;")
        self._update_rec_label()

    def _stop_recording(self):
        if self._csv_writer is not None:
            self._csv_file.flush()
            self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None
        was_recording = self.recording
        self.recording = False
        self.btn_rec.setText("开始录制")
        self.btn_rec.setStyleSheet("")
        self.rec_label.setStyleSheet("")
        if was_recording and self._rec_path:
            self.rec_label.setText(
                f"已保存：{os.path.basename(self._rec_path)}"
                f"（{self._rec_count} 个采样点）"
            )
        else:
            self.rec_label.setText("录制：未开始")

    def _update_rec_label(self):
        if self.recording and self._rec_path:
            self.rec_label.setText(
                f"录制中：{self._rec_count} 个采样点 · "
                f"{os.path.basename(self._rec_path)}"
            )

    def _on_rate_clicked(self, sps):
        if self._rate_callback is not None:
            self._rate_callback(sps)
