"""
右侧常驻控制栏（控制 + 调试两个标签页）

「控制」页：把所有页面的操作类控件集中在一处，切页面时它一直在
    设备状态   —— 连接状态 / 采样率 / 接收帧率
    采集(硬件) —— 开始采集 / 停止采集 / 1000-2000Hz
    滤波       —— 带通 20–450 Hz、带阻 50 Hz 两个开关
    录制波形   —— 开始录制 / 停止录制 + 打开目录
    自采数据集 —— 开始采集 / 停止并保存 + 保存路径 + 打开目录
    标定采集   —— 开始训练 / 停止

「调试」页：整个设备调试功能（命令控制台 + 引脚 + 传感器调试）常驻在这里，
不再单独占一个导航页面。

实现方式：主窗口把各页面里**已经存在**的按钮/标签搬到这里（Qt 里换父窗口是
允许的），控件仍然由原来的页面对象维护状态，所以这里只负责摆位置 +
统一在未连接时禁用采集类操作，不重复实现业务逻辑。
"""
import os
import subprocess

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QScrollArea, QTabWidget,
                             QVBoxLayout, QWidget)

from core.paths import data_dir as user_data_dir
from ui.themes.pixel_theme import PixelTheme
from ui.widgets.imu_view import IMU面板


class 右侧控制栏(QFrame):
    """常驻右侧的控制栏：控制页 + 调试页。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlPanel")
        self.setFixedWidth(300)

        C = PixelTheme.COLORS
        self.setStyleSheet(
            "QFrame#controlPanel {"
            f"background-color:{C['surface']};"
            f"border-left:1px solid {C['border']};"
            "}"
            f"QLabel#ctrlGroupTitle {{color:{C['primary']};"
            "font-size:14px;font-weight:bold;background:transparent;}"
            f"QLabel#ctrlValue {{color:{C['text_primary']};"
            "font-size:15px;font-weight:bold;background:transparent;}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs)

        # ---------------- 控制页 ----------------
        self.tab_control = QWidget()
        ctrl_outer = QVBoxLayout(self.tab_control)
        ctrl_outer.setContentsMargins(0, 0, 0, 0)
        ctrl_outer.setSpacing(0)

        # 窗口不高时这一页可以自己滚动，不挤压内容
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        scroll.setWidget(inner)
        ctrl_outer.addWidget(scroll)

        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(12, 14, 12, 14)
        self.body.setSpacing(10)

        sub = QLabel("采集、录制与保存集中于此栏，切换页面时保持可用")
        sub.setObjectName("hintText")
        sub.setWordWrap(True)
        self.body.addWidget(sub)

        self.grp_device, self.lay_device = self._group("设备状态")
        self.grp_capture, self.lay_capture = self._group("采集（硬件）")
        self.grp_filter, self.lay_filter = self._group("滤波")
        self.grp_imu, self.lay_imu = self._group("IMU 姿态（LSM6DSR）")
        self.grp_record, self.lay_record = self._group("录制波形")
        self.grp_dataset, self.lay_dataset = self._group("自采数据集")
        self.grp_train, self.lay_train = self._group("标定采集")
        for card in (self.grp_device, self.grp_capture, self.grp_filter,
                     self.grp_imu,
                     self.grp_record, self.grp_dataset, self.grp_train):
            self.body.addWidget(card)
        self.body.addStretch(1)

        # 设备状态
        self.conn_label = QLabel("未连接")
        self.conn_label.setObjectName("ctrlValue")
        self.lay_device.addWidget(self.conn_label)
        self.device_name_label = QLabel("—")
        self.device_name_label.setObjectName("hintText")
        self.device_name_label.setWordWrap(True)
        self.lay_device.addWidget(self.device_name_label)

        # 滤波开关（默认全开）
        self.chk_bandpass = QCheckBox("带通 20–450 Hz")
        self.chk_bandpass.setChecked(True)
        self.chk_bandpass.setCursor(Qt.PointingHandCursor)
        self.chk_notch = QCheckBox("带阻 50 Hz（工频陷波）")
        self.chk_notch.setChecked(True)
        self.chk_notch.setCursor(Qt.PointingHandCursor)
        self.lay_filter.addWidget(self.chk_bandpass)
        self.lay_filter.addWidget(self.chk_notch)
        ftip = QLabel("关闭后该级被旁路，波形为原始码值（直流阻断始终保留）")
        ftip.setObjectName("hintText")
        ftip.setWordWrap(True)
        self.lay_filter.addWidget(ftip)

        # IMU 姿态（3D 方块 + 读数 + 轮询开关 + 零位校准）
        self.imu_panel = IMU面板()
        self.lay_imu.addWidget(self.imu_panel)

        self.tabs.addTab(self.tab_control, "控制")

        # ---------------- 调试页 ----------------
        self.tab_debug = QWidget()
        self.debug_lay = QVBoxLayout(self.tab_debug)
        self.debug_lay.setContentsMargins(0, 0, 0, 0)
        self.debug_lay.setSpacing(0)
        self.tabs.addTab(self.tab_debug, "调试")

        # 各页面挂上来的控件（attach 时填充）
        self.monitor = None
        self.capture = None
        self.training = None

    # ==================== 内部小工具 ====================

    def _group(self, title):
        """一个带标题的小卡片，返回 (卡片, 卡片里的竖排布局)"""
        card = QFrame()
        card.setObjectName("infoCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(8)
        cap = QLabel(title)
        cap.setObjectName("ctrlGroupTitle")
        lay.addWidget(cap)
        return card, lay

    @staticmethod
    def _pair(*widgets):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for w in widgets:
            row.addWidget(w)
        return row

    def _open_btn(self, folder_name):
        显示名 = {
            "captured_data": "采集数据",
            "training_data": "训练数据",
            "recordings": "波形录制",
        }
        btn = QPushButton(f"打开 {显示名.get(folder_name, folder_name)} 目录")
        btn.setObjectName("funcBtn")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: self._open_folder(folder_name))
        return btn

    @staticmethod
    def _open_folder(name):
        path = user_data_dir(name)
        try:
            os.startfile(path)          # Windows
        except Exception:
            try:
                subprocess.Popen(["explorer", path])
            except Exception:
                pass

    # ==================== 对接各页面 ====================

    def attach(self, monitor, capture, training, debug_view=None):
        """把各页面里的操作控件、以及整个设备调试视图搬进这一栏。"""
        self.monitor = monitor
        self.capture = capture
        self.training = training

        # 设备状态：采样率 / 接收帧率徽标本来就由监视页的定时器刷新
        self.lay_device.addWidget(monitor.fs_badge)
        self.lay_device.addWidget(monitor.fps_badge)

        # 采集（硬件）
        self.lay_capture.addLayout(
            self._pair(monitor.btn_acq_start, monitor.btn_acq_stop))
        self.lay_capture.addWidget(monitor.rate_box)

        # 录制波形（监视页那份 CSV）
        self.lay_record.addLayout(self._pair(monitor.btn_rec))
        self.lay_record.addWidget(monitor.rec_label)
        self.lay_record.addWidget(self._open_btn("recordings"))

        # 自采数据集（数据采集页那份，带 meta.json）
        self.lay_dataset.addLayout(
            self._pair(capture.btn_start, capture.btn_stop))
        self.lay_dataset.addWidget(capture.状态)
        self.lay_dataset.addWidget(capture.路径)
        self.lay_dataset.addWidget(self._open_btn("captured_data"))

        # 标定采集（用户迁移 → 标定采集 页那两个按钮）
        self.lay_train.addLayout(self._pair(training.sb, training.eb))
        tip = QLabel("动作提示见「用户迁移 → 标定采集」页")
        tip.setObjectName("hintText")
        tip.setWordWrap(True)
        self.lay_train.addWidget(tip)

        # 设备调试（整页搬进来常驻）
        if debug_view is not None:
            self.debug_lay.addWidget(debug_view)

    def set_connected(self, connected, device_name=""):
        """连接状态变化：更新状态文字，并统一禁用/放开采集类操作。"""
        C = PixelTheme.COLORS
        self.conn_label.setText("已连接" if connected else "未连接")
        self.conn_label.setStyleSheet(
            f"color:{C['success'] if connected else C['text_dim']};"
            "font-size:15px;font-weight:bold;background:transparent;"
        )
        self.device_name_label.setText(device_name or "—")
        # 滤波开关和设备状态不禁用；调试页始终可用（断线后还要看日志）
        for card in (self.grp_capture, self.grp_record,
                     self.grp_dataset, self.grp_train):
            card.setEnabled(bool(connected))
