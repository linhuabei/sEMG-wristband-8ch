"""
主窗口 - 顶栏品牌区 + 侧边栏导航版
"""
from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFrame, QListWidget, QListWidgetItem,
                             QGraphicsOpacityEffect, QCheckBox, QSizePolicy,
                             QScrollArea)
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve
from ui.themes.pixel_theme import PixelTheme
from ui.views.monitor_view import ChannelMonitorView
from ui.views.user_transfer_view import 用户迁移视图
from ui.views.data_capture_view import 数据采集视图
from ui.views.device_detail_view import DeviceDetailView
from ui.widgets.control_panel import 右侧控制栏
from core import paths as 路径
from core.device_manager import DeviceManager
from core.signal_processor import SignalProcessor
from core.gesture_classifier import GestureClassifier


class AnimatedNavButton(QPushButton):
    """导航按钮：选中态带左侧高亮条 + 渐隐底色"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._active = False
        self.setMinimumHeight(48)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setObjectName("navBtn")

    def set_active(self, active):
        self._active = active
        c = PixelTheme.COLORS
        if active:
            css = (
                f"QPushButton{{"
                f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                f"stop:0 {PixelTheme.rgba(c['primary'], 0.18)},"
                f"stop:0.7 {PixelTheme.rgba(c['primary'], 0.05)},"
                f"stop:1 {PixelTheme.rgba(c['primary'], 0)});"
                f"color:{c['primary']};font-weight:bold;border:none;"
                f"border-left:3px solid {c['primary']};"
                "text-align:left;padding-left:16px;font-size:16px;}"
                f"QPushButton:hover{{color:{c['primary']};}}"
            )
        else:
            css = (
                f"QPushButton{{background-color:transparent;"
                f"color:{c['text_secondary']};border:none;"
                "text-align:left;padding-left:19px;font-size:16px;}"
                f"QPushButton:hover{{background-color:{c['hover']};"
                f"color:{c['text_primary']};}}"
            )
        self.setStyleSheet(css)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Node 御识 v260920")

        # 跟随屏幕可用区域：小屏自动最大化，大屏用 1500x900 左右
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        if avail is not None:
            win_w = min(1500, int(avail.width() * 0.92))
            win_h = min(900, int(avail.height() * 0.90))
            # 右边多了一条常驻控制栏；页面本身可滚动，所以最小尺寸可以放小一点
            self.setMinimumSize(min(1100, win_w), min(640, win_h))
            self.resize(win_w, win_h)
            self._start_maximized = avail.width() < 1400 or avail.height() < 860
        else:
            self.setMinimumSize(1100, 640)
            self.resize(1400, 860)
            self._start_maximized = False

        self._device_name = ""
        self.device_manager = DeviceManager()
        self.signal_processor = SignalProcessor(channels=8)
        self.gesture_classifier = GestureClassifier()

        self.current_view_index = None
        self._init_ui()
        self._connect_signals()
        self.center_on_screen()
        self.switch_view(0, animated=False)      # 启动停在「连接与监视」
        self._update_all_pages_enabled(False)
        self._render_connection(False)

    # ==================== UI ====================

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(264)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        self._create_device_panel(sidebar_layout)

        sep = QFrame()
        sep.setObjectName("sep")
        sep.setFrameShape(QFrame.HLine)
        sidebar_layout.addWidget(sep)
        self._create_nav_menu(sidebar_layout)
        sidebar_layout.addStretch(1)

        footer = QLabel("Powered by LinHuabei Lab")
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(footer)
        body.addWidget(self.sidebar)

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("content")

        self.monitor_view = ChannelMonitorView(self.signal_processor)
        self.monitor_view.set_acquisition_control(
            self._acq_start, self._acq_stop
        )
        self.monitor_view.set_rate_control(self._request_sample_rate)
        self.数据采集 = 数据采集视图()
        # 用户迁移 = 标定采集 + 实时识别（个人化专用）
        self.用户迁移 = 用户迁移视图(self.gesture_classifier)
        self.gesture_view = self.用户迁移.实时识别        # 保留旧名字，兼容其它引用
        self.training_view = self.用户迁移.标定采集
        self.device_detail_view = DeviceDetailView(self.device_manager)
        # 常驻右侧「硬件控制」栏：把各页面的操作控件都收到这里
        self.control_panel = 右侧控制栏()
        self.control_panel.attach(
            self.monitor_view, self.数据采集, self.training_view,
            debug_view=self.device_detail_view,
        )
        self.control_panel.set_connected(False)
        # 在右侧栏点「开始」时自动跳到对应页面，方便立刻看到波形/提示
        self.数据采集.btn_start.clicked.connect(lambda: self.switch_view(1))
        self.training_view.sb.clicked.connect(self._focus_training_page)

        # 页面顺序按工作流程排：连接监视 → 数据采集 → 用户迁移
        # （设备调试整块并进右侧控制栏的「调试」页，不再单独占一个页面）
        self.stacked_widget.addWidget(self._scrollable(self.monitor_view))
        self.stacked_widget.addWidget(self._scrollable(self.数据采集))
        self.stacked_widget.addWidget(self._scrollable(self.用户迁移))
        body.addWidget(self.stacked_widget)
        body.addWidget(self.control_panel)
        main_layout.addLayout(body)

    @staticmethod
    def _scrollable(widget):
        """把页面装进滚动区：屏幕小/卡片放不下时可以滚动，内容不会被裁掉"""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setWidget(widget)
        return area

    def _build_header(self):
        header = QWidget()
        header.setFixedHeight(64)
        header.setObjectName("header")
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 0, 18, 0)
        h.setSpacing(12)

        h.addStretch()

        # 右侧状态区
        self.conn_pill = QFrame()
        self.conn_pill.setObjectName("connPill")
        pill_layout = QHBoxLayout(self.conn_pill)
        pill_layout.setContentsMargins(10, 3, 10, 3)
        pill_layout.setSpacing(6)
        self.connection_status = QLabel("未连接")
        self.connection_status.setObjectName("connStatus")
        pill_layout.addWidget(self.connection_status)

        h.addWidget(self.conn_pill)
        return header

    def _create_device_panel(self, parent_layout):
        panel = QFrame()
        panel.setObjectName("devicePanel")
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 16, 14, 8)
        panel_layout.setSpacing(8)

        section_title = QLabel("设备连接")
        section_title.setObjectName("sectionTitle")
        panel_layout.addWidget(section_title)

        self.device_list = QListWidget()
        self.device_list.setMaximumHeight(150)
        self.device_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        item = QListWidgetItem("未连接 · 请先扫描设备")
        item.setFlags(Qt.NoItemFlags)
        self.device_list.addItem(item)
        panel_layout.addWidget(self.device_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.scan_btn = QPushButton("扫描")
        self.scan_btn.setObjectName("scanBtn")
        self.scan_btn.setMinimumHeight(38)
        self.scan_btn.clicked.connect(self._on_scan_clicked)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.setEnabled(False)
        self.connect_btn.setMinimumHeight(38)
        self.connect_btn.clicked.connect(self._on_connect_device)

        btn_row.addWidget(self.scan_btn)
        btn_row.addWidget(self.connect_btn)
        btn_row.addStretch()
        panel_layout.addLayout(btn_row)
        parent_layout.addWidget(panel)

        self.device_list.itemClicked.connect(
            lambda: self.connect_btn.setEnabled(True)
        )
        self.device_manager.devices_found.connect(self._on_devices_found)
        self.device_manager.connection_changed.connect(self._on_connection_changed)

    def _create_nav_menu(self, parent_layout):
        nav_label = QLabel("功能导航")
        nav_label.setObjectName("sectionTitle")
        nav_label.setContentsMargins(14, 8, 0, 8)
        parent_layout.addWidget(nav_label)

        nav_items = [
            ("01  连接与监视", 0),
            ("02  数据采集", 1),
            ("03  用户迁移", 2),
        ]
        self.nav_buttons = []
        for text, index in nav_items:
            btn = AnimatedNavButton(text)
            btn.clicked.connect(lambda checked, i=index: self.switch_view(i))
            parent_layout.addWidget(btn)
            self.nav_buttons.append(btn)

    def _render_connection(self, connected):
        """渲染顶栏连接状态胶囊"""
        c = PixelTheme.COLORS
        if connected:
            text = "已连接"
            color = c['success']
        else:
            text = "未连接"
            color = c['text_dim']

        self.connection_status.setText(text)
        self.connection_status.setStyleSheet(
            f"color:{c['text_primary'] if connected else c['text_dim']};"
            "background:transparent;font-size:14px;font-weight:bold;"
        )
        # 带选择器，避免边框级联到里面的 LED/文字标签变成“两个框”
        self.conn_pill.setStyleSheet(
            "QFrame#connPill {"
            f"background-color:{PixelTheme.rgba(color, 0.10)};"
            f"border:1px solid {PixelTheme.rgba(color, 0.45)};"
            "border-radius:13px;"
            "}"
        )

    # ==================== 页面切换 ====================

    def switch_view(self, index, animated=True):
        if self.current_view_index == index:
            return
        self.current_view_index = index
        # 每次切换都刷新全部按钮，避免没点过的按钮残留默认边框样式
        for i, btn in enumerate(self.nav_buttons):
            btn.set_active(i == index)
        self.stacked_widget.setCurrentIndex(index)

    def _update_all_pages_enabled(self, enabled):
        # 设备调试不在这里禁用：它在右侧控制栏常驻，断线后还要看命令台日志
        self.monitor_view.setEnabled(enabled)
        self.gesture_view.setEnabled(enabled)
        self.training_view.setEnabled(enabled)
        self.数据采集.setEnabled(enabled)

    # ==================== 设备事件 ====================

    def _on_scan_clicked(self):
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("扫描中…")
        self.device_manager.scan_devices()

    def _on_devices_found(self, devices):
        self.device_list.clear()
        if not devices:
            item = QListWidgetItem("未发现设备")
            item.setFlags(Qt.NoItemFlags)
            self.device_list.addItem(item)
        else:
            for device in devices:
                item = QListWidgetItem(f"  {device['name'][:25]}")
                item.setData(Qt.UserRole, device)
                self.device_list.addItem(item)
        self.connect_btn.setEnabled(False)
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("扫描")

    def _on_connect_device(self):
        ci = self.device_list.currentItem()
        if ci and ci.data(Qt.UserRole):
            self._device_name = (ci.data(Qt.UserRole) or {}).get('name', '')
            self.connect_btn.setEnabled(False)
            self.connect_btn.setText("连接中…")
            self.device_manager.connect_device(ci.data(Qt.UserRole))

    def _on_connection_changed(self, connected):
        self._render_connection(connected)
        self._update_all_pages_enabled(connected)
        self.monitor_view.set_rate_control_enabled(connected)
        self.control_panel.set_connected(
            connected, self._device_name if connected else ""
        )
        # 连接后按勾选状态启动 IMU 轮询；断开就停掉并清空姿态
        self.device_manager.set_imu_polling(
            bool(connected and self.control_panel.imu_panel.is_polling_enabled())
        )
        if not connected:
            self.control_panel.imu_panel.reset()
        if connected:
            self.connect_btn.setText("已连接")
            self.connect_btn.setEnabled(True)
        else:
            self.connect_btn.setText("连接")
            self.connect_btn.setEnabled(True)

    # ==================== 数据 / 信号 ====================

    def _connect_signals(self):
        self.device_manager.data_received.connect(self.signal_processor.process_data)
        self.signal_processor.filtered_data_ready.connect(self.monitor_view.update_plot)
        self.signal_processor.filtered_data_ready.connect(self.training_view.update_emg_data)
        self.signal_processor.filtered_data_ready.connect(self.数据采集.update_plot)
        self.signal_processor.features_ready.connect(self.gesture_classifier.classify)
        self.gesture_classifier.gesture_detected.connect(self.gesture_view.show_gesture)
        self.gesture_classifier.gesture_detected.connect(self.training_view.on_gesture_detected)
        self.device_manager.pin_status_updated.connect(
            self.device_detail_view._on_status_updated
        )
        self.device_manager.emg_config_changed.connect(self._on_emg_config)
        # 滤波开关（右侧控制栏的复选框直接控制信号处理链路）
        self.control_panel.chk_bandpass.toggled.connect(
            self.signal_processor.set_bandpass)
        self.control_panel.chk_notch.toggled.connect(
            self.signal_processor.set_notch)
        # IMU 姿态：固件回的 IMU 行 -> 3D 姿态；勾选框控制 20Hz 轮询
        self.device_manager.imu_sample.connect(
            self.control_panel.imu_panel.on_sample)
        self.device_manager.imu_unavailable.connect(
            self.control_panel.imu_panel.on_unavailable)
        self.control_panel.imu_panel.poll_toggled.connect(
            self.device_manager.set_imu_polling)

    def _acq_start(self):
        if not self.device_manager.connected:
            return
        self.device_manager.start_emg()

    def _acq_stop(self):
        if not self.device_manager.connected:
            return
        self.device_manager.stop_emg()

    def _request_sample_rate(self, sps):
        """监视页点击 1000/2000 Hz：下发 EMG_SPS 命令"""
        if not self.device_manager.connected:
            return
        self.device_manager.set_emg_sps(sps)

    def _on_emg_config(self, sps, bits, channels):
        """固件 EMG_CFG 上报：按实际采样率重配滤波与页面标识"""
        self.signal_processor.configure_sampling_rate(sps)
        self.monitor_view.set_sampling_rate(sps)
        self.数据采集.set_sampling_rate(sps)

    def _focus_training_page(self):
        """右侧栏点「开始训练」时跳到 用户迁移 → 标定采集，方便看动作提示"""
        self.switch_view(2)
        self.用户迁移.tabs.setCurrentIndex(0)

    def _update_curve_colors(self, palette=None):
        """按 8 通道调色板重绘监视/训练波形"""
        colors = list(palette if palette is not None else PixelTheme.TRACE_COLORS)
        for view in (self.monitor_view, self.training_view):
            if hasattr(view, "apply_trace_colors"):
                view.apply_trace_colors(colors)

    def closeEvent(self, event):
        if hasattr(self.monitor_view, "_stop_recording"):
            self.monitor_view._stop_recording()
        event.accept()

    def center_on_screen(self):
        if getattr(self, "_start_maximized", False):
            self.setWindowState(self.windowState() | Qt.WindowMaximized)
            return
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        size = self.frameGeometry()
        self.move(geo.center().x() - size.width() // 2,
                  geo.center().y() - size.height() // 2)
