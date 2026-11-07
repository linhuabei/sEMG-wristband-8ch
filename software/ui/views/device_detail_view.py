"""
设备调试视图（引脚控制 + 调试工具 + 命令台）

现在整块常驻在右侧「硬件控制」栏的「调试」标签页里，所以是紧凑单列布局，
不再单独占一个导航页面。
"""
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QScrollArea, QTextEdit,
                             QVBoxLayout, QWidget)

from core.esp32_controller import PIN_CONFIG
from ui.themes.pixel_theme import PixelTheme


class DeviceDetailView(QWidget):
    def __init__(self, device_manager):
        super().__init__()
        self.device_manager = device_manager
        self.pin_states = {i: 0 for i in range(1, 14)}
        self.pin_labels = {}
        self._init_ui()
        self._connect_signals()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_status)
        self.refresh_timer.start(2000)

    # ==================== UI ====================

    def _init_ui(self):
        """紧凑单列布局：整个调试功能要塞进右侧常驻栏（约 300px 宽）里"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        tip = QLabel("ESP32-S3 引脚级调试 · ADS1298 传感器控制")
        tip.setObjectName("hintText")
        tip.setWordWrap(True)
        main_layout.addWidget(tip)

        main_layout.addWidget(self._build_func_card())
        main_layout.addWidget(self._build_pin_panel())
        main_layout.addWidget(self._build_console_card(), 1)

    def _build_pin_panel(self):
        """紧凑引脚列表：GPIO + 名称 + 状态 + [1][0][读]（整块可滚动）"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setObjectName("pinScroll")
        scroll.setMinimumHeight(150)
        scroll.setMaximumHeight(250)

        panel = QFrame()
        panel.setObjectName("pinPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 10)
        panel_layout.setSpacing(4)

        title_row = QHBoxLayout()
        title = QLabel("引脚控制")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        panel_layout.addLayout(title_row)

        for pin in range(1, 14):
            cfg = PIN_CONFIG.get(pin, {"name": "NC", "type": "-", "direction": "-"})

            row = QHBoxLayout()
            row.setSpacing(4)

            number = QLabel(f"GPIO{pin}")
            number.setObjectName("pinNumber")
            number.setFixedWidth(52)
            number.setAlignment(Qt.AlignCenter)
            row.addWidget(number)

            name = QLabel(cfg["name"])
            name.setObjectName("pinName")
            row.addWidget(name, 1)

            state = QLabel("—")
            state.setObjectName("pinState")
            state.setFixedWidth(20)
            state.setAlignment(Qt.AlignCenter)
            row.addWidget(state)
            self.pin_labels[pin] = state

            if cfg["direction"] in ("out", "inout"):
                for text, value in (("1", 1), ("0", 0)):
                    btn = QPushButton(text)
                    btn.setObjectName("pinOnBtn" if value else "pinOffBtn")
                    btn.setFixedSize(26, 24)
                    btn.setCursor(Qt.PointingHandCursor)
                    btn.clicked.connect(
                        lambda checked, p=pin, v=value: self._set_pin(p, v))
                    row.addWidget(btn)

            if cfg["direction"] in ("in", "inout"):
                btn = QPushButton("读")
                btn.setObjectName("pinReadBtn")
                btn.setFixedSize(30, 24)
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda checked, p=pin: self._read_pin(p))
                row.addWidget(btn)

            panel_layout.addLayout(row)

        panel_layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _build_func_card(self):
        card = QFrame()
        card.setObjectName("funcCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("高级功能")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        def add_group(caption, groups):
            cap = QLabel(caption)
            cap.setObjectName("funcCaption")
            layout.addWidget(cap)
            for row_items in groups:
                row = QHBoxLayout()
                row.setSpacing(8)
                for text, handler in row_items:
                    btn = QPushButton(text)
                    btn.setObjectName("funcBtn")
                    btn.setCursor(Qt.PointingHandCursor)
                    btn.setMinimumHeight(38)
                    btn.clicked.connect(handler)
                    row.addWidget(btn)
                layout.addLayout(row)

        # 开始/停止采集、1000/2000Hz 已经并进右侧常驻「硬件控制」栏，
        # 这一页只留调试工具（控制台里照样可以手敲 EMG_START 等命令）。
        add_group("调试工具", [
            [("I2C 扫描", self._scan_i2c),
             ("SPI 测试", self._spi_test)],
            [("读取 IMU", self._read_imu),
             ("读取全部", self._read_all)],
            [("ADS1298 复位", self._ads_init),
             ("内部测试信号", self._ads_test_toggle)],
        ])
        return card

    def _build_console_card(self):
        card = QFrame()
        card.setObjectName("consoleCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("命令控制台")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 12))
        self.log_text.setMinimumHeight(140)
        self.log_text.setPlaceholderText("设备响应将显示在这里")
        layout.addWidget(self.log_text)

        cmd_row = QHBoxLayout()
        cmd_row.setSpacing(8)
        self.cmd_input = QLineEdit()
        self.cmd_input.setFont(QFont("Consolas", 13))
        self.cmd_input.setPlaceholderText("输入命令（例：PIN8=1）")
        self.cmd_input.setMinimumHeight(40)
        self.cmd_input.returnPressed.connect(self._send_command)
        send_btn = QPushButton("发送")
        send_btn.setObjectName("sendBtn")
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.setFixedSize(86, 40)
        send_btn.clicked.connect(self._send_command)
        cmd_row.addWidget(self.cmd_input, 1)
        cmd_row.addWidget(send_btn)
        layout.addLayout(cmd_row)
        return card

    def _connect_signals(self):
        self.device_manager.esp_response.connect(self._on_response)
        self.device_manager.pin_status_updated.connect(self._on_status_updated)
        self.device_manager.error_occurred.connect(
            lambda msg: self._log(f"! {msg}")
        )

    # ==================== 命令 ====================

    def _set_pin(self, pin, value):
        self.device_manager.set_pin(pin, value)
        self._log(f"> PIN{pin}={value}")

    def _read_pin(self, pin):
        self.device_manager.read_pin(pin)
        self._log(f"> READ{pin}")

    def _read_all(self):
        self.device_manager.read_all_status()
        self._log("> ALL")

    def _scan_i2c(self):
        self.device_manager.scan_i2c()
        self._log("> I2C_SCAN")

    def _spi_test(self):
        self.device_manager.spi_test()
        self._log("> SPI_TEST")

    def _ads_init(self):
        """重新初始化 ADS1298 寄存器"""
        self.device_manager.ads_init()
        self._log("> ADS_INIT")

    def _ads_test_toggle(self):
        """切换 ADS1298 内部测试信号（约 2Hz 方波），验证 ADC+SPI+蓝牙整条链路"""
        self._ads_test_on = not getattr(self, "_ads_test_on", False)
        cmd = "ADS_TEST=1" if self._ads_test_on else "ADS_TEST=0"
        self.device_manager.send_esp_command(cmd)
        self._log(f"> {cmd}")

    def _start_emg(self):
        """开启 ADS1298 流式采集"""
        self.device_manager.start_emg()
        self._log("> EMG_START")

    def _stop_emg(self):
        """停止流式采集"""
        self.device_manager.stop_emg()
        self._log("> EMG_STOP")

    def _read_imu(self):
        """读一次 LSM6DSR 加速度/角速度"""
        self.device_manager.read_imu()
        self._log("> IMU")

    def _sps_1000(self):
        self.device_manager.set_emg_sps(1000)
        self._log("> EMG_SPS=1000")

    def _sps_2000(self):
        self.device_manager.set_emg_sps(2000)
        self._log("> EMG_SPS=2000")

    def _send_command(self):
        cmd = self.cmd_input.text().strip()
        if cmd:
            self.device_manager.send_esp_command(cmd)
            self._log(f"> {cmd}")
            self.cmd_input.clear()

    # ==================== 响应 / 状态 ====================

    def _on_response(self, response):
        self._log(f"< {response}")
        if response.startswith("STATUS:"):
            from core.esp32_controller import ESP32Controller
            status = ESP32Controller()._parse_status(response)
            self._update_pin_display(status)
        elif "=" in response and not response.startswith("ERROR"):
            try:
                parts = response.replace("OK:", "").strip().split("=")
                if len(parts) == 2 and "PIN" in parts[0]:
                    pin = int(parts[0].replace("PIN", ""))
                    self._set_pin_label(pin, parts[1])
            except (ValueError, IndexError):
                pass

    def _on_status_updated(self, status):
        self._update_pin_display(status)

    def _update_pin_display(self, status):
        mapping = {"LED": 8, "DRDY": 11, "RESET": 9, "START": 12, "PWDN": 13}
        for name, value in status.items():
            pin = mapping.get(name)
            if pin and pin in self.pin_labels:
                self._set_pin_label(pin, str(value))

    def _set_pin_label(self, pin, value):
        if pin not in self.pin_labels:
            return
        label = self.pin_labels[pin]
        label.setText(str(value))
        color = (PixelTheme.COLORS['success']
                 if str(value) == "1" else PixelTheme.COLORS['error'])
        label.setStyleSheet(
            f"color:{color};font-size:16px;font-weight:bold;background:transparent;"
        )

    def _refresh_status(self):
        if self.device_manager.connected:
            self.device_manager.read_all_status()

    def _log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
