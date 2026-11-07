"""
Node 浅色主题 - 统一科研浅色风（深色模式已移除）

所有颜色 / 圆角 / 间距集中在此维护；主题只负责配色与排版，不涉及数据来源。
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout


class PixelTheme:
    """浅色主题管理类"""

    COLORS = {
        # 基础表面（暖灰白，页面更安静）
        'background': '#f5f6f8',
        'surface': '#ffffff',
        'card': '#ffffff',
        'hover': '#eef2f6',
        'disabled': '#f0f2f4',
        # 边框（细、浅，不抢内容）
        'border': '#d4dae1',
        'border_light': '#b9c3ce',
        'border_active': '#1f4e79',
        # 强调色（学术蓝，去掉霓虹绿）
        'primary': '#1f4e79',
        'primary_soft': '#4a7fb5',
        'secondary': '#2e6da4',
        'accent': '#8a6d1f',
        # 文本
        'text_primary': '#20242a',
        'text_secondary': '#59636f',
        'text_dim': '#8a94a0',
        # 状态（克制一点的红绿）
        'success': '#2e7d32',
        'warning': '#9a6700',
        'error': '#b3261e',
    }
    # 8 通道波形配色（浅底上保持可读）
    TRACE_COLORS = [
        '#1f6fb2', '#c05a1f', '#1a7f5a', '#a63d40',
        '#5f5aa2', '#0e7490', '#7a8c2d', '#5c6b7a',
    ]
    FONT_FAMILY = "Microsoft YaHei"
    FONT_SIZE_SMALL = 14
    FONT_SIZE_NORMAL = 16
    FONT_SIZE_LARGE = 21
    FONT_SIZE_TITLE = 24

    CARD_NAMES = (
        "QFrame#devicePanel, QFrame#gestureCard, QFrame#pinPanel, QFrame#funcCard, "
        "QFrame#consoleCard, QFrame#trainGestureCard, QFrame#trainRoundCard, "
        "QFrame#trainProgressCard, QFrame#trainStatusCard, QFrame#demoCard, "
        "QFrame#trainPlanCard, QFrame#algoBar, QFrame#infoCard"
    )

    # ==================== 颜色工具 ====================

    @classmethod
    def _rgb(cls, color):
        h = color.lstrip('#')
        return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"

    @classmethod
    def rgba(cls, color, alpha):
        return f"rgba({cls._rgb(color)},{alpha})"

    # ==================== 页面头 / 徽标 ====================

    @staticmethod
    def page_header(title, subtitle=None, badges=None):
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(3)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        left.addWidget(title_label)
        if subtitle:
            sub_label = QLabel(subtitle)
            sub_label.setObjectName("pageSub")
            left.addWidget(sub_label)

        box.addLayout(left)
        box.addStretch(1)
        if badges:
            items = badges if isinstance(badges, (list, tuple)) else [badges]
            for w in items:
                box.addWidget(w)
        return box, title_label

    @classmethod
    def badge(cls, text, color=None, dot=True):
        """朴素文字标签，不再做发光胶囊"""
        c = color or cls.COLORS['text_secondary']
        label = QLabel(text)
        label.setStyleSheet(
            f"color:{c};"
            "background:transparent;"
            "border:none;"
            "padding:2px 0;"
            "font-size:13px;"
        )
        return label

    # ==================== 应用主题 ====================

    @classmethod
    def apply(cls, app):
        app.setStyleSheet(cls.get_stylesheet())
        app.setFont(QFont(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL))

    @classmethod
    def get_stylesheet(cls):
        C = cls.COLORS
        card_qss = cls.CARD_NAMES

        return f"""
        /* ================= 基础 ================= */
        QWidget {{
            background-color: transparent;
            color: {C['text_primary']};
            font-family: "{cls.FONT_FAMILY}";
            font-size: {cls.FONT_SIZE_NORMAL}px;
        }}
        QMainWindow, QDialog {{
            background-color: {C['background']};
        }}
        QToolTip {{
            background-color: {C['surface']};
            color: {C['text_primary']};
            border: 1px solid {C['border_light']};
            border-radius: 4px;
            padding: 4px 8px;
        }}

        /* ================= 文本 ================= */
        QLabel {{
            color: {C['text_primary']};
            background: transparent;
        }}
        QLabel#pageTitle {{
            font-size: {cls.FONT_SIZE_TITLE}px;
            font-weight: bold;
            color: #14202d;
        }}
        QLabel#pageSub {{
            font-size: {cls.FONT_SIZE_SMALL - 1}px;
            color: {C['text_dim']};
        }}
        QLabel#sectionTitle {{
            font-size: 18px;
            font-weight: bold;
            color: {C['text_primary']};
        }}
        QLabel#hintText {{
            color: {C['text_dim']};
            font-size: {cls.FONT_SIZE_SMALL - 1}px;
        }}
        QLabel#funcCaption {{
            color: {C['text_secondary']};
            font-size: 13px;
            font-weight: bold;
            padding-left: 2px;
        }}
        QLabel#gestureIcon {{
            color: {C['primary']};
            font-size: 46px;
            font-weight: bold;
        }}
        QLabel#gestureName {{
            color: #14202d;
            font-size: 28px;
            font-weight: bold;
        }}
        QLabel#gestureConf {{
            color: {C['text_secondary']};
            font-size: {cls.FONT_SIZE_SMALL}px;
        }}
        QLabel#roundNumber {{
            color: {C['primary']};
            font-size: 32px;
            font-weight: bold;
        }}
        QLabel#roundCaption, QLabel#countdownCaption {{
            color: {C['text_dim']};
            font-size: 14px;
        }}
        QLabel#countdown {{
            color: {C['secondary']};
            font-size: 26px;
            font-weight: bold;
        }}
        QLabel#statusText {{
            color: {C['text_secondary']};
            font-size: {cls.FONT_SIZE_SMALL}px;
        }}
        QLabel#statusFile {{
            color: {C['text_dim']};
            font-size: 14px;
        }}
        QLabel#pinHeader {{
            color: {C['text_dim']};
            font-size: {cls.FONT_SIZE_SMALL - 1}px;
            font-weight: bold;
        }}
        QLabel#pinNumber {{
            color: {C['secondary']};
            font-size: {cls.FONT_SIZE_SMALL + 1}px;
            font-weight: bold;
        }}
        QLabel#pinName {{
            color: {C['text_primary']};
            font-size: {cls.FONT_SIZE_SMALL}px;
        }}
        QLabel#pinType {{
            color: {C['text_dim']};
            font-size: 14px;
        }}
        QLabel#pinState {{
            color: {C['text_dim']};
            font-size: {cls.FONT_SIZE_SMALL + 1}px;
            font-weight: bold;
        }}
        QScrollArea {{
            background: transparent;
            border: none;
        }}

        /* ================= 按钮 ================= */
        QPushButton {{
            background-color: {C['surface']};
            color: {C['text_secondary']};
            border: 1px solid {C['border']};
            border-radius: 4px;
            padding: 7px 16px;
            font-size: {cls.FONT_SIZE_NORMAL}px;
        }}
        QPushButton:hover {{
            background-color: {C['hover']};
            border-color: {C['primary']};
            color: {C['primary']};
        }}
        QPushButton:pressed {{
            background-color: {C['primary']};
            border-color: {C['primary']};
            color: #ffffff;
        }}
        QPushButton:checked {{
            background-color: {cls.rgba(C['primary'], 0.12)};
            color: {C['primary']};
            border: 1px solid {C['primary']};
            font-weight: bold;
        }}
        QPushButton#rateBtn {{
            background-color: transparent;
            color: {C['text_secondary']};
            border: 1px solid {C['border_light']};
            border-radius: 3px;
            padding: 5px 12px;
            font-size: 14px;
        }}
        QPushButton#rateBtn:hover {{
            color: {C['primary']};
            border-color: {C['primary']};
            background-color: {cls.rgba(C['primary'], 0.06)};
        }}
        QPushButton#rateBtn:checked {{
            background-color: {C['primary']};
            color: #ffffff;
            border-color: {C['primary']};
            font-weight: bold;
        }}
        QPushButton#acqStartBtn {{
            background-color: {C['primary']};
            color: #ffffff;
            border: 1px solid {C['primary']};
            border-radius: 4px;
            padding: 6px 14px;
            font-weight: bold;
        }}
        QPushButton#acqStartBtn:hover {{
            background-color: {C['primary_soft']};
        }}
        QPushButton#acqStopBtn {{
            background-color: transparent;
            color: {C['error']};
            border: 1px solid {C['error']};
            border-radius: 4px;
            padding: 6px 14px;
            font-weight: bold;
        }}
        QPushButton#acqStopBtn:hover {{
            background-color: {C['error']};
            color: #ffffff;
        }}
        QPushButton#recBtn {{
            background-color: transparent;
            color: {C['accent']};
            border: 1px solid {C['accent']};
            border-radius: 4px;
            padding: 6px 14px;
            font-weight: bold;
        }}
        QPushButton#recBtn:hover {{
            background-color: {C['accent']};
            color: #ffffff;
        }}
        QPushButton#recBtn:disabled {{
            background-color: transparent;
            border-color: {C['border']};
            color: {C['text_dim']};
        }}
        QLabel#recLabel {{
            color: {C['text_dim']};
            font-size: 14px;
        }}
        QPushButton:disabled {{
            background-color: {C['disabled']};
            border-color: {C['border']};
            color: {C['text_dim']};
        }}
        QPushButton#demoBtn {{
            background-color: #f6f9fc;
            border-radius: 4px;
            padding: 5px 8px;
            color: {C['text_secondary']};
        }}
        QPushButton#demoBtn:hover {{
            border-color: {C['secondary']};
            color: {C['secondary']};
            background-color: {cls.rgba(C['secondary'], 0.07)};
        }}

        QPushButton#scanBtn {{
            background-color: transparent;
            color: {C['text_secondary']};
            border: 1px solid #b9c8d6;
            border-radius: 3px;
            padding: 6px 10px;
        }}
        QPushButton#scanBtn:hover {{
            color: {C['secondary']};
            border-color: {C['secondary']};
            background-color: {cls.rgba(C['secondary'], 0.07)};
        }}
        QPushButton#scanBtn:pressed {{
            background-color: {C['secondary']};
            color: #ffffff;
        }}

        QPushButton#connectBtn {{
            background-color: {cls.rgba(C['primary'], 0.10)};
            color: {C['primary']};
            border: 1px solid {C['primary']};
            border-radius: 3px;
            padding: 6px 10px;
            font-weight: bold;
        }}
        QPushButton#connectBtn:hover {{
            background-color: {C['primary']};
            color: #ffffff;
        }}
        QPushButton#connectBtn:disabled {{
            background-color: transparent;
            border-color: {C['border']};
            color: {C['text_dim']};
        }}

        QPushButton#funcBtn, QPushButton#sendBtn, QPushButton#trainStartBtn,
        QPushButton#trainStopBtn {{
            border-radius: 4px;
            padding: 8px 14px;
        }}
        QPushButton#sendBtn, QPushButton#trainStartBtn {{
            background-color: {cls.rgba(C['primary'], 0.10)};
            color: {C['primary']};
            border: 1px solid {C['primary']};
            font-weight: bold;
        }}
        QPushButton#sendBtn:hover, QPushButton#trainStartBtn:hover {{
            background-color: {C['primary']};
            color: #ffffff;
        }}
        QPushButton#sendBtn:disabled, QPushButton#trainStartBtn:disabled {{
            background-color: transparent;
            color: {C['text_dim']};
            border-color: {C['border']};
        }}
        QPushButton#trainStopBtn {{
            background-color: {cls.rgba(C['error'], 0.08)};
            color: {C['error']};
            border: 1px solid {C['error']};
        }}
        QPushButton#trainStopBtn:hover {{
            background-color: {C['error']};
            color: #ffffff;
        }}

        QPushButton#pinOnBtn, QPushButton#pinOffBtn, QPushButton#pinReadBtn {{
            border-radius: 3px;
            padding: 4px;
            font-weight: bold;
            font-size: {cls.FONT_SIZE_SMALL}px;
        }}
        QPushButton#pinOnBtn {{
            background-color: transparent;
            color: {C['success']};
            border: 1px solid {cls.rgba(C['success'], 0.65)};
        }}
        QPushButton#pinOnBtn:hover {{
            background-color: {C['success']};
            color: #ffffff;
        }}
        QPushButton#pinOffBtn {{
            background-color: transparent;
            color: {C['error']};
            border: 1px solid {cls.rgba(C['error'], 0.65)};
        }}
        QPushButton#pinOffBtn:hover {{
            background-color: {C['error']};
            color: #ffffff;
        }}
        QPushButton#pinReadBtn {{
            background-color: transparent;
            color: {C['secondary']};
            border: 1px solid {cls.rgba(C['secondary'], 0.65)};
        }}
        QPushButton#pinReadBtn:hover {{
            background-color: {C['secondary']};
            color: #ffffff;
        }}

        /* ================= 卡片 ================= */
        {card_qss} {{
            background-color: {C['card']};
            border: 1px solid {C['border']};
            border-radius: 3px;
        }}
        QFrame#trainGestureCard {{
            border-radius: 12px;
        }}

        /* ================= 列表 ================= */
        QListWidget {{
            background-color: {C['surface']};
            border: 1px solid {C['border']};
            border-radius: 4px;
            color: {C['text_primary']};
            padding: 3px;
            font-size: {cls.FONT_SIZE_SMALL}px;
            outline: none;
        }}
        QListWidget::item {{
            padding: 7px 9px;
            margin: 1px 2px;
            border-radius: 3px;
            color: {C['text_secondary']};
        }}
        QListWidget::item:hover {{
            background-color: {cls.rgba(C['secondary'], 0.06)};
            color: {C['text_primary']};
        }}
        QListWidget::item:selected {{
            background-color: {cls.rgba(C['primary'], 0.12)};
            color: {C['primary']};
            border-left: 2px solid {C['primary']};
        }}

        /* ================= 输入 / 输出 ================= */
        QLineEdit, QTextEdit {{
            background-color: {C['surface']};
            border: 1px solid {C['border']};
            border-radius: 4px;
            color: {C['text_primary']};
            padding: 5px 10px;
            selection-background-color: {cls.rgba(C['primary'], 0.25)};
        }}
        QLineEdit:focus, QTextEdit:focus {{
            border: 1px solid {C['primary']};
        }}

        /* ================= 进度条 ================= */
        QProgressBar {{
            background-color: #e8eef5;
            border: 1px solid {C['border']};
            border-radius: 3px;
            text-align: center;
            color: {C['text_secondary']};
            font-size: {cls.FONT_SIZE_SMALL - 1}px;
        }}
        QProgressBar::chunk {{
            background-color: {C['primary']};
            border-radius: 2px;
        }}

        /* ================= 滚动条 ================= */
        QScrollBar:vertical {{
            background: transparent;
            width: 8px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background: {C['border']};
            border-radius: 4px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {C['border_light']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            height: 0;
            background: transparent;
        }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 8px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal {{
            background: {C['border']};
            border-radius: 4px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {C['border_light']};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            width: 0;
            background: transparent;
        }}

        /* ================= 复选框 ================= */
        QCheckBox {{
            color: {C['text_secondary']};
            font-size: {cls.FONT_SIZE_SMALL}px;
            spacing: 8px;
        }}
        QCheckBox:hover {{
            color: {C['text_primary']};
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {C['border_light']};
            border-radius: 4px;
            background-color: {C['surface']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {C['primary']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {C['primary']};
            border-color: {C['primary']};
        }}

        /* ================= 菜单（右键） ================= */
        QMenu {{
            background-color: {C['surface']};
            color: {C['text_primary']};
            border: 1px solid {C['border_light']};
            border-radius: 3px;
            padding: 4px;
        }}
        QMenu::item {{
            padding: 6px 22px;
            border-radius: 4px;
            font-size: {cls.FONT_SIZE_SMALL}px;
        }}
        QMenu::item:selected {{
            background-color: {C['hover']};
            color: {C['primary']};
        }}

        /* ================= 顶栏文字 ================= */
        QLabel#appTitle {{
            color: #14202d;
            font-size: 18px;
            font-weight: bold;
        }}
        QLabel#appSub {{
            color: {C['text_dim']};
            font-size: 13px;
        }}
        QLabel#connLed {{
            background: transparent;
            font-size: 12px;
        }}
        QLabel#connStatus {{
            background: transparent;
            font-size: 14px;
            font-weight: bold;
        }}

        /* ================= 主窗口骨架 ================= */
        QWidget#header {{
            background-color: {C['surface']};
            border-bottom: 1px solid {C['border']};
        }}
        QFrame#sidebar {{
            background-color: {C['surface']};
            border-right: 1px solid {C['border']};
        }}
        QStackedWidget#content {{
            background-color: {C['background']};
            border: none;
        }}
        QFrame#sep {{
            background-color: {C['border']};
            max-height: 1px;
            margin: 6px 16px;
            border: none;
        }}
        QLabel#footer {{
            color: {C['text_dim']};
            font-size: 13px;
            padding: 14px 16px;
        }}
        """

    @classmethod
    def get_pixel_font(cls, size=None):
        if size is None:
            size = cls.FONT_SIZE_NORMAL
        return QFont(cls.FONT_FAMILY, size)
