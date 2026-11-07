"""
用户迁移视图

把原来的「手势交互」和「交互训练」合成一个功能区：
    ① 实时识别  —— 看当前识别结果（手势 / 关节角）
    ② 标定采集  —— 让新用户按提示做固定动作，采集用于个人化的数据

上位机主窗口只挂载这一个页面；两个子页面的接口保持原样，
主窗口仍然可以用 用户迁移.实时识别 / 用户迁移.标定采集 访问它们。
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from ui.themes.pixel_theme import PixelTheme
from ui.views.gesture_view import GestureInteractionView
from ui.views.training_view import InteractionTrainingView


class 用户迁移视图(QWidget):
    """用户迁移功能区：实时识别 + 标定采集。"""

    def __init__(self, gesture_classifier):
        super().__init__()
        self.gesture_classifier = gesture_classifier

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(10)

        header, _ = PixelTheme.page_header(
            "用户迁移",
            "新用户完成一轮标定动作后生成其专属模型；可同时查看实时识别结果",
            badges=[PixelTheme.badge("个人化", PixelTheme.COLORS['secondary'])],
        )
        layout.addLayout(header)

        tip = QLabel("使用流程：① 标定采集（按提示完成一轮固定动作）→ ② 实时识别"
                     "（通过「个人化迁移」生成该用户的模型，随后在此查看识别结果）。")
        tip.setObjectName("hintText")
        tip.setWordWrap(True)
        layout.addWidget(tip)

        # 迁移入口放在页面级工具条上，两个标签页都能看到
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton
        bar = QFrame()
        bar.setObjectName("algoBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 8, 16, 8)
        self.migrate_btn = QPushButton("个人化迁移（生成该用户模型）")
        self.migrate_btn.setObjectName("funcBtn")
        self.migrate_btn.setCursor(Qt.PointingHandCursor)
        bl.addWidget(self.migrate_btn)
        bl.addStretch()
        layout.addWidget(bar)

        self.实时识别 = GestureInteractionView(gesture_classifier)
        self.标定采集 = InteractionTrainingView()

        self.tabs = QTabWidget()
        self.tabs.setObjectName("migrateTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.标定采集, "① 标定采集（新用户执行一轮）")
        self.tabs.addTab(self.实时识别, "② 实时识别与迁移")
        layout.addWidget(self.tabs)
        self.migrate_btn.clicked.connect(self.实时识别._personalize)

    # ---- 给主窗口用的转发接口 ----
    def clear_data(self):
        self.标定采集.clear_data()

    def 当前子页(self) -> str:
        return "标定采集" if self.tabs.currentIndex() == 0 else "实时识别"
