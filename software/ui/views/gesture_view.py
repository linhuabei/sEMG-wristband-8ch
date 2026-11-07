"""
手势交互界面（支持外部算法导入）
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QGridLayout, QPushButton, QFrame,
                             QProgressBar, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt

from core.algorithm_loader import load_algorithm, AlgorithmError
from ui.themes.pixel_theme import PixelTheme


class GestureInteractionView(QWidget):
    """手势实时识别展示页"""

    DEMO_GESTURES = [
        "握拳", "张开", "捏合", "屈腕",
        "伸腕", "左摆", "右摆", "休息",
    ]

    def __init__(self, gesture_classifier):
        super().__init__()
        self.gesture_classifier = gesture_classifier

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(14)

        live_badge = PixelTheme.badge("实时识别", PixelTheme.COLORS['primary'])
        header, _ = PixelTheme.page_header(
            "实时识别",
            "8 通道 sEMG → 手势 / 关节角度（支持导入外部算法）",
            badges=[live_badge],
        )
        layout.addLayout(header)

        # ---- 算法工具条 ----
        algo_bar = QFrame()
        algo_bar.setObjectName("algoBar")
        bar = QHBoxLayout(algo_bar)
        bar.setContentsMargins(16, 10, 16, 10)
        bar.setSpacing(10)
        self.algo_label = QLabel(f"识别算法: {self.gesture_classifier.algorithm_label()}")
        self.algo_label.setObjectName("statusText")
        bar.addWidget(self.algo_label)
        bar.addStretch()
        self.import_btn = QPushButton("导入算法 .py")
        self.import_btn.setObjectName("funcBtn")
        self.import_btn.setCursor(Qt.PointingHandCursor)
        self.import_btn.clicked.connect(self._import_algorithm)
        bar.addWidget(self.import_btn)
        self.reset_btn = QPushButton("恢复内置")
        self.reset_btn.setObjectName("funcBtn")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self._reset_algorithm)
        bar.addWidget(self.reset_btn)
        self.migrate_btn = QPushButton("个人化迁移")
        self.migrate_btn.setObjectName("funcBtn")
        self.migrate_btn.setCursor(Qt.PointingHandCursor)
        self.migrate_btn.setToolTip(
            "依据新用户的标定数据生成其专属模型：\n"
            "选择「标定采集」生成的 session 目录，程序将生成可双击执行的迁移脚本。")
        self.migrate_btn.clicked.connect(self._personalize)
        bar.addWidget(self.migrate_btn)
        layout.addWidget(algo_bar)

        # ---- 当前手势大卡片 ----
        gesture_card = QFrame()
        gesture_card.setObjectName("gestureCard")
        gc_layout = QHBoxLayout(gesture_card)
        gc_layout.setContentsMargins(24, 18, 24, 18)
        gc_layout.setSpacing(20)

        icon_well = QFrame()
        icon_well.setFixedSize(120, 120)
        icon_well.setStyleSheet(
            f"background-color:{PixelTheme.rgba(PixelTheme.COLORS['primary'], 0.06)};"
            f"border:1px solid {PixelTheme.rgba(PixelTheme.COLORS['primary'], 0.35)};"
            "border-radius:6px;"
        )
        iw_layout = QVBoxLayout(icon_well)
        iw_layout.setContentsMargins(0, 0, 0, 0)
        self.gesture_icon = QLabel("[ ]")
        self.gesture_icon.setObjectName("gestureIcon")
        self.gesture_icon.setAlignment(Qt.AlignCenter)
        iw_layout.addWidget(self.gesture_icon)

        info_col = QVBoxLayout()
        info_col.setSpacing(8)
        self.gesture_name = QLabel("等待手势")
        self.gesture_name.setObjectName("gestureName")
        self.confidence_label = QLabel("置信度 --")
        self.confidence_label.setObjectName("gestureConf")

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(6)
        self.confidence_bar.setObjectName("confidenceBar")

        hint = QLabel("佩戴设备并执行动作后可查看实时识别结果")
        hint.setObjectName("hintText")
        hint.setWordWrap(True)

        info_col.addWidget(self.gesture_name)
        info_col.addWidget(self.confidence_label)
        info_col.addWidget(self.confidence_bar)
        info_col.addStretch()
        info_col.addWidget(hint)

        gc_layout.addWidget(icon_well)
        gc_layout.addLayout(info_col, 1)
        layout.addWidget(gesture_card)

        # ---- 角度输出（外部 angles 算法时显示） ----
        self.angle_card = QFrame()
        self.angle_card.setObjectName("demoCard")
        angle_l = QVBoxLayout(self.angle_card)
        angle_l.setContentsMargins(20, 12, 20, 12)
        angle_title = QLabel("关节角度输出（度）")
        angle_title.setObjectName("sectionTitle")
        self.angle_text = QLabel("--")
        self.angle_text.setObjectName("statusText")
        self.angle_text.setWordWrap(True)
        angle_l.addWidget(angle_title)
        angle_l.addWidget(self.angle_text)
        self.angle_card.hide()
        layout.addWidget(self.angle_card)

        # ---- 动作演示网格 ----
        layout.addLayout(self._build_demo_grid(), 2)

        self.gesture_classifier.gesture_detected.connect(self.show_gesture)
        self.gesture_classifier.angles_ready.connect(self.show_angles)

    # ---------- 算法导入 ----------
    def _import_algorithm(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入算法", "", "Python 算法 (*.py)"
        )
        if not path:
            return
        try:
            algo = load_algorithm(path)
        except AlgorithmError as exc:
            self.algo_label.setText(f"识别算法: 导入失败 - {exc}")
            return
        self.gesture_classifier.set_algorithm(algo)
        self.algo_label.setText(f"识别算法: {algo['name']}（{algo['type']}）")
        self.angle_card.setVisible(algo["type"] == "angles")
        self.gesture_icon.setText("[ 等待 ]")
        self.gesture_name.setText("等待数据")
        self.confidence_label.setText("置信度 --")
        self.confidence_bar.setValue(0)

    def _reset_algorithm(self):
        self.gesture_classifier.clear_algorithm()
        self.algo_label.setText(f"识别算法: {self.gesture_classifier.algorithm_label()}")
        self.angle_card.hide()
        self.angle_text.setText("--")

    # ---------- 个人化迁移（用户迁移） ----------
    def _project_root(self):
        """上位机 目录的上一级 = 项目根目录（算法/ 就在它下面）。"""
        return Path(__file__).resolve().parents[2].parent

    def _personalize(self):
        """选一个 session 目录 → 生成"personalize.bat" → 提示操作步骤。"""
        from core.paths import data_dir as user_data_dir
        start = user_data_dir("training_data")
        folder = QFileDialog.getExistingDirectory(
            self, "选择刚采集完成的 session 目录（位于「training_data」目录下）", start)
        if not folder:
            return
        algo_src = self._project_root() / "算法" / "src"
        script = algo_src / "个人化迁移.py"
        if not script.exists():
            QMessageBox.warning(self, "找不到迁移脚本",
                                f"未找到迁移脚本：\n{script}\n\n"
                                "请确认 算法/src/个人化迁移.py 存在。")
            return
        bat = Path(folder) / "personalize.bat"
        bat.write_text(
            "@echo off\r\n"
            "chcp 65001 >nul\r\n"
            "rem 可选：先设置环境变量 EMG_PYTHON 指向已安装 PyTorch 的 python.exe\r\n"
            "set \"PY=%EMG_PYTHON%\"\r\n"
            "if not defined PY set \"PY=python\"\r\n"
            "\"%PY%\" -c \"import torch\" 2>nul\r\n"
            "if errorlevel 1 (\r\n"
            "  echo [错误] 当前 Python 环境未安装 PyTorch，无法执行模型微调。\r\n"
            "  echo 请在已安装 PyTorch 的计算机上运行本脚本。\r\n"
            "  pause & exit /b 1\r\n"
            ")\r\n"
            f"cd /d \"{algo_src}\"\r\n"
            f"\"%PY%\" \"个人化迁移.py\" --session \"{folder}\" --subject 1 --install\r\n"
            "pause\r\n",
            encoding="utf-8")
        QMessageBox.information(
            self, "已生成个人化迁移脚本",
            f"已生成迁移脚本：\n{bat}\n\n"
            "使用步骤：\n"
            "1）在「用户迁移 → 标定采集」载入「个人化标定方案」，由新用户按提示完成一轮\n"
            "2）返回本页点击「个人化迁移」，选择刚生成的 session 目录\n"
            "3）在已安装 PyTorch 的计算机上双击生成的「personalize.bat」\n"
            "   —— 该脚本将依次执行：数据整理 → 伪标签生成 → 模型微调 → 覆盖插件权重\n"
            "4）返回本页重新点击「导入算法 .py」，即可使用该用户对应的模型\n\n"
            "（若配备数据手套，可将角度 CSV 放入同一目录以获得更好效果）")

    # ---------- 展示 ----------
    def show_gesture(self, gesture, confidence):
        self.gesture_icon.setText(f"[ {gesture} ]")
        self.gesture_name.setText(gesture)
        self.confidence_label.setText(f"置信度 {confidence:.0%}")
        self.confidence_bar.setValue(int(confidence * 100))

    def show_angles(self, angles):
        arr = np.asarray(angles, dtype=np.float64)
        parts = [f"角度{i + 1}: {v:.1f}°" for i, v in enumerate(arr)]
        self.angle_text.setText("    ".join(parts))
        self.gesture_name.setText("连续输出中")
        self.confidence_label.setText("外部算法 · angles 模式")
        self.confidence_bar.setValue(100)

    def _build_demo_grid(self):
        card = QFrame()
        card.setObjectName("demoCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 20, 18)
        card_layout.setSpacing(12)

        title = QLabel("动作演示")
        title.setObjectName("sectionTitle")
        card_layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for i, name in enumerate(self.DEMO_GESTURES):
            btn = QPushButton(name)
            btn.setObjectName("demoBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(42)
            grid.addWidget(btn, i // 4, i % 4)
        for c in range(4):
            grid.setColumnStretch(c, 1)
        card_layout.addLayout(grid)
        card_layout.addStretch()

        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(card)
        return box
