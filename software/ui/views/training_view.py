"""
交互训练界面
"""
import csv
import os
import random
import time as tm
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
                             QWidget, QCheckBox, QSpinBox, QDoubleSpinBox, QFileDialog)
import pyqtgraph as pg

from core.paths import data_dir as user_data_dir
from core.training_plan import (default_plan, default_plan_dir, total_rounds,
                               build_queue, save_plan, load_plan)
from ui.themes.pixel_theme import PixelTheme


class InteractionTrainingView(QWidget):
    """带标签的交互式 sEMG 数据采集训练页"""

    GESTURES = [
        ("握拳", "1", 1),
        ("十指张开", "2", 2),
        ("OK手势", "3", 3),
        ("比大拇指", "4", 4),
        ("比耶", "5", 5),
        ("捏合", "6", 6),
    ]
    REST_CODE = 0

    def __init__(self):
        super().__init__()
        self.is_training = False
        self.current_gesture = None
        self.current_code = self.REST_CODE
        self.last_code = None
        self.round = 0
        self.total_rounds = 10
        self.plan = default_plan()
        self._queue = []
        self.round_duration = float(self.plan.get('duration_seconds', 3.0))
        self._plan_file = ''
        self.collecting = False
        self.collect_time_left = 0
        # 轮次之间的休息（方案 JSON 里写了 rest_seconds 才启用；旧方案不受影响）
        self._resting = False
        self._round_done = False
        self.channels = 8
        self.buffer_size = 600
        self.data_buffers = [np.zeros(self.buffer_size) for _ in range(self.channels)]
        self.current_index = 0
        self.emg_data_ready = False
        self.csv_file = None
        self.csv_writer = None
        # 用绝对路径（exe 旁），避免从别处启动时把目录建在奇怪的地方
        self.data_dir = user_data_dir("training_data")

        self.round_timer = QTimer(self)
        self.round_timer.timeout.connect(self._next_round)
        self.collect_timer = QTimer(self)
        self.collect_timer.timeout.connect(self._collect_tick)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_plot)

        self._init_ui()
        self.pg.setMaximum(total_rounds(self.plan))
        self.br.setText(f"0 / {total_rounds(self.plan)}")
        self._ensure_data_dir()
        self.update_timer.start(33)

    def _ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    # ==================== UI ====================

    def _init_ui(self):
        ml = QVBoxLayout(self)
        ml.setContentsMargins(24, 20, 24, 18)
        ml.setSpacing(14)

        badge = PixelTheme.badge("自动打标", PixelTheme.COLORS['secondary'])
        header, _ = PixelTheme.page_header(
            "标定采集",
            "按提示完成动作，软件自动为 8 通道数据写入标签",
            badges=[badge],
        )
        ml.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(16)

        # ---- 左：提示卡 + 波形 ----
        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(self._build_plan_card())
        left.addWidget(self._build_prompt_card())

        pg.setConfigOptions(antialias=True)
        self.pw = pg.GraphicsLayoutWidget()
        self.pw.setBackground('#ffffff')
        self.pw.setMinimumHeight(260)
        self.plots = []
        self.curves = []
        colors = list(PixelTheme.TRACE_COLORS)
        for i in range(self.channels):
            p = self.pw.addPlot(row=i // 2, col=i % 2)
            p.showGrid(x=True, y=True, alpha=0.16)
            p.setYRange(-1.2, 1.2)
            p.hideButtons()
            p.setMenuEnabled(False)
            p.setTitle(f'CH{i + 1}', color='#7d9abc', size='11pt')

            la = p.getAxis('left')
            la.setPen(pg.mkPen('#d3dde7'))
            la.setTextPen('#5f7083')
            la.setWidth(42)
            la.setStyle(tickLength=-4)
            la.setTickFont(QFont("Microsoft YaHei", 10))

            ba = p.getAxis('bottom')
            ba.setPen(pg.mkPen('#d3dde7'))
            ba.setTextPen('#5f7083')
            ba.setStyle(tickLength=-4)
            ba.setTickFont(QFont("Microsoft YaHei", 10))

            self.plots.append(p)
            self.curves.append(p.plot(pen=pg.mkPen(color=colors[i], width=1.0)))
        left.addWidget(self.pw, 1)
        body.addLayout(left, 3)

        # ---- 右：状态 / 进度 / 操作 ----
        right = QVBoxLayout()
        right.setSpacing(12)
        right.setContentsMargins(0, 0, 0, 0)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMinimumWidth(270)
        right_widget.setMaximumWidth(320)

        right.addWidget(self._build_round_card())
        right.addWidget(self._build_progress_card())
        right.addWidget(self._build_status_card())
        right.addStretch(1)

        # 开始/停止这两个按钮由主窗口搬进右侧常驻「硬件控制」栏，
        # 这里只创建控件本体（ui/widgets/control_panel.py 负责摆放）。
        self.sb = QPushButton("开 始 训 练", self)
        self.sb.setObjectName("trainStartBtn")
        self.sb.setCursor(Qt.PointingHandCursor)
        self.sb.setMinimumHeight(42)
        self.sb.clicked.connect(self._start_training)
        self.eb = QPushButton("停 止", self)
        self.eb.setObjectName("trainStopBtn")
        self.eb.setCursor(Qt.PointingHandCursor)
        self.eb.setMinimumHeight(42)
        self.eb.setEnabled(False)
        self.eb.clicked.connect(self._stop_training)
        tip = QLabel("开始/停止位于右侧「硬件控制 → 标定采集」栏")
        tip.setObjectName("hintText")
        tip.setWordWrap(True)
        right.addWidget(tip)

        body.addWidget(right_widget, 1)
        ml.addLayout(body, 1)

    def _build_plan_card(self):
        """测试方案：选动作、定次数和时长，可保存/载入 JSON"""
        card = QFrame()
        card.setObjectName("trainPlanCard")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(18, 12, 18, 12)
        cl.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel("测试方案")
        title.setObjectName("sectionTitle")
        self.plan_name_label = QLabel("")
        self.plan_name_label.setObjectName("hintText")
        head.addWidget(title)
        head.addWidget(self.plan_name_label)
        head.addStretch()
        load_btn = QPushButton("载入")
        load_btn.setObjectName("funcBtn")
        load_btn.setCursor(Qt.PointingHandCursor)
        load_btn.clicked.connect(self._load_plan)
        save_btn = QPushButton("保存")
        save_btn.setObjectName("funcBtn")
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._save_plan)
        head.addWidget(load_btn)
        head.addWidget(save_btn)
        cl.addLayout(head)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        self._plan_widgets = []
        for i, g in enumerate(self.plan["gestures"]):
            row = i // 2
            col = i % 2
            cell = QHBoxLayout()
            cell.setSpacing(6)
            chk = QCheckBox(g["name"])
            chk.setChecked(int(g.get("rounds", 0)) > 0)
            spin = QSpinBox()
            spin.setRange(0, 50)
            spin.setValue(int(g.get("rounds", 0)))
            spin.setFixedWidth(64)
            spin.setSuffix(" 次")
            chk.stateChanged.connect(lambda _=None: self._refresh_plan_summary())
            spin.valueChanged.connect(lambda _=None: self._refresh_plan_summary())
            cell.addWidget(chk)
            cell.addWidget(spin)
            cell.addStretch()
            grid.addLayout(cell, row, col)
            self._plan_widgets.append((g, chk, spin))
        cl.addLayout(grid)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        dur_lbl = QLabel("每轮时长")
        dur_lbl.setObjectName("hintText")
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(1.0, 20.0)
        self.dur_spin.setSingleStep(0.5)
        self.dur_spin.setValue(float(self.plan.get("duration_seconds", 3.0)))
        self.dur_spin.setSuffix(" 秒")
        self.dur_spin.valueChanged.connect(lambda _=None: self._refresh_plan_summary())
        self.plan_summary = QLabel("")
        self.plan_summary.setObjectName("hintText")
        bottom.addWidget(dur_lbl)
        bottom.addWidget(self.dur_spin)
        bottom.addStretch()
        bottom.addWidget(self.plan_summary)
        cl.addLayout(bottom)
        self._refresh_plan_summary()
        return card

    def _build_prompt_card(self):
        gc = QFrame()
        gc.setObjectName("trainGestureCard")
        gc.setFixedHeight(138)
        gl = QHBoxLayout(gc)
        gl.setContentsMargins(24, 14, 24, 14)
        gl.setSpacing(22)

        self.gd = QLabel("—")
        self.gd.setObjectName("gestureIcon")
        self.gd.setAlignment(Qt.AlignCenter)
        self.gd.setFixedWidth(86)
        gl.addWidget(self.gd)

        gi = QVBoxLayout()
        gi.setSpacing(4)
        self.gn = QLabel("准备开始")
        self.gn.setObjectName("gestureName")
        self.ri = QLabel("")
        self.ri.setObjectName("roundCaption")
        gi.addWidget(self.gn)
        gi.addWidget(self.ri)
        gi.addStretch()
        gl.addLayout(gi)
        gl.addStretch()
        return gc

    def _build_round_card(self):
        rc = QFrame()
        rc.setObjectName("trainRoundCard")
        rl = QVBoxLayout(rc)
        rl.setContentsMargins(18, 14, 18, 14)
        rl.setSpacing(2)
        rt = QLabel("训练轮次")
        rt.setObjectName("roundCaption")
        rt.setAlignment(Qt.AlignCenter)
        self.br = QLabel("0 / 10")
        self.br.setObjectName("roundNumber")
        self.br.setAlignment(Qt.AlignCenter)
        rl.addWidget(rt)
        rl.addWidget(self.br)
        return rc

    def _build_progress_card(self):
        pc = QFrame()
        pc.setObjectName("trainProgressCard")
        pl = QVBoxLayout(pc)
        pl.setContentsMargins(18, 14, 18, 14)
        pl.setSpacing(8)
        pt = QLabel("本轮剩余")
        pt.setObjectName("roundCaption")
        pt.setAlignment(Qt.AlignCenter)
        self.cd = QLabel("3.0 秒")
        self.cd.setObjectName("countdown")
        self.cd.setAlignment(Qt.AlignCenter)
        pl.addWidget(pt)
        pl.addWidget(self.cd)

        self.pg = QProgressBar()
        self.pg.setMaximum(self.total_rounds)
        self.pg.setValue(0)
        self.pg.setFormat("%v / %m 轮")
        self.pg.setFixedHeight(12)
        self.pg.setObjectName("trainProgress")
        pl.addWidget(self.pg)
        return pc

    def _build_status_card(self):
        sc = QFrame()
        sc.setObjectName("trainStatusCard")
        sl = QVBoxLayout(sc)
        sl.setContentsMargins(18, 14, 18, 14)
        sl.setSpacing(6)
        st_title = QLabel("当前状态")
        st_title.setObjectName("roundCaption")
        self.st = QLabel("点击「开始训练」以启动采集")
        self.st.setObjectName("statusText")
        self.st.setWordWrap(True)
        self.ds = QLabel("")
        self.ds.setObjectName("statusFile")
        self.ds.setWordWrap(True)
        sl.addWidget(st_title)
        sl.addWidget(self.st)
        sl.addWidget(self.ds)
        sl.addStretch()
        return sc

    # ==================== 数据 ====================

    def update_emg_data(self, data):
        dc = data.copy()
        for i in range(min(len(dc), self.channels)):
            self.data_buffers[i][self.current_index] = dc[i]
        self.current_index = (self.current_index + 1) % self.buffer_size
        self.emg_data_ready = True
        if self.csv_writer:
            row = [f"{tm.time():.3f}", str(self.current_code)]
            row += [f"{v:.6f}" for v in dc]
            self.csv_writer.writerow(row)

    def _update_plot(self):
        if self.emg_data_ready:
            for i in range(self.channels):
                if self.plots[i].isVisible():
                    self.curves[i].setData(
                        np.roll(self.data_buffers[i], -self.current_index)
                    )

    # ==================== CSV ====================

    def _open_csv(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{self.data_dir}/training_{ts}.csv"
        self.csv_file = open(fn, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["timestamp", "gesture_code"]
        header += [f"CH{i + 1}" for i in range(self.channels)]
        self.csv_writer.writerow(header)
        self.ds.setText(f"记录文件：{os.path.basename(fn)}")

    def _close_csv(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    # ==================== 训练流程 ====================

    def _start_training(self):
        self.is_training = True
        self.round = 0
        self.last_code = None
        self._resting = False
        self._round_done = False
        st = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = os.path.join(user_data_dir("training_data"), f"session_{st}")
        self._ensure_data_dir()
        self._open_csv()

        self.plan = self._sync_plan_from_ui()
        self._queue, self.round_duration = build_queue(self.plan)
        self.total_rounds = len(self._queue)
        if self.total_rounds == 0:
            self.is_training = False
            self.st.setText("方案中未勾选任何动作，请先勾选")
            return
        self.pg.setMaximum(self.total_rounds)
        self.pg.setValue(0)
        self.br.setText(f"0 / {self.total_rounds}")
        self.sb.setEnabled(False)
        self.eb.setEnabled(True)
        self.current_gesture = "休息"
        self.current_code = self.REST_CODE
        self.gd.setText("—")
        self.gn.setText("准备开始")
        self.ri.setText(f"方案：{self.plan.get('name', '未命名')}")
        self.st.setText("请保持放松，即将开始")
        self.cd.setText("准备")
        QTimer.singleShot(3000, self._next_round)

    def _stop_training(self):
        self.is_training = False
        self.round_timer.stop()
        self.collect_timer.stop()
        self.current_code = self.REST_CODE
        self._resting = False
        self._round_done = False
        self._close_csv()
        self.gd.setText("—")
        self.gn.setText("训练结束")
        self.ri.setText("")
        self.cd.setText("—")
        self.sb.setEnabled(True)
        self.eb.setEnabled(False)
        self.st.setText(f"训练结束，共 {self.round} 轮")
        self.ds.setText(f"已保存至 {self.data_dir}")

    def _next_round(self):
        if not self.is_training:
            return
        rest_sec = float(self.plan.get("rest_seconds", 0.0) or 0.0)
        if self._resting:                       # 休息结束 → 接着做下一轮
            self._resting = False
        elif rest_sec > 0 and self._round_done:  # 一轮刚做完 → 先休息
            self._round_done = False
            if self.round >= self.total_rounds or not self._queue:
                self._stop_training()
                return
            self._enter_rest(rest_sec)
            return
        if self.round >= self.total_rounds:
            self._stop_training()
            return
        if not self._queue:
            self._stop_training()
            return
        item = self._queue.pop(0)
        name = item["name"]
        code = item["code"]
        self.round += 1
        self.pg.setValue(self.round)
        self.br.setText(f"{self.round} / {self.total_rounds}")
        self.current_gesture = name
        self.current_code = code
        self.last_code = code
        self.gd.setText(str(code))
        self.gn.setText(name)
        self.ri.setText(f"第 {self.round}/{self.total_rounds} 轮")
        self.st.setText(f"请做「{name}」")
        self.collect_time_left = self.round_duration
        self.collecting = True
        self.cd.setText(f"{self.round_duration:.1f} 秒")
        self.collect_timer.start(50)
        self.round_timer.start(int(self.round_duration * 1000))
        self._round_done = True

    def _enter_rest(self, seconds):
        """轮次之间的休息段：期间写标签 0（休息），用于给模型提供"休息类"样本。

        只有在方案 JSON 里写了 "rest_seconds" 才会走到这里；旧方案行为不变。
        """
        self._resting = True
        self.collecting = False
        self.current_gesture = "休息"
        self.current_code = self.REST_CODE
        self.gn.setText("休息")
        self.gd.setText("0")
        self.ri.setText("休息中")
        self.st.setText("请放松，保持手部不动")
        self.cd.setText(f"{seconds:.1f} 秒")
        self.round_timer.start(int(max(0.5, seconds) * 1000))

    # ==================== 测试方案 ====================

    def _sync_plan_from_ui(self):
        gestures = []
        for g, chk, spin in self._plan_widgets:
            gestures.append({
                "name": g["name"],
                "code": g["code"],
                "rounds": spin.value() if chk.isChecked() else 0,
            })
        self.plan["gestures"] = gestures
        self.plan["duration_seconds"] = float(self.dur_spin.value())
        return self.plan

    def _refresh_plan_summary(self):
        self._sync_plan_from_ui()
        n = total_rounds(self.plan)
        self.plan_name_label.setText(self.plan.get("name", ""))
        self.plan_summary.setText(f"共 {n} 轮 · {self.dur_spin.value():.1f}s/轮")

    def _load_plan(self):
        folder = default_plan_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, "载入测试方案", folder, "方案文件 (*.json)"
        )
        if not path:
            return
        try:
            plan = load_plan(path)
        except Exception as exc:
            self.st.setText(f"载入失败: {exc}")
            return
        self.plan = plan
        self._plan_file = path
        for g, chk, spin in self._plan_widgets:
            match = next((pg for pg in plan["gestures"]
                          if int(pg["code"]) == int(g["code"])), None)
            rounds = int(match["rounds"]) if match else 0
            chk.setChecked(rounds > 0)
            spin.setValue(rounds)
        self.dur_spin.setValue(float(plan.get("duration_seconds", 3.0)))
        self._refresh_plan_summary()
        self.st.setText(f"已载入方案: {os.path.basename(path)}")

    def _save_plan(self):
        plan = self._sync_plan_from_ui()
        folder = default_plan_dir()
        name = plan.get("name", "方案").strip() or "方案"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存测试方案",
            os.path.join(folder, name + ".json"),
            "方案文件 (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_plan(plan, path)
        except Exception as exc:
            self.st.setText(f"保存失败: {exc}")
            return
        self._plan_file = path
        self.st.setText(f"已保存: {os.path.basename(path)}")

    def _collect_tick(self):
        self.collect_time_left -= 0.05
        if self.collect_time_left <= 0:
            self.collect_timer.stop()
            self.collecting = False
            self.cd.setText("完成")
        else:
            self.cd.setText(f"{self.collect_time_left:.1f} 秒")

    def clear_data(self):
        """模式切换时清空显示缓冲"""
        self.current_index = 0
        self.emg_data_ready = False
        for i in range(self.channels):
            self.data_buffers[i][:] = 0.0

    def apply_trace_colors(self, colors):
        """跟随监视页通道配色"""
        palette = list(colors)
        while len(palette) < self.channels:
            palette.extend(palette)
        for i in range(self.channels):
            self.curves[i].setPen(pg.mkPen(color=palette[i], width=1.0))

    def on_gesture_detected(self, gesture, confidence):
        """预留：后续接入实时模型后，可在训练中做一致性校验"""
        pass
