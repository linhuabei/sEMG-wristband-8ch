"""
手势识别分类器模块

默认是内置 RMS 阈值；也可以从“手势交互”页导入外部 .py 算法，
TYPE=gesture 走手势信号，TYPE=angles 走 angles_ready 信号。
"""
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np


class GestureClassifier(QObject):
    gesture_detected = pyqtSignal(str, float)
    angles_ready = pyqtSignal(np.ndarray)

    THRESHOLD_STRONG = 0.30
    THRESHOLD_WEAK = 0.15

    def __init__(self):
        super().__init__()
        self.current_gesture = '无'
        self.algorithm = None      # load_algorithm() 返回的 dict

    # ---------- 外部算法管理 ----------
    def set_algorithm(self, algo):
        self.algorithm = algo
        self.current_gesture = '无'

    def clear_algorithm(self):
        self.algorithm = None
        self.current_gesture = '无'

    def algorithm_label(self):
        if not self.algorithm:
            return "内置阈值"
        return self.algorithm["name"]

    # ---------- 分类 ----------
    def classify(self, features):
        if self.algorithm is not None:
            self._classify_external(features)
            return
        self._classify_builtin(features)

    def _classify_external(self, features):
        algo_type = self.algorithm["type"]
        try:
            result = self.algorithm["module"].predict({"features": features})
        except Exception as exc:
            self.gesture_detected.emit("算法异常", 0.0)
            return

        if algo_type == "angles":
            try:
                arr = np.asarray(result, dtype=np.float64).reshape(-1)
            except Exception:
                arr = np.zeros(1)
            self.angles_ready.emit(arr)
            return

        if isinstance(result, (tuple, list)) and len(result) == 2:
            gesture, conf = result[0], float(result[1])
        else:
            gesture, conf = str(result), 0.9
        self.current_gesture = str(gesture)
        self.gesture_detected.emit(str(gesture), max(0.0, min(1.0, conf)))

    def _classify_builtin(self, features):
        rms = np.asarray(features.get('rms', [0.0]), dtype=np.float64)
        peak = float(np.max(np.abs(rms))) if rms.size else 0.0

        if peak > self.THRESHOLD_STRONG:
            gesture = '握拳'
        elif peak > self.THRESHOLD_WEAK:
            gesture = '张开'
        else:
            gesture = '无'

        self.current_gesture = gesture
        confidence = min(0.99, 0.55 + peak * 0.9) if gesture != '无' else 0.5
        self.gesture_detected.emit(gesture, confidence)
