"""
IMU 姿态显示（LSM6DSR）

固件里的 `IMU` 命令每次返回一行：
    IMU:AX=0.012,AY=-0.003,AZ=0.998,GX=0.12,GY=-0.34,GZ=0.05
其中加速度单位 g，角速度单位 dps。上位机按 20 Hz 轮询拿这些数据。

姿态解算用 Mahony 互补滤波（加速度计修横滚/俯仰，陀螺仪积分补动态）。
注意：没有磁力计，航向角（yaw）会缓慢漂移，只有横滚/俯仰是绝对姿态。

3D 显示用 QPainter 手绘线框/实体方块，不依赖 OpenGL（打包体积和兼容性都好）。
"""
import numpy as np
from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from ui.themes.pixel_theme import PixelTheme


# ==================== 姿态解算 ====================

def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q):
    """四元数 (w,x,y,z) -> 旋转矩阵（体系 -> 世界系）"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class 姿态估计:
    """Mahony 互补滤波：加速度计 + 陀螺仪 -> 姿态四元数"""

    def __init__(self, kp=2.0, ki=0.05):
        self.kp = kp
        self.ki = ki
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.integral = np.zeros(3)
        self.last_t = None
        self.ready = False

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.integral[:] = 0.0
        self.last_t = None
        self.ready = False

    def matrix(self):
        return quat_to_matrix(self.q)

    def euler(self):
        """返回 (横滚 roll, 俯仰 pitch, 航向 yaw)，单位度"""
        R = self.matrix()
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        return float(roll), float(pitch), float(yaw)

    def _init_from_accel(self, a):
        """用静止时的重力方向直接给出初始姿态，避免起步时慢慢收敛"""
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(a, z)
        c = float(np.dot(a, z))
        if np.linalg.norm(v) < 1e-6:
            self.q = np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1.0, 0, 0])
        else:
            axis = v / np.linalg.norm(v)
            ang = np.arccos(np.clip(c, -1.0, 1.0))
            s = np.sin(ang / 2.0)
            self.q = np.array([np.cos(ang / 2.0),
                               axis[0] * s, axis[1] * s, axis[2] * s])
        self.q = self.q / np.linalg.norm(self.q)

    def update(self, ax, ay, az, gx, gy, gz, t):
        """喂一帧 IMU 数据（g / dps / 秒），返回当前旋转矩阵"""
        a = np.array([ax, ay, az], dtype=float)
        g = np.radians([gx, gy, gz])
        norm = float(np.linalg.norm(a))

        if self.last_t is None:
            self.last_t = t
            if norm > 1e-3:
                self._init_from_accel(a / norm)
                self.ready = True
            return self.matrix()

        dt = float(np.clip(t - self.last_t, 1e-3, 0.2))
        self.last_t = t

        if norm > 0.1:                      # 运动太剧烈时别用加速度计修正
            a = a / norm
            v = quat_to_matrix(self.q).T @ np.array([0.0, 0.0, 1.0])
            e = np.cross(a, v)
            self.integral += e * dt
            g = g + self.kp * e + self.ki * self.integral

        dq = 0.5 * _qmul(self.q, np.array([0.0, g[0], g[1], g[2]])) * dt
        self.q = self.q + dq
        self.q = self.q / np.linalg.norm(self.q)
        return self.matrix()


# ==================== 3D 姿态方块 ====================

class 姿态指示器(QWidget):
    """把小方块按当前姿态转起来（QPainter 手绘，不用 OpenGL）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.R = np.eye(3)          # 体系 -> 世界系
        self.R0 = np.eye(3)         # 零位校准参考

    def set_rotation(self, R):
        self.R = np.asarray(R, dtype=float)
        self.update()

    def set_zero(self, R):
        self.R0 = np.asarray(R, dtype=float)
        self.update()

    def clear_zero(self):
        self.R0 = np.eye(3)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        C = PixelTheme.COLORS

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        scale = min(w, h) * 0.30
        R = self.R0.T @ self.R          # 相对零位的姿态

        def project(points):
            """世界系坐标 -> 屏幕坐标（X 右、Z 上、Y 为深度）"""
            pts = np.atleast_2d(points)
            world = pts @ R.T
            depth = world[:, 1]
            persp = 6.0 / (6.0 + depth)
            sx = cx + world[:, 0] * scale * persp
            sy = cy - world[:, 2] * scale * persp
            return np.stack([sx, sy], axis=1), depth

        # ---- 方块 ----
        hx, hy, hz = 1.0, 0.62, 0.30
        verts = np.array([
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ], dtype=float)
        faces = [
            ((0, 1, 2, 3), (0, 0, -1)), ((4, 5, 6, 7), (0, 0, 1)),
            ((0, 1, 5, 4), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
            ((1, 2, 6, 5), (1, 0, 0)), ((0, 3, 7, 4), (-1, 0, 0)),
        ]
        light = np.array([-0.35, -0.55, 0.76])
        light = light / np.linalg.norm(light)
        base = QColor(C['secondary'])

        screen, depth = project(verts)
        draw_order = []
        for idx, normal in faces:
            n_world = R @ np.asarray(normal, dtype=float)
            shade = 0.45 + 0.55 * max(0.0, float(np.dot(n_world, light)))
            mean_depth = float(np.mean(depth[list(idx)]))
            draw_order.append((mean_depth, idx, shade))
        draw_order.sort(key=lambda item: -item[0])      # 远的先画

        for _, idx, shade in draw_order:
            poly = QPolygonF([self._qpoint(screen[i]) for i in idx])
            color = QColor(int(base.red() * shade),
                           int(base.green() * shade),
                           int(base.blue() * shade))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(C['border']), 1))
            painter.drawPolygon(poly)

        # ---- 三个坐标轴 ----
        origin, _ = project([[0.0, 0.0, 0.0]])
        axes = [
            (np.array([1.05, 0, 0]), '#b3261e', 'X'),
            (np.array([0, 1.05, 0]), '#2e7d32', 'Y'),
            (np.array([0, 0, 1.05]), '#1f4e79', 'Z'),
        ]
        painter.setFont(QFont(PixelTheme.FONT_FAMILY, 8))
        for vec, color, label in axes:
            tip, _ = project([vec])
            painter.setPen(QPen(QColor(color), 1.6))
            painter.drawLine(self._qpoint(origin[0]), self._qpoint(tip[0]))
            painter.drawText(self._qpoint(tip[0]) + self._offset(), label)

        painter.end()

    @staticmethod
    def _qpoint(xy):
        return QPointF(float(xy[0]), float(xy[1]))

    @staticmethod
    def _offset():
        return QPointF(5.0, -5.0)


# ==================== 面板（读数 + 开关 + 校准） ====================

class IMU面板(QWidget):
    """3D 姿态 + 数值 + 轮询开关 + 零位校准"""

    poll_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.估计 = 姿态估计()
        self.有数据 = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.view = 姿态指示器()
        self.view.setMinimumHeight(150)
        layout.addWidget(self.view)

        self.att_label = QLabel("姿态：横滚 --° · 俯仰 --° · 航向 --°")
        self.att_label.setObjectName("hintText")
        self.att_label.setWordWrap(True)
        layout.addWidget(self.att_label)

        self.raw_label = QLabel("加速度 -- g · 角速度 -- dps")
        self.raw_label.setObjectName("hintText")
        self.raw_label.setWordWrap(True)
        layout.addWidget(self.raw_label)

        self.chk_poll = QCheckBox("实时读取 IMU（20 Hz）")
        self.chk_poll.setChecked(True)
        self.chk_poll.setCursor(Qt.PointingHandCursor)
        self.chk_poll.toggled.connect(self.poll_toggled.emit)
        layout.addWidget(self.chk_poll)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_zero = QPushButton("零位校准")
        self.btn_zero.setObjectName("funcBtn")
        self.btn_zero.setCursor(Qt.PointingHandCursor)
        self.btn_zero.clicked.connect(self.calibrate_zero)
        self.btn_clear = QPushButton("取消零位")
        self.btn_clear.setObjectName("funcBtn")
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_zero)
        row.addWidget(self.btn_zero)
        row.addWidget(self.btn_clear)
        layout.addLayout(row)

        self.status_label = QLabel("等待 IMU 数据…")
        self.status_label.setObjectName("hintText")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # ---- 外部接口 ----

    def is_polling_enabled(self):
        return self.chk_poll.isChecked()

    def calibrate_zero(self):
        """把当前姿态记为零位（显示相对角度）"""
        self.view.set_zero(self.估计.matrix())
        self.status_label.setText("已记录零位：现在显示的是相对该姿态的角度")

    def clear_zero(self):
        self.view.clear_zero()
        self.status_label.setText("已取消零位校准")

    def reset(self):
        self.估计.reset()
        self.有数据 = False
        self.att_label.setText("姿态：横滚 --° · 俯仰 --° · 航向 --°")
        self.raw_label.setText("加速度 -- g · 角速度 -- dps")

    def on_unavailable(self, message):
        """固件说 IMU 读不到：停掉轮询并把原因显示出来"""
        self.chk_poll.setChecked(False)
        self.status_label.setText(message)

    def on_sample(self, data):
        R = self.估计.update(
            data["ax"], data["ay"], data["az"],
            data["gx"], data["gy"], data["gz"], data["t"],
        )
        self.view.set_rotation(R)
        self.有数据 = True

        roll, pitch, yaw = self.估计.euler()
        self.att_label.setText(
            f"姿态：横滚 {roll:.1f}° · 俯仰 {pitch:.1f}° · 航向 {yaw:.1f}°"
        )
        self.raw_label.setText(
            f"加速度 {data['ax']:.2f} / {data['ay']:.2f} / {data['az']:.2f} g · "
            f"角速度 {data['gx']:.0f} / {data['gy']:.0f} / {data['gz']:.0f} dps"
        )
        self.status_label.setText(
            "无磁力计：横滚/俯仰是绝对姿态，航向角会缓慢漂移"
        )
