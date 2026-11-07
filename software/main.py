"""
Node（御识）八通道sEMG采集手环上位机系统
"""
import os
import sys

if hasattr(sys, 'frozen'):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'PyQt5', 'Qt5', 'plugins')
else:
    try:
        import PyQt5
        qt_path = os.path.dirname(PyQt5.__file__)
        plugin_path = os.path.join(qt_path, 'Qt5', 'plugins')
        if os.path.exists(plugin_path):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    except:
        pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from ui.main_window import MainWindow
from ui.themes.pixel_theme import PixelTheme
from core.paths import _resource_root


_main_windows = []  # 持有主窗口引用，避免被垃圾回收


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName("Node 御识")
    app.setWindowIcon(QIcon(os.path.join(_resource_root(), "icon.ico")))

    PixelTheme.apply(app)

    window = MainWindow()
    window.show()
    _main_windows.append(window)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
