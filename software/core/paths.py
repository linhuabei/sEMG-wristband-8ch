"""
路径工具：区分“程序内资源”和“用户数据”。

打包成 exe 后：
  - 资源（图标、示例算法）在 PyInstaller 解压的临时目录，只读；
  - 用户数据（录制的 CSV、plans、training_data）放在 exe 旁边，要能长期保存。

（早期版本有个"模拟模式"，用来把测试模式的假数据单独存放；
  模拟数据源已删除，这里也一并去掉了。）
"""
import os
import sys

def _resource_root():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_root():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_dir(name):
    d = os.path.join(_resource_root(), name)
    os.makedirs(d, exist_ok=True)
    return d


def data_dir(name):
    d = os.path.join(_data_root(), name)
    os.makedirs(d, exist_ok=True)
    return d
