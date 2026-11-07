"""
Node 御识 完整性检查

用法:
    python check.py
    (或双击 start.bat 后传入 check 参数)

检查内容: 关键文件 / 源码语法 / 模块导入 / 运行依赖 / 数据目录
"""
import ast
import importlib
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))

KEY_FILES = [
    ("main.py", "程序入口"),
    ("requirements.txt", "依赖清单"),
    ("icon.ico", "应用图标"),
    (os.path.join("resources", "icons", "icon.ico"), "资源图标"),
    (os.path.join("resources", "fonts", "PixelFont.ttf"), "像素字体"),
    (os.path.join("ui", "themes", "pixel_theme.py"), "主题模块"),
    (os.path.join("core", "esp32_controller.py"), "BLE 控制器"),
    (os.path.join("core", "device_manager.py"), "设备管理"),
    (os.path.join("core", "signal_processor.py"), "信号处理"),
    (os.path.join("core", "gesture_classifier.py"), "手势分类器"),
    (os.path.join("core", "emg_protocol.py"), "EMG 协议"),
    (os.path.join("ui", "main_window.py"), "主窗口"),
]

# 运行期真正需要的第三方库
REQUIRED_DEPS = [
    ("PyQt5", "GUI"),
    ("pyqtgraph", "波形绘图"),
    ("numpy", "数值计算"),
    ("scipy", "信号滤波"),
    ("bleak", "BLE 通信"),
]

# 可选依赖（当前没有必须安装的可选项）
OPTIONAL_DEPS = []

SRC_MODULES = [
    "main",
    "ui.main_window",
    "ui.themes.pixel_theme",
    "ui.views.monitor_view",
    "ui.views.user_transfer_view",
    "ui.views.data_capture_view",
    "ui.views.device_detail_view",
    "core.device_manager",
    "core.esp32_controller",
    "core.signal_processor",
    "core.gesture_classifier",
    "core.emg_protocol",
]


def source_files():
    files = [os.path.join(ROOT, "main.py")]
    for folder in ("core", "ui", "examples"):
        base = os.path.join(ROOT, folder)
        for dirpath, _, names in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for name in names:
                if name.endswith(".py"):
                    files.append(os.path.join(dirpath, name))
    return files


def main():
    failures = 0
    print("=" * 52)
    print("  Node 御识 · 程序完整性检查")
    print("=" * 52)

    # 1. 关键文件
    print("\n[1/6] 关键文件")
    for rel, label in KEY_FILES:
        path = os.path.join(ROOT, rel)
        ok = os.path.isfile(path)
        print(f"  [{'OK ' if ok else '缺失'}] {label:<14} {rel}")
        failures += (not ok)

    # 2. 源码语法
    print("\n[2/6] 源码语法")
    py_files = source_files()
    for path in py_files:
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                ast.parse(fh.read(), filename=path)
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"  [错误] {os.path.relpath(path, ROOT)}: {exc}")
            failures += 1
        else:
            print(f"  [OK ] {os.path.relpath(path, ROOT)}")

    # 3. 模块导入
    print("\n[3/6] 模块导入")
    for name in SRC_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            print(f"  [错误] {name}: {exc}")
            failures += 1
        else:
            print(f"  [OK ] {name}")

    # 4. 依赖
    print("\n[4/6] 运行依赖")
    for name, label in REQUIRED_DEPS:
        try:
            importlib.import_module(name)
        except Exception as exc:
            print(f"  [错误] {label:<8} {name}: {exc}")
            failures += 1
        else:
            print(f"  [OK ] {label:<8} {name}")
    for name, label in OPTIONAL_DEPS:
        try:
            importlib.import_module(name)
        except Exception:
            print(f"  [可选] {label:<8} {name} 未安装(不影响启动)")
        else:
            print(f"  [OK ] {label:<8} {name}")

    # 5. 数据目录
    print("\n[5/6] 数据目录")
    data_dir = os.path.join(ROOT, "training_data")
    if os.path.isdir(data_dir):
        sessions = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        print(f"  [OK ] training_data 目录存在, 共 {len(sessions)} 个采集会话")
    else:
        print("  [提示] training_data 目录不存在(首次训练时自动创建)")

    # 6. EMG 协议往返自检
    print("\n[6/6] EMG 协议自检")
    from core.emg_protocol import self_test
    if self_test():
        print("  [OK ] 编解码 / 丢包重同步 / 配置解析 通过")
    else:
        print("  [错误] EMG 协议自检失败")
        failures += 1

    print("\n" + "=" * 52)
    if failures:
        print(f"  检查完成: {failures} 项异常, 请修复后重试")
        return 1
    print("  检查完成: 全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
