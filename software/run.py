# -*- coding: utf-8 -*-
"""启动器（文件名和内容都保持 ASCII，避免 cmd 读取批处理时的编码问题）。

批处理 start.bat 只调用本文件；真正要运行的入口文件名写在 Python 源码里，
Python 3 默认按 UTF-8 读源码，批处理只负责找到 run.py，不会再被中文编码搞坏。

用法：
    python run.py            # 启动上位机（等价于运行 main.py）
    python run.py --check    # 完整性检查（等价于运行 check.py）
"""
import os
import runpy
import sys

本目录 = os.path.dirname(os.path.abspath(__file__))


def main():
    参数 = list(sys.argv[1:])
    目标 = "main.py"
    if "--check" in 参数:
        参数.remove("--check")
        目标 = "check.py"
    目标路径 = os.path.join(本目录, 目标)
    if not os.path.isfile(目标路径):
        sys.stderr.write(f"[错误] 找不到 {目标}，请确认 run.py 与它放在同一目录\n")
        return 2
    sys.argv = [目标路径] + 参数
    runpy.run_path(目标路径, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
