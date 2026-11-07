# -*- coding: utf-8 -*-
"""英文名别名（兼容旧路径）：真正的实现在 手势识别_CNNTransformer.py。

早先的文档里写的是 cnn_transformer_gesture.py，为了不让旧说明失效保留这个文件：
它只做一件事——把中文名插件里的 NAME / TYPE / predict 转发出来。
"""
import importlib.util as _ilu
from pathlib import Path as _Path

_中文插件 = _Path(__file__).resolve().with_name("手势识别_CNNTransformer.py")
_spec = _ilu.spec_from_file_location("手势识别_CNNTransformer", str(_中文插件))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NAME = _mod.NAME
TYPE = _mod.TYPE
predict = _mod.predict
reset = getattr(_mod, "reset", None)
