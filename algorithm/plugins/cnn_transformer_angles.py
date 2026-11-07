# -*- coding: utf-8 -*-
"""英文名别名（兼容旧路径）：真正的实现在 连续关节角_CNNTransformer.py。"""
import importlib.util as _ilu
from pathlib import Path as _Path

_中文插件 = _Path(__file__).resolve().with_name("连续关节角_CNNTransformer.py")
_spec = _ilu.spec_from_file_location("连续关节角_CNNTransformer", str(_中文插件))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

NAME = _mod.NAME
TYPE = _mod.TYPE
predict = _mod.predict
reset = getattr(_mod, "reset", None)
