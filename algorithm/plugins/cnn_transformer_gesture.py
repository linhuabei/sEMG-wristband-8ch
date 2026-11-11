# -*- coding: utf-8 -*-

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
