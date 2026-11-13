"""
加载 .py 算法插件。
约定
  NAME  = "算法名"           必填
  TYPE  = "gesture" 或 "angles"   必填
  predict(ctx)              必填，ctx 里带 features 字典

  TYPE="gesture"  -> predict 返回 ("手势名", 置信度0~1)
  TYPE="angles"   -> predict 返回 数组/列表，元素是各关节角度
"""
import importlib.util
import os
import uuid


class AlgorithmError(Exception):
    pass


def default_algorithms_dir():
    from core.paths import resource_dir
    return resource_dir("examples")


def list_builtin_algorithms():
    folder = default_algorithms_dir()
    result = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".py") and name != "__init__.py":
                result.append(os.path.join(folder, name))
    return result


def load_algorithm(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path) or not path.endswith(".py"):
        raise AlgorithmError("请选择 .py 算法文件")

    spec = importlib.util.spec_from_file_location(
        "user_algorithm_" + uuid.uuid4().hex[:8], path
    )
    if spec is None or spec.loader is None:
        raise AlgorithmError("无法加载该 .py 文件")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    name = getattr(module, "NAME", None)
    if not name:
        name = os.path.splitext(os.path.basename(path))[0]

    algo_type = getattr(module, "TYPE", "gesture")
    if algo_type not in ("gesture", "angles"):
        raise AlgorithmError("TYPE 字段必须为 gesture 或 angles")

    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise AlgorithmError("算法中未找到 predict(ctx) 函数")

    return {
        "path": path,
        "name": str(name),
        "type": algo_type,
        "module": module,
    }
