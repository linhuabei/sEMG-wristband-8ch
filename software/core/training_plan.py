"""
测试方案（训练计划）数据层。

一个方案就是一个 JSON：
  name               方案名
  duration_seconds   每个动作录几秒
  gestures           [{name, code, rounds}, ...]  rounds=0 表示不采集

存在上位机目录下的 plans/ 文件夹里。
"""
import json
import os
import random

GESTURES = [
    ("握拳", 1),
    ("十指张开", 2),
    ("OK手势", 3),
    ("比大拇指", 4),
    ("比耶", 5),
    ("捏合", 6),
]


def default_plan():
    return {
        "name": "默认方案",
        "duration_seconds": 3.0,
        "gestures": [
            {"name": name, "code": code, "rounds": 2}
            for name, code in GESTURES
        ],
    }


def default_plan_dir():
    from core.paths import data_dir
    return data_dir("plans")


def total_rounds(plan):
    return sum(int(g.get("rounds", 0)) for g in plan.get("gestures", []))


def build_queue(plan):
    """按方案展开成动作队列，随机打乱且尽量不让同一个动作连做两次。"""
    duration = float(plan.get("duration_seconds", 3.0))
    queue = []
    for g in plan.get("gestures", []):
        rounds = int(g.get("rounds", 0))
        if rounds <= 0:
            continue
        queue.extend([{"name": g["name"], "code": int(g["code"]), "rounds": 1}] * rounds)

    if len(queue) < 2:
        return queue, duration

    codes = [g["code"] for g in queue]
    unique = len(set(codes))
    if unique > 1:
        for _ in range(60):
            random.shuffle(queue)
            ok = all(
                queue[i]["code"] != queue[i - 1]["code"]
                for i in range(1, len(queue))
            )
            if ok:
                break
    return queue, duration


def save_plan(plan, path):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(plan, fh, ensure_ascii=False, indent=2)
    return path


def load_plan(path):
    with open(path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    if "gestures" not in plan:
        raise ValueError("方案文件中缺少 gestures 字段")
    plan.setdefault("name", "未命名方案")
    plan.setdefault("duration_seconds", 3.0)
    return plan
