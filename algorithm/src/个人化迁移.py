# -*- coding: utf-8 -*-
r"""一键个人化迁移：把「上位机采的一段标定数据」变成「这个人的专用模型」。

流程（全自动串起来）：
    上位机采集的 CSV
        ↓  ① 读出来（timestamp / gesture_code / CH1-8）
        ↓  ② 如果有参考角度 CSV → 按时间戳合并；没有 → 用动作模板生成伪标签
        ↓  ③ 微调（冻结卷积前端 + 小学习率，默认 16 轮）
        ↓  ④ 导出新的权重，并自动备份/覆盖 plugins 里的那份

用法（最简，只有上位机采的数据）：
    python 个人化迁移.py --session "上位机/training_data/session_20260912_101530"

带参考角度（有手套/动捕时，效果更好）：
    python 个人化迁移.py --session "…” --ref-csv "D:/手套/角度.csv" --subject 1

顺便生成一个双击就能跑的批处理：
    python 个人化迁移.py --session "…” --make-bat
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from config import OUTPUT_DIR, PLUGIN_DIR
from merge_app_csv import _read_csv, _pick, merge

本目录 = Path(__file__).resolve().parent
默认模板 = OUTPUT_DIR / "运动学模板.npz"
默认权重 = PLUGIN_DIR / "模型权重_连续关节角.npz"


def 找采集文件(会话目录):
    """在 session_xxx 目录里找"上位机采集的"CSV。

    注意：参考角度 CSV 也可能放在同一个目录里，所以要按表头判断——
    必须含 gesture_code / label / code 这一列，才是上位机采集的文件。
    """
    def 有标签列(f):
        try:
            with open(f, "r", encoding="utf-8-sig", newline="") as fh:
                head = next(csv.reader(fh), [])
        except Exception:                                   # noqa: BLE001
            return False
        return any(h.strip().lower() in ("gesture_code", "label", "code") for h in head)

    d = Path(会话目录)
    if d.is_file() and d.suffix.lower() == ".csv":
        return d
    候选 = sorted(d.rglob("training_*.csv")) or sorted(d.rglob("*.csv"))
    for f in 候选:
        if 有标签列(f):
            return f
    raise FileNotFoundError(
        f"{d} 里没找到上位机采集的 CSV（需要表头含 gesture_code 那一列）。"
        f"找到的文件：{[f.name for f in 候选]}")


def 只存肌电(app_csv, out_path, subject):
    """没有参考角度时：先把肌电+标签存成自采格式（角度留 0，稍后用模板填）。"""
    head, A = _read_csv(app_csv)
    t = A[_pick(head, ["timestamp", "time", "t"], "时间戳列")]
    code = A[_pick(head, ["gesture_code", "label", "code"], "标签列")]
    emg_cols = sorted([h for h in head if h.lower().startswith("ch")],
                      key=lambda s: int("".join(c for c in s if c.isdigit()) or 0))
    emg = np.stack([A[c] for c in emg_cols], axis=1).astype(np.float32)
    dt = np.median(np.diff(t))
    fs = int(round(1.0 / dt)) if dt > 0 else 2000
    trial = np.zeros(len(code), dtype=np.int16)
    seen, prev = {}, None
    for i, c in enumerate(code):
        if c != prev:
            seen[c] = seen.get(c, 0) + 1
            prev = c
        trial[i] = min(seen[c], 127)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, emg=emg, fs=np.int32(fs), trial=trial, label=code.astype(np.int16),
             angles=np.zeros((len(code), 21), dtype=np.float32),
             angle_names=np.array([f"关节{i + 1}" for i in range(21)]),
             subject=np.int16(subject), session=np.int16(1))
    print(f"  已整理成自采格式：{out_path.name}（{len(t)} 点，{fs} Hz，"
          f"{len(t) / fs:.0f} s，动作 {sorted(set(code.astype(int)))}）")
    return out_path


def 跑(命令, 说明):
    print(f"\n>>> {说明}")
    print("    " + " ".join(str(c) for c in 命令))
    # 子进程强制用 UTF-8，否则 Windows 下中文/上下标字符会因 GBK 编码报错
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([str(命令[0]), "-X", "utf8"] + [str(c) for c in 命令[1:]],
                       cwd=str(本目录), env=env)
    if r.returncode != 0:
        raise SystemExit(f"这一步失败（退出码 {r.returncode}），请把上面的输出发给开发者")


def main():
    ap = argparse.ArgumentParser(description="一键个人化迁移")
    ap.add_argument("--session", required=True, help="上位机的 session 目录或 CSV 文件")
    ap.add_argument("--ref-csv", default=None, help="（可选）参考角度 CSV，有手套时给上")
    ap.add_argument("--subject", type=int, default=1, help="受试者/用户编号（用于命名）")
    ap.add_argument("--template", default=str(默认模板), help="动作模板库 .npz")
    ap.add_argument("--base", default=str(默认权重), help="预训练权重 .npz")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no-freeze", action="store_true", help="不冻结卷积前端（数据多时可用）")
    ap.add_argument("--work-dir", default=None, help="中间文件放哪（默认 outputs/个人化迁移_Sx）")
    ap.add_argument("--out", default=None, help="导出的权重路径")
    ap.add_argument("--install", action="store_true",
                    help="直接覆盖 plugins 里的权重（会先备份旧的）")
    ap.add_argument("--make-bat", action="store_true", help="在会话目录里生成双击可跑的 .bat")
    args = ap.parse_args()

    csv_path = 找采集文件(args.session)
    工作目录 = Path(args.work_dir) if args.work_dir else (OUTPUT_DIR / f"个人化迁移_S{args.subject}")
    工作目录.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] 采集文件：{csv_path}\n      工作目录：{工作目录}")

    # ---- ① / ② 生成自采格式（真标签 或 伪标签）----
    原始 = 工作目录 / f"S{args.subject:02d}_ses1.npz"
    训练数据目录 = 工作目录 / "训练数据"
    训练数据目录.mkdir(parents=True, exist_ok=True)
    if args.ref_csv:
        print("[2/5] 有参考角度 → 按时间戳合并（真标签）")
        merge(csv_path, args.ref_csv, 原始, subject=args.subject, session=1)
        shutil.copyfile(原始, 训练数据目录 / 原始.name)
        评估目录 = 训练数据目录
        模式 = "真标签"
    else:
        print("[2/5] 没有参考角度 → 先整理肌电，再用动作模板生成伪标签")
        只存肌电(csv_path, 原始, args.subject)
        if not Path(args.template).exists():
            raise SystemExit(f"缺少模板库 {args.template}；"
                             f"先用 shared_kinematics.py --save-template 生成，"
                             f"或者提供 --ref-csv")
        跑([sys.executable, "make_pseudo_labels.py", "--template", args.template,
             "--in-dir", 工作目录, "--out-dir", 训练数据目录,
             "--subjects", args.subject], "生成伪标签")
        评估目录 = 训练数据目录
        模式 = "伪标签"

    # ---- ③ 微调 ----
    导出权重 = Path(args.out) if args.out else (工作目录 / f"个人化权重_S{args.subject}.npz")
    命令 = [sys.executable, "finetune.py", "--base", args.base,
            "--data-dir", 训练数据目录, "--eval-data-dir", 评估目录,
            "--subjects", args.subject, "--epochs", args.epochs, "--lr", args.lr,
            "--out", 导出权重]
    if not args.no_freeze:
        命令.append("--freeze-front")
    跑(命令, f"微调（{模式}，{args.epochs} 轮）")

    # ---- ④ 安装 ----
    目标 = PLUGIN_DIR / "模型权重_连续关节角.npz"
    if args.install:
        if 目标.exists():
            备份 = 目标.with_name(目标.stem + "_备份.npz")
            shutil.copyfile(目标, 备份)
            print(f"\n[4/5] 已备份原权重 -> {备份.name}")
        shutil.copyfile(导出权重, 目标)
        print(f"      已安装到插件目录：{目标.name}")
    else:
        print(f"\n[4/5] 权重已生成：{导出权重}")
        print(f"      想装进上位机：把它复制成 {目标}")
        print(f"      或重新运行本命令时加上 --install")

    print(f"\n[5/5] 完成（{模式}）。下一步：上位机「手势交互」→ 重新导入插件，"
          f"界面上会显示新的识别结果。")
    if args.make_bat:
        bat = csv_path.parent / "一键个人化迁移.bat"
        内容 = (
            "@echo off\r\n"
            "chcp 65001 >nul\r\n"
            f'cd /d "{本目录}"\r\n'
            f'"{sys.executable}" "个人化迁移.py" --session "{csv_path.parent}" '
            f"--subject {args.subject} --install\r\n"
            "pause\r\n"
        )
        bat.write_text(内容, encoding="utf-8")
        print(f"已生成双击可运行的批处理：{bat}")


if __name__ == "__main__":
    main()
