# -*- coding: utf-8 -*-
r"""用「动作模板库」给新用户的数据生成**伪标签**（不需要数据手套）。

思路（就是"用起止手势标定"那套）：
  1. 新用户在上位机里做固定动作，程序知道"每一段是什么动作"（gesture_code / label 切换处）；
  2. 每个动作的关节角轨迹形状，用**别人的平均模板**（模板库）代替；
  3. 段内把模板按时间拉伸到该段实际长度，休息段填"静息姿态"；
  4. 得到逐采样点的伪角度 → 拿去微调模型（真手套数据只留作最终评估）。

用法：
    python make_pseudo_labels.py --template ../outputs/运动学模板.npz \
        --in-dir ../outputs/自采格式样例 --out-dir ../outputs/自采格式样例_伪标签 --subjects 2
"""
import argparse
from pathlib import Path

import numpy as np

from shared_kinematics import action_segments, resample_time


def build_pseudo(label, 模板, 静息, n_angles, T0):
    """按 label 的连续段，把模板铺成逐采样点的伪角度。"""
    out = np.tile(静息.astype(np.float64), (len(label), 1))
    for m in range(1, 18):
        if f"动作{m}" not in 模板:
            continue
        tpl = np.asarray(模板[f"动作{m}"], dtype=np.float64)      # (21, T0)
        for a, b in action_segments(label == m):
            L = b - a
            if L < 40:
                continue
            seg = resample_time(tpl.T, L)                          # (L, 21)
            out[a:b] = seg
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subjects", default="2")
    ap.add_argument("--pattern", default="*.npz")
    args = ap.parse_args()

    with np.load(args.template, allow_pickle=False) as z:
        模板 = {k: z[k] for k in z.files}
    静息 = 模板.pop("静息", np.zeros(21))
    T0 = int(模板.get("T0", 100))
    模板.pop("mean", None)
    模板.pop("std", None)
    模板.pop("T0", None)
    模板.pop("供体受试者", None)

    subs = [int(x) for x in str(args.subjects).split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(args.in_dir).glob(args.pattern)):
        name_sub = None
        for s in subs:
            if f"S{s:02d}" in f.name or f"S{s}_" in f.name:
                name_sub = s
        if name_sub is None:
            continue
        with np.load(f, allow_pickle=True) as z:
            rec = {k: z[k] for k in z.files}
        label = np.asarray(rec["label"]).ravel().astype(int)
        n_angles = 21 if "angles" not in rec else np.asarray(rec["angles"]).shape[1]
        pseudo = build_pseudo(label, 模板, 静息, n_angles, T0)
        out_path = out_dir / f.name
        np.savez(out_path, **{**rec, "angles": pseudo})
        msg = f"  {f.name} -> {out_path.name}：伪标签 {pseudo.shape}"
        if "angles" in rec:                     # 有真值就顺手量一下伪标签质量（仅用于分析）
            true = np.asarray(rec["angles"], dtype=np.float64)
            rs = [np.corrcoef(pseudo[:, c], true[:, c])[0, 1]
                  for c in range(pseudo.shape[1])
                  if pseudo[:, c].std() > 1e-9 and true[:, c].std() > 1e-9]
            rmse = np.sqrt(((pseudo - true) ** 2).mean())
            msg += f"，与真值相关 {np.mean(rs):.3f}，RMSE(原始单位) {rmse:.1f}"
        print(msg)
    print(f"\n伪标签数据已写入 {out_dir}，可直接用于微调：")
    print(f'  python finetune.py --base ../plugins/模型权重_连续关节角.npz '
          f'--data-dir "{out_dir}" --subjects {subs[0]} --epochs 16 --lr 2e-4 '
          f'--freeze-front --out ../outputs/伪标签微调.npz')


if __name__ == "__main__":
    main()
