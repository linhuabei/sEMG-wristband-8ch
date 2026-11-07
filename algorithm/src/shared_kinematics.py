# -*- coding: utf-8 -*-
r"""检验「跨人运动学是否相似」，并顺便导出"动作模板库"（供伪标签用）。

核心假设：同一个人做某个动作（比如握拳）时，关节角的**变化轨迹**在人与人之间是相似的；
个体差异主要体现在"肌电→角度"的映射上（电极位置、肌肉构型、阻抗）。

做法：从 NinaPro 的 Cyberglove 角度里，把每个动作的每一次重复裁出来
（去掉过渡段），时间归一化到固定长度，然后比较三类相似度：
  * 同一个人不同重复（组内上限）
  * 不同人做同一个动作（跨人）
  * 不同动作（对照下限）

用法：
    python shared_kinematics.py --subjects 1-8                 # 出统计报告
    python shared_kinematics.py --subjects 1,4,5,6 --save-template ../outputs/运动学模板.npz
"""
import argparse
from pathlib import Path

import numpy as np

from config import Config
from ninapro import load_subject

T0 = 100            # 时间归一化后的长度
TRIM = 0.2          # 每段首尾各裁掉 20%（避开过渡动作）


def action_segments(mask):
    """把布尔数组切成连续段 [(起, 止), ...]。"""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    return list(zip(idx[::2], idx[1::2]))


def resample_time(x, n=T0):
    """(T, C) 沿时间线性重采样到 n 点。"""
    if len(x) < 8:
        return None
    t = np.linspace(0, len(x) - 1, n)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (t - i0)[:, None]
    return x[i0] * (1 - w) + x[i1] * w


def collect(subjects, cfg):
    """返回 (轨迹字典, 保持姿态字典)。保持姿态 = 动作中段稳定时那 21 路角度。"""
    out = {}
    holds = {}
    for s in subjects:
        d = load_subject(s, cfg.data.exercise, cfg.data)
        glove = np.asarray(d["glove"], dtype=np.float64)[:, cfg.data.angle_keep]
        lab = np.asarray(d["restimulus"]).ravel().astype(int)
        for m in range(1, 18):
            traj = []
            hold = []
            for a, b in action_segments(lab == m):
                if b - a < 200:                      # 太短的段不要
                    continue
                cut = int((b - a) * TRIM)
                seg = glove[a + cut:b - cut]
                r = resample_time(seg)
                if r is not None:
                    traj.append(r.T)                 # (21, T0)
                    mid = glove[a + (b - a) // 2 - 100:a + (b - a) // 2 + 100]
                    hold.append(mid.mean(axis=0))    # 姿态向量 (21,)
            if traj:
                out[(s, m)] = traj
                holds[(s, m)] = np.array(hold)
    return out, holds


def corr(a, b, per_channel_norm=True):
    """两条轨迹 (C,T) 的平均逐通道相关系数。"""
    if per_channel_norm:
        a = (a - a.mean(1, keepdims=True)) / np.maximum(a.std(1, keepdims=True), 1e-9)
        b = (b - b.mean(1, keepdims=True)) / np.maximum(b.std(1, keepdims=True), 1e-9)
    rs = []
    for c in range(a.shape[0]):
        if a[c].std() > 1e-9 and b[c].std() > 1e-9:
            rs.append(np.corrcoef(a[c], b[c])[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-8")
    ap.add_argument("--exercise", type=int, default=1)
    ap.add_argument("--save-template", default=None)
    args = ap.parse_args()
    subs = []
    for chunk in args.subjects.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            subs.extend(range(int(a), int(b) + 1))
        else:
            subs.append(int(chunk))
    cfg = Config()
    cfg.data.exercise = args.exercise

    data, holds = collect(subs, cfg)
    print(f"受试者 {subs}，共 {len(data)} 组（人×动作），每组平均 "
          f"{np.mean([len(v) for v in data.values()]):.1f} 次重复\n")

    # ---- 三类相似度 ----
    within, cross, cross_action = [], [], []
    for key, trajs in data.items():
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                within.append(corr(trajs[i], trajs[j]))
    keys = list(data)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            a = data[k1][0]
            b = data[k2][0]
            (cross if k1[1] == k2[1] else cross_action).append(corr(a, b))
    print("相似度（逐通道相关再平均；1.0 = 完全一致）")
    print(f"  同一人不同重复（组内上限） : {np.nanmean(within):.3f} ± {np.nanstd(within):.3f}"
          f"  (n={len(within)})")
    print(f"  不同人 同一动作（跨人）      : {np.nanmean(cross):.3f} ± {np.nanstd(cross):.3f}"
          f"  (n={len(cross)})")
    print(f"  不同动作（对照下限）          : {np.nanmean(cross_action):.3f} ± "
          f"{np.nanstd(cross_action):.3f}  (n={len(cross_action)})")

    # ---- 静态姿态（"起止手势"）跨人一致性 ----
    print("\n【关键】动作稳定段的“姿态”跨人一致性：")
    pose_r, amp_cv = [], []
    for m in range(1, 18):
        ks = [k for k in holds if k[1] == m]
        vecs = {k: holds[k].mean(axis=0) for k in ks}
        for i, k1 in enumerate(ks):
            for k2 in ks[i + 1:]:
                u, v = vecs[k1], vecs[k2]
                if u.std() > 1e-9 and v.std() > 1e-9:
                    pose_r.append(np.corrcoef(u, v)[0, 1])
        # 幅度（同一姿态在不同人身上的绝对值差多少）
        if len(ks) > 2:
            M = np.array([vecs[k] for k in ks])
            amp_cv.append(np.mean(M.std(axis=0) / np.maximum(np.abs(M.mean(axis=0)), 1e-9)))
    print(f"  同一动作的保持姿态，跨人相关 : {np.nanmean(pose_r):.3f} ± {np.nanstd(pose_r):.3f}"
          f"  (n={len(pose_r)})")
    print(f"  同姿态的绝对幅度，跨人变异系数 : {np.nanmean(amp_cv):.3f}"
          f"（0.2 ≈ 不同人之间幅度差 20%）")

    # 逐动作看哪些动作最"通用"
    print("\n逐动作的跨人一致性（高 = 这个动作大家做得像）：")
    rows = []
    for m in range(1, 18):
        vals = []
        ks = [k for k in keys if k[1] == m]
        for i, k1 in enumerate(ks):
            for k2 in ks[i + 1:]:
                vals.append(corr(data[k1][0], data[k2][0]))
        if vals:
            rows.append((m, float(np.nanmean(vals)), len(vals)))
    for m, v, n in sorted(rows, key=lambda r: -r[1]):
        print(f"   动作 {m:2d}: 跨人相关 {v:.3f}  (n={n})")

    if args.save_template:
        # 模板 = 所有供体在同一动作上的平均轨迹（原始手套单位），以及全局标准化参数
        tmpl, rest = {}, []
        for m in range(1, 18):
            stack = [t for k, ts in data.items() if k[1] == m for t in ts]
            if stack:
                tmpl[m] = np.mean(stack, axis=0)          # (21, T0)
        all_glove = np.concatenate([t for ts in data.values() for t in ts], axis=1)
        # 静息姿态：供体在休息段的手套读数（伪标签里休息段要填这个）
        rest_vecs = []
        for s in subs:
            d = load_subject(s, cfg.data.exercise, cfg.data)
            g = np.asarray(d["glove"], dtype=np.float64)[:, cfg.data.angle_keep]
            l = np.asarray(d["restimulus"]).ravel().astype(int)
            sel = l == 0
            if sel.sum() > 1000:
                rest_vecs.append(g[sel][::200].mean(axis=0))
        out = Path(args.save_template)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **{f"动作{m}": v for m, v in tmpl.items()},
                 mean=all_glove.mean(axis=1), std=np.maximum(all_glove.std(axis=1), 1e-9),
                 静息=np.mean(rest_vecs, axis=0) if rest_vecs else np.zeros(21),
                 T0=np.int32(T0), 供体受试者=np.array(subs))
        print(f"\n模板库已保存：{out}（{len(tmpl)} 个动作，每个 {tmpl[1].shape}）")


if __name__ == "__main__":
    main()
