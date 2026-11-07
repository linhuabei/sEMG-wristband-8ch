# -*- coding: utf-8 -*-
"""划分审计：检查训练/验证/测试三份数据之间到底有没有泄漏。

查三件事：
  1. 三份数据的窗口有没有混进「别的重复段」的采样点（窗口 100 ms，正好可能跨重复边界）；
  2. 混进来的采样点是不是「动作」采样点（混进休息段无所谓，混进动作段才是真污染）；
  3. 逐段地列出每个窗口的标签在时间上属于哪一段重复，确认没有交叉。

用法：python audit_split.py --subjects 1-4
"""
import argparse

import numpy as np

from config import Config
from dataset import build_index, prepare_subject


def audit(sig, pos, name, allowed, cfg, W):
    reps, labels = sig.reps, sig.labels
    n_bad = n_bad_move = n_cross = 0
    worst = 0
    end_labels = []
    for _, st in pos:
        seg_rep = reps[st:st + W]
        seg_lab = labels[st:st + W]
        outside = ~np.isin(seg_rep, list(allowed))
        if outside.any():
            n_bad += 1
            if (seg_lab[outside] > 0).any():
                n_bad_move += 1
            worst = max(worst, int(outside.sum()))
            end_labels.append(int(seg_lab[-1]))
        # 窗内出现了「不属于本划分」的重复号 = 跨集污染
        if set(np.unique(seg_rep)) - allowed:
            n_cross += 1
    print(f"  S{sig.subject:<2} {name}：共 {len(pos):5d} 个窗口 | "
          f"窗内混入异段采样点 {n_bad:4d} 个（{100 * n_bad / max(1, len(pos)):.3f}%）| "
          f"其中混入动作采样点 {n_bad_move:3d} 个 | 最多混入 {worst} 点"
          f"（{worst / cfg.pre.fs * 1000:.0f} ms）"
          + (f" | 这些窗口的末端标签：休息 {sum(1 for v in end_labels if v == 0)} / "
             f"动作 {sum(1 for v in end_labels if v > 0)}" if end_labels else ""))
    return n_bad, n_bad_move, n_cross, len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-4")
    ap.add_argument("--exercise", type=int, default=1)
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
    W = cfg.window.length
    print(f"划分：训练=重复{sorted(cfg.data.train_repetitions)}  "
          f"验证=重复{sorted(cfg.data.val_repetitions)}  "
          f"测试=重复{sorted(cfg.data.test_repetitions)}；窗口 {W} 点（{W / cfg.pre.fs * 1000:.0f} ms）")
    print("检查：每个窗口的 200 个采样点里，有没有混进别的重复段的采样点\n")
    tot = {"训练": [0, 0, 0], "验证": [0, 0, 0], "测试": [0, 0, 0]}
    n_win = dict.fromkeys(tot, 0)
    for s in subs:
        sig = prepare_subject(s, cfg)
        tr, va, te = build_index([sig], cfg)
        for name, pos, allowed in (("训练", tr, set(cfg.data.train_repetitions)),
                                   ("验证", va, set(cfg.data.val_repetitions)),
                                   ("测试", te, set(cfg.data.test_repetitions))):
            b, bm, cr, n = audit(sig, pos, name, allowed, cfg, W)
            tot[name][0] += b
            tot[name][1] += bm
            tot[name][2] += cr
            n_win[name] += n
        print()
    print("汇总：")
    for name in ("训练", "验证", "测试"):
        b, bm, cr = tot[name]
        n = n_win[name]
        print(f"  {name}：{n} 个窗口，跨段窗口 {b} 个（{100 * b / max(1, n):.3f}%），"
              f"含他段动作的 {bm} 个，跨划分的 {cr} 个")
    print("\n说明：跨段窗口是「上一次重复的动作尾部 / 休息段」正好落在重复边界上，"
          "窗口末端才是本段的标签。")
    print("      要 100% 干净：把 WindowConfig.require_same_repetition 打开，"
          "只保留完全落在同一重复段内的窗口；")
    print("      已经训好的模型可以用 recheck_clean.py 直接在严格划分上重新评估，不用重训。")


if __name__ == "__main__":
    main()
