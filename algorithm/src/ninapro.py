# -*- coding: utf-8 -*-
"""读取 NinaPro 公开数据集（DB2）。

数据来源就在本仓库里：算法/公开数据集/Ninapro/DB2/
  * DB2_sN.zip              —— 每位受试者一个压缩包（内部是 SN_E1/E2/E3_A1.mat）
  * S1_E1_A1.mat 等         —— 已经解压出来的（优先用 zip，不用解压也能读）

每个 .mat 里用到的变量：
  emg         (T, 12)  12 路 sEMG，2 kHz
  glove       (T, 22)  Cyberglove 22 路关节角（原始未标定值）
  stimulus    (T, 1)   动作标签（按提示影片）
  restimulus  (T, 1)   官方重标注后的动作标签（时间对齐更好，默认用它）
  repetition / rerepetition (T, 1)  第几次重复（用作训练/测试划分，避免重叠窗口泄漏）

本模块只做“读 + 缓存”，滤波和滑窗分别在 preprocess.py / dataset.py 里。
"""
import os
import time
import zipfile

import numpy as np
import scipy.io as sio

from config import DB2_DIR

VARS = ["emg", "glove", "stimulus", "restimulus", "repetition", "rerepetition"]


def _open_member(subject, exercise):
    """打开某位受试者某个练习的 .mat（优先从 zip 里直接流式读，不解压到磁盘）。"""
    zip_path = DB2_DIR / f"DB2_s{subject}.zip"
    if zip_path.exists():
        zf = zipfile.ZipFile(zip_path)
        names = [n for n in zf.namelist()
                 if n.endswith(f"S{subject}_E{exercise}_A1.mat")]
        if names:
            return zf.open(names[0]), f"{zip_path.name}!{names[0]}", zf
        zf.close()
    direct = DB2_DIR / f"S{subject}_E{exercise}_A1.mat"
    if direct.exists():
        return str(direct), direct.name, None
    raise FileNotFoundError(
        f"找不到 DB2 受试者 {subject} 练习 {exercise} 的数据：{zip_path} / {direct}")


def read_subject_mat(subject, exercise):
    """读取原始 .mat（只取需要的那几个变量，省内存）。"""
    fh, desc, zf = _open_member(subject, exercise)
    try:
        md = sio.loadmat(fh, variable_names=VARS)
    finally:
        if hasattr(fh, "close"):
            fh.close()
        if zf is not None:
            zf.close()
    out = {}
    for k in VARS:
        if k in md:
            out[k] = np.asarray(md[k])
    out["_source"] = desc
    return out


def build_cache(subject, exercise, data_cfg, force=False, verbose=True):
    """把想要的通道抽出来存成小 npz，后面训练直接读缓存。

    emg 只存 1-8 路（约 58 MB/人），glove 存全部 22 路（供角度回归用）。
    """
    path = data_cfg.cache_path(subject)
    if path.exists() and not force:
        return path
    t0 = time.time()
    md = read_subject_mat(subject, exercise)
    emg = np.ascontiguousarray(md["emg"][:, data_cfg.emg_slice], dtype=np.float32)
    payload = {
        "emg": emg,
        "stimulus": np.asarray(md.get("stimulus", np.zeros((emg.shape[0], 1))),
                               dtype=np.int8).ravel(),
        "restimulus": np.asarray(md.get("restimulus", np.zeros((emg.shape[0], 1))),
                                 dtype=np.int8).ravel(),
        "repetition": np.asarray(md.get("repetition", np.zeros((emg.shape[0], 1))),
                                 dtype=np.int8).ravel(),
        "rerepetition": np.asarray(md.get("rerepetition", np.zeros((emg.shape[0], 1))),
                                   dtype=np.int8).ravel(),
        "subject": np.int16(subject),
        "exercise": np.int8(exercise),
    }
    if "glove" in md:
        payload["glove"] = np.asarray(md["glove"], dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    if verbose:
        print(f"  [cache] S{subject} E{exercise}: {emg.shape[0]} 点 "
              f"({emg.shape[0] / 2000:.0f} s) -> {path.name}  {time.time() - t0:.1f}s")
    return path


def load_subject(subject, exercise, data_cfg):
    """读缓存（没有就先建）。"""
    path = data_cfg.cache_path(subject)
    if not path.exists():
        build_cache(subject, exercise, data_cfg, verbose=False)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def ensure_cache(data_cfg, verbose=True, force=False):
    """把配置里所有受试者的缓存准备好，返回成功的受试者列表。"""
    ok, bad = [], []
    for s in data_cfg.subject_list:
        try:
            build_cache(s, data_cfg.exercise, data_cfg, force=force, verbose=verbose)
            ok.append(s)
        except Exception as exc:                      # noqa: BLE001
            bad.append((s, str(exc)))
            if verbose:
                print(f"  [cache] S{s} 读取失败：{exc}")
    if bad and verbose:
        print(f"  [cache] 共 {len(bad)} 位受试者读取失败：{[b[0] for b in bad]}")
    return ok


def cache_size_mb(data_cfg):
    total = 0
    for s in data_cfg.subject_list:
        p = data_cfg.cache_path(s)
        if p.exists():
            total += os.path.getsize(p)
    return total / 1024 ** 2
