# -*- coding: utf-8 -*-
"""
00_common.py —— QuadFP-Cascade 公共常量、路径与工具函数
================================================================
论文：《多尺度指纹级联框架 QuadFP-Cascade》（Computer Communications 251 (2026) 108495）
主题：网络隐蔽定时信道（Covert Timing Channel, CTC）检测。

本文件集中存放论文里出现的固定参数与若干工具函数，供 01~14 各脚本复用。
所有随机过程都用固定种子，保证结果可复算。
"""

import os
import numpy as np

# ----------------------------------------------------------------------
# 0. 复现实验用的固定随机种子（论文未指定种子，这里取一个教学种子）
# ----------------------------------------------------------------------
SEED = 20261014

# ----------------------------------------------------------------------
# 1. 路径：所有中间产物（.npz）都放在本目录的 outputs/ 子目录
# ----------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def out_path(filename):
    """返回 outputs 目录下某文件的绝对路径。"""
    return os.path.join(OUT_DIR, filename)


# ----------------------------------------------------------------------
# 2. 论文中的固定参数
# ----------------------------------------------------------------------
# §3.2.1 外层分段：每段 L=2000 个 IPD，相邻段 25% 重叠（步进 = 1500）
SEG_LEN = 2000
SEG_OVERLAP = 0.25
SEG_STEP = int(round(SEG_LEN * (1 - SEG_OVERLAP)))   # 1500

# §3.2.1 / Table 2 内层多尺度滑窗：(窗口 win, 步长 step)
SCALES = [(32, 8), (64, 16), (128, 32)]

# 七级符号：-3 ... +3
SYMBOL_LEVELS = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int64)
NUM_SYMBOLS = 7

# 每个窗口的指纹长度（128 位，公式 2）
FP_BITS = 128

# Table 1：相邻 IPD 变化率 Δ 的七级符号映射阈值（语义见 03 脚本）
#   +3: Δ >= 0.15
#   +2: 0.06 <= Δ < 0.15
#   +1: 0.02 <= Δ < 0.06
#    0: -0.02 < Δ < 0.02
#   -1: -0.06 < Δ <= -0.02
#   -2: -0.15 < Δ <= -0.06
#   -3: Δ <= -0.15
DELTA_BINS = [-0.15, -0.06, -0.02, 0.02, 0.06, 0.15]   # 与 -3..+3 配合使用

# LSH 教学配置：把 128 位指纹切成 8 个 band，每 band 16 位（8×16=128）
# 注意：论文只写了 "low-overhead LSH bucketing"，未给 band 参数，
#       这里采用标准 SimHash-LSH 的常见教学配置。
LSH_BANDS = 8
LSH_BAND_BITS = 16   # 8*16 = 128

# 阈值标定：论文用合法-合法段对相似度 c(·) 的 Q0.30 / Q0.60 分位作 θ0 / θ1
THETA_LO_Q = 0.30
THETA_HI_Q = 0.60

# 5 类隐蔽信道（论文实验对象）
CHANNELS = ["IPCTC", "LNCTC", "TRCTC", "Jitterbug", "EpsilonKlibur"]
LEGIT_PROTOCOLS = ["HTTP", "HTTPS", "SSH"]


# ----------------------------------------------------------------------
# 3. 工具函数
# ----------------------------------------------------------------------
def rng(seed=SEED):
    """返回一个固定种子的 numpy 随机数生成器。"""
    return np.random.default_rng(seed)


def banner(title):
    """打印分节标题。"""
    line = "=" * 70
    print("\n" + line)
    print(title)
    print(line)


def sub_banner(title):
    print("\n" + "-" * 66)
    print(title)
    print("-" * 66)


def count_windows(n_points, win, step):
    """
    长度为 n_points 的序列上，用窗口 win、步长 step 滑窗，能得到多少个窗口。
    公式：floor((N - win)/step) + 1
    例：N=2000, win=32, step=8 -> floor(1968/8)+1 = 247
    """
    if n_points < win:
        return 0
    return (n_points - win) // step + 1


def segment_series(seq, seg_len=SEG_LEN, step=SEG_STEP):
    """
    把一条长 IPD 序列切成若干段（25% 重叠）。
    返回形状 (num_seg, seg_len) 的数组；不足一整段的尾部丢弃。
    """
    n = len(seq)
    if n < seg_len:
        return np.zeros((0, seg_len), dtype=seq.dtype)
    num_seg = (n - seg_len) // step + 1
    starts = np.arange(num_seg) * step
    return np.stack([seq[s:s + seg_len] for s in starts])


def save_npz(filename, **arrays):
    """保存中间结果到 outputs/。"""
    path = out_path(filename)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(filename, allow_pickle=False):
    """读取 outputs/ 下的中间结果。"""
    return np.load(out_path(filename), allow_pickle=allow_pickle)


def sigmoid(x):
    """数值稳定的 sigmoid。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def set_data_source(tag):
    """在 outputs/data_source.txt 写入当前数据来源标记（synthetic / CAIDA:bignet ...）。"""
    with open(out_path("data_source.txt"), "w", encoding="utf-8") as f:
        f.write(str(tag) + "\n")


def get_data_source(default="synthetic"):
    """读取当前数据来源标记；不存在时返回 default。"""
    p = out_path("data_source.txt")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            tag = f.read().strip()
        if tag:
            return tag
    return default


if __name__ == "__main__":
    banner("00_common.py 自检：论文固定参数")
    print(f"段长 L = {SEG_LEN}，段步进 = {SEG_STEP}（重叠 {int(SEG_OVERLAP*100)}%）")
    print(f"内层三尺度 (win, step) = {SCALES}")
    print(f"指纹位数 = {FP_BITS}，符号级 = {list(SYMBOL_LEVELS)}")
    for (w, s) in SCALES:
        print(f"  win={w:3d}, step={s:2d} -> 每段窗口数 = {count_windows(SEG_LEN, w, s)}")
    print(f"LSH 教学配置：{LSH_BANDS} band × {LSH_BAND_BITS} 位 = {LSH_BANDS*LSH_BAND_BITS} 位")
    print(f"\n固定种子 SEED = {SEED}")
    print(f"输出目录 = {OUT_DIR}")
