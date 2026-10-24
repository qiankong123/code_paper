# -*- coding: utf-8 -*-
"""
04_multiscale_windows.py —— 多尺度滑窗 + 7 维计数向量 v
================================================================
论文 §3.2.1 / Table 2：在每个段内用三个尺度 (win, step) = (32,8)/(64,16)/(128,32)
在符号序列上滑窗；每个窗口统计 −3…+3 七个符号的出现频次，得到 7 维计数向量 v∈ℝ^7。

口径对齐：段内有 2000 个 IPD，但相邻变化率只有 1999 个。令第一个 IPD 无前驱、
其 Δ 记 0，把符号序列补齐到 2000，于是：
  窗口数 = floor((2000−win)/step)+1 = 247 / 122 / 59，
  每个窗口 7 个计数之和 = 窗口大小（32 / 64 / 128）。

输出：outputs/04_count_vectors.npz
  V32 : (num_seg, 247, 7)
  V64 : (num_seg, 122, 7)
  V128: (num_seg,  59, 7)
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)


def count_vectors(sym_seq, win, step):
    """
    对长度 2000 的符号序列滑窗，统计每个窗口内 −3…+3 的频次。
    返回 (num_win, 7) 整数矩阵；每行 7 个数之和 = win。
    """
    n = len(sym_seq)
    nw = common.count_windows(n, win, step)
    V = np.zeros((nw, common.NUM_SYMBOLS), dtype=np.int64)
    idx = sym_seq + 3   # 符号 −3..3 映射到数组下标 0..6
    for i in range(nw):
        start = i * step
        chunk = idx[start:start + win]
        V[i] = np.bincount(chunk, minlength=common.NUM_SYMBOLS)
    return V


def main():
    data = common.load_npz("03_symbols.npz", allow_pickle=True)
    seg_symbols = data["seg_symbols"]; seg_channel = data["seg_channel"]
    num_seg = seg_symbols.shape[0]

    common.banner("04 多尺度滑窗与 7 维计数向量 v")
    # 段首补一个 0，使符号长度对齐为 2000
    sym_padded = np.pad(seg_symbols, ((0, 0), (1, 0)), mode="constant",
                        constant_values=0)  # (num_seg, 2000)

    outputs = {}
    for (win, step) in common.SCALES:
        nw = common.count_windows(common.SEG_LEN, win, step)
        V = np.stack([count_vectors(sym_padded[i], win, step) for i in range(num_seg)])
        outputs[f"V{win}"] = V
        print(f"  win={win:3d}, step={step:2d} -> 每段 {V.shape[1]:3d} 个窗口，"
              f"每窗口计数和={V.shape[2]}个符号、合计={V.shape[1] and int(V[0].sum(1)[0])}")

    V32 = outputs["V32"]; V64 = outputs["V64"]; V128 = outputs["V128"]

    common.sub_banner("win=32 时第一个段的前 3 个计数向量 v（7 列：−3…+3）")
    print("  符号:", list(common.SYMBOL_LEVELS))
    for i in range(3):
        print(f"  窗口{i+1}: {V32[0, i].tolist()}  和={V32[0,i].sum()}")

    common.sub_banner("合法段 vs IPCTC 段的平均计数向量（win=32，归一化为占比）")
    chnames = data["channel_names"]
    legit_idx = np.where(seg_channel == 0)[0][0]
    ipctc_idx = np.where(seg_channel == 1)[0][0]
    for tag, ix in [("LEGIT", legit_idx), ("IPCTC", ipctc_idx)]:
        p = V32[ix].mean(0) / 32.0 * 100
        print(f"  {tag:6s} " + " ".join(f"{v:6.1f}" for v in p))

    path = common.save_npz(
        "04_count_vectors.npz",
        V32=V32, V64=V64, V128=V128,
        seg_label=data["seg_label"], seg_channel=seg_channel,
        seg_conn_id=data["seg_conn_id"], seg_index=data["seg_index"],
        channel_names=chnames,
    )
    print(f"\n输出形状：V32={V32.shape}, V64={V64.shape}, V128={V128.shape}")
    print(f"已保存：{path}")


if __name__ == "__main__":
    main()
