# -*- coding: utf-8 -*-
"""
06_hamming_similarity.py —— 汉明距离与窗口级相似度（公式 3）
================================================================
论文 §3.2.1 公式(3)：
      sim(h_a, h_b) = 1 − Ham(h_a, h_b) / 128
  Ham 为两个 128 位指纹中不同的位数（汉明距离）。
  完全相同 → 距离 0、相似度 1；128 位全不同 → 相似度 0。

本脚本同时给出可复用的"打包字节异或 + 查表 popcount"高速实现，
供 07（鲁棒聚合）、08（LSH）调用。

输出：outputs/06_pairwise_demo.npz（仅演示用的少量段对相似度）
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

# 256 个字节各自的 popcount（1 的个数）查表
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_bits(Ha, Hb):
    """
    对打包指纹（最后一维 16 字节 = 128 位）计算不同的位数。
    支持任意前置形状（..., 16）-> (..., )。
    """
    xor = Ha.astype(np.uint8) ^ Hb.astype(np.uint8)
    return POPCOUNT[xor].sum(axis=-1).astype(np.int64)


def hamming_similarity(Ha, Hb):
    """公式(3)：1 − Ham/128。输入为打包指纹。"""
    d = hamming_bits(Ha, Hb)
    return 1.0 - d / float(common.FP_BITS)


def main():
    data = common.load_npz("05_fingerprints.npz", allow_pickle=True)
    H32 = data["H32"]; seg_channel = data["seg_channel"]
    chnames = data["channel_names"]

    common.banner("06 汉明距离与窗口级相似度（公式 3）")
    # 基本验证
    a = H32[0, 0]; a2 = H32[0, 0]; b = H32[0, 1]
    d_same = hamming_bits(a, a2)
    d_diff = hamming_bits(a, b)
    print(f"同一窗口指纹自身比较：Ham={d_same:3d}, sim={1-d_same/128:.4f}")
    print(f"同段相邻窗口 0 vs 1：Ham={d_diff:3d}, sim={1-d_diff/128:.4f}")

    # 段对：对应位置窗口（不做位移）的相似度序列
    legit_ids = np.where(seg_channel == 0)[0]
    ipctc_ids = np.where(seg_channel == 1)[0]
    i, j = legit_ids[0], legit_ids[1]
    k = ipctc_ids[0]

    common.sub_banner("win=32 下，段对的对应窗口相似度（247 个窗口，取前 8 个）")
    for tag, p, q in [("合法-合法", i, j), ("合法-IPCTC", i, k)]:
        sims = hamming_similarity(H32[p], H32[q])
        print(f"  {tag:10s} 前8个: " + " ".join(f"{v:.3f}" for v in sims[:8])
              + f"  均值={sims.mean():.4f}")

    # 自身 vs 自身
    sims_self = hamming_similarity(H32[i], H32[i])
    print(f"  {'段自身':10s} 均值={sims_self.mean():.4f}（应=1）")

    # 手算 6 位的玩具：构造两个只差 6 位的指纹
    base = np.zeros((1, 16), np.uint8)
    bbits = np.zeros((1, 128), np.uint8)
    for pos in [28, 62, 67, 74, 99, 102]:
        bbits[0, pos] = 1
    other = np.packbits(bbits, axis=1)
    d6 = int(hamming_bits(base, other)[0])
    print(f"\n手算玩具：人为把 6 位翻成 1 -> Ham={d6}, sim=1−{d6}/128={1-d6/128:.4f}")

    path = common.save_npz(
        "06_pairwise_demo.npz",
        ll=hamming_similarity(H32[i], H32[j]),
        lc=hamming_similarity(H32[i], H32[k]),
    )
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
