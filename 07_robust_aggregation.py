# -*- coding: utf-8 -*-
"""
07_robust_aggregation.py —— 有限位移对齐 + top 小比例平均，得段对三尺度相似度
================================================================
论文 §3.2.1：
  "robust aggregation is applied over a limited shift range. Similarity is
   computed under each shift and then aggregated ... A robust similarity c(·)
   is obtained by averaging the top small portion of similarity values, which
   reduces extreme bias caused by occasional good alignments."

做法（对一对段、某一个尺度），两级聚合：
  1) 两段的窗口指纹序列允许在有限范围 ±τ 内相对滑动（隐蔽编码可能只造成小相位偏移）；
  2) 每个 shift 下，把重叠窗口逐对算汉明相似度（公式 3），取【中位数】作为该
     相位的对齐质量——中位数不会被个别偶然相同的窗口抬高；
  3) 得到 2τ+1 个"相位级分数"，再取其中最高的一小部分（top 3 个相位）平均，
     对应论文 "averaging the top small portion of similarity values"。
  三个尺度分别得到 w1/w2/w3，触发分 m=max(w1,w2,w3)。

说明：若直接把所有(位移×窗口)相似度混在一起取 top，会因 0 符号占主导、大量窗口
指纹雷同而把模式不同的段对也虚高到 1；两级聚合（先窗口级中位数、再相位级 top）
既容忍小相位偏移，又保留"整体模式差异"（如合法 vs IPCTC 仍显著更低）。

输出：outputs/07_pair_similarities.npz（代表性段对的 w1/w2/w3）
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


common = _load("00_common")
m06 = _load("06_hamming_similarity")

# 有限位移范围（以"窗口步长"为单位，论文未给具体值，取 ±5 的教学范围）
MAX_SHIFT = 5
# 相位级取最高的 3 个位移分数平均（top small portion，论文未给具体比例，教学值）
TOP_SHIFTS = 3


def scale_similarity(Ha, Hb, max_shift=MAX_SHIFT, top_shifts=TOP_SHIFTS):
    """
    单个尺度下，两段窗口指纹序列的鲁棒相似度 c。
    Ha,Hb: (num_win, 16) 打包指纹；返回一个标量相似度。
    两级聚合：窗口级中位数（每位移）→ 位移级取 top_shifts 个平均。
    """
    nw = len(Ha)
    ia0 = np.arange(nw)
    shift_scores = []
    for tau in range(-max_shift, max_shift + 1):
        ib = ia0 + tau
        mask = (ib >= 0) & (ib < nw)
        if mask.any():
            sims = m06.hamming_similarity(Ha[ia0[mask]], Hb[ib[mask]])
            shift_scores.append(float(np.median(sims)))#把中位数当做同尺度下同比特窗口下的两个不同段的相似度。
    ss = np.sort(np.asarray(shift_scores))
    k = min(top_shifts, len(ss))
    return float(ss[-k:].mean())# 取最大的


def three_scale_similarity(H32a, H64a, H128a, H32b, H64b, H128b):
    """返回 (w1,w2,w3,m)：三尺度鲁棒相似度及触发分 m=max。"""
    w1 = scale_similarity(H32a, H32b)
    w2 = scale_similarity(H64a, H64b)
    w3 = scale_similarity(H128a, H128b)
    return w1, w2, w3, max(w1, w2, w3)


def first_id(seg_channel, cid):
    """返回该类别第一个段索引；无该类样本时返回 None（真实数据可能缺 ε）。"""
    arr = np.where(seg_channel == cid)[0]
    return int(arr[0]) if len(arr) else None


def main():
    data = common.load_npz("05_fingerprints.npz", allow_pickle=True)
    H32, H64, H128 = data["H32"], data["H64"], data["H128"]
    seg_channel = data["seg_channel"]; chnames = data["channel_names"]

    li = first_id(seg_channel, 0)
    lj = (li + 1) if li is not None else None
    ip = first_id(seg_channel, 1)
    ip2 = (ip + 1) if ip is not None else None
    jt = first_id(seg_channel, 4)
    ep = first_id(seg_channel, 5)

    candidate_pairs = [
        ("合法-合法", li, lj),
        ("合法-IPCTC", li, ip),
        ("合法-Jitterbug", li, jt),
        ("合法-εklibur", li, ep),
        ("IPCTC-IPCTC", ip, ip2),
    ]
    pairs = [(t, a, b) for (t, a, b) in candidate_pairs
             if a is not None and b is not None]

    common.banner("07 有限位移对齐 + top 平均：段对三尺度相似度 w1/w2/w3")
    print(f"位移范围 ±{MAX_SHIFT}（窗口步长）；每个位移取窗口相似度中位数，")
    print(f"再取最高的 {TOP_SHIFTS} 个位移分数平均（top small portion）。")
    print(f"{'段对':16s}{'w1(32)':>9}{'w2(64)':>9}{'w3(128)':>9}{'m=max':>9}")
    rows = {}
    for tag, a, b in pairs:
        w1, w2, w3, m = three_scale_similarity(
            H32[a], H64[a], H128[a], H32[b], H64[b], H128[b])
        rows[tag] = (w1, w2, w3, m)
        print(f"{tag:16s}{w1:9.4f}{w2:9.4f}{w3:9.4f}{m:9.4f}")

    print("\n解读：同类段对（合法-合法、IPCTC-IPCTC）相似度高，跨类（合法-IPCTC）明显低；")
    if ep is not None:
        print("      合法-εklibur 可能接近合法（最隐蔽）。")
    else:
        print("      本数据集不含 ε-xlibur，故无“合法-εklibur”演示对。")

    save = {}
    for tag, vals in rows.items():
        save[tag] = np.array(vals)
    path = common.save_npz("07_pair_similarities.npz", **save)
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
