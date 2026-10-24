# -*- coding: utf-8 -*-
"""
12_graph_features.py —— 每个段的 4 维检测特征（公式 7–10）
================================================================
论文 §3.2.3："After GNN refinement, four features are extracted from each
traffic segment, and a 4-dimensional feature vector is constructed for
classifier decision making."

  段自身（不依赖图）：
    ① H_sym 符号熵（公式7）：H_sym = − Σ_{k=1}^7 p(k) log p(k)
       衡量段内 7 级符号分布的均匀程度。注意：熵的高低方向取决于编码模式，
       不能一概而论。本合成数据里，合法平稳段符号集中在 0（熵≈0.9）；
       大倍率编码把符号推到 ±2/±3 各档、分布更均匀（熵≈1.4–1.7）；
       jitter/ε 仍接近合法。分类器只需学到"偏离正常分布"即可。
    ② L_cy 循环均值（公式8）：L_cy = (Ū + D̄)/2
       Ū、D̄ 分别为符号序列中相邻"正跳""负跳"位置的平均间隔，
       周期性注入会形成稳定的小间隔节奏（L_cy 偏小）。

  图结构（依赖邻居边强 s，k 取 top-k，论文未给 k，教学取 k=5）：
    ③ ρ 边权累加（公式9）：ρ = log(1 + Σ_{i∈top-k} s_i)
    ④ √S_flow（公式10）：√S_flow = sqrt(Σ top_scores / k)

输出：outputs/12_features.npz
  feat : (num_seg, 4)，列依次为 H_sym, L_cy, ρ, √S_flow
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

TOP_K = 5


def symbol_entropy(sym_seq):
    """公式7：七级符号经验分布的香农熵。"""
    counts = np.bincount(sym_seq + 3, minlength=common.NUM_SYMBOLS).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def cycle_mean(sym_seq):
    """公式8：相邻正跳、负跳间隔均值的平均。"""
    pos = np.where(sym_seq > 0)[0]
    neg = np.where(sym_seq < 0)[0]
    U = np.mean(np.diff(pos)) if len(pos) > 1 else 0.0
    D = np.mean(np.diff(neg)) if len(neg) > 1 else 0.0
    return float((U + D) / 2.0)


def graph_topk_features(edges, s, num_seg, k=TOP_K):
    """公式9、10：对每个段取其邻边中最大的 k 个 s，算 ρ 与 √S_flow。"""
    acc = [[] for _ in range(num_seg)]
    for e, sv in zip(edges, s):
        a, b = int(e[0]), int(e[1])
        acc[a].append(sv); acc[b].append(sv)
    rho = np.zeros(num_seg); sflow = np.zeros(num_seg)
    for i in range(num_seg):
        if len(acc[i]) == 0:
            rho[i] = 0.0; sflow[i] = 0.0
            continue
        top = np.sort(np.asarray(acc[i]))[-k:]
        rho[i] = np.log(1.0 + top.sum())
        sflow[i] = np.sqrt(top.sum() / k)   # 缺失边贡献 0，分母固定 k
    return rho, sflow


def main():
    sym = common.load_npz("03_symbols.npz", allow_pickle=True)
    seg_symbols = sym["seg_symbols"]; seg_channel = sym["seg_channel"]
    f = common.load_npz("11_fusion.npz", allow_pickle=True)
    edges = f["edges"]; s = f["s"]
    num_seg = seg_symbols.shape[0]

    common.banner("12 每段 4 维特征：H_sym / L_cy / ρ / √S_flow")
    H = np.array([symbol_entropy(x) for x in seg_symbols])
    L = np.array([cycle_mean(x) for x in seg_symbols])
    rho, sflow = graph_topk_features(edges, s, num_seg)
    feat = np.stack([H, L, rho, sflow], axis=1)

    common.sub_banner("各类别特征均值（看区分度；top-k 中 k=%d）" % TOP_K)
    print(f"{'类别':14s}{'H_sym熵':>10}{'L_cy节奏':>10}{'ρ累加':>9}{'√S_flow':>9}")
    chnames = f["channel_names"]
    for cid, name in enumerate(chnames):
        ix = np.where(seg_channel == cid)[0]
        print(f"{str(name):14s}{H[ix].mean():10.3f}{L[ix].mean():10.3f}"
              f"{rho[ix].mean():9.3f}{sflow[ix].mean():9.3f}")

    print("\n要点（本合成数据）：")
    print("  · 大倍率编码 IPCTC/LNCTC/TRCTC：H_sym 高、L_cy 小，且 ρ/√S_flow=0（孤立）→ 最易识别；")
    print("  · 合法/jitter/ε 都在大团内（ρ/√S_flow 高）；jitter 靠 L_cy 偏小、H_sym 略高识别；")
    print("  · ε 的 H_sym/L_cy 与合法几乎无差异（ε-不可区分，最隐蔽）。")

    print("\n前 5 个段的特征向量:")
    for i in range(5):
        print(f"  段{i}: H_sym={H[i]:.3f}, L_cy={L[i]:.2f}, ρ={rho[i]:.3f}, "
              f"√S_flow={sflow[i]:.3f}")

    path = common.save_npz(
        "12_features.npz", feat=feat,
        seg_label=f["seg_label"], seg_channel=seg_channel,
        seg_conn_id=f["seg_conn_id"], seg_index=f["seg_index"],
        channel_names=chnames,
    )
    print(f"\n已保存：{path}（feat 形状 {feat.shape}）")


if __name__ == "__main__":
    main()
