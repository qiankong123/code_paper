# -*- coding: utf-8 -*-
"""
11_edge_fusion.py —— 边强度融合（公式 6）
================================================================
论文 §3.2.2 公式(6)：
      s_uv = α1·w1 + α2·w2 + α3·w3 + β_gnn·φ_uv
  · w1/w2/w3：三个尺度的原始汉明相似度
  · φ_uv：GNN 精修后的边置信度
  · α 满足 sum-to-one，α 与 β 在离线验证集上做约束搜索
    （联合考虑 AUC/F1、候选边规模与稳定性）。

本复现没有真实验证集，用【合成标签】做一个教学网格搜索：
  目标 = 合法-合法边的 s 均值 − 合法-隐蔽边的 s 均值（两类边分离度，越大越好）。
  （隐蔽-隐蔽边构成"异常团"，其性质在 12/13 由段自身特征处理。）

输出：outputs/11_fusion.npz
  s : (E,) 每条边的最终强度；alpha(3,)、beta
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)


def main():
    g = common.load_npz("10_gnn.npz", allow_pickle=True)
    edges = g["edges"]; phi = g["phi"]; seg_label = g["seg_label"]
    w1 = edges[:, 2]; w2 = edges[:, 3]; w3 = edges[:, 4]
    u = edges[:, 0].astype(int); v = edges[:, 1].astype(int)

    common.banner("11 边强度融合  s = α1w1 + α2w2 + α3w3 + β·φ（公式6）")

    ll = (seg_label[u] == 0) & (seg_label[v] == 0)      # 合法-合法
    cross = (seg_label[u] != seg_label[v])              # 合法-隐蔽

    # 网格搜索：α 在单纯形上（step 0.2），β 一维
    best = None
    grid = np.arange(0, 1.0001, 0.2)
    for a1 in grid:
        for a2 in grid:
            if a1 + a2 > 1.0 + 1e-9:
                continue
            a3 = 1.0 - a1 - a2
            for beta in [0.0, 0.25, 0.5, 1.0, 2.0]:
                s = a1 * w1 + a2 * w2 + a3 * w3 + beta * phi
                if ll.sum() > 0 and cross.sum() > 0:
                    sep = s[ll].mean() - s[cross].mean()
                else:
                    sep = 0.0
                if best is None or sep > best[0]:
                    best = (sep, a1, a2, a3, beta)

    sep, a1, a2, a3, beta = best
    alpha = np.array([a1, a2, a3])
    s = a1 * w1 + a2 * w2 + a3 * w3 + beta * phi

    print(f"网格搜索最优：α = [{a1:.1f}, {a2:.1f}, {a3:.1f}]（sum=1），β = {beta:.2f}")
    print(f"两类边分离度（合法-合法均值 − 合法-隐蔽均值）= {sep:.4f}")

    common.sub_banner("融合前后对比（边强度均值）")
    print(f"{'边类型':14s}{'边数':>6}{'w1均值':>9}{'w3均值':>9}{'φ均值':>9}{'s均值':>9}")
    def row(name, mask):
        print(f"{name:14s}{int(mask.sum()):6d}{w1[mask].mean():9.3f}{w3[mask].mean():9.3f}"
              f"{phi[mask].mean():9.3f}{s[mask].mean():9.3f}")
    bb = (seg_label[u] == 1) & (seg_label[v] == 1)
    row("合法-合法", ll)
    row("合法-隐蔽", cross)
    row("隐蔽-隐蔽", bb)

    print("\n说明：合法-合法边强度高；合法-隐蔽边弱（跨不进大团）；")
    print("      隐蔽-隐蔽若形成异常团，强度也会高——这部分在 12/13 靠段自身特征识别。")

    path = common.save_npz(
        "11_fusion.npz",
        s=s, alpha=alpha, beta=np.array([beta]),
        edges=edges, edge_type=g["edge_type"], phi=phi,
        X=g["X"], A=g["A"], theta=g["theta"],
        seg_label=seg_label, seg_channel=g["seg_channel"],
        seg_conn_id=g["seg_conn_id"], seg_index=g["seg_index"],
        channel_names=g["channel_names"],
    )
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
