# -*- coding: utf-8 -*-
"""
09_threshold_graph.py —— 阈值标定 θ0/θ1、边分流、建稀疏相似图
================================================================
论文 §3.2.1–§3.2.2：
  · 对 LSH 候选对算三尺度相似度，m=max(w1,w2,w3) 作为建图触发分；
  · 阈值在"合法-合法"段对的鲁棒相似度 c(·) 上标定，
    取分位数 Q0.30、Q0.60 作为 θ0、θ1；
  · m < θ0 丢弃；θ0 ≤ m < θ1 为候选边；m ≥ θ1 为告警边；
  · 以段为节点、候选/告警边为边，建稀疏相似图 G=(V,E)；
  · 节点初始化特征只有两个统计量：节点度、邻边平均权重。

输出：outputs/09_graph.npz
  theta     : (θ0, θ1)
  edges     : (E, 6) 每行 [i, j, w1, w2, w3, m]
  edge_type : (E,)   0=候选边, 1=告警边
  X         : (num_seg, 2) 节点特征 [度, 邻边均权]
  A         : (num_seg, num_seg) 带权邻接矩阵（权重=m）
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


common = _load("00_common")
m07 = _load("07_robust_aggregation")


def main():
    # 05 指纹
    fp = common.load_npz("05_fingerprints.npz", allow_pickle=True)
    H32, H64, H128 = fp["H32"], fp["H64"], fp["H128"]
    seg_channel = fp["seg_channel"]; seg_label = fp["seg_label"]
    # 08 候选
    cand = common.load_npz("08_candidates.npz", allow_pickle=True)
    candidates = cand["candidates"]
    num_seg = H32.shape[0] #段数量（一段2000个IPD）

    common.banner("09 阈值标定、边分流与建图")

    # 1) 对候选对算三尺度 w 与 m
    rec = []
    for t, (i, j) in enumerate(candidates):
        w1, w2, w3, m = m07.three_scale_similarity(
            H32[i], H64[i], H128[i], H32[j], H64[j], H128[j])
        rec.append((i, j, w1, w2, w3, m))
        if (t + 1) % 300 == 0:
            print(f"  已计算 {t+1}/{len(candidates)} 个候选对 ...")
    rec = np.array(rec)
    if len(rec) == 0:
        raise RuntimeError("LSH 候选对为 0（真实指纹相似度过低），无法建图；"
                           "请检查 IPD 清洗/尺度，或放宽 08 的证据阈值。")

    # 2) 在合法-合法对上标定 θ0/θ1（Q0.30/Q0.60）
    ll_mask = (seg_channel[rec[:, 0].astype(int)] == 0) & \
              (seg_channel[rec[:, 1].astype(int)] == 0)
    ll_m = rec[ll_mask, 5]
    if ll_m.size == 0:
        raise RuntimeError("没有合法-合法候选对，无法在合法分布上标定 θ0/θ1；"
                           "请增加合法连接数或检查真实数据清洗。")
    theta0 = float(np.quantile(ll_m, common.THETA_LO_Q))
    theta1 = float(np.quantile(ll_m, common.THETA_HI_Q))
    print(f"\n合法-合法候选对 m 分布：min={ll_m.min():.3f}, "
          f"Q30={theta0:.3f}, Q60={theta1:.3f}, max={ll_m.max():.3f}")

    # 3) 边分流
    mvals = rec[:, 5]
    keep = mvals >= theta0
    edges = rec[keep]
    edge_type = (edges[:, 5] >= theta1).astype(np.int64)   # 0 候选, 1 告警

    # 4) 带权邻接矩阵与节点特征
    A = np.zeros((num_seg, num_seg), dtype=np.float64)
    for e in edges:
        i, j, m = int(e[0]), int(e[1], ), e[5]
        A[i, j] = m; A[j, i] = m
    degree = (A > 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_w = np.where(degree > 0, A.sum(axis=1) / np.maximum(degree, 1), 0.0)
    X = np.stack([degree.astype(np.float64), mean_w], axis=1)

    common.sub_banner("建图结果")
    print(f"  候选对 {len(rec)} 个 -> 丢弃(m<θ0) {int((~keep).sum())} 个，"
          f"保留边 {len(edges)} 条")
    print(f"  其中候选边 {int((edge_type==0).sum())} 条，告警边 {int((edge_type==1).sum())} 条")
    print(f"  节点数（段）= {num_seg}，孤立节点（度=0）= {int((degree==0).sum())}")

    common.sub_banner("按真实类别看平均度/均权（验证团簇结构）")
    print(f"{'类别':14s}{'节点数':>7}{'平均度':>9}{'平均邻边均权':>13}{'孤立占比':>10}")
    chnames = fp["channel_names"]
    for cid, name in enumerate(chnames):
        ix = np.where(seg_channel == cid)[0]
        if len(ix) == 0:
            print(f"{str(name):14s}{'0':>7s}    （无样本）")
            continue
        iso = np.mean(degree[ix] == 0) * 100
        print(f"{str(name):14s}{len(ix):7d}{degree[ix].mean():9.2f}"
              f"{mean_w[ix].mean():13.3f}{iso:9.1f}%")

    print("\n节点特征 X（前 6 个节点，列为 [度, 邻边均权]）:")
    for i in range(6):
        print(f"  段{i:3d} (channel={str(chnames[seg_channel[i]]):10s}): "
              f"度={int(X[i,0]):3d}, 均权={X[i,1]:.4f}")

    path = common.save_npz(
        "09_graph.npz",
        theta=np.array([theta0, theta1]),
        edges=edges, edge_type=edge_type, X=X, A=A,
        seg_label=seg_label, #零是正常，一是异常。
        seg_channel=seg_channel,#零是没有方法和流量，一是第一个将方法流量....
        seg_conn_id=fp["seg_conn_id"], seg_index=fp["seg_index"],# cong05拿到的数据直接存储在09
        channel_names=chnames,
    )
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
