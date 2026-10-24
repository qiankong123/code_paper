# -*- coding: utf-8 -*-
"""
13_two_level_classifier.py —— 两级阈值级联分类器（公式 11）
================================================================
论文 §3.2.3/§3.2.4 的两级判定思想：
  (0) 图拓扑硬规则：两级阈值 θ0/θ1 之下，与正常基线明显不同的段对"无边"，
      于是大倍率编码（IPCTC/LNCTC/TRCTC）在图中成为【孤立节点】（ρ=0、√S_flow=0）。
      孤立节点直接判隐蔽——这是阈值级联最确定的判定。
  (1) 第一级（召回导向 τ1）：对仍在团内（高相似）的节点，pro1 < τ1 直接判正常，
      否则进候选（对应 Jitterbug/ε 这类高相似隐蔽信道）；
  (2) 第二级（精度导向 τ2）：候选中 pro2 ≥ τ2 才判隐蔽，宁可漏检最隐蔽的 ε，
      也不误报合法。
  公式(11)：final_label(x) = 1  if pro1(x) ≥ τ1 ∧ pro2(x) ≥ τ2
                               0  otherwise（孤立节点恒为 1）

论文用 LightGBM；本机未装 lightgbm，用 sklearn.ensemble.GradientBoostingClassifier
（GBDT）替代，原理一致。大倍率由拓扑规则处理，GBDT 只在【团内】节点
（合法/Jitterbug/ε）上训练，避免被大倍率占主导。

阈值标定：按连接级划分（同连接段不跨集，论文 §4.1），测试集 = conn%3==0；
两级阈值都在连接级 5 折 out-of-fold 概率上标定，避免训练集自标定过拟合。

输出：outputs/13_prediction.npz（测试段索引、两级概率、最终标签、阈值）
"""

import numpy as np
import pickle, importlib.util, os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

RHO, SFLOW = 2, 3     # 特征列：H_sym, L_cy, ρ, √S_flow
ISO_EPS = 1e-6        # ρ/√S_flow 视为孤立（无边）的阈值
CV = GroupKFold(5)


def oof_proba(model, X, y, groups):
    return cross_val_predict(model, X, y, groups=groups, cv=CV, method="predict_proba")[:, 1]


def best_precision_threshold(p, y, min_precision=0.97):
    """在 OOF 概率上选阈值：满足 precision≥min_precision 前提下召回最大。"""
    best = None
    for t in np.linspace(0.01, 0.99, 99):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision >= min_precision and (best is None or recall > best[1]):
            best = (t, recall, precision)
    return best[0] if best else None


def main():
    d = common.load_npz("12_features.npz", allow_pickle=True)
    feat = d["feat"]; y = d["seg_label"]; conn = d["seg_conn_id"]
    chnames = d["channel_names"]; chan = d["seg_channel"]

    test_mask = (conn % 3 == 0)
    train_mask = ~test_mask
    iso = feat[:, RHO] < ISO_EPS          # 孤立节点（大倍率）
    clique_tr = train_mask & ~iso         # 训练中在团内的节点（合法/Jitterbug/ε）

    Xtr, ytr, gtr = feat[clique_tr], y[clique_tr], conn[clique_tr]
    Xte, yte = feat[test_mask], y[test_mask]
    test_idx = np.where(test_mask)[0]
    iso_te = iso[test_mask]

    common.banner("13 两级阈值级联分类器（公式 11）")
    n_iso_tr = int((train_mask & iso).sum())
    print(f"连接级划分：训练段 {int(train_mask.sum())}（其中孤立段 {n_iso_tr} 由拓扑规则直接判隐蔽，"
          f"团内 {len(ytr)} 段训练 GBDT），测试段 {len(yte)}")
    print(f"测试段中孤立节点 {int(iso_te.sum())} 个 → 直接判隐蔽（大倍率编码）")

    # ---- 团内两级 GBDT ----
    first = GradientBoostingClassifier(
        n_estimators=150, max_depth=2, min_samples_leaf=3,
        learning_rate=0.05, random_state=common.SEED)
    p1_oof = oof_proba(first, Xtr, ytr, gtr)
    first.fit(Xtr, ytr)

    common.sub_banner("团内第一级 OOF 概率按类别（min/中位/max）")
    print(f"{'类别':14s}{'n':>4}{'min':>8}{'中位':>8}{'max':>8}")
    for cid in range(len(chnames)):
        v = p1_oof[chan[clique_tr] == cid]
        if len(v):
            print(f"{str(chnames[cid]):14s}{len(v):4d}{v.min():8.3f}{np.median(v):8.3f}{v.max():8.3f}")

    tau1 = float(np.quantile(p1_oof[ytr == 1], 0.01))   # 召回导向
    print(f"\n第一级 τ1 = {tau1:.4f}（团内正类 OOF 概率 1% 分位，召回导向）")

    cand = p1_oof >= tau1
    second = GradientBoostingClassifier(
        n_estimators=40, max_depth=2, min_samples_leaf=8,
        learning_rate=0.1, random_state=common.SEED + 1)
    if len(np.unique(ytr[cand])) == 2:
        p2_oof = oof_proba(second, Xtr[cand], ytr[cand], gtr[cand])
        second.fit(Xtr[cand], ytr[cand])
        # 精度导向：τ2 取候选中【合法段】OOF 第二级概率的 99% 分位，
        # 即"高于几乎所有看起来可疑的合法段"才判隐蔽，直接保护合法；
        # 可分的高相似信道（Jitterbug，H_sym 高/L_cy 小）自然越过阈值，
        # 与合法重叠的 ε 漏检（ε-不可区分的理论代价）。
        p2_legit = p2_oof[ytr[cand] == 0]
        p2_jit = p2_oof[ytr[cand] == 1]
        tau2 = float(np.quantile(p2_legit, 0.99))
        print(f"第二级 τ2 = {tau2:.4f}（候选中合法 OOF 概率的 99% 分位，精度导向）")
        print(f"    候选合法 OOF pro2 中位 {np.median(p2_legit):.3f}/最大 {p2_legit.max():.3f}；"
              f"候选隐蔽 OOF pro2 中位 {np.median(p2_jit):.3f}")
    else:
        print("（候选仅含单类，第二级退化为复用第一级）")
        second = first; tau2 = tau1

    # ---- 测试：孤立硬规则 + 团内级联 ----
    p1 = first.predict_proba(Xte)[:, 1]
    final = np.zeros(len(yte), dtype=int)
    p2 = np.full(len(yte), np.nan)
    final[iso_te] = 1                    # 孤立节点直接判隐蔽
    cand_te = (~iso_te) & (p1 >= tau1)
    if cand_te.any():
        p2[cand_te] = second.predict_proba(Xte[cand_te])[:, 1]
        final[cand_te] = (p2[cand_te] >= tau2).astype(int)

    common.sub_banner("测试级联过程")
    print(f"  测试段总数 {len(yte)}：孤立判隐蔽 {int(iso_te.sum())} 段")
    print(f"  团内第一级 pro1<τ1 直接判正常：{int((~iso_te & ~cand_te).sum())} 段")
    print(f"  团内进入第二级候选：{int(cand_te.sum())} 段，判隐蔽 {int((final & cand_te).sum())} 段")

    common.sub_banner("测试段判定示例（前 12 个）")
    print(f"{'段':>5}{'真实':>5}{'pro1':>8}{'pro2':>8}{'最终':>5}")
    for t in range(min(12, len(yte))):
        tag = "孤立" if iso_te[t] else ("—" if np.isnan(p2[t]) else f"{p2[t]:.3f}")
        print(f"{test_idx[t]:5d}{int(yte[t]):5d}{p1[t]:8.3f}{tag:>8}{int(final[t]):5d}")

    with open(common.out_path("13_models.pkl"), "wb") as fpkl:
        pickle.dump({"first": first, "second": second, "tau1": tau1,
                     "tau2": tau2, "iso_eps": ISO_EPS}, fpkl)
    path = common.save_npz(
        "13_prediction.npz",
        test_idx=test_idx, test_y=yte, pro1=p1, pro2=p2, final=final,
        iso=iso_te.astype(int), tau=np.array([tau1, tau2]), channel_names=chnames,
    )
    print(f"\n已保存：{path}")
    print(f"模型已保存：{common.out_path('13_models.pkl')}")


if __name__ == "__main__":
    main()
