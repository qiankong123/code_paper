# -*- coding: utf-8 -*-
"""
14_evaluation.py —— 评估指标（混淆矩阵 / 召回率 / 精确率 / F1 / FPR / AUC）
================================================================
论文 §4.2：CTC 为正类、合法为负类，两级级联给出最终标签。
  · 召回率 Recall（公式12，即 TPR）= TP / (TP + FN)
  · 精确率 Precision = TP / (TP + FP)
  · 误报率 FPR = FP / (FP + TN)
  · F1 = 2·P·R/(P+R)
  · AUC：用第一级连续概率 pro1 计算 ROC-AUC
并按 5 类信道分别统计召回（对应论文 Table 4 的分信道 Recall）。

注意：本结果来自【合成数据】，用于验证流程，不代表论文在 NCTC 上的真实数值。
"""

import numpy as np
import pickle, importlib.util, os
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)


def main():
    d = common.load_npz("13_prediction.npz", allow_pickle=True)
    test_idx, yte, p1, final = d["test_idx"], d["test_y"], d["pro1"], d["final"]
    iso_te = d["iso"].astype(bool)
    chnames = d["channel_names"]

    # 测试段的真实信道（从 12 特征文件取 channel）
    fd = common.load_npz("12_features.npz", allow_pickle=True)
    test_channel = fd["seg_channel"][test_idx]

    common.banner("14 评估指标（合成数据 · 测试集）")

    tp = int(((final == 1) & (yte == 1)).sum())
    tn = int(((final == 0) & (yte == 0)).sum())
    fp = int(((final == 1) & (yte == 0)).sum())
    fn = int(((final == 0) & (yte == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # 完整级联的置信度：孤立节点（ρ=0）由拓扑硬规则置 1（确定异常），团内节点用 pro1
    final_score = np.where(iso_te, 1.0, p1)
    auc = roc_auc_score(yte, final_score)

    common.sub_banner("混淆矩阵与整体指标")
    print("                    预测合法(0)   预测隐蔽(1)")
    print(f"  实际合法(0)        TN={tn:<8d}   FP={fp:<8d}")
    print(f"  实际隐蔽(1)        FN={fn:<8d}   TP={tp:<8d}")
    print(f"\n  准确率 Accuracy = {accuracy:.4f}")
    print(f"  精确率 Precision= {precision:.4f}")
    print(f"  召回率 Recall/TPR= {recall:.4f}   （公式12）")
    print(f"  误报率 FPR       = {fpr:.4f}")
    print(f"  F1               = {f1:.4f}")
    print(f"  ROC-AUC（完整级联置信度：孤立置1、团内 pro1）= {auc:.4f}")

    common.sub_banner("分信道召回率（对应论文 Table 4 的 Recall 列）")
    print(f"{'信道':16s}{'测试样本':>8}{'TP':>6}{'FN':>6}{'召回率':>10}")
    per_channel = {}
    for cid in range(1, len(chnames)):
        m = (test_channel == cid)
        if m.sum() == 0:
            continue
        tp_c = int(((final == 1) & (yte == 1) & m).sum())
        fn_c = int(((final == 0) & (yte == 1) & m).sum())
        r = tp_c / max(tp_c + fn_c, 1)
        per_channel[str(chnames[cid])] = r
        print(f"{str(chnames[cid]):16s}{int(m.sum()):8d}{tp_c:6d}{fn_c:6d}{r:10.4f}")

    # 特征重要性
    with open(common.out_path("13_models.pkl"), "rb") as fpkl:
        models = pickle.load(fpkl)
    imp = models["first"].feature_importances_
    feats = ["H_sym", "L_cy", "ρ", "√S_flow"]
    common.sub_banner("第一级 GBDT 特征重要性（4 个特征的相对贡献）")
    for name, v in sorted(zip(feats, imp), key=lambda x: -x[1]):
        print(f"  {name:8s} {v:6.1f}  " + "#" * int(v / max(imp.max(), 1) * 30))

    # 两级阈值（ROC 参考）
    fprs, tprs, _ = roc_curve(yte, final_score)
    common.sub_banner("说明")
    print("  · 判定路径：①孤立节点（ρ=0，两级阈值下与任何段都不成边，对应大倍率")
    print("    IPCTC/LNCTC/TRCTC）由图拓扑硬规则直接判隐蔽；②仍在团内的高相似段")
    print("    （合法/Jitterbug/ε）走两级 GBDT：τ1 召回导向、τ2 精度导向。")
    print("  · 本结果来自【合成数据】，仅用于验证计算链路，不代表论文真实 NCTC 数值。")
    print("  · 本合成实验中 ε-klibur 的周期微调被指数平滑完全吸收、达到 ε-不可区分极限，")
    print("    其段特征与合法重叠，无法在不误报合法的前提下检出；第二级精度导向选择")
    print("    保护合法，故 ε 召回为 0（体现其理论最难）。真实 NCTC 数据含更丰富痕迹，")
    print("    论文 Table 4 五类信道 AUC≥0.997、Recall 均很高。")
    print(f"  · 第一级 τ1={models['tau1']:.4f} 保召回；第二级 τ2={models['tau2']:.4f} 保精度。")

    path = common.save_npz(
        "14_metrics.npz",
        confusion=np.array([[tn, fp], [fn, tp]]),
        precision=np.array([precision]), recall=np.array([recall]),
        fpr=np.array([fpr]), f1=np.array([f1]), auc=np.array([auc]),
        roc_fpr=fprs, roc_tpr=tprs,
    )
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
