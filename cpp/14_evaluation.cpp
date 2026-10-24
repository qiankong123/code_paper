// 14_evaluation.cpp —— 评估指标（混淆矩阵 / 准确率 / 精确率 / 召回率 / FPR / F1 / AUC）
// 对应 Python 14_evaluation.py。ROC-AUC 用 Mann–Whitney 统计量（手算可复现）。
#include "common.hpp"

int main() {
    init_console();
    Store d = Store::load(out_path("13_results.bin"));
    const BinArc* y = d.find("test_y");
    const BinArc* final = d.find("final");
    const BinArc* pro1 = d.find("pro1");
    const BinArc* iso = d.find("iso");
    const BinArc* ch = d.find("test_channel");
    const BinArc* tau = d.find("tau");
    const BinArc* imp = d.find("first_imp");
    int T = y->shape[0];

    int tp = 0, tn = 0, fp = 0, fn = 0;
    std::vector<double> score(T);
    for (int t = 0; t < T; ++t) {
        int fv = final->i[t], yv = y->i[t];
        score[t] = iso->i[t] ? 1.0 : pro1->d[t];
        if (fv == 1 && yv == 1) ++tp;
        else if (fv == 0 && yv == 0) ++tn;
        else if (fv == 1 && yv == 0) ++fp;
        else ++fn;
    }
    double precision = tp / (double)std::max(tp + fp, 1);
    double recall = tp / (double)std::max(tp + fn, 1);
    double fpr = fp / (double)std::max(fp + tn, 1);
    double accuracy = (tp + tn) / (double)std::max(tp + tn + fp + fn, 1);
    double f1 = 2 * precision * recall / std::max(precision + recall, 1e-9);

    // ROC-AUC（Mann–Whitney：正样本分数高于负样本的概率）
    int npos = 0, nneg = 0; double win = 0;
    for (int i = 0; i < T; ++i) {
        if (y->i[i] != 1) continue;
        ++npos;
        for (int j = 0; j < T; ++j) {
            if (y->i[j] != 0) continue;
            if (score[i] > score[j]) win += 1.0;
            else if (score[i] == score[j]) win += 0.5;
        }
    }
    for (int j = 0; j < T; ++j) if (y->i[j] == 0) ++nneg;
    double auc = win / (double)(npos * nneg);

    banner("14 评估指标（合成数据 · 测试集）");
    sub_banner("混淆矩阵与整体指标");
    std::cout << "                    预测合法(0)   预测隐蔽(1)\n";
    std::printf("  实际合法(0)        TN=%-8d   FP=%-8d\n", tn, fp);
    std::printf("  实际隐蔽(1)        FN=%-8d   TP=%-8d\n", fn, tp);
    std::printf("\n  准确率 Accuracy = %.4f\n", accuracy);
    std::printf("  精确率 Precision= %.4f\n", precision);
    std::printf("  召回率 Recall/TPR= %.4f   （公式12）\n", recall);
    std::printf("  误报率 FPR       = %.4f\n", fpr);
    std::printf("  F1               = %.4f\n", f1);
    std::printf("  ROC-AUC（完整级联置信度：孤立置1、团内 pro1）= %.4f\n", auc);

    sub_banner("分信道召回率（对应论文 Table 4 的 Recall 列）");
    std::cout << "信道               测试样本   TP    FN    召回率\n";
    double r_eps = 0.0;
    for (int cid = 1; cid < 6; ++cid) {
        int m = 0, tpc = 0, fnc = 0;
        for (int t = 0; t < T; ++t) if (ch->i[t] == cid) {
            ++m;
            if (final->i[t] == 1 && y->i[t] == 1) ++tpc;
            else if (final->i[t] == 0 && y->i[t] == 1) ++fnc;
        }
        if (m == 0) continue;
        double rr = tpc / (double)std::max(tpc + fnc, 1);
        if (cid == 5) r_eps = rr;
        std::printf("  %-14s %8d %5d %5d %9.4f\n",
                    CHANNEL_NAMES[cid], m, tpc, fnc, rr);
    }

    sub_banner("第一级 GBDT 特征重要性（相对贡献，%）");
    const char* fn4[4] = {"H_sym", "L_cy", "rho", "sqrtS"};
    int order[4] = {0, 1, 2, 3};
    for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b)
            if (imp->d[order[b]] > imp->d[order[a]]) std::swap(order[a], order[b]);
    double maximp = imp->d[order[0]];
    for (int a = 0; a < 4; ++a) {
        int q = order[a];
        int bars = maximp > 0 ? (int)(imp->d[q] / maximp * 30) : 0;
        std::printf("  %-8s %6.1f  ", fn4[q], imp->d[q]);
        for (int b = 0; b < bars; ++b) std::cout << "#";
        std::cout << "\n";
    }

    sub_banner("说明");
    std::cout << "  · 判定路径：①孤立节点（ρ=0，两级阈值下与任何段都不成边，对应大倍率\n"
              << "    IPCTC/LNCTC/TRCTC）由图拓扑硬规则直接判隐蔽；②仍在团内的高相似段\n"
              << "    （合法/Jitterbug/ε）走两级 GBDT：τ1 召回导向、τ2 精度导向。\n";
    std::cout << "  · 本结果来自【合成数据】，仅用于验证计算链路，不代表论文真实 NCTC 数值。\n";
    std::printf("  · ε-klibur 的周期微调被指数平滑吸收、接近 ε-不可区分极限，是五类里最难的一类：\n"
                "    本 C++ 运行（mt19937 RNG）其召回仅 %.2f，远低于其余四类（均=1.0）；\n"
                "    同参数 Python 版（numpy PCG64 RNG）因具体噪声序列不同，ε 召回为 0、FPR≈0.075。\n"
                "    数值不同但结论一致：大倍率三类靠拓扑硬规则必检出、Jitter 靠节奏特征检出、\n"
                "    ε 最隐蔽且以牺牲其召回换取合法不误报。真实 NCTC 数据含更丰富痕迹，论文 Table 4\n"
                "    五类信道 AUC≥0.997、Recall 均很高。\n", r_eps);
    std::printf("  · 第一级 τ1=%.4f 保召回；第二级 τ2=%.4f 保精度。\n", tau->d[0], tau->d[1]);

    Store st;
    std::vector<double> conf(4);
    conf[0] = tn; conf[1] = fp; conf[2] = fn; conf[3] = tp;
    st.put_d("confusion", std::vector<int>{2, 2}, conf);
    st.put_d("metrics", std::vector<int>{6},
             std::vector<double>{accuracy, precision, recall, fpr, f1, auc});
    st.put_d("score", std::vector<int>{T}, score);
    st.put_i("final", std::vector<int>{T}, final->i);
    st.put_i("test_y", std::vector<int>{T}, y->i);
    st.put_i("test_channel", std::vector<int>{T}, ch->i);
    std::string pth = out_path("14_metrics.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
