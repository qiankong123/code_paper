// 09_threshold_graph.cpp —— 阈值标定 θ0/θ1、边分流、建稀疏相似图
// 对应 Python 09_threshold_graph.py。节点=段，节点特征仅 [度, 邻边均权]。
#include "common.hpp"

int main() {
    init_console();
    Store fp = Store::load(out_path("05_fingerprints.bin"));
    const BinArc* H[3] = {fp.find("H32"), fp.find("H64"), fp.find("H128")};
    const BinArc* ch = fp.find("seg_channel");
    const BinArc* lab = fp.find("seg_label");
    Store cd = Store::load(out_path("08_candidates.bin"));
    const BinArc* cand = cd.find("candidates");
    int num_seg = H[0]->shape[0];
    int ncand = cand->shape[0];

    auto fp_ptr = [&](int sc, int g) -> const unsigned char* {
        return &H[sc]->b[(long long)g * H[sc]->shape[1] * FP_BYTES];
    };

    banner("09 阈值标定、边分流与建图");
    // 1) 候选对三尺度相似度
    std::vector<double> rec;   // i,j,w1,w2,w3,m 平铺
    std::vector<double> ll_m;
    for (int t = 0; t < ncand; ++t) {
        int i = cand->i[2 * t], j = cand->i[2 * t + 1];
        double w[4];
        for (int sc = 0; sc < 3; ++sc)
            w[sc] = scale_similarity(fp_ptr(sc, i), fp_ptr(sc, j), H[sc]->shape[1]);
        w[3] = std::max(std::max(w[0], w[1]), w[2]);
        for (int q = 0; q < 4; ++q) rec.push_back(w[q]);
        rec.push_back((double)i); rec.push_back((double)j);
        if (ch->i[i] == 0 && ch->i[j] == 0) ll_m.push_back(w[3]);
        if ((t + 1) % 2000 == 0) std::cout << "  已计算 " << t + 1 << "/" << ncand << " 个候选对 ...\n";
    }
    // rec 现在每行按 [w1,w2,w3,m,i,j] 存放，重新整理为 [i,j,w1,w2,w3,m]
    std::vector<double> rec6((long long)ncand * 6);
    for (int t = 0; t < ncand; ++t) {
        rec6[(long long)t * 6 + 0] = rec[(long long)t * 6 + 4]; // i
        rec6[(long long)t * 6 + 1] = rec[(long long)t * 6 + 5]; // j
        rec6[(long long)t * 6 + 2] = rec[(long long)t * 6 + 0];
        rec6[(long long)t * 6 + 3] = rec[(long long)t * 6 + 1];
        rec6[(long long)t * 6 + 4] = rec[(long long)t * 6 + 2];
        rec6[(long long)t * 6 + 5] = rec[(long long)t * 6 + 3];
    }

    double theta0 = quantile_vec(ll_m, THETA_LO_Q);
    double theta1 = quantile_vec(ll_m, THETA_HI_Q);
    std::vector<double> srt = ll_m; std::sort(srt.begin(), srt.end());
    std::cout << "\n合法-合法候选对 m 分布：min=" << srt.front()
              << ", Q30=" << theta0 << ", Q60=" << theta1 << ", max=" << srt.back() << "\n";

    // 2) 边分流 + 邻接矩阵
    Mat A(num_seg, num_seg);
    std::vector<double> edges;
    std::vector<int> etype;
    int dropped = 0;
    for (int t = 0; t < ncand; ++t) {
        int i = (int)rec6[(long long)t * 6 + 0], j = (int)rec6[(long long)t * 6 + 1];
        double m = rec6[(long long)t * 6 + 5];
        if (m < theta0) { ++dropped; continue; }
        for (int q = 0; q < 6; ++q) edges.push_back(rec6[(long long)t * 6 + q]);
        etype.push_back(m >= theta1 ? 1 : 0);
        A.at(i, j) = m; A.at(j, i) = m;
    }
    int E = etype.size();
    std::vector<int> degree(num_seg, 0);
    std::vector<double> mean_w(num_seg, 0.0);
    for (int i = 0; i < num_seg; ++i) {
        double s = 0; int d = 0;
        for (int j = 0; j < num_seg; ++j) if (A.get(i, j) > 0) { ++d; s += A.get(i, j); }
        degree[i] = d; mean_w[i] = d > 0 ? s / d : 0.0;
    }

    sub_banner("建图结果");
    int nwarn = 0; for (int x : etype) if (x == 1) ++nwarn;
    std::cout << "  候选对 " << ncand << " 个 -> 丢弃(m<θ0) " << dropped
              << " 个，保留边 " << E << " 条\n";
    std::cout << "  其中候选边 " << E - nwarn << " 条，告警边 " << nwarn << " 条\n";
    int niso = 0; for (int d : degree) if (d == 0) ++niso;
    std::cout << "  节点数（段）= " << num_seg << "，孤立节点（度=0）= " << niso << "\n";

    sub_banner("按真实类别看平均度/均权（验证团簇结构）");
    std::cout << "类别              节点数    平均度  平均邻边均权   孤立占比\n";
    for (int cid = 0; cid < 6; ++cid) {
        long long cnt = 0; double sd = 0, sw = 0; int iso = 0;
        for (int g = 0; g < num_seg; ++g) if (ch->i[g] == cid) {
            ++cnt; sd += degree[g]; sw += mean_w[g]; if (degree[g] == 0) ++iso;
        }
        if (cnt == 0) {
            std::printf("  %-14s %6lld %26s\n", CHANNEL_NAMES[cid], cnt, "（无样本）");
            continue;
        }
        std::printf("  %-14s %6lld %9.2f %13.3f %9.1f%%\n",
                    CHANNEL_NAMES[cid], cnt, sd / cnt, sw / cnt, iso / (double)cnt * 100);
    }
    std::cout << "\n节点特征 X（前 6 个节点，列为 [度, 邻边均权]）:\n";
    for (int g = 0; g < 6; ++g)
        std::printf("  段%3d (channel=%-10s): 度=%3d, 均权=%.4f\n",
                    g, CHANNEL_NAMES[ch->i[g]], degree[g], mean_w[g]);

    std::vector<double> X((long long)num_seg * 2);
    for (int g = 0; g < num_seg; ++g) {
        X[g * 2] = degree[g]; X[g * 2 + 1] = mean_w[g];
    }
    Store st;
    st.put_d("theta", std::vector<int>{2}, std::vector<double>{theta0, theta1});
    st.put_d("edges", std::vector<int>{E, 6}, edges);
    st.put_i("edge_type", std::vector<int>{E}, etype);
    st.put_d("X", std::vector<int>{num_seg, 2}, X);
    st.put_d("A", std::vector<int>{num_seg, num_seg}, A.a);
    st.put_i("seg_label", std::vector<int>{num_seg}, lab->i);
    st.put_i("seg_channel", std::vector<int>{num_seg}, ch->i);
    st.put_i("seg_conn_id", std::vector<int>{num_seg}, fp.find("seg_conn_id")->i);
    st.put_i("seg_index", std::vector<int>{num_seg}, fp.find("seg_index")->i);
    std::string pth = out_path("09_graph.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
