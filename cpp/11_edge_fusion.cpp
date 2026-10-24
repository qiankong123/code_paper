// 11_edge_fusion.cpp —— 边强度融合 s = α1w1 + α2w2 + α3w3 + β·φ（公式 6）
// 对应 Python 11_edge_fusion.py。教学网格搜索：最大化合法-合法边与合法-隐蔽边的分离度。
#include "common.hpp"

int main() {
    init_console();
    Store g = Store::load(out_path("10_gnn.bin"));
    const BinArc* edges = g.find("edges");
    const BinArc* phi = g.find("phi");
    const BinArc* lab = g.find("seg_label");
    int E = edges->shape[0];

    std::vector<double> w1(E), w2(E), w3(E), pv(E);
    std::vector<int> u(E), v(E), yl(E);
    for (int e = 0; e < E; ++e) {
        u[e] = (int)edges->d[(long long)e * 6 + 0];
        v[e] = (int)edges->d[(long long)e * 6 + 1];
        w1[e] = edges->d[(long long)e * 6 + 2];
        w2[e] = edges->d[(long long)e * 6 + 3];
        w3[e] = edges->d[(long long)e * 6 + 4];
        pv[e] = phi->d[e];
        yl[e] = lab->i[u[e]];   // 用段标签（下面取两端）
    }
    auto isll = [&](int e) { return lab->i[u[e]] == 0 && lab->i[v[e]] == 0; };
    auto iscross = [&](int e) { return lab->i[u[e]] != lab->i[v[e]]; };
    auto isbb = [&](int e) { return lab->i[u[e]] == 1 && lab->i[v[e]] == 1; };
    double gr[6] = {0, 0.2, 0.4, 0.6, 0.8, 1.0};
    double betas[5] = {0, 0.25, 0.5, 1.0, 2.0};

    banner("11 边强度融合  s = α1w1 + α2w2 + α3w3 + β·φ（公式6）");
    double best_sep = -1e9, ba1 = 0, ba2 = 0, ba3 = 1, bb = 0;
    for (int x = 0; x < 6; ++x) for (int y = 0; y < 6; ++y) {
        double a1 = gr[x], a2 = gr[y], a3 = 1.0 - a1 - a2;
        if (a3 < -1e-9) continue;
        for (int z = 0; z < 5; ++z) {
            double beta = betas[z];
            double sl = 0, sc = 0; int nl = 0, nc = 0;
            for (int e = 0; e < E; ++e) {
                double s = a1 * w1[e] + a2 * w2[e] + a3 * w3[e] + beta * pv[e];
                if (isll(e)) { sl += s; ++nl; }
                if (iscross(e)) { sc += s; ++nc; }
            }
            double sep = (nl > 0 && nc > 0) ? (sl / nl - sc / nc) : 0.0;
            if (sep > best_sep) { best_sep = sep; ba1 = a1; ba2 = a2; ba3 = a3; bb = beta; }
        }
    }
    std::vector<double> s(E);
    for (int e = 0; e < E; ++e) s[e] = ba1 * w1[e] + ba2 * w2[e] + ba3 * w3[e] + bb * pv[e];
    std::printf("网格搜索最优：α = [%.1f, %.1f, %.1f]（sum=1），β = %.2f\n", ba1, ba2, ba3, bb);
    std::printf("两类边分离度（合法-合法均值 − 合法-隐蔽均值）= %.4f\n", best_sep);

    sub_banner("融合前后对比（边强度均值）");
    std::cout << "边类型           边数  w1均值  w3均值    φ均值   s均值\n";
    {
        double a=0,c=0,pp=0,ss=0; int cnt=0;
        for (int e=0;e<E;++e) if (isll(e)){++cnt;a+=w1[e];c+=w3[e];pp+=pv[e];ss+=s[e];}
        std::printf("  %-14s %5d %8.3f %8.3f %8.3f %8.3f\n","合法-合法",cnt,a/cnt,c/cnt,pp/cnt,ss/cnt);
    }
    {
        double a=0,c=0,pp=0,ss=0; int cnt=0;
        for (int e=0;e<E;++e) if (iscross(e)){++cnt;a+=w1[e];c+=w3[e];pp+=pv[e];ss+=s[e];}
        std::printf("  %-14s %5d %8.3f %8.3f %8.3f %8.3f\n","合法-隐蔽",cnt,a/cnt,c/cnt,pp/cnt,ss/cnt);
    }
    {
        double a=0,c=0,pp=0,ss=0; int cnt=0;
        for (int e=0;e<E;++e) if (isbb(e)){++cnt;a+=w1[e];c+=w3[e];pp+=pv[e];ss+=s[e];}
        std::printf("  %-14s %5d %8.3f %8.3f %8.3f %8.3f\n","隐蔽-隐蔽",cnt,a/cnt,c/cnt,pp/cnt,ss/cnt);
    }
    std::cout << "\n说明：合法-合法边强度高；合法-隐蔽边弱（跨不进大团）；\n"
              << "      隐蔽-隐蔽若形成异常团，在 12/13 靠段自身特征识别。\n";

    Store st;
    st.put_d("s", std::vector<int>{E}, s);
    st.put_d("alpha", std::vector<int>{3}, std::vector<double>{ba1, ba2, ba3});
    st.put_d("beta", std::vector<int>{1}, std::vector<double>(1, bb));
    st.put_d("edges", std::vector<int>{E, 6}, edges->d);
    st.put_i("edge_type", std::vector<int>{E}, g.find("edge_type")->i);
    st.put_d("phi", std::vector<int>{E}, phi->d);
    st.put_d("X", std::vector<int>{g.find("X")->shape[0], 2}, g.find("X")->d);
    st.put_d("A", std::vector<int>{g.find("A")->shape[0], g.find("A")->shape[1]}, g.find("A")->d);
    st.put_d("theta", std::vector<int>{2}, g.find("theta")->d);
    st.put_i("seg_label", std::vector<int>{g.find("seg_label")->shape[0]}, g.find("seg_label")->i);
    st.put_i("seg_channel", std::vector<int>{g.find("seg_channel")->shape[0]}, g.find("seg_channel")->i);
    st.put_i("seg_conn_id", std::vector<int>{g.find("seg_conn_id")->shape[0]}, g.find("seg_conn_id")->i);
    st.put_i("seg_index", std::vector<int>{g.find("seg_index")->shape[0]}, g.find("seg_index")->i);
    std::string pth = out_path("11_fusion.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
