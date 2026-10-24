// 12_graph_features.cpp —— 每段 4 维检测特征 H_sym / L_cy / ρ / √S_flow（公式 7–10）
// 对应 Python 12_graph_features.py。
#include "common.hpp"
#include <unordered_map>

static const int TOP_K = 5;

static double symbol_entropy(const std::vector<int>& sym, int len) {
    long long cnt[7] = {0};
    for (int t = 0; t < len; ++t) ++cnt[sym[t] + 3];
    double H = 0, tot = (double)len;
    for (int q = 0; q < 7; ++q) if (cnt[q] > 0) {
        double p = cnt[q] / tot;
        H -= p * std::log(p);
    }
    return H;
}
static double cycle_mean(const std::vector<int>& sym, int len) {
    std::vector<int> pos, neg;
    for (int t = 0; t < len; ++t) {
        if (sym[t] > 0) pos.push_back(t);
        else if (sym[t] < 0) neg.push_back(t);
    }
    double U = 0, D = 0;
    if (pos.size() > 1) {
        double s = 0; for (size_t k = 1; k < pos.size(); ++k) s += pos[k] - pos[k - 1];
        U = s / (pos.size() - 1);
    }
    if (neg.size() > 1) {
        double s = 0; for (size_t k = 1; k < neg.size(); ++k) s += neg[k] - neg[k - 1];
        D = s / (neg.size() - 1);
    }
    return (U + D) / 2.0;
}

int main() {
    init_console();
    Store sym_st = Store::load(out_path("03_symbols.bin"));
    const BinArc* sym = sym_st.find("seg_symbols");
    const BinArc* ch = sym_st.find("seg_channel");
    int num_seg = sym->shape[0], slen = sym->shape[1];
    Store fst = Store::load(out_path("11_fusion.bin"));
    const BinArc* edges = fst.find("edges");
    const BinArc* sarc = fst.find("s");
    int E = edges->shape[0];

    std::vector<double> H(num_seg), L(num_seg);
    std::vector<int> oneseg(slen);
    for (int g = 0; g < num_seg; ++g) {
        for (int t = 0; t < slen; ++t) oneseg[t] = sym->i[(long long)g * slen + t];
        H[g] = symbol_entropy(oneseg, slen);
        L[g] = cycle_mean(oneseg, slen);
    }

    // 图 top-k 特征
    std::unordered_map<int, std::vector<double> > acc;
    for (int e = 0; e < E; ++e) {
        int a = (int)edges->d[(long long)e * 6 + 0], b = (int)edges->d[(long long)e * 6 + 1];
        double sv = sarc->d[e];
        acc[a].push_back(sv); acc[b].push_back(sv);
    }
    std::vector<double> rho(num_seg, 0), sflow(num_seg, 0);
    for (int g = 0; g < num_seg; ++g) {
        std::unordered_map<int, std::vector<double> >::iterator it = acc.find(g);
        if (it == acc.end() || it->second.empty()) { rho[g] = 0; sflow[g] = 0; continue; }
        std::vector<double> v = it->second;
        std::sort(v.begin(), v.end());
        int k = std::min(TOP_K, (int)v.size());
        double sum = 0;
        for (int t = (int)v.size() - k; t < (int)v.size(); ++t) sum += v[t];
        rho[g] = std::log(1.0 + sum);
        sflow[g] = std::sqrt(sum / TOP_K);
    }

    banner("12 每段 4 维特征：H_sym / L_cy / ρ / √S_flow");
    sub_banner("各类别特征均值（看区分度；top-k 中 k=5）");
    std::cout << "类别             H_sym熵   L_cy节奏   ρ累加   √S_flow\n";
    for (int cid = 0; cid < 6; ++cid) {
        double a = 0, b = 0, c = 0, dd = 0; long long cnt = 0;
        for (int g = 0; g < num_seg; ++g) if (ch->i[g] == cid) {
            a += H[g]; b += L[g]; c += rho[g]; dd += sflow[g]; ++cnt;
        }
        std::printf("  %-14s %9.3f %10.3f %8.3f %9.3f\n",
                    CHANNEL_NAMES[cid], a / cnt, b / cnt, c / cnt, dd / cnt);
    }
    std::cout << "\n要点（本合成数据）：\n"
              << "  · 大倍率编码 IPCTC/LNCTC/TRCTC：H_sym 高、L_cy 小，且 ρ/√S_flow=0（孤立）；\n"
              << "  · 合法/jitter/ε 都在大团内；jitter 靠 L_cy 偏小、H_sym 略高识别；\n"
              << "  · ε 的 H_sym/L_cy 与合法几乎无差异（ε-不可区分，最隐蔽）。\n\n";
    std::cout << "前 5 个段的特征向量:\n";
    for (int g = 0; g < 5; ++g)
        std::printf("  段%d: H_sym=%.3f, L_cy=%.2f, ρ=%.3f, √S_flow=%.3f\n",
                    g, H[g], L[g], rho[g], sflow[g]);

    std::vector<double> feat((long long)num_seg * 4);
    for (int g = 0; g < num_seg; ++g) {
        feat[g * 4 + 0] = H[g]; feat[g * 4 + 1] = L[g];
        feat[g * 4 + 2] = rho[g]; feat[g * 4 + 3] = sflow[g];
    }
    Store st;
    st.put_d("feat", std::vector<int>{num_seg, 4}, feat);
    st.put_i("seg_label", std::vector<int>{num_seg}, fst.find("seg_label")->i);
    st.put_i("seg_channel", std::vector<int>{num_seg}, ch->i);
    st.put_i("seg_conn_id", std::vector<int>{num_seg}, fst.find("seg_conn_id")->i);
    st.put_i("seg_index", std::vector<int>{num_seg}, fst.find("seg_index")->i);
    std::string pth = out_path("12_features.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "（feat 形状 (" << num_seg << ",4)）\n";
    return 0;
}
