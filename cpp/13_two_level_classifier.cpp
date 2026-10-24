// 13_two_level_classifier.cpp —— 两级阈值级联分类器（公式 11）
// 对应 Python 13_two_level_classifier.py。纯标准库实现：
//   (0) 孤立硬规则：ρ=0（大倍率编码）直接判隐蔽；
//   (1) 第一级 GBDT（召回导向）τ1；(2) 第二级 GBDT（精度导向）τ2。
//   GBDT = 二分类 logistic 梯度提升 + 深度受限回归树（等价 sklearn GradientBoosting）。
//   连接级 OOF：fold = conn_id % 5（与 sklearn GroupKFold(5) 同思想、确定性等价）。
#include "common.hpp"

static const int RHO = 2;
static const double ISO_EPS = 1e-6;
static const int N_FOLD = 5;

struct GBDTConfig { int n_est, max_depth, min_leaf; double lr; };
struct TNode {
    bool leaf; int feature; double threshold; int left, right; double value;
    TNode() : leaf(false), feature(-1), threshold(0), left(-1), right(-1), value(0) {}
};

class GBDT {
public:
    GBDTConfig cfg;
    std::vector<std::vector<TNode> > trees;
    double F0;
    double imp[4];
    GBDT() { F0 = 0; for (int q = 0; q < 4; ++q) imp[q] = 0; }

    int build(const Mat& X, const std::vector<int>& idx, const std::vector<double>& g,
              const std::vector<double>& p, int depth, std::vector<TNode>& t) {
        int n = (int)idx.size();
        double sumg = 0, sumd = 0, sumg2 = 0;
        for (int i = 0; i < n; ++i) {
            double gg = g[idx[i]]; double pp = p[idx[i]];
            sumg += gg; sumg2 += gg * gg; sumd += pp * (1.0 - pp);
        }
        TNode node;
        if (depth == cfg.max_depth || n < 2 * cfg.min_leaf) {
            node.leaf = true; node.value = sumg / (sumd + 1e-12);
            t.push_back(node); return (int)t.size() - 1;
        }
        double parent_sse = sumg2 - sumg * sumg / n;
        double best_gain = 1e-12; int bf = -1; double bthr = 0;
        std::vector<int> bL, bR;
        for (int f = 0; f < X.c; ++f) {
            std::vector<std::pair<double, int> > order;
            order.reserve(n);
            for (int i = 0; i < n; ++i) order.push_back(std::make_pair(X.get(idx[i], f), idx[i]));
            std::sort(order.begin(), order.end());
            double cumg = 0, cumg2 = 0;
            for (int k = 1; k < n; ++k) {
                double gv = g[order[k - 1].second]; cumg += gv; cumg2 += gv * gv;
                if (k < cfg.min_leaf || n - k < cfg.min_leaf) continue;
                if (order[k - 1].first == order[k].first) continue;
                double lg = cumg, lg2 = cumg2, rg = sumg - cumg, rg2 = sumg2 - cumg2;
                double sse = (lg2 - lg * lg / k) + (rg2 - rg * rg / (n - k));
                double gain = parent_sse - sse;
                if (gain > best_gain) {
                    best_gain = gain; bf = f;
                    bthr = (order[k - 1].first + order[k].first) / 2.0;
                }
            }
        }
        if (bf < 0) {
            node.leaf = true; node.value = sumg / (sumd + 1e-12);
            t.push_back(node); return (int)t.size() - 1;
        }
        imp[bf] += best_gain;
        for (int i = 0; i < n; ++i) {
            if (X.get(idx[i], bf) <= bthr) bL.push_back(idx[i]); else bR.push_back(idx[i]);
        }
        node.feature = bf; node.threshold = bthr;
        t.push_back(node);
        int id = (int)t.size() - 1;
        int lc = build(X, bL, g, p, depth + 1, t);
        int rc = build(X, bR, g, p, depth + 1, t);
        t[id].left = lc; t[id].right = rc;
        return id;
    }

    void fit(const Mat& X, const std::vector<int>& y, const GBDTConfig& c) {
        cfg = c;
        int n = X.r;
        for (int q = 0; q < 4; ++q) imp[q] = 0;
        trees.clear();
        int pos = 0; for (int i = 0; i < n; ++i) pos += y[i];
        int neg = n - pos;
        if (pos == 0) F0 = -10.0; else if (neg == 0) F0 = 10.0;
        else F0 = std::log((double)pos / neg);
        std::vector<double> F(n, F0), p(n), g(n);
        std::vector<int> all(n);
        for (int i = 0; i < n; ++i) all[i] = i;
        for (int m = 0; m < cfg.n_est; ++m) {
            for (int i = 0; i < n; ++i) { p[i] = sigmoid(F[i]); g[i] = y[i] - p[i]; }
            std::vector<TNode> t;
            build(X, all, g, p, 0, t);
            trees.push_back(t);
            for (int i = 0; i < n; ++i) {
                int node = 0;
                while (!t[node].leaf)
                    node = (X.get(i, t[node].feature) <= t[node].threshold) ? t[node].left : t[node].right;
                F[i] += cfg.lr * t[node].value;
            }
        }
    }
    double predict_one(const Mat& X, int row) const {
        double F = F0;
        for (size_t m = 0; m < trees.size(); ++m) {
            const std::vector<TNode>& t = trees[m];
            int node = 0;
            while (!t[node].leaf)
                node = (X.get(row, t[node].feature) <= t[node].threshold) ? t[node].left : t[node].right;
            F += cfg.lr * t[node].value;
        }
        return sigmoid(F);
    }
    std::vector<double> predict(const Mat& X) const {
        std::vector<double> o(X.r);
        for (int i = 0; i < X.r; ++i) o[i] = predict_one(X, i);
        return o;
    }
};

// 连接级 out-of-fold 概率（fold = conn % 5）
static std::vector<double> oof_predict(const std::vector<int>& idx, const Mat& Fall,
                                       const std::vector<int>& yall, const std::vector<int>& conn,
                                       const GBDTConfig& cfg) {
    int K = (int)idx.size();
    std::vector<double> oof(K, 0.5);
    for (int f = 0; f < N_FOLD; ++f) {
        std::vector<int> tr, va;
        for (int k = 0; k < K; ++k)
            (conn[idx[k]] % N_FOLD == f ? va : tr).push_back(k);
        if (tr.empty() || va.empty()) continue;
        int npos = 0;
        for (size_t k = 0; k < tr.size(); ++k) npos += yall[idx[tr[k]]];
        Mat Xtr((int)tr.size(), 4);
        std::vector<int> ytr;
        for (size_t k = 0; k < tr.size(); ++k) {
            for (int q = 0; q < 4; ++q) Xtr.at((int)k, q) = Fall.get(idx[tr[k]], q);
            ytr.push_back(yall[idx[tr[k]]]);
        }
        if (npos == 0 || npos == (int)tr.size()) {
            double cv = npos == 0 ? 0.0 : 1.0;
            for (size_t k = 0; k < va.size(); ++k) oof[va[k]] = cv;
            continue;
        }
        GBDT model; model.fit(Xtr, ytr, cfg);
        for (size_t k = 0; k < va.size(); ++k) {
            Mat row(1, 4);
            for (int q = 0; q < 4; ++q) row.at(0, q) = Fall.get(idx[va[k]], q);
            oof[va[k]] = model.predict_one(row, 0);
        }
    }
    return oof;
}

int main() {
    init_console();
    Store d = Store::load(out_path("12_features.bin"));
    const BinArc* feat = d.find("feat");
    const BinArc* lab = d.find("seg_label");
    const BinArc* chan = d.find("seg_channel");
    const BinArc* conn = d.find("seg_conn_id");
    int num_seg = feat->shape[0];
    Mat F(num_seg, 4);
    F.a = feat->d;
    std::vector<int> y(lab->i), chh(chan->i), cn(conn->i);

    std::vector<int> test_idx, clique_idx;
    std::vector<int> iso_te_all(num_seg, 0);
    for (int g = 0; g < num_seg; ++g) {
        bool test = (cn[g] % 3 == 0);
        bool iso = (feat->d[(long long)g * 4 + RHO] < ISO_EPS);
        iso_te_all[g] = iso ? 1 : 0;
        if (test) test_idx.push_back(g);
        else if (!iso) clique_idx.push_back(g);
    }
    int n_iso_tr = 0;
    for (int g = 0; g < num_seg; ++g) if (cn[g] % 3 != 0 && iso_te_all[g]) ++n_iso_tr;
    int n_iso_te = 0;
    for (size_t k = 0; k < test_idx.size(); ++k) n_iso_te += iso_te_all[test_idx[k]];

    banner("13 两级阈值级联分类器（公式 11）");
    std::cout << "连接级划分：训练段 " << (num_seg - (int)test_idx.size())
              << "（其中孤立段 " << n_iso_tr << " 由拓扑规则直接判隐蔽，团内 "
              << clique_idx.size() << " 段训练 GBDT），测试段 " << test_idx.size() << "\n";
    std::cout << "测试段中孤立节点 " << n_iso_te << " 个 → 直接判隐蔽（大倍率编码）\n";

    GBDTConfig c1 = {150, 2, 3, 0.05};
    GBDTConfig c2 = {40, 2, 8, 0.10};

    // 第一级 OOF
    std::vector<int> ytr_c(clique_idx.size());
    for (size_t k = 0; k < clique_idx.size(); ++k) ytr_c[k] = y[clique_idx[k]];
    std::vector<double> p1_oof = oof_predict(clique_idx, F, y, cn, c1);
    GBDT first;
    {
        Mat Xtr((int)clique_idx.size(), 4);
        for (size_t k = 0; k < clique_idx.size(); ++k)
            for (int q = 0; q < 4; ++q) Xtr.at((int)k, q) = F.get(clique_idx[k], q);
        first.fit(Xtr, ytr_c, c1);
    }

    sub_banner("团内第一级 OOF 概率按类别（min/中位/max）");
    std::cout << "类别              n     min     中位     max\n";
    for (int cid = 0; cid < 6; ++cid) {
        std::vector<double> v;
        for (size_t k = 0; k < clique_idx.size(); ++k)
            if (chh[clique_idx[k]] == cid) v.push_back(p1_oof[k]);
        if (v.empty()) continue;
        std::sort(v.begin(), v.end());
        std::printf("  %-14s %4d %8.3f %8.3f %8.3f\n", CHANNEL_NAMES[cid], (int)v.size(),
                    v.front(), median_vec(v), v.back());
    }
    std::vector<double> p1_pos;
    for (size_t k = 0; k < clique_idx.size(); ++k) if (ytr_c[k] == 1) p1_pos.push_back(p1_oof[k]);
    double tau1 = quantile_vec(p1_pos, 0.01);
    std::printf("\n第一级 τ1 = %.4f（团内正类 OOF 概率 1%% 分位，召回导向）\n", tau1);

    // 第二级：候选（p1_oof>=tau1）
    std::vector<int> cand_idx;
    for (size_t k = 0; k < clique_idx.size(); ++k)
        if (p1_oof[k] >= tau1) cand_idx.push_back(clique_idx[k]);
    std::vector<int> ytr_cand(cand_idx.size());
    int npos = 0;
    for (size_t k = 0; k < cand_idx.size(); ++k) { ytr_cand[k] = y[cand_idx[k]]; npos += y[cand_idx[k]]; }
    double tau2 = tau1;
    std::vector<double> p2_oof;
    GBDT second;
    bool two_class = npos > 0 && npos < (int)cand_idx.size();
    if (two_class) {
        p2_oof = oof_predict(cand_idx, F, y, cn, c2);
        Mat Xtr((int)cand_idx.size(), 4);
        for (size_t k = 0; k < cand_idx.size(); ++k)
            for (int q = 0; q < 4; ++q) Xtr.at((int)k, q) = F.get(cand_idx[k], q);
        second.fit(Xtr, ytr_cand, c2);
        std::vector<double> p2_legit, p2_pos;
        for (size_t k = 0; k < cand_idx.size(); ++k)
            (ytr_cand[k] == 0 ? p2_legit : p2_pos).push_back(p2_oof[k]);
        std::sort(p2_legit.begin(), p2_legit.end());
        tau2 = quantile_vec(p2_legit, 0.99);
        std::printf("第二级 τ2 = %.4f（候选中合法 OOF 概率 99%% 分位，精度导向）\n", tau2);
        std::printf("    候选合法 OOF pro2 中位 %.3f/最大 %.3f；候选隐蔽 OOF pro2 中位 %.3f\n",
                    median_vec(p2_legit), p2_legit.back(), median_vec(p2_pos));
    } else {
        std::cout << "（候选仅含单类，第二级退化为复用第一级）\n";
        second = first;
    }

    // ---- 测试级联 ----
    int T = (int)test_idx.size();
    std::vector<double> pro1(T), pro2(T, -1.0);
    std::vector<int> final(T), iso_t(T);
    int n_direct_normal = 0, n_cand = 0, n_cand_pos = 0;
    for (int t = 0; t < T; ++t) {
        int g = test_idx[t];
        Mat row(1, 4);
        for (int q = 0; q < 4; ++q) row.at(0, q) = F.get(g, q);
        pro1[t] = first.predict_one(row, 0);
        bool iso = iso_te_all[g];
        iso_t[t] = iso ? 1 : 0;
        if (iso) { final[t] = 1; }
        else if (pro1[t] < tau1) { final[t] = 0; ++n_direct_normal; }
        else {
            ++n_cand;
            if (two_class) {
                double p2v = second.predict_one(row, 0);
                pro2[t] = p2v;
                final[t] = (p2v >= tau2) ? 1 : 0;
            } else {
                final[t] = 1;
            }
            if (final[t] == 1) ++n_cand_pos;
        }
    }

    sub_banner("测试级联过程");
    std::cout << "  测试段总数 " << T << "：孤立判隐蔽 " << n_iso_te << " 段\n";
    std::cout << "  团内第一级 pro1<τ1 直接判正常：" << n_direct_normal << " 段\n";
    std::cout << "  团内进入第二级候选：" << n_cand << " 段，判隐蔽 " << n_cand_pos << " 段\n";

    sub_banner("测试段判定示例（前 12 个）");
    std::cout << "   段   真实    pro1    pro2   最终\n";
    for (int t = 0; t < std::min(12, T); ++t) {
        char buf[32];
        if (iso_t[t]) std::snprintf(buf, sizeof(buf), "孤立");
        else if (pro2[t] < 0) std::snprintf(buf, sizeof(buf), "—");
        else std::snprintf(buf, sizeof(buf), "%.3f", pro2[t]);
        std::printf("%5d %5d %8.3f %8s %5d\n",
                    test_idx[t], y[test_idx[t]], pro1[t], buf, final[t]);
    }

    // 保存
    std::vector<int> test_y(T), test_channel(T);
    for (int t = 0; t < T; ++t) { test_y[t] = y[test_idx[t]]; test_channel[t] = chh[test_idx[t]]; }
    std::vector<double> first_imp(4);
    double impsum = 0; for (int q = 0; q < 4; ++q) impsum += first.imp[q];
    for (int q = 0; q < 4; ++q) first_imp[q] = impsum > 0 ? first.imp[q] / impsum * 100.0 : 0;
    Store st;
    st.put_i("test_idx", std::vector<int>{T}, test_idx);
    st.put_i("test_y", std::vector<int>{T}, test_y);
    st.put_i("test_channel", std::vector<int>{T}, test_channel);
    st.put_d("pro1", std::vector<int>{T}, pro1);
    st.put_d("pro2", std::vector<int>{T}, pro2);
    st.put_i("final", std::vector<int>{T}, final);
    st.put_i("iso", std::vector<int>{T}, iso_t);
    st.put_d("tau", std::vector<int>{2}, std::vector<double>{tau1, tau2});
    st.put_d("first_imp", std::vector<int>{4}, first_imp);
    std::string pth = out_path("13_results.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
