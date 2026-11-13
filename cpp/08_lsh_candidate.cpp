// 08_lsh_candidate.cpp —— LSH 分 band 粗筛候选对 + 召回率推导
// 对应 Python 08_lsh_candidate.py。窗口级指纹 + 位置对齐 + 多点碰撞证据。
#include "common.hpp"
#include <unordered_map>
#include <map>

static const int POS_SHIFT = 2;        // 位置容差（教学值）

static double theo_recall(double s, int b = LSH_BAND_BITS, int B = LSH_BANDS) {
    return 1.0 - std::pow(1.0 - std::pow(s, b), B);
}
static double mc_recall(double s, int n_pairs, RNG& r) {
    int hit = 0;
    for (int t = 0; t < n_pairs; ++t) {
        bool any = false;
        for (int band = 0; band < LSH_BANDS && !any; ++band) {
            bool all = true;
            for (int bit = 0; bit < LSH_BAND_BITS; ++bit)
                if (r.uniform01() >= s) { all = false; break; }
            if (all) any = true;
        }
        if (any) ++hit;
    }
    return hit / (double)n_pairs;
}

int main() {
    init_console();
    Store data = Store::load(out_path("05_fingerprints.bin"));
    const BinArc* H64 = data.find("H64");
    const BinArc* ch = data.find("seg_channel");
    const BinArc* lab = data.find("seg_label");
    int num_seg = H64->shape[0], nw = H64->shape[1];

    banner("08 LSH 位置对齐粗筛候选对（8 band × 16 位，需成片碰撞）");
    sub_banner("单点召回率 R(s)=1−(1−s^16)^8：理论 vs 蒙特卡洛(20000 对)");
    std::cout << "    相似度 s    理论召回   蒙特卡洛\n";
    RNG rmc(SEED);
    double ss[5] = {0.95, 0.90, 0.85, 0.70, 0.50};
    for (int t = 0; t < 5; ++t)
        std::printf("%12.2f %9.2f%% %9.2f%%\n", ss[t],
                    theo_recall(ss[t]) * 100, mc_recall(ss[t], 20000, rmc) * 100);

    // 桶：key = (band<<23)|(sig<<7)|pbin
    std::unordered_map<uint64_t, std::vector<int> > bucket;
    for (int g = 0; g < num_seg; ++g)
        for (int wi = 0; wi < nw; ++wi) {
            const unsigned char* h = &H64->b[((long long)g * nw + wi) * FP_BYTES];
            int lo = std::max(0, wi - POS_SHIFT), hi = std::min(nw - 1, wi + POS_SHIFT);
            for (int b = 0; b < LSH_BANDS; ++b) {
                int sig = band_signature(h, b);
                for (int pbin = lo; pbin <= hi; ++pbin) {
                    uint64_t key = ((uint64_t)b << 23) | ((uint64_t)sig << 7) | (uint64_t)pbin;
                    bucket[key].push_back(g);
                }
            }
        }
    // 段对碰撞证据
    std::unordered_map<uint64_t, int> ev;
    for (std::unordered_map<uint64_t, std::vector<int> >::iterator it = bucket.begin();
         it != bucket.end(); ++it) {
        std::vector<int>& v = it->second;
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (size_t x = 0; x < v.size(); ++x)
            for (size_t y = x + 1; y < v.size(); ++y)
                ++ev[(uint64_t)v[x] * num_seg + v[y]];
    }
    auto evof = [&](int i, int j) -> int {
        if (i > j) std::swap(i, j);
        std::unordered_map<uint64_t, int>::iterator it = ev.find((uint64_t)i * num_seg + j);
        return it == ev.end() ? 0 : it->second;
    };

    long long total_pairs = (long long)num_seg * (num_seg - 1) / 2;

    // 分组统计
    std::vector<double> g_ll, g_bb;
    std::vector<std::vector<double> > g_cross(6);
    for (int i = 0; i < num_seg; ++i)
        for (int j = i + 1; j < num_seg; ++j) {
            int e = evof(i, j);
            if (ch->i[i] == 0 && ch->i[j] == 0) g_ll.push_back(e);
            else if (ch->i[i] == 0 || ch->i[j] == 0) {
                int o = ch->i[i] == 0 ? ch->i[j] : ch->i[i];
                g_cross[o].push_back(e);
            } else g_bb.push_back(e);
        }
    int MIN_EVIDENCE = (int)quantile_vec(g_ll, 0.02);

    sub_banner("段对碰撞证据数分布（按类别）");
    std::cout << "段对类型             中位数     均值    最小    最大\n";
    auto print_row = [&](const std::string& nm, const std::vector<double>& v) {
        if (v.empty()) return;
        double mn = *std::min_element(v.begin(), v.end());
        double mx = *std::max_element(v.begin(), v.end());
        std::printf("  %-16s %8.0f %8.1f %7.0f %7.0f\n",
                    nm.c_str(), median_vec(v), mean_vec(v), mn, mx);
    };
    print_row("合法-合法", g_ll);
    for (int cid = 1; cid <= 5; ++cid)
        print_row(std::string("合法-") + CHANNEL_NAMES[cid], g_cross[cid]);
    print_row("隐蔽-隐蔽", g_bb);

    // 候选对
    std::vector<int> cand, evarr;
    for (int i = 0; i < num_seg; ++i)
        for (int j = i + 1; j < num_seg; ++j) {
            int e = evof(i, j);
            if (e >= MIN_EVIDENCE) {
                cand.push_back(i); cand.push_back(j);
                evarr.push_back(i); evarr.push_back(j); evarr.push_back(e);
            }
        }
    int ncand = (int)cand.size() / 2;
    sub_banner("候选对规模（证据数 ≥ " + std::to_string(MIN_EVIDENCE) + "，= 合法-合法证据 2% 分位）");
    std::cout << "  全配对 " << total_pairs << " 个，LSH 保留 " << ncand
              << " 个（压缩到 " << ncand / (double)total_pairs * 100 << "%）\n";

    int n_legit = 0; for (int g = 0; g < num_seg; ++g) if (ch->i[g] == 0) ++n_legit;
    auto count_pair = [&](int want) {
        int hit = 0;
        for (int t = 0; t < ncand; ++t) {
            int i = cand[2 * t], j = cand[2 * t + 1];
            if ((ch->i[i] == 0 && ch->i[j] == want) ||
                (ch->i[j] == 0 && ch->i[i] == want)) ++hit;
        }
        return hit;
    };
    int ll = count_pair(0);
    long long ll_total = (long long)n_legit * (n_legit - 1) / 2;
    std::cout << "  合法-合法对召回 = " << ll << "/" << ll_total << " = "
              << ll / (double)ll_total * 100 << "%\n";
    for (int cid = 1; cid <= 5; ++cid) {
        int nc = 0; for (int g = 0; g < num_seg; ++g) if (ch->i[g] == cid) ++nc;
        if (nc == 0) continue;                 // 本数据集无该类（真实数据缺 ε）
        long long tot = (long long)n_legit * nc;
        int hit = count_pair(cid);
        std::printf("  合法-%-14s 保留 %5d/%-5d = %5.1f%%\n",
                    CHANNEL_NAMES[cid], hit, (int)tot, hit / (double)tot * 100);
    }

    Store st;
    st.put_i("candidates", std::vector<int>{ncand, 2}, cand);
    st.put_i("evidence", std::vector<int>{(int)evarr.size() / 3, 3}, evarr);
    st.put_i("seg_channel", std::vector<int>{num_seg}, ch->i);
    st.put_i("seg_label", std::vector<int>{num_seg}, lab->i);
    std::string pth = out_path("08_candidates.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "（候选对数组 (" << ncand << ",2)）\n";
    return 0;
}
