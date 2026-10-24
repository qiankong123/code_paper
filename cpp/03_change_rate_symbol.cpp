// 03_change_rate_symbol.cpp —— 相邻变化率 Δ 与七级符号映射（Table 1）
// 对应 Python 03_change_rate_symbol.py。
#include "common.hpp"

static inline int map_symbol(double d) {
    if (d >= 0.15) return 3;
    if (d >= 0.06) return 2;
    if (d >= 0.02) return 1;
    if (d > -0.02) return 0;
    if (d > -0.06) return -1;
    if (d > -0.15) return -2;
    return -3;
}

int main() {
    init_console();
    Store data = Store::load(out_path("02_segments.bin"));
    const BinArc* sipd = data.find("seg_ipd");
    const BinArc* ch = data.find("seg_channel");
    int num_seg = sipd->shape[0];

    banner("03 相邻变化率 Δ 与七级符号映射（Table 1）");
    // 玩具手算
    double toy[6] = {100, 103, 103.2, 120, 95, 100};
    std::cout << "玩具 IPD: 100 103 103.2 120 95 100\n  Δ  : ";
    for (int t = 0; t < 5; ++t)
        std::printf("%+.3f ", toy[t + 1] / toy[t] - 1.0);
    std::cout << "\n  符号: ";
    for (int t = 0; t < 5; ++t)
        std::printf("%+d ", map_symbol(toy[t + 1] / toy[t] - 1.0));
    std::cout << "\n  （例：120/103.2−1=+0.163 → +3；95/120−1=−0.208 → −3）\n";

    int delta_len = SEG_LEN - 1;   // 1999
    std::vector<double> delta((long long)num_seg * delta_len);
    std::vector<int> symbols((long long)num_seg * delta_len);
    for (int g = 0; g < num_seg; ++g) {
        for (int t = 0; t < delta_len; ++t) {
            double a = sipd->d[(long long)g * SEG_LEN + t];
            double b = sipd->d[(long long)g * SEG_LEN + t + 1];
            double dlt = (b - a) / a;
            delta[(long long)g * delta_len + t] = dlt;
            symbols[(long long)g * delta_len + t] = map_symbol(dlt);
        }
    }

    sub_banner("各信道符号分布（−3…+3 占比 %）——合法分散、编码集中");
    std::cout << "信道             -3     -2     -1      0     +1     +2     +3\n";
    for (int cid = 0; cid < 6; ++cid) {
        long long cnt[7] = {0}; long long tot = 0;
        for (int g = 0; g < num_seg; ++g) if (ch->i[g] == cid) {
            for (int t = 0; t < delta_len; ++t)
                ++cnt[symbols[(long long)g * delta_len + t] + 3];
            tot += delta_len;
        }
        std::cout << "  " << CHANNEL_NAMES[cid]
                  << std::string(14 - std::string(CHANNEL_NAMES[cid]).size(), ' ');
        for (int q = 0; q < 7; ++q) std::printf("%7.1f", cnt[q] / (double)tot * 100.0);
        std::cout << "\n";
    }

    std::cout << "\n输出形状：seg_delta=(" << num_seg << "," << delta_len
              << "), seg_symbols=(" << num_seg << "," << delta_len << ")\n";
    Store st;
    st.put_d("seg_delta", std::vector<int>{num_seg, delta_len}, delta);
    st.put_i("seg_symbols", std::vector<int>{num_seg, delta_len}, symbols);
    st.put_i("seg_label", std::vector<int>{num_seg}, data.find("seg_label")->i);
    st.put_i("seg_channel", std::vector<int>{num_seg}, ch->i);
    st.put_i("seg_conn_id", std::vector<int>{num_seg}, data.find("seg_conn_id")->i);
    st.put_i("seg_index", std::vector<int>{num_seg}, data.find("seg_index")->i);
    std::string pth = out_path("03_symbols.bin");
    st.save(pth);
    std::cout << "已保存：" << pth << "\n";
    return 0;
}
