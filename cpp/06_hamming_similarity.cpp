// 06_hamming_similarity.cpp —— 汉明距离与窗口级相似度（公式 3）
// 对应 Python 06_hamming_similarity.py。打包字节异或 + popcount。
#include "common.hpp"

int main() {
    init_console();
    Store data = Store::load(out_path("05_fingerprints.bin"));
    const BinArc* H32 = data.find("H32");
    const BinArc* ch = data.find("seg_channel");
    int num_seg = H32->shape[0], nw = H32->shape[1];

    // 找前两个合法段、第一个 IPCTC 段
    int li = -1, lj = -1, ip = -1;
    for (int g = 0; g < num_seg; ++g) {
        if (ch->i[g] == 0) { if (li < 0) li = g; else if (lj < 0) lj = g; }
        if (ch->i[g] == 1 && ip < 0) ip = g;
    }

    banner("06 汉明距离与窗口级相似度（公式 3）");
    const unsigned char* a = &H32->b[(long long)li * nw * FP_BYTES];
    const unsigned char* a1 = &H32->b[(long long)li * nw * FP_BYTES + FP_BYTES];
    std::cout << "同一窗口指纹自身比较：Ham=" << hamming_bits(a, a)
              << ", sim=1.0000\n";
    std::cout << "同段相邻窗口 0 vs 1：Ham=" << hamming_bits(a, a1)
              << ", sim=" << hamming_sim(a, a1) << "\n";

    auto win_sims = [&](int p, int q, std::vector<double>& out) {
        out.resize(nw);
        for (int wi = 0; wi < nw; ++wi)
            out[wi] = hamming_sim(&H32->b[(long long)p * nw * FP_BYTES + (long long)wi * FP_BYTES],
                                  &H32->b[(long long)q * nw * FP_BYTES + (long long)wi * FP_BYTES]);
    };
    sub_banner("win=32 下，段对的对应窗口相似度（前 8 个）");
    std::vector<double> ll, lc, self;
    win_sims(li, lj, ll); win_sims(li, ip, lc); win_sims(li, li, self);
    for (int t = 0; t < 2; ++t) {
        std::cout << "  " << (t == 0 ? "合法-合法 " : "合法-IPCTC ") << "前8个: ";
        std::vector<double>& v = t == 0 ? ll : lc;
        for (int wi = 0; wi < 8; ++wi) std::printf("%.3f ", v[wi]);
        std::cout << " 均值=" << mean_vec(v) << "\n";
    }
    std::cout << "  段自身      均值=" << mean_vec(self) << "（应=1）\n";

    // 手算：把 6 位翻成 1
    int bits[FP_BITS];
    for (int t = 0; t < FP_BITS; ++t) bits[t] = 0;
    int flips[6] = {28, 62, 67, 74, 99, 102};
    for (int t = 0; t < 6; ++t) bits[flips[t]] = 1;
    unsigned char zero[FP_BYTES] = {0}, other[FP_BYTES];
    pack_bits(bits, other);
    int d6 = hamming_bits(zero, other);
    std::cout << "\n手算玩具：人为把 6 位翻成 1 -> Ham=" << d6
              << ", sim=1−" << d6 << "/128=" << 1.0 - d6 / 128.0 << "\n";

    Store st;
    st.put_d("ll", std::vector<int>{nw}, ll);
    st.put_d("lc", std::vector<int>{nw}, lc);
    std::string pth = out_path("06_pairwise_demo.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
