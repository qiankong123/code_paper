// 04_multiscale_windows.cpp —— 多尺度滑窗 + 7 维计数向量 v（Table 2）
// 对应 Python 04_multiscale_windows.py。段首补一个 0，使符号长度对齐为 2000。
#include "common.hpp"

// 对长度 2000 的符号序列（int8：-3..3），在某尺度下生成 (nw,7) 计数
static std::vector<int> count_vectors(const std::vector<int>& sym2000,
                                      int win, int step, int nw) {
    std::vector<int> V((long long)nw * NUM_SYMBOLS, 0);
    for (int wi = 0; wi < nw; ++wi) {
        int start = wi * step;
        for (int t = start; t < start + win; ++t)
            ++V[(long long)wi * NUM_SYMBOLS + (sym2000[t] + 3)];
    }
    return V;
}

int main() {
    init_console();
    Store data = Store::load(out_path("03_symbols.bin"));
    const BinArc* sym = data.find("seg_symbols");
    const BinArc* ch = data.find("seg_channel");
    int num_seg = sym->shape[0], delta_len = sym->shape[1];

    banner("04 多尺度滑窗与 7 维计数向量 v");

    // 每尺度计数（段首补 0 -> 2000）
    std::vector<int> V[3];
    int nws[3];
    for (int sc = 0; sc < N_SCALES; ++sc) {
        int win = SCALE_WIN[sc], step = SCALE_STEP[sc];
        int nw = count_windows(SEG_LEN, win, step);
        nws[sc] = nw;
        V[sc].assign((long long)num_seg * nw * NUM_SYMBOLS, 0);
        std::vector<int> sym2000(SEG_LEN, 0);
        for (int g = 0; g < num_seg; ++g) {
            // 段首 0 + 1999 个符号
            for (int t = 0; t < delta_len; ++t)
                sym2000[t + 1] = sym->i[(long long)g * delta_len + t];
            std::vector<int> vg = count_vectors(sym2000, win, step, nw);
            for (long long q = 0; q < (long long)nw * NUM_SYMBOLS; ++q)
                V[sc][(long long)g * nw * NUM_SYMBOLS + q] = vg[q];
        }
        std::cout << "  win=" << win << ", step=" << step
                  << " -> 每段 " << nw << " 个窗口，每窗口计数和=" << win << "\n";
    }

    sub_banner("win=32 时第一个段的前 3 个计数向量 v（7 列：−3…+3）");
    std::cout << "  符号: -3  -2  -1   0  +1  +2  +3\n";
    for (int wi = 0; wi < 3; ++wi) {
        std::cout << "  窗口" << wi + 1 << ":";
        int sum = 0;
        for (int q = 0; q < 7; ++q) {
            int val = V[0][(long long)wi * 7 + q];
            std::printf("%4d", val); sum += val;
        }
        std::cout << "  和=" << sum << "\n";
    }

    sub_banner("合法段 vs IPCTC 段的平均计数向量（win=32，归一化为占比 %）");
    auto first_of = [&](int cid) {
        for (int g = 0; g < num_seg; ++g) if (ch->i[g] == cid) return g;
        return 0;
    };
    int li = first_of(0), ip = first_of(1);
    for (int tag = 0; tag < 2; ++tag) {
        int g = tag == 0 ? li : ip;
        std::cout << "  " << (tag == 0 ? "LEGIT " : "IPCTC ") << " ";
        double avg[7] = {0};
        for (int wi = 0; wi < nws[0]; ++wi)
            for (int q = 0; q < 7; ++q)
                avg[q] += V[0][(long long)g * nws[0] * 7 + (long long)wi * 7 + q];
        for (int q = 0; q < 7; ++q) std::printf("%6.1f", avg[q] / nws[0] / 32.0 * 100.0);
        std::cout << "\n";
    }

    Store st;
    st.put_i("V32", std::vector<int>{num_seg, nws[0], 7}, V[0]);
    st.put_i("V64", std::vector<int>{num_seg, nws[1], 7}, V[1]);
    st.put_i("V128", std::vector<int>{num_seg, nws[2], 7}, V[2]);
    st.put_i("seg_label", std::vector<int>{num_seg}, data.find("seg_label")->i);
    st.put_i("seg_channel", std::vector<int>{num_seg}, ch->i);
    st.put_i("seg_conn_id", std::vector<int>{num_seg}, data.find("seg_conn_id")->i);
    st.put_i("seg_index", std::vector<int>{num_seg}, data.find("seg_index")->i);
    std::string pth = out_path("04_count_vectors.bin");
    st.save(pth);
    std::cout << "\n输出形状：V32=(" << num_seg << "," << nws[0] << ",7), V64=("
              << num_seg << "," << nws[1] << ",7), V128=(" << num_seg << "," << nws[2] << ",7)\n";
    std::cout << "已保存：" << pth << "\n";
    return 0;
}
