// 07_robust_aggregation.cpp —— 有限位移对齐 + top 平均，得段对三尺度相似度 w1/w2/w3
// 对应 Python 07_robust_aggregation.py。两级聚合：位移内中位数 -> 位移间 top3 均值。
#include "common.hpp"

int main() {
    init_console();
    Store data = Store::load(out_path("05_fingerprints.bin"));
    const BinArc* H[3] = {data.find("H32"), data.find("H64"), data.find("H128")};
    const BinArc* ch = data.find("seg_channel");
    int num_seg = H[0]->shape[0];

    auto first_of = [&](int cid) {
        for (int g = 0; g < num_seg; ++g) if (ch->i[g] == cid) return g;
        return -1;
    };
    int li = first_of(0);
    int lj = (li >= 0) ? li + 1 : -1;
    int ip = first_of(1);
    int ip2 = (ip >= 0) ? ip + 1 : -1;
    int jt = first_of(4);
    int ep = first_of(5);

    auto fp = [&](int sc, int g) -> const unsigned char* {
        return &H[sc]->b[(long long)g * H[sc]->shape[1] * FP_BYTES];
    };
    auto three = [&](int a, int b, double w[4]) {
        for (int sc = 0; sc < 3; ++sc)
            w[sc] = scale_similarity(fp(sc, a), fp(sc, b), H[sc]->shape[1]);
        w[3] = std::max(std::max(w[0], w[1]), w[2]);
    };

    struct P { const char* tag; int a, b; };
    std::vector<P> pairs;
    auto add_pair = [&](const char* t, int a, int b) {
        if (a >= 0 && b >= 0) { P pp; pp.tag = t; pp.a = a; pp.b = b; pairs.push_back(pp); }
    };
    add_pair("合法-合法", li, lj);
    add_pair("合法-IPCTC", li, ip);
    add_pair("合法-Jitterbug", li, jt);
    add_pair("合法-εklibur", li, ep);
    add_pair("IPCTC-IPCTC", ip, ip2);

    banner("07 有限位移对齐 + top 平均：段对三尺度相似度 w1/w2/w3");
    std::cout << "位移范围 ±" << MAX_SHIFT << "（窗口步长）；每个位移取窗口相似度中位数，\n"
              << "再取最高的 " << TOP_SHIFTS << " 个位移分数平均（top small portion）。\n";
    std::cout << "段对               w1(32)  w2(64)  w3(128)  m=max\n";

    Store st;
    for (size_t t = 0; t < pairs.size(); ++t) {
        double w[4]; three(pairs[t].a, pairs[t].b, w);
        std::printf("  %-16s %7.4f %7.4f %7.4f %7.4f\n",
                    pairs[t].tag, w[0], w[1], w[2], w[3]);
        std::vector<double> wv(w, w + 4);
        st.put_d(std::string("p") + std::to_string(t),
                 std::vector<int>{4}, wv);
    }
    std::cout << "\n解读：同类段对（合法-合法、IPCTC-IPCTC）相似度高，跨类（合法-IPCTC）明显低；\n";
    if (ep >= 0)
        std::cout << "      合法-εklibur 可能接近合法（最隐蔽）。\n";
    else
        std::cout << "      本数据集不含 ε-xlibur，故无“合法-εklibur”演示对。\n";
    std::string pth = out_path("07_pair_similarities.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
