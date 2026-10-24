// 05_random_matrix_fingerprint.cpp —— 固定随机矩阵 R 与 128 位 SimHash 指纹（公式 2）
// 对应 Python 05_random_matrix_fingerprint.py。R 用固定种子生成、全程冻结。
#include "common.hpp"

int main() {
    init_console();
    Store data = Store::load(out_path("04_count_vectors.bin"));
    const BinArc* Vs[3] = {data.find("V32"), data.find("V64"), data.find("V128")};
    const char* keys[3] = {"V32", "V64", "V128"};
    int num_seg = Vs[0]->shape[0];

    // 固定随机矩阵 R（128×7，标准正态），按 numpy 行优先顺序连续抽样
    RNG r(R_SEED);
    double R[FP_BITS][NUM_SYMBOLS];
    for (int k = 0; k < FP_BITS; ++k)
        for (int q = 0; q < NUM_SYMBOLS; ++q)
            R[k][q] = r.standard_normal();

    banner("05 固定随机矩阵 R 与 128 位 SimHash 指纹");
    std::cout << "R 形状 = (128, 7)，固定种子 R_SEED = " << R_SEED << "\nR 前 3 行 × 7 列:\n";
    for (int k = 0; k < 3; ++k) {
        std::cout << "  ";
        for (int q = 0; q < 7; ++q) std::printf("%+6.3f ", R[k][q]);
        std::cout << "\n";
    }

    // 玩具 v
    int toy[7] = {1, 0, 2, 18, 8, 3, 0};
    int toybits[FP_BITS];
    for (int k = 0; k < FP_BITS; ++k) {
        double proj = 0; for (int q = 0; q < 7; ++q) proj += R[k][q] * toy[q];
        toybits[k] = proj > 0 ? 1 : 0;
    }
    unsigned char toypack[FP_BYTES]; pack_bits(toybits, toypack);
    sub_banner("玩具 v = [1,0,2,18,8,3,0] 的 128 位指纹");
    int none = 0;
    std::cout << "  前 32 位: "; for (int k = 0; k < 32; ++k) { std::cout << toybits[k]; none += toybits[k]; }
    std::cout << "\n  后 32 位: "; for (int k = 96; k < 128; ++k) { std::cout << toybits[k]; none += toybits[k]; }
    // 统计完整 1 的个数
    int ones = 0; for (int k = 0; k < 128; ++k) ones += toybits[k];
    std::cout << "\n  共 128 位，其中 1 的个数 = " << ones << "，打包为 16 字节\n";

    // 全量三尺度指纹
    Store st;
    std::vector<double> Rflat(FP_BITS * NUM_SYMBOLS);
    for (int k = 0; k < FP_BITS; ++k)
        for (int q = 0; q < NUM_SYMBOLS; ++q) Rflat[k * NUM_SYMBOLS + q] = R[k][q];
    st.put_d("R", std::vector<int>{FP_BITS, NUM_SYMBOLS}, Rflat);

    for (int sc = 0; sc < N_SCALES; ++sc) {
        int nw = Vs[sc]->shape[1];
        std::vector<unsigned char> H((long long)num_seg * nw * FP_BYTES, 0);
        for (int g = 0; g < num_seg; ++g)
            for (int wi = 0; wi < nw; ++wi) {
                const int* v = &Vs[sc]->i[(long long)g * nw * 7 + (long long)wi * 7];
                int bits[FP_BITS];
                for (int k = 0; k < FP_BITS; ++k) {
                    double proj = 0;
                    for (int q = 0; q < 7; ++q) proj += R[k][q] * v[q];
                    bits[k] = proj > 0 ? 1 : 0;
                }
                pack_bits(bits, &H[((long long)g * nw + wi) * FP_BYTES]);
            }
        std::vector<int> sh{num_seg, nw, FP_BYTES};
        st.put_b(std::string("H") + std::to_string(SCALE_WIN[sc]), sh, H);
        std::cout << "  win=" << SCALE_WIN[sc] << ": 指纹数组 (" << num_seg << ","
                  << nw << ",16)（段×窗口×16字节，对应128位）\n";
    }
    st.put_i("seg_label", std::vector<int>{num_seg}, data.find("seg_label")->i);
    st.put_i("seg_channel", std::vector<int>{num_seg}, data.find("seg_channel")->i);
    st.put_i("seg_conn_id", std::vector<int>{num_seg}, data.find("seg_conn_id")->i);
    st.put_i("seg_index", std::vector<int>{num_seg}, data.find("seg_index")->i);
    std::string pth = out_path("05_fingerprints.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
