// 01_generate_ipd.cpp —— 生成合成 IPD 数据（合法 HTTP/HTTPS/SSH + 5 类隐蔽定时信道）
// 对应 Python 01_generate_ipd.py。统计模型与参数完全一致，仅 RNG 不同（mt19937 代替 numpy PCG64）。
#include "common.hpp"

// 合法 log-IPD 一阶自相关参数（与 Python 完全一致）
static const double MU[3]  = {std::log(0.12), std::log(0.10), std::log(0.09)};
static const double PHI[3] = {0.60, 0.58, 0.55};
static const double SIG[3] = {0.065, 0.060, 0.055};
static const double BURST[3] = {0.003, 0.002, 0.001};

// 合法连接：log-AR(1) + 少量突发
static std::vector<double> legit_connection(int n, RNG& r, int proto) {
    double mu = MU[proto], phi = PHI[proto], sigma = SIG[proto];
    std::vector<double> x(n), ipd(n);
    // 先抽全部驱动噪声（与 numpy 调用顺序一致）
    std::vector<double> eps(n);
    for (int t = 0; t < n; ++t) eps[t] = r.standard_normal();
    x[0] = mu;
    for (int t = 1; t < n; ++t)
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * eps[t];
    for (int t = 0; t < n; ++t) ipd[t] = std::exp(x[t]);
    int nb = (int)(n * BURST[proto]);
    for (int q = 0; q < nb; ++q) {
        int idx = r.integer(1, n);                 // [1,n)
        double m = r.uniform(0.30, 0.60);
        ipd[idx] *= m;
    }
    return ipd;
}

// 在合法基线上叠加隐蔽定时编码
static std::vector<double> covert_encode(const std::vector<double>& base, RNG& r, int ch) {
    std::vector<double> ipd = base;
    int n = (int)ipd.size();
    if (ch == 1) {                       // IPCTC：两档大倍率
        int T = 8;
        for (int j = 0; j < n / T; ++j) {
            int b = r.integer(0, 2);
            ipd[j * T] = base[j * T] * (b == 0 ? 0.45 : 2.20);
        }
    } else if (ch == 2) {                // LNCTC：四档
        int T = 6;
        static const double mult[4] = {0.50, 0.75, 1.33, 2.00};
        for (int j = 0; j < n / T; ++j) {
            int s = r.integer(0, 4);
            ipd[j * T] = base[j * T] * mult[s];
        }
    } else if (ch == 3) {                // TRCTC：周期 ×2.00
        int T = 12;
        for (int i = 0; i < n; i += T) ipd[i] = base[i] * 2.00;
    } else if (ch == 4) {                // Jitterbug：周期 T=4 ×0.91
        int T = 4;
        for (int i = 0; i < n; i += T) ipd[i] = base[i] * 0.91;
    } else if (ch == 5) {                // ε-klibur：更弱更稀疏 T=8 ×0.95
        int T = 8;
        for (int i = 0; i < n; i += T) ipd[i] = base[i] * 0.95;
    }
    return ipd;
}

int main() {
    init_console();
    RNG r(SEED);
    std::vector<std::vector<double> > conn;
    std::vector<int> labels, channels, protos;

    // 合法：3 协议 × 8
    for (int p = 0; p < 3; ++p)
        for (int k = 0; k < N_CONN_PER_KIND; ++k) {
            conn.push_back(legit_connection(CONN_LEN, r, p));
            labels.push_back(0); channels.push_back(0); protos.push_back(p);
        }
    // 隐蔽：5 类 × 8
    for (int ch = 1; ch <= 5; ++ch)
        for (int k = 0; k < N_CONN_PER_KIND; ++k) {
            int p = k % 3;
            std::vector<double> base = legit_connection(CONN_LEN, r, p);
            conn.push_back(covert_encode(base, r, ch));
            labels.push_back(1); channels.push_back(ch); protos.push_back(p);
        }

    int N = (int)conn.size();
    banner("01 生成合成 IPD 数据集（合法 + 5 类隐蔽定时信道）");
    std::cout << "连接总数 = " << N << "，每条长度约 " << CONN_LEN << " IPD\n";

    // 扁平化
    std::vector<double> ipd_flat((long long)N * CONN_LEN);
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < CONN_LEN; ++t)
            ipd_flat[(long long)i * CONN_LEN + t] = conn[i][t];

    sub_banner("各类连接条数与示例 IPD（前 8 个）");
    for (int cid = 0; cid < 6; ++cid) {
        int first = -1, cnt = 0;
        for (int i = 0; i < N; ++i) if (channels[i] == cid) { if (first < 0) first = i; ++cnt; }
        std::cout << "  " << CHANNEL_NAMES[cid];
        for (int pad = (int)std::string(CHANNEL_NAMES[cid]).size(); pad < 14; ++pad) std::cout << ' ';
        std::cout << " n=" << cnt << "  label=" << labels[first] << "  示例IPD: [";
        for (int t = 0; t < 8; ++t) std::printf("%s%.4f", t ? ", " : "", conn[first][t]);
        std::cout << ", ...]\n";
    }

    sub_banner("编码后相邻变化率 |Δ| 的粗略对比（取前 2000 个 IPD）");
    for (int cid = 0; cid < 6; ++cid) {
        int first = -1;
        for (int i = 0; i < N; ++i) if (channels[i] == cid) { first = i; break; }
        int sharp = 0; double sumabs = 0.0;
        for (int t = 0; t < 2000; ++t) {
            double d = (conn[first][t + 1] - conn[first][t]) / conn[first][t];
            if (std::fabs(d) >= 0.06) ++sharp;
            sumabs += std::fabs(d);
        }
        std::cout << "  " << CHANNEL_NAMES[cid];
        for (int pad = (int)std::string(CHANNEL_NAMES[cid]).size(); pad < 14; ++pad) std::cout << ' ';
        std::printf(" |Δ|≥0.06(±2 及以上)占比 = %5.1f%%   mean|Δ| = %.4f\n",
                    sharp / 2000.0 * 100.0, sumabs / 2000.0);
    }

    Store st;
    st.put_d("ipd_list", std::vector<int>{N, CONN_LEN}, ipd_flat);
    st.put_i("labels", std::vector<int>{N}, labels);
    st.put_i("channel_id", std::vector<int>{N}, channels);
    st.put_i("protocol", std::vector<int>{N}, protos);
    std::string pth = out_path("01_connections.bin");
    st.save(pth);
    {
        std::ofstream sf(out_path("data_source.txt").c_str(), std::ios::binary);
        sf << "synthetic\n";
    }
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
