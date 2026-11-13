// 01_load_caida.cpp —— 加载【真实 CAIDA / NCTC 数据】（替代合成 IPD，01_generate_ipd.cpp 保留）
// 对应 Python 01_load_caida.py，切块/抽样/清洗/类别顺序完全一致，仅语言不同。
// ============================================================================
// 数据：../data/<subset>/ 下单列 CSV（表头 "IPDs"，每行一个包间时间间隔，单位秒）。
//   subset = bignet（Backbone 广域网）或 labnet（Lab 实验室网）。
//   合法：http.csv / https.csv / ssh.csv；
//   异常：IPCTC_1/2、LNCTC_1/2、TRCTC_1/2、Jitterbug_1/2（论文 Table 3 两种配置）；
//   【不含 ε-xlibur】，channel_id=5 样本数为 0，下游自动跳过。
//
// 做法：每个 CSV 按 CONN_LEN=8000 切【互不重叠】连接块（与 01_generate_ipd 同 schema）；
//   每类用【确定性均匀抽样】取 k 个块（纯算术、无随机数，与 Python 逐块一致）；
//   0/负/非有限 IPD 统一裁剪到 1e-6 s（抓包时间戳误差）。
//
// 用法：01_load_caida.exe [bignet|labnet] [每类抽样数K]   （默认 bignet，K=16）
// 输出：outputs/01_connections.bin（与 01_generate_ipd 同名、同 schema），
//       并写 outputs/data_source.txt 标记数据来源（供 14 标题/说明使用）。
#include "common.hpp"

static const double CAIDA_MIN_IPD = 1e-6;
static const int DEFAULT_K = 16;
static const char* DATA_ROOT = "../data";

// 读取单列 IPD CSV（跳过表头），返回清洗后的 double 序列
static std::vector<double> read_ipd_csv(const std::string& path) {
    std::vector<double> x;
    std::ifstream f(path.c_str());
    if (!f) { std::cerr << "无法打开 " << path << "\n"; return x; }
    std::string line;
    std::getline(f, line);                       // 表头
    while (std::getline(f, line)) {
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;   // 空行（真实数据通常没有）
        size_t b = line.find_last_not_of(" \t\r\n");
        std::string t = line.substr(a, b - a + 1);
        const char* cs = t.c_str();
        char* end = NULL;
        double v = std::strtod(cs, &end);
        if (end == cs) v = CAIDA_MIN_IPD;        // 解析失败
        if (!std::isfinite(v) || v < CAIDA_MIN_IPD) v = CAIDA_MIN_IPD;
        x.push_back(v);
    }
    return x;
}

// 确定性均匀选块（与 Python uniform_pick 完全一致）：idx_j = floor(j*(n-1)/(k-1)+0.5)
static std::vector<int> uniform_pick(int n_total, int k) {
    std::vector<int> idx;
    if (k <= 0) return idx;
    if (n_total <= k) { for (int i = 0; i < n_total; ++i) idx.push_back(i); return idx; }
    if (k == 1) { idx.push_back(0); return idx; }
    for (int j = 0; j < k; ++j)
        idx.push_back((int)(j * (double)(n_total - 1) / (k - 1) + 0.5));
    return idx;
}

// 把 total 个名额按顺序分到 m 个文件，受容量 caps 限制（与 Python allocate_quotas 一致）
static std::vector<int> allocate_quotas(int total, const std::vector<int>& caps) {
    int m = (int)caps.size();
    int base = total / m, rem = total % m;
    std::vector<int> q(m);
    for (int j = 0; j < m; ++j) q[j] = base + (j < rem ? 1 : 0);
    for (int i = 0; i < m; ++i) {
        if (q[i] > caps[i]) {
            int extra = q[i] - caps[i];
            q[i] = caps[i];
            for (int j = 0; j < m; ++j)
                if (j != i && q[j] < caps[j]) {
                    int add = std::min(extra, caps[j] - q[j]);
                    q[j] += add; extra -= add;
                }
        }
    }
    return q;
}

struct Rec {
    std::vector<double> ipd;
    int label, channel, proto;
};

int main(int argc, char** argv) {
    init_console();
    std::string subset = (argc > 1) ? std::string(argv[1]) : std::string("bignet");
    if (subset != "bignet" && subset != "labnet") {
        std::cerr << "未知子集 " << subset << "，应为 bignet 或 labnet\n"; return 1;
    }
    int k = (argc > 2) ? std::atoi(argv[2]) : DEFAULT_K;
    if (k <= 0) k = DEFAULT_K;

    std::string scene = std::string(DATA_ROOT) + "/" + subset;
    banner("01 加载真实 CAIDA/NCTC 数据（subset=" + subset + "，每类 k=" + std::to_string(k) + "）");
    std::cout << "数据目录 = " << scene << "\n";
    std::cout << "每条连接长度 = " << CONN_LEN << " IPD；每类抽取 " << k
              << " 条；IPD 下限 " << CAIDA_MIN_IPD << "s\n";

    std::vector<Rec> recs;

    // 文件清单（与 Python 01_load_caida.py 完全一致）
    struct LegitFile { const char* fn; int proto; };
    LegitFile legit_files[3] = {
        {"http.csv", 0}, {"https.csv", 1}, {"ssh.csv", 2}};
    struct CovertFile { int ch; const char* f1; const char* f2; };
    CovertFile covert_files[4] = {
        {1, "IPCTC_1.csv", "IPCTC_2.csv"},
        {2, "LNCTC_1.csv", "LNCTC_2.csv"},
        {3, "TRCTC_1.csv", "TRCTC_2.csv"},
        {4, "Jitterbug_1.csv", "Jitterbug_2.csv"}};

    sub_banner("各文件可用块数与实际抽取");
    std::printf("%-24s %12s %8s %6s %6s\n", "文件", "原始IPD", "可用块", "计划", "实际");

    auto append_blocks = [&](const std::vector<double>& x, int want, int label, int ch,
                             int proto, const std::string& tag) {
        int nblk = (int)(x.size() / CONN_LEN);
        std::vector<int> picks = uniform_pick(nblk, std::min(want, nblk));
        int got = 0;
        for (size_t qi = 0; qi < picks.size(); ++qi) {
            int b = picks[qi];
            Rec r;
            r.ipd.assign(x.begin() + (long long)b * CONN_LEN,
                         x.begin() + (long long)(b + 1) * CONN_LEN);
            if ((int)r.ipd.size() < CONN_LEN) continue;
            r.label = label; r.channel = ch; r.proto = proto;
            recs.push_back(r);
            ++got;
        }
        std::printf("%-24s %12zu %8d %6d %6d\n", tag.c_str(), x.size(), nblk, want, got);
    };

    // ---- 合法：http -> https -> ssh，每协议 k 条 ----
    for (int p = 0; p < 3; ++p) {
        std::string path = scene + "/" + legit_files[p].fn;
        std::ifstream test(path.c_str());
        if (!test) {
            std::printf("%-24s %12d %8d %6d %6d  [警告] 缺少合法文件，跳过\n",
                        legit_files[p].fn, 0, 0, k, 0);
            continue;
        }
        test.close();
        std::vector<double> x = read_ipd_csv(path);
        append_blocks(x, k, 0, 0, legit_files[p].proto,
                      std::string(PROTOCOL_NAMES[legit_files[p].proto]) + ":" + legit_files[p].fn);
    }
    // ---- 隐蔽：每类 k 条，在两个配置文件间均分 ----
    for (int c = 0; c < 4; ++c) {
        int ch = covert_files[c].ch;
        const char* fns[2] = {covert_files[c].f1, covert_files[c].f2};
        std::vector<std::vector<double> > xs(2);
        std::vector<int> caps(2, 0);
        for (int u = 0; u < 2; ++u) {
            std::string path = scene + "/" + fns[u];
            std::ifstream test(path.c_str());
            if (!test) {
                std::printf("[警告] 缺少隐蔽文件 %s，跳过。\n", fns[u]);
                continue;
            }
            test.close();
            xs[u] = read_ipd_csv(path);
            caps[u] = (int)(xs[u].size() / CONN_LEN);
        }
        std::vector<int> quota = allocate_quotas(k, caps);
        for (int u = 0; u < 2; ++u) {
            if (quota[u] <= 0 || xs[u].empty()) continue;
            append_blocks(xs[u], quota[u], 1, ch, 0,
                          std::string(CHANNEL_NAMES[ch]) + ":" + fns[u]);
        }
    }

    int N = (int)recs.size();
    int n_legit = 0, n_cov = 0;
    for (int i = 0; i < N; ++i) { if (recs[i].label == 0) ++n_legit; else ++n_cov; }

    sub_banner("各类连接条数与示例 IPD（前 8 个）");
    for (int cid = 0; cid < 6; ++cid) {
        int first = -1, cnt = 0;
        for (int i = 0; i < N; ++i) if (recs[i].channel == cid) { if (first < 0) first = i; ++cnt; }
        if (cnt == 0) {
            std::cout << "  " << CHANNEL_NAMES[cid]
                      << std::string(14 - std::string(CHANNEL_NAMES[cid]).size(), ' ')
                      << " n= 0  （本数据集无该类样本）\n";
            continue;
        }
        std::cout << "  " << CHANNEL_NAMES[cid]
                  << std::string(14 - std::string(CHANNEL_NAMES[cid]).size(), ' ')
                  << " n=" << cnt << "  label=" << recs[first].label << "  示例IPD: [";
        for (int t = 0; t < 8; ++t)
            std::printf("%s%.5f", t ? ", " : "", recs[first].ipd[t]);
        std::cout << ", ...]\n";
    }
    std::cout << "\n连接总数 = " << N << "（合法 " << n_legit << " / 隐蔽 " << n_cov
              << "），预计 02 切段后每连接 5 段、共 " << N * 5 << " 段\n";

    // 扁平化（每条固定 CONN_LEN）
    std::vector<double> ipd_flat((long long)N * CONN_LEN);
    std::vector<int> labels(N), channels(N), protos(N);
    for (int i = 0; i < N; ++i) {
        for (int t = 0; t < CONN_LEN; ++t)
            ipd_flat[(long long)i * CONN_LEN + t] = recs[i].ipd[t];
        labels[i] = recs[i].label; channels[i] = recs[i].channel; protos[i] = recs[i].proto;
    }
    Store st;
    st.put_d("ipd_list", std::vector<int>{N, CONN_LEN}, ipd_flat);
    st.put_i("labels", std::vector<int>{N}, labels);
    st.put_i("channel_id", std::vector<int>{N}, channels);
    st.put_i("protocol", std::vector<int>{N}, protos);
    std::string pth = out_path("01_connections.bin");
    st.save(pth);

    // 数据来源标记
    std::ofstream sf(out_path("data_source.txt").c_str(), std::ios::binary);
    sf << ("CAIDA/" + subset) << "\n";
    sf.close();

    std::cout << "\n已保存：" << pth << "\n";
    std::cout << "数据来源标记：outputs/data_source.txt = CAIDA/" << subset << "\n";
    return 0;
}
