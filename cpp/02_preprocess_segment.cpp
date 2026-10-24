// 02_preprocess_segment.cpp —— 指数平滑（α=0.30）+ L=2000、25% 重叠分段（段=图节点）
// 对应 Python 02_preprocess_segment.py。
#include "common.hpp"

static const double ALPHA = 0.30;

static std::vector<double> exponential_smoothing(const std::vector<double>& x) {
    int n = (int)x.size();
    std::vector<double> s(n);
    s[0] = x[0];
    for (int t = 1; t < n; ++t)
        s[t] = ALPHA * x[t] + (1.0 - ALPHA) * s[t - 1];
    return s;
}

int main() {
    init_console();
    Store data = Store::load(out_path("01_connections.bin"));
    const BinArc* ipd = data.find("ipd_list");
    const BinArc* lab = data.find("labels");
    const BinArc* ch = data.find("channel_id");
    int N = ipd->shape[0], L = ipd->shape[1];

    banner("02 指数平滑 + 分段（L=2000，25% 重叠）");
    // 平滑去尖峰演示
    double demo[8] = {10, 10, 10, 10, 30, 10, 10, 10};
    std::vector<double> dv(demo, demo + 8), sm = exponential_smoothing(dv);
    std::cout << "平滑去尖峰演示（尖峰 30）:\n  原始:";
    for (int t = 0; t < 8; ++t) std::printf("%6.1f", demo[t]);
    std::cout << "\n  平滑:";
    for (int t = 0; t < 8; ++t) std::printf("%6.1f", sm[t]);
    std::cout << "\n";

    int n_per_conn = count_windows(L, SEG_LEN, SEG_STEP);   // 5
    int num_seg = N * n_per_conn;                           // 320
    std::vector<double> seg_flat((long long)num_seg * SEG_LEN);
    std::vector<int> seg_label, seg_channel, seg_conn, seg_index;

    std::vector<double> one(L);
    for (int ci = 0; ci < N; ++ci) {
        for (int t = 0; t < L; ++t) one[t] = ipd->d[(long long)ci * L + t];
        std::vector<double> smoothed = exponential_smoothing(one);
        for (int si = 0; si < n_per_conn; ++si) {
            int start = si * SEG_STEP;
            int gi = ci * n_per_conn + si;
            for (int t = 0; t < SEG_LEN; ++t)
                seg_flat[(long long)gi * SEG_LEN + t] = smoothed[start + t];
            seg_label.push_back(lab->i[ci]);
            seg_channel.push_back(ch->i[ci]);
            seg_conn.push_back(ci);
            seg_index.push_back(si);
        }
    }

    sub_banner("分段结果：每连接长度 8000，L=2000，步进=1500");
    std::cout << "  每连接段数 = floor((8000-2000)/1500)+1 = " << n_per_conn << "\n";
    std::cout << "  连接数 = " << N << "，段总数（图的节点数）= " << num_seg << "\n";
    std::cout << "  合法段 = ";
    int nleg = 0; for (int x : seg_label) if (x == 0) ++nleg;
    std::cout << nleg << "，隐蔽段 = " << num_seg - nleg << "\n";
    for (int cid = 0; cid < 6; ++cid) {
        int cnt = 0; for (int x : seg_channel) if (x == cid) ++cnt;
        std::cout << "    " << CHANNEL_NAMES[cid]
                  << std::string(14 - std::string(CHANNEL_NAMES[cid]).size(), ' ')
                  << cnt << " 段\n";
    }
    std::cout << "  相邻段重叠 IPD 数 = " << SEG_LEN - SEG_STEP << "（25%）\n";

    Store st;
    st.put_d("seg_ipd", std::vector<int>{num_seg, SEG_LEN}, seg_flat);
    st.put_i("seg_label", std::vector<int>{num_seg}, seg_label);
    st.put_i("seg_channel", std::vector<int>{num_seg}, seg_channel);
    st.put_i("seg_conn_id", std::vector<int>{num_seg}, seg_conn);
    st.put_i("seg_index", std::vector<int>{num_seg}, seg_index);
    std::string pth = out_path("02_segments.bin");
    st.save(pth);
    std::cout << "\n已保存：" << pth << "\n";
    return 0;
}
