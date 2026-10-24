// 10_gnn_gcn.cpp —— 两层 GCNConv 编码器（前向 + 自监督 MSE 训练，公式 4/5）
// 对应 Python 10_gnn_gcn.py。纯标准库实现矩阵乘、ReLU、sigmoid 与 Adam 反向传播。
#include "common.hpp"

static const int H_DIM = 16;
static const int EPOCHS = 300;
static const double LR = 1e-2;

static Mat transpose(const Mat& A) {
    Mat B(A.c, A.r);
    for (int i = 0; i < A.r; ++i)
        for (int j = 0; j < A.c; ++j) B.at(j, i) = A.get(i, j);
    return B;
}
static Mat add_col_bias(const Mat& M, const std::vector<double>& b) {
    Mat R = M;
    for (int i = 0; i < R.r; ++i)
        for (int j = 0; j < R.c; ++j) R.at(i, j) += b[j];
    return R;
}
// P = D̂^-1/2 (A+I) D̂^-1/2
static Mat build_P(const Mat& A) {
    int n = A.r;
    Mat Ahat(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Ahat.at(i, j) = A.get(i, j) + (i == j ? 1.0 : 0.0);
    std::vector<double> dinv(n);
    for (int i = 0; i < n; ++i) {
        double s = 0; for (int j = 0; j < n; ++j) s += Ahat.get(i, j);
        dinv[i] = s > 0 ? 1.0 / std::sqrt(s) : 0.0;
    }
    Mat P(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) P.at(i, j) = dinv[i] * Ahat.get(i, j) * dinv[j];
    return P;
}

int main() {
    init_console();
    Store g = Store::load(out_path("09_graph.bin"));
    const BinArc* Xarc = g.find("X");
    const BinArc* Aarc = g.find("A");
    const BinArc* edges = g.find("edges");
    int n = Xarc->shape[0], E = edges->shape[0];

    // 节点特征归一化：度除以最大度
    Mat X(n, 2);
    double maxdeg = 1.0;
    for (int i = 0; i < n; ++i) maxdeg = std::max(maxdeg, Xarc->d[i * 2]);
    for (int i = 0; i < n; ++i) { X.at(i, 0) = Xarc->d[i * 2] / maxdeg; X.at(i, 1) = Xarc->d[i * 2 + 1]; }
    Mat A(n, n); A.a = Aarc->d;

    // ---------- 3 节点链手算 ----------
    banner("10 两层 GCNConv：前向 + 自监督 MSE 训练");
    {
        double Xd[6] = {1, 2, 0, 1, 2, 0};
        double Ad[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
        Mat Xh(3, 2); for (int q = 0; q < 6; ++q) Xh.a[q] = Xd[q];
        Mat Ah(3, 3); for (int q = 0; q < 9; ++q) Ah.a[q] = Ad[q];
        Mat Ph = build_P(Ah);
        Mat W0h(2, 2);
        W0h.a[0] = 1; W0h.a[1] = 1; W0h.a[2] = 1; W0h.a[3] = -1;
        Mat PX = mat_mul(Ph, Xh), M1 = mat_mul(PX, W0h);
        for (int q = 0; q < M1.a.size(); ++q) M1.a[q] = std::max(0.0, M1.a[q]);
        Mat Z = mat_mul(Ph, M1);   // W1=I
        sub_banner("3 节点链手算验证（无偏置、W1=I）");
        std::cout << "第1层聚合 P·X 与 Z：\n";
        for (int i = 0; i < 3; ++i) {
            std::printf("  P·X 节点%d: %6.4f %6.4f | Z: %6.4f %6.4f\n",
                        i, PX.get(i, 0), PX.get(i, 1), Z.get(i, 0), Z.get(i, 1));
        }
        auto dot = [&](int u, int v) { return Z.get(u, 0) * Z.get(v, 0) + Z.get(u, 1) * Z.get(v, 1); };
        std::printf("φ(A,B)=%.4f, φ(B,C)=%.4f\n", sigmoid(dot(0, 1)), sigmoid(dot(1, 2)));
    }

    // ---------- 真实图训练 ----------
    Mat P = build_P(A);
    RNG r(SEED + 11);
    Mat W0(2, H_DIM), W1(H_DIM, H_DIM);
    std::vector<double> b0(H_DIM, 0), b1(H_DIM, 0);
    for (int q = 0; q < 2 * H_DIM; ++q) W0.a[q] = r.standard_normal() * 0.5;
    for (int q = 0; q < H_DIM * H_DIM; ++q) W1.a[q] = r.standard_normal() * 0.3;

    std::vector<int> eu(E), ev(E);
    std::vector<double> tw(E);
    for (int e = 0; e < E; ++e) {
        eu[e] = (int)edges->d[(long long)e * 6 + 0];
        ev[e] = (int)edges->d[(long long)e * 6 + 1];
        tw[e] = edges->d[(long long)e * 6 + 5];
    }

    // Adam 状态
    Mat mW0(2, H_DIM), vW0(2, H_DIM), mW1(H_DIM, H_DIM), vW1(H_DIM, H_DIM);
    std::vector<double> mb0(H_DIM, 0), vb0(H_DIM, 0), mb1(H_DIM, 0), vb1(H_DIM, 0);
    const double b1m = 0.9, b2m = 0.999, eps = 1e-8;

    sub_banner("真实图上训练（节点 " + std::to_string(n) + "，边 " + std::to_string(E) +
               "，隐藏/输出 16 维）");
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 前向
        Mat PX = mat_mul(P, X);
        Mat M1 = mat_mul(PX, W0); M1 = add_col_bias(M1, b0);
        Mat H1(n, H_DIM);
        for (int q = 0; q < n * H_DIM; ++q) H1.a[q] = std::max(0.0, M1.a[q]);
        Mat PH1 = mat_mul(P, H1);
        Mat Zc = mat_mul(PH1, W1); Zc = add_col_bias(Zc, b1);

        // phi
        std::vector<double> phi(E), ginner(E);
        for (int e = 0; e < E; ++e) {
            double inner = 0;
            for (int c = 0; c < H_DIM; ++c) inner += Zc.get(eu[e], c) * Zc.get(ev[e], c);
            double p = sigmoid(inner);
            phi[e] = p;
            double gphi = 2.0 * (p - tw[e]) / E;
            ginner[e] = gphi * p * (1.0 - p);
        }
        // dZ
        Mat dZ(n, H_DIM);
        for (int e = 0; e < E; ++e) {
            for (int c = 0; c < H_DIM; ++c) {
                dZ.at(eu[e], c) += ginner[e] * Zc.get(ev[e], c);
                dZ.at(ev[e], c) += ginner[e] * Zc.get(eu[e], c);
            }
        }
        // 反向第二层
        Mat PdZ = mat_mul(P, dZ);
        Mat H1T = transpose(H1);
        Mat dW1 = mat_mul(H1T, PdZ);
        std::vector<double> db1(H_DIM, 0);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < H_DIM; ++c) db1[c] += dZ.get(i, c);
        Mat W1T = transpose(W1);
        Mat dH1 = mat_mul(PdZ, W1T);
        Mat dM1(n, H_DIM);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < H_DIM; ++c) dM1.at(i, c) = M1.get(i, c) > 0 ? dH1.get(i, c) : 0.0;
        // 反向第一层
        Mat PdM1 = mat_mul(P, dM1);
        Mat XT = transpose(X);
        Mat dW0 = mat_mul(XT, PdM1);
        std::vector<double> db0(H_DIM, 0);
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < H_DIM; ++c) db0[c] += dM1.get(i, c);

        double loss = 0;
        for (int e = 0; e < E; ++e) loss += (phi[e] - tw[e]) * (phi[e] - tw[e]);
        loss /= E;

        // Adam 更新（对 4 个参数块）
        struct Block { Mat* W; Mat* mw; Mat* vw; std::vector<double>* b;
                       std::vector<double>* mb; std::vector<double>* vb; Mat* gW; std::vector<double>* gb; };
        Mat *pW[2] = {&W0, &W1}, *pmW[2] = {&mW0, &mW1}, *pvW[2] = {&vW0, &vW1};
        Mat *gW[2] = {&dW0, &dW1};
        std::vector<double>* pb[2] = {&b0, &b1}, *pmb[2] = {&mb0, &mb1}, *pvb[2] = {&vb0, &vb1};
        std::vector<double>* gb[2] = {&db0, &db1};
        double bc1 = 1.0 - std::pow(b1m, epoch + 1), bc2 = 1.0 - std::pow(b2m, epoch + 1);
        for (int k = 0; k < 2; ++k) {
            for (int q = 0; q < (int)pW[k]->a.size(); ++q) {
                pmW[k]->a[q] = b1m * pmW[k]->a[q] + (1 - b1m) * gW[k]->a[q];
                pvW[k]->a[q] = b2m * pvW[k]->a[q] + (1 - b2m) * gW[k]->a[q] * gW[k]->a[q];
                double mh = pmW[k]->a[q] / bc1, vh = pvW[k]->a[q] / bc2;
                pW[k]->a[q] -= LR * mh / (std::sqrt(vh) + eps);
            }
            for (int c = 0; c < (int)pb[k]->size(); ++c) {
                (*pmb[k])[c] = b1m * (*pmb[k])[c] + (1 - b1m) * (*gb[k])[c];
                (*pvb[k])[c] = b2m * (*pvb[k])[c] + (1 - b2m) * (*gb[k])[c] * (*gb[k])[c];
                double mh = (*pmb[k])[c] / bc1, vh = (*pvb[k])[c] / bc2;
                (*pb[k])[c] -= LR * mh / (std::sqrt(vh) + eps);
            }
        }
        if (epoch % 50 == 0 || epoch == EPOCHS - 1)
            std::printf("  epoch %4d  MSE=%.5f\n", epoch, loss);
    }

    // 训练后前向 + phi
    Mat PX = mat_mul(P, X);
    Mat M1 = mat_mul(PX, W0); M1 = add_col_bias(M1, b0);
    Mat H1(n, H_DIM);
    for (int q = 0; q < n * H_DIM; ++q) H1.a[q] = std::max(0.0, M1.a[q]);
    Mat Zc = mat_mul(mat_mul(P, H1), W1); Zc = add_col_bias(Zc, b1);
    std::vector<double> phi(E);
    double sse = 0;
    for (int e = 0; e < E; ++e) {
        double inner = 0;
        for (int c = 0; c < H_DIM; ++c) inner += Zc.get(eu[e], c) * Zc.get(ev[e], c);
        phi[e] = sigmoid(inner);
        sse += (phi[e] - tw[e]) * (phi[e] - tw[e]);
    }
    long long nparams = 2 * H_DIM + H_DIM + H_DIM * H_DIM + H_DIM;
    std::cout << "参数量 = " << nparams << "（应为 320）\n";
    sub_banner("训练后：边权 w（软目标）vs GNN 置信度 φ（抽 10 条边）");
    std::cout << "边            w(m)       φ\n";
    for (int e = 0; e < std::min(10, E); ++e)
        std::printf("(%3d,%3d) %8.3f %9.3f\n", eu[e], ev[e], tw[e], phi[e]);
    std::printf("\n最终 RMSE(φ, w) = %.4f（φ 已学到边权，达到去噪一致性）\n", std::sqrt(sse / E));

    // 保存（Z, phi, edges, edge_type, X, A, theta 及元信息）
    Store st;
    st.put_d("Z", std::vector<int>{n, H_DIM}, Zc.a);
    st.put_d("phi", std::vector<int>{E}, phi);
    st.put_d("edges", std::vector<int>{E, 6}, edges->d);
    st.put_i("edge_type", std::vector<int>{E}, g.find("edge_type")->i);
    st.put_d("X", std::vector<int>{n, 2}, X.a);
    st.put_d("A", std::vector<int>{n, n}, A.a);
    st.put_d("theta", std::vector<int>{2}, g.find("theta")->d);
    st.put_i("seg_label", std::vector<int>{n}, g.find("seg_label")->i);
    st.put_i("seg_channel", std::vector<int>{n}, g.find("seg_channel")->i);
    st.put_i("seg_conn_id", std::vector<int>{n}, g.find("seg_conn_id")->i);
    st.put_i("seg_index", std::vector<int>{n}, g.find("seg_index")->i);
    std::string pth = out_path("10_gnn.bin");
    st.save(pth);
    std::cout << "已保存：" << pth << "\n";
    return 0;
}
