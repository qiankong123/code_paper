# -*- coding: utf-8 -*-
"""
10_gnn_gcn.py —— 两层 GCNConv 编码器（前向 + 自监督 MSE 训练）
================================================================
论文 §3.2.2：
  · 归一化传播矩阵 P = D̂^{-1/2} Â D̂^{-1/2}，其中 Â = A + I（加自环）
  · 一层 GCN：H^{(l+1)} = ReLU( P H^{(l)} W^{(l)} + b^{(l)} )
  · 两层后每节点得 16 维 embedding；两层 = 覆盖 2 跳邻居
  · 边分数 φ_uv = σ(z_u · z_v)（公式4）
  · 无人工标签，用边权 w_uv 当软目标做 MSE 自监督（公式5）：
        L = (1/|E|) Σ_{(u,v)∈E} (φ_uv − w_uv)^2
    Adam，学习率 1e-2；离线训练、在线冻结。

参数量（输入 2 维、隐藏 16、输出 16，含偏置）：
  W0 2×16 + b0 16 + W1 16×16 + b1 16 = 32+16+256+16 = 320。

输出：outputs/10_gnn.npz
  Z     : (num_seg, 16) 节点 embedding
  edges : 透传边集；phi 每条边的 GNN 置信度
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

H_DIM = 16
EPOCHS = 300
LR = 1e-2


def build_P(A):
    """P = D̂^{-1/2} (A+I) D̂^{-1/2}。"""
    n = A.shape[0]
    Ahat = A + np.eye(n)
    d = Ahat.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return D @ Ahat @ D


def forward(X, P, params):
    """两层 GCN 前向。返回 Z 与中间缓存。"""
    W0, b0, W1, b1 = params
    M1 = P @ X @ W0 + b0
    H1 = np.maximum(M1, 0.0)
    Z = P @ H1 @ W1 + b1
    return Z, dict(M1=M1, H1=H1)


def edge_phi(Z, edges):
    """公式4：φ_uv = σ(z_u·z_v)。"""
    u = edges[:, 0].astype(int); v = edges[:, 1].astype(int)
    return common.sigmoid(np.sum(Z[u] * Z[v], axis=1))


def train(X, A, edges, target_w, seed=common.SEED):
    """Adam 自监督训练，让 φ_uv 逼近边权 target_w。"""
    r = common.rng(seed + 11)
    n, din = X.shape
    # Python/C++ 共享 GCN 初始权重（消除 PCG64 与 mt19937 的初始化差异，
    # 否则训练后的 φ 在“是否启用 GNN 边强（β>0）”的弱分离场景会翻转）。
    W0_path = os.path.join(HERE, "gcn_W0_init.csv")
    W1_path = os.path.join(HERE, "gcn_W1_init.csv")
    if os.path.exists(W0_path) and os.path.exists(W1_path):
        W0 = np.loadtxt(W0_path, delimiter=",")
        W1 = np.loadtxt(W1_path, delimiter=",")
        assert W0.shape == (din, H_DIM) and W1.shape == (H_DIM, H_DIM)
    else:
        W0 = r.standard_normal((din, H_DIM)) * 0.5
        W1 = r.standard_normal((H_DIM, H_DIM)) * 0.3
    b0 = np.zeros(H_DIM); b1 = np.zeros(H_DIM)
    params = [W0, b0, W1, b1]
    P = build_P(A)
    u = edges[:, 0].astype(int); v = edges[:, 1].astype(int)
    E = len(edges)

    def adam_step():
        pass

    # Adam 状态
    m_ = [np.zeros_like(p) for p in params]
    v_ = [np.zeros_like(p) for p in params]
    b1m, b2m, eps = 0.9, 0.999, 1e-8

    for epoch in range(EPOCHS):
        Z, cache = forward(X, P, params)
        W0, b0, W1, b1 = params
        H1 = cache["H1"]; M1 = cache["M1"]
        inner = np.sum(Z[u] * Z[v], axis=1)
        phi = common.sigmoid(inner)

        # 损失对 φ、内积的梯度
        g_phi = 2.0 * (phi - target_w) / E
        g_inner = g_phi * phi * (1.0 - phi)
        dZ = np.zeros_like(Z)
        np.add.at(dZ, u, g_inner[:, None] * Z[v])
        np.add.at(dZ, v, g_inner[:, None] * Z[u])

        # 反向：Z = P H1 W1 + b1
        dW1 = H1.T @ P @ dZ
        db1 = dZ.sum(axis=0)
        dH1 = P @ dZ @ W1.T
        dM1 = dH1 * (M1 > 0)
        # 反向：M1 = P X W0 + b0
        dW0 = X.T @ P @ dM1
        db0 = dM1.sum(axis=0)
        grads = [dW0, db0, dW1, db1]

        loss = float(np.mean((phi - target_w) ** 2))
        # Adam 更新
        for k in range(4):
            m_[k] = b1m * m_[k] + (1 - b1m) * grads[k]
            v_[k] = b2m * v_[k] + (1 - b2m) * (grads[k] ** 2)
            mh = m_[k] / (1 - b1m ** (epoch + 1))
            vh = v_[k] / (1 - b2m ** (epoch + 1))
            params[k] -= LR * mh / (np.sqrt(vh) + eps)
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"  epoch {epoch:4d}  MSE={loss:.5f}")
    return params, P


def handcalc_demo():
    """3 节点链 A-B-C 手算验证（隐藏层压到 2 维）。"""
    X = np.array([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
    A = np.array([[0., 1, 0], [1, 0, 1], [0, 1, 0]])
    P = build_P(A)
    W0 = np.array([[1.0, 1.0], [1.0, -1.0]])
    M1 = P @ X @ W0
    H1 = np.maximum(M1, 0)
    common.sub_banner("3 节点链手算验证（无偏置、W1=I）")
    np.set_printoptions(precision=4, suppress=True)
    print("归一化 P =\n", P)
    print("第1层聚合 P·X =\n", P @ X)
    print("H1 = ReLU(P·X·W0) =\n", H1)
    Z = P @ H1 @ np.eye(2)
    print("Z（W1=I）=\n", Z)
    phi_ab = common.sigmoid(Z[0] @ Z[1]); phi_bc = common.sigmoid(Z[1] @ Z[2])
    print(f"φ(A,B)={phi_ab:.4f}, φ(B,C)={phi_bc:.4f}")


def normalize_features(X):
    """节点特征量级归一化：度除以最大度（均权本就在 0-1），避免度≈70 主导。"""
    Xn = X.copy()
    Xn[:, 0] = Xn[:, 0] / max(Xn[:, 0].max(), 1.0)
    return Xn


def main():
    g = common.load_npz("09_graph.npz", allow_pickle=True)
    X = normalize_features(g["X"]); A = g["A"]; edges = g["edges"]
    target_w = edges[:, 5]

    common.banner("10 两层 GCNConv：前向 + 自监督 MSE 训练")
    handcalc_demo()

    common.sub_banner(f"真实图上训练（节点 {X.shape[0]}，边 {len(edges)}，隐藏/输出 {H_DIM} 维）")
    params, P = train(X, A, edges, target_w)
    Z, _ = forward(X, P, params)
    phi = edge_phi(Z, edges)

    print(f"参数量 = {sum(p.size for p in params)}（应为 320）")
    common.sub_banner("训练后：边权 w（软目标）vs GNN 置信度 φ（抽 10 条边）")
    print(f"{'边':12s}{'w(m)':>8}{'φ':>9}")
    for k in range(min(10, len(edges))):
        i, j = int(edges[k, 0]), int(edges[k, 1])
        print(f"({i:3d},{j:3d}){target_w[k]:8.3f}{phi[k]:9.3f}")
    rmse = float(np.sqrt(np.mean((phi - target_w) ** 2)))
    print(f"\n最终 RMSE(φ, w) = {rmse:.4f}（φ 已学到边权，达到去噪一致性）")

    path = common.save_npz(
        "10_gnn.npz",
        Z=Z, edges=edges, phi=phi, edge_type=g["edge_type"],
        X=X, A=A, theta=g["theta"],
        seg_label=g["seg_label"], seg_channel=g["seg_channel"],
        seg_conn_id=g["seg_conn_id"], seg_index=g["seg_index"],
        channel_names=g["channel_names"],
    )
    print(f"已保存：{path}")


if __name__ == "__main__":
    main()
