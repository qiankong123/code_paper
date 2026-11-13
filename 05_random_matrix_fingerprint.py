# -*- coding: utf-8 -*-
"""
05_random_matrix_fingerprint.py —— 固定随机矩阵 R 与 128 位 SimHash 指纹
================================================================
论文 §3.2.1 公式(2)：
      h = sign(R v) ∈ {0,1}^128 ,   R ∈ ℝ^{128×7} 固定随机矩阵
  · v：一个窗口的 7 维计数向量
  · R v：把 v 投影到 128 个随机超平面上（128 维）
  · sign：投影为正记 1、为负记 0，得到 128 位指纹
这就是随机超平面 LSH（SimHash）：
  两个 v 方向越接近，128 位指纹中相同的位就越多，无需对原始高维序列做对齐。

R 如何"确定"：只在系统初始化时用固定随机种子生成一次，之后全程冻结、
训练/在线都用同一份（指纹才可比）。本复现用固定种子生成 R。

为节省存储，128 位指纹用 np.packbits 打包成 16 字节（16 个 uint8）。

输出：outputs/05_fingerprints.npz
  R          : (128, 7) 固定随机矩阵
  H32/H64/H128 : (num_seg, 窗口数, 16) uint8（128 位打包后的字节）
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

# 生成 R 的专用种子（固定，保证每次运行得到同一份 R）
R_SEED = common.SEED + 7

# Python/C++ 共享的固定随机矩阵文件（128×7，%.17g 文本，double 可逐位往返）。
# 两语言都从这里加载同一份 R，避免 numpy PCG64 与 C++ mt19937 生成不同的 R，
# 进而导致 SimHash 指纹、相似度、图边（ρ/√S 特征）不一致。
R_PATH = os.path.join(HERE, "random_matrix_R.csv")


def make_random_matrix(seed=R_SEED):
    """固定随机矩阵 R∈ℝ^{128×7}。优先加载共享文件（与 C++ 逐位一致）；
    缺失时才用固定种子生成（同分布、同参数，但不保证与 C++ 逐位相同）。"""
    if os.path.exists(R_PATH):
        R = np.loadtxt(R_PATH, delimiter=",")
        assert R.shape == (common.FP_BITS, common.NUM_SYMBOLS)
        return R
    r = common.rng(seed)
    return r.standard_normal((common.FP_BITS, common.NUM_SYMBOLS))


def project_bits(V, R):
    """
    V: (..., 7) 计数向量；R: (128,7)。
    返回 128 位 0/1 指纹（投影 >0 记 1）。
    """
    proj = np.asarray(V, dtype=np.float64) @ R.T
    return (proj > 0).astype(np.uint8)


def pack_fingerprints(V, R):
    """对 (num_win, 7) 计数矩阵生成打包指纹 (num_win, 16) uint8。"""
    bits = project_bits(V, R)                 # (num_win, 128)
    return np.packbits(bits, axis=-1)         # (num_win, 16)


def main():
    data = common.load_npz("04_count_vectors.npz", allow_pickle=True)
    V = {"32": data["V32"], "64": data["V64"], "128": data["V128"]}

    R = make_random_matrix()
    common.banner("05 固定随机矩阵 R 与 128 位 SimHash 指纹")
    print(f"R 形状 = {R.shape}，固定种子 R_SEED = {R_SEED}")
    print("R 前 3 行 × 7 列:")
    for i in range(3):
        print("  ", " ".join(f"{v:+.3f}" for v in R[i]))

    # 玩具：一个典型合法窗口计数向量
    v_toy = np.array([[1, 0, 2, 18, 8, 3, 0]])
    bits = project_bits(v_toy, R)[0]
    packed = np.packbits(bits)
    common.sub_banner("玩具 v = [1,0,2,18,8,3,0] 的 128 位指纹")
    s = "".join(str(b) for b in bits)
    print("  前 32 位:", s[:32])
    print("  后 32 位:", s[-32:])
    print(f"  共 128 位，其中 1 的个数 = {int(bits.sum())}，打包为 {packed.shape[0]} 字节")

    # 全量（每尺度每段打包）
    packed = {}
    for key in ["32", "64", "128"]:
        num_seg, nw, _ = V[key].shape
        H = np.stack([pack_fingerprints(V[key][i], R) for i in range(num_seg)])
        packed[key] = H
        print(f"  win={key:>3s}: 指纹数组 {H.shape}（段×窗口×16字节，对应128位）")

    path = common.save_npz(
        "05_fingerprints.npz",
        R=R, H32=packed["32"], H64=packed["64"], H128=packed["128"],
        seg_label=data["seg_label"], seg_channel=data["seg_channel"],
        seg_conn_id=data["seg_conn_id"], seg_index=data["seg_index"],
        channel_names=data["channel_names"],
    )
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
