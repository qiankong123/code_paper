# -*- coding: utf-8 -*-
"""
make_shared_constants.py —— 生成 Python / C++ 两端共享的固定随机常量
================================================================================

为保证 Python（numpy PCG64）与 C++（mt19937）两套实现"逐位一致"，下列三处随机
量不能各自生成，必须由本脚本统一导出为 CSV（%.17g、无表头、逗号分隔），两端在
对应步骤加载：

  1) random_matrix_R.csv   R ∈ ℝ^{128×7}  第 05 步 SimHash 随机投影矩阵
                           生成器：numpy default_rng(SEED+7).standard_normal(128,7)
  2) gcn_W0_init.csv       W0 ∈ ℝ^{2×16}   第 10 步两层 GCN 第 1 层初始权重（×0.5）
  3) gcn_W1_init.csv       W1 ∈ ℝ^{16×16}  第 10 步两层 GCN 第 2 层初始权重（×0.3）
                           生成器：numpy default_rng(SEED+11)，先 W0 后 W1

C++ 端在 cpp/ 目录下运行，以 "../" 前缀读取同一份 CSV；Python 端直接读当前目录。

用法：
    python make_shared_constants.py
若 CSV 已存在，脚本只做"重新生成值与现有文件逐元素一致"的校验，不会覆盖（防止
误改种子导致跨语言结果漂移）；如需重新生成，先手动删除这三个 CSV 再运行。
"""
import os
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_common():
    spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate():
    common = _load_common()
    # 05 步 SimHash 随机矩阵 R：rng(SEED+7)
    r_r = common.rng(common.SEED + 7)
    R = r_r.standard_normal((common.FP_BITS, common.NUM_SYMBOLS))
    # 10 步 GCN 初始权重：rng(SEED+11)，W0 先、W1 后（与 10_gnn_gcn.py 初始化顺序一致）
    h_dim, d_in = 16, 2
    r_g = common.rng(common.SEED + 11)
    W0 = r_g.standard_normal((d_in, h_dim)) * 0.5
    W1 = r_g.standard_normal((h_dim, h_dim)) * 0.3
    return R, W0, W1


def _write_or_verify(path, arr):
    if os.path.exists(path):
        old = np.loadtxt(path, delimiter=",")
        old = np.atleast_2d(old)
        if old.shape != arr.shape:
            raise SystemExit(f"[错误] 现有 {path} 形状 {old.shape} 与新生成 {arr.shape} 不一致，已拒绝覆盖。")
        if not np.array_equal(old, arr):
            maxd = np.max(np.abs(old - arr))
            raise SystemExit(f"[错误] 现有 {path} 与按固定种子重新生成的结果不一致（最大差 {maxd:.3e}），"
                             f"已拒绝覆盖。若确认要换种子，请先手动删除该 CSV。")
        print(f"[校验一致] {os.path.basename(path):20s} shape={arr.shape}，保留现有文件。")
    else:
        np.savetxt(path, arr, delimiter=",", fmt="%.17g")
        back = np.atleast_2d(np.loadtxt(path, delimiter=","))
        assert np.array_equal(back, arr), "写盘读回不一致！"
        print(f"[新建写入] {os.path.basename(path):20s} shape={arr.shape}，写盘读回一致。")


def main():
    R, W0, W1 = generate()
    _write_or_verify(os.path.join(HERE, "random_matrix_R.csv"), R)
    _write_or_verify(os.path.join(HERE, "gcn_W0_init.csv"), W0)
    _write_or_verify(os.path.join(HERE, "gcn_W1_init.csv"), W1)
    print("\n共享常量就绪：Python 与 C++ 将加载同一份 R / W0 / W1，保证跨语言逐位一致。")


if __name__ == "__main__":
    main()
