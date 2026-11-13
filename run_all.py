# -*- coding: utf-8 -*-
"""
run_all.py —— 顺序执行 01~14，完整复现 QuadFP-Cascade 计算链路
================================================================
用法：
    python run_all.py                     # 默认：合成数据（01_generate_ipd.py）
    python run_all.py synthetic           # 显式：合成数据
    python run_all.py caida               # 真实 CAIDA/NCTC，bignet（广域网/大网）
    python run_all.py caida labnet        # 真实 CAIDA/NCTC，labnet（实验室网）
    python run_all.py 09                  # 旧用法：从第 9 步开始（需上游 outputs 已存在）

无论使用哪种数据源，02~14 完全相同；区别仅在第一步（生成合成 IPD 或 加载真实 CSV）。
合成生成器 01_generate_ipd.py 始终保留；真实数据由 01_load_caida.py 加载。

中间产物写入 outputs/ 目录的 .npz 文件；每个脚本会打印关键计算结果。
"""

import os
import sys
import time
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))

# 02~14 与数据源无关的公共步骤
STEPS_02_14 = [
    "02_preprocess_segment.py",
    "03_change_rate_symbol.py",
    "04_multiscale_windows.py",
    "05_random_matrix_fingerprint.py",
    "06_hamming_similarity.py",
    "07_robust_aggregation.py",
    "08_lsh_candidate.py",
    "09_threshold_graph.py",
    "10_gnn_gcn.py",
    "11_edge_fusion.py",
    "12_graph_features.py",
    "13_two_level_classifier.py",
    "14_evaluation.py",
]
ALL_SCRIPTS = ["01_generate_ipd.py"] + STEPS_02_14


def run_script(script, script_args=None):
    """以 __main__ 运行单个脚本，并临时注入它的 sys.argv。"""
    old_argv = sys.argv
    sys.argv = [script] + (script_args or [])
    try:
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main():
    args = sys.argv[1:]

    # 旧用法：从第 N 步开始
    if args and args[0].isdigit():
        start = int(args[0])
        files = ALL_SCRIPTS[start - 1:] if 1 <= start <= 14 else ALL_SCRIPTS
        for f in files:
            step(f, [])
        return

    # 选择数据源
    if args and args[0].lower() == "caida":
        subset = args[1].strip().lower() if len(args) > 1 else "bignet"
        if subset not in ("bignet", "labnet"):
            print(f"未知子集 {subset}，应为 bignet 或 labnet"); sys.exit(1)
        k_arg = args[2].strip() if len(args) > 2 else ""
        load_args = [subset] + ([k_arg] if k_arg else [])
        print(f"### 数据源：真实 CAIDA/NCTC（subset={subset}"
              f"{('，每类 k=' + k_arg) if k_arg else '，默认 k'}），加载器 01_load_caida.py ###")
        step("01_load_caida.py", load_args)
    else:
        print("### 数据源：合成数据，生成器 01_generate_ipd.py ###")
        step("01_generate_ipd.py", [])

    for f in STEPS_02_14:
        step(f, [])

    print("\n" + "#" * 70)
    print("全部步骤执行完毕。中间产物见 outputs/ 目录。")
    print("#" * 70)


def step(f, script_args):
    t0 = time.time()
    run_script(f, script_args)
    print(f"\n>>> {f} 完成，用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
