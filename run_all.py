# -*- coding: utf-8 -*-
"""
run_all.py —— 顺序执行 01~14，完整复现 QuadFP-Cascade 计算链路
================================================================
用法：
    python run_all            # 从 01 到 14 顺序执行
    python run_all 09         # 从 09 开始（但需要上游 outputs 已存在，建议从头跑）
    python 10_gnn_gcn.py      # 也可单独运行某个脚本（每个脚本都自带演示）

中间产物写入 outputs/ 目录的 .npz 文件；每个脚本会打印关键计算结果。
"""

import os
import sys
import time
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "01_generate_ipd.py",
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


def main():
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
        files = SCRIPTS[start - 1:] if 1 <= start <= 14 else SCRIPTS
    else:
        files = SCRIPTS
    for f in files:
        t0 = time.time()
        runpy.run_path(os.path.join(HERE, f), run_name="__main__")
        print(f"\n>>> {f} 完成，用时 {time.time()-t0:.1f}s")
    print("\n" + "#" * 70)
    print("全部步骤执行完毕。中间产物见 outputs/ 目录。")
    print("#" * 70)


if __name__ == "__main__":
    main()
