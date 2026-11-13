# -*- coding: utf-8 -*-
"""
01_load_caida.py —— 加载【真实 CAIDA / NCTC 数据】（替代合成 IPD，01_generate_ipd.py 保留）
================================================================
数据来源：data/<subset>/ 下的单列 CSV（表头 "IPDs"，每行一个包间时间间隔，单位秒）。
  subset = bignet（Backbone 广域网/大网）或 labnet（Lab 实验室网）。
  论文（沿用 Li et al. 的 NCTC 数据集）在 Lab 与 Backbone 两个环境分别评估，
  对应 Table 3 的两类参数配置：
     http.csv / https.csv / ssh.csv ........ 合法流量（三种协议）
     IPCTC_1.csv,IPCTC_2.csv .............. IPCTC（不同 timing interval）
     LNCTC_1.csv,LNCTC_2.csv .............. LNCTC（modulo ω=5 / 10）
     TRCTC_1.csv,TRCTC_2.csv .............. TRCTC（两组参数）
     Jitterbug_1.csv,Jitterbug_2.csv ...... Jitterbug（modulo ω=5 / 10）
  注意：本数据集【不含 ε-xlibur】（论文 Table 4 的第 5 类信道），
       故 channel_id=5 的样本数为 0，下游评估会自动跳过该类。

为什么要把 CSV 切成"连接"：
  每个 CSV 是一整条 trace、约 200 万个 IPD，而后续 02 按"每条连接 L=2000、25% 重叠"
  切段。这里把长 trace 按 CONN_LEN=8000 切成【互不重叠】的连接块，与合成版
  01_generate_ipd.py 输出的每条连接 8000 IPD 完全对齐，因此 02~14 无需改动。
  为控制规模（08/09 是 O(N²)），每类用【确定性均匀抽样】取 K 个连接，覆盖整条 trace。

数据清洗：
  真实正常流量存在少量 0 / 负 IPD（抓包时间戳误差），会导致变化率 Δ 除零。
  统一裁剪到 MIN_IPD=1e-6 s（1 微秒）下限。

用法：
  python 01_load_caida.py [bignet|labnet] [每类抽样数K]   （默认 bignet，K=16）

输出：outputs/01_connections.npz（与 01_generate_ipd.py 同名、同 schema）
  ipd_list / labels / channel_id / protocols / channel_names
  另写 outputs/data_source.txt 标记数据来源（供 14 标题显示）。
"""

import os
import sys
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

# ----------------------------------------------------------------------
# 与合成版一致的结构参数
# ----------------------------------------------------------------------
CONN_LEN = 8000          # 每个"连接"块长度，02 会把它切成 5 段（L=2000，步进1500）
DEFAULT_K = 16           # 每类（每种合法协议 / 每类隐蔽信道）默认抽取的连接条数；
                         # 可用第 2 个命令行参数覆盖。真实数据每类有 129~250 块可用，
                         # 取 16 让 13 步 GBDT 训练样本翻倍（每类约 50 段），阈值更稳。
MIN_IPD = 1e-6           # IPD 下限（1 微秒），裁剪 0/负值

# channel_id 与名称（与合成版保持同一套枚举；ε 本数据无样本）
CH_LEGIT, CH_IPCTC, CH_LNCTC, CH_TRCTC, CH_JITTER, CH_EPS = 0, 1, 2, 3, 4, 5
CHANNEL_NAMES = ["LEGIT", "IPCTC", "LNCTC", "TRCTC", "Jitterbug", "EpsilonKlibur"]

DATA_DIR = os.path.join(HERE, "data")

# 合法文件：文件名 -> 协议
LEGIT_FILES = [
    ("http.csv",  "HTTP"),
    ("https.csv", "HTTPS"),
    ("ssh.csv",   "SSH"),
]
# 隐蔽信道：channel_id -> [配置文件1, 配置文件2]（论文 Table 3 的 _1/_2）
COVERT_FILES = {
    CH_IPCTC:  ["IPCTC_1.csv", "IPCTC_2.csv"],
    CH_LNCTC:  ["LNCTC_1.csv", "LNCTC_2.csv"],
    CH_TRCTC:  ["TRCTC_1.csv", "TRCTC_2.csv"],
    CH_JITTER: ["Jitterbug_1.csv", "Jitterbug_2.csv"],
}


# ----------------------------------------------------------------------
# 读取单列 IPD CSV
# ----------------------------------------------------------------------
def read_ipd_csv(path):
    """读单列 CSV（第一行为表头），返回 float64 一维 numpy 数组。"""
    # 优先 pandas（200 万行也很快）；缺失时回退标准库。
    try:
        import pandas as pd
        df = pd.read_csv(path, header=0)
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    except Exception:
        vals = []
        with open(path, "r", newline="") as f:
            f.readline()           # 表头
            for line in f:
                s = line.strip()
                if s:
                    vals.append(float(s))
        x = np.asarray(vals, dtype=np.float64)
    # 清洗：非有限值 -> 下限；0/负值 -> 下限
    x = np.where(np.isfinite(x), x, MIN_IPD)
    x = np.maximum(x, MIN_IPD)
    return x


def n_available_blocks(n_rows, conn_len=CONN_LEN):
    """长度 n_rows 的 trace 能切成多少个不重叠的 conn_len 连接块（尾部丢弃）。"""
    return n_rows // conn_len


def uniform_pick(n_total, k):
    """
    从 [0, n_total) 个块中【确定性均匀】抽取 k 个块索引（含首尾）。
    纯算术、无随机数：idx_j = round(j*(n_total-1)/(k-1))，
    用 floor(x+0.5) 实现四舍五入，保证 Python / C++ 结果完全一致。
    """
    if k <= 0:
        return []
    if n_total <= k:
        return list(range(n_total))
    if k == 1:
        return [0]
    return [int(j * (n_total - 1) / (k - 1) + 0.5) for j in range(k)]


def allocate_quotas(total, caps):
    """把 total 个名额按顺序分到 m 个文件，受每个文件容量 caps 限制。"""
    m = len(caps)
    base = total // m
    rem = total % m
    quotas = [base + (1 if j < rem else 0) for j in range(m)]
    for i in range(m):
        if quotas[i] > caps[i]:
            extra = quotas[i] - caps[i]
            quotas[i] = caps[i]
            for j in range(m):           # 多余名额补给仍有容量的文件
                if j != i and quotas[j] < caps[j]:
                    add = min(extra, caps[j] - quotas[j])
                    quotas[j] += add
                    extra -= add
    return quotas


def build_scene(subset, k):
    scene_dir = os.path.join(DATA_DIR, subset)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"找不到数据目录：{scene_dir}")

    ipd_list, labels, channels, protocols = [], [], [], []
    per_file_info = []

    def take_blocks(x, want, label, ch, proto, tag):
        nblk = n_available_blocks(len(x))
        picks = uniform_pick(nblk, min(want, nblk))
        got = 0
        for b in picks:
            seg = x[b * CONN_LEN:(b + 1) * CONN_LEN].astype(np.float64)
            if seg.shape[0] < CONN_LEN:
                continue
            ipd_list.append(seg)
            labels.append(label); channels.append(ch); protocols.append(proto)
            got += 1
        per_file_info.append((subset, tag, len(x), nblk, want, got))
        return got

    # ---- 合法：三种协议（按 http/https/ssh 顺序），每协议 k 条 ----
    for fn, proto in LEGIT_FILES:
        p = os.path.join(scene_dir, fn)
        if not os.path.exists(p):
            per_file_info.append((subset, fn, 0, 0, k, 0))
            print(f"  [警告] 缺少合法文件 {fn}，跳过（协议 {proto}）。")
            continue
        x = read_ipd_csv(p)
        take_blocks(x, k, 0, CH_LEGIT, proto, f"{proto}:{fn}")

    # ---- 隐蔽：每类 k 条，在两个配置文件间均分 ----
    for ch_id, files in COVERT_FILES.items():
        # 先读各文件、统计容量
        xs, caps = [], []
        for fn in files:
            p = os.path.join(scene_dir, fn)
            if not os.path.exists(p):
                xs.append(None); caps.append(0)
                print(f"  [警告] 缺少隐蔽文件 {fn}，跳过。")
            else:
                x = read_ipd_csv(p)
                xs.append(x); caps.append(n_available_blocks(len(x)))
        quotas = allocate_quotas(k, caps)
        for fn, x, want in zip(files, xs, quotas):
            if x is None or want <= 0:
                continue
            take_blocks(x, want, 1, ch_id, "NA", f"{CHANNEL_NAMES[ch_id]}:{fn}")

    return (np.array(ipd_list, dtype=object),
            np.array(labels, dtype=np.int64),
            np.array(channels, dtype=np.int64),
            np.array(protocols, dtype=object),
            per_file_info)


def main():
    subset = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "bignet"
    if subset not in ("bignet", "labnet"):
        print(f"未知子集 {subset}，应为 bignet 或 labnet"); sys.exit(1)
    k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_K

    common.banner(f"01 加载真实 CAIDA/NCTC 数据（subset={subset}，每类 k={k}）")
    print(f"数据目录 = {os.path.join(DATA_DIR, subset)}")
    print(f"每条连接长度 = {CONN_LEN} IPD；每类抽取 {k} 条；IPD 下限 {MIN_IPD:g}s")

    ipd_list, labels, channels, protocols, info = build_scene(subset, k)

    common.sub_banner("各文件可用块数与实际抽取")
    print(f"{'文件':24s}{'原始IPD':>12s}{'可用块':>8s}{'计划':>6s}{'实际':>6s}")
    for _subset, tag, nrows, nblk, want, got in info:
        print(f"{tag:24s}{nrows:12d}{nblk:8d}{want:6d}{got:6d}")

    common.sub_banner("各类连接条数与示例 IPD（前 8 个）")
    for cid, name in enumerate(CHANNEL_NAMES):
        idx = np.where(channels == cid)[0]
        if len(idx) == 0:
            print(f"  {name:14s} n= 0  （本数据集无该类样本）")
            continue
        ex = ipd_list[idx[0]]
        head = ", ".join(f"{v:.5f}" for v in ex[:8])
        print(f"  {name:14s} n={len(idx):2d}  label={int(labels[idx[0]])}  示例IPD: [{head}, ...]")

    n_legit = int((labels == 0).sum()); n_cov = int((labels == 1).sum())
    print(f"\n连接总数 = {len(ipd_list)}（合法 {n_legit} / 隐蔽 {n_cov}），"
          f"预计 02 切段后每连接 5 段、共 {len(ipd_list)*5} 段")

    path = common.save_npz(
        "01_connections.npz",
        ipd_list=ipd_list, labels=labels, channel_id=channels,
        protocols=protocols, channel_names=np.array(CHANNEL_NAMES),
    )
    src_tag = f"CAIDA/{subset}"
    with open(common.out_path("data_source.txt"), "w", encoding="utf-8") as f:
        f.write(src_tag + "\n")
    print(f"已保存：{path}")
    print(f"数据来源标记：{common.out_path('data_source.txt')} = {src_tag}")


if __name__ == "__main__":
    main()
