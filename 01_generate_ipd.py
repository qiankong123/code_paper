# -*- coding: utf-8 -*-
"""
01_generate_ipd.py —— 生成合成 IPD 数据集（合法流量 + 5 类隐蔽定时信道）
================================================================
论文使用公开 NCTC 数据集（Li et al.），本复现无法附带该数据集，
因此用统计模型【合成】结构相近的 IPD（包间时间间隔）序列，用于打通并
验证整条计算链路。所有分布参数均为教学近似，非论文真实数据。

标签约定：
  label = 0 合法（LEGIT，HTTP/HTTPS/SSH）
  label = 1 隐蔽（IPCTC / LNCTC / TRCTC / Jitterbug / EpsilonKlibur）

输出：outputs/01_connections.npz
  ipd_list     : object 数组，每个元素是一条连接的 IPD（长度约 8000）
  labels       : 每条连接 0/1
  channel_id   : 0=合法, 1..5 = 五类隐蔽信道
  channel_names: 名称数组
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)
rng = common.rng
SEED = common.SEED

# 每条连接长度（约 8000 个 IPD），后续 02 脚本会切成 L=2000 的段
CONN_LEN = 8000
# 每种（协议/信道）生成的连接条数
N_CONN_PER_KIND = 8

# channel_id 与名称
CH_LEGIT, CH_IPCTC, CH_LNCTC, CH_TRCTC, CH_JITTER, CH_EPS = 0, 1, 2, 3, 4, 5
CHANNEL_NAMES = ["LEGIT", "IPCTC", "LNCTC", "TRCTC", "Jitterbug", "EpsilonKlibur"]


# ----------------------------------------------------------------------
# 合法流量：对 log(IPD) 建 AR(1) 自相关过程，再叠加少量"突发"极小间隔。
# 网络 IPD 为正、右偏、带自相关，AR(1) on log-scale 是常见的近似模型。
# ----------------------------------------------------------------------
LEGIT_PARAMS = {
    "HTTP":  dict(mu=np.log(0.12), phi=0.60, sigma=0.065, burst=0.003),
    "HTTPS": dict(mu=np.log(0.10), phi=0.58, sigma=0.060, burst=0.002),
    "SSH":   dict(mu=np.log(0.09), phi=0.55, sigma=0.055, burst=0.001),
}


def legit_connection(n, r, protocol):
    p = LEGIT_PARAMS[protocol]
    mu, phi, sigma = p["mu"], p["phi"], p["sigma"]
    eps = r.normal(0.0, 1.0, n)
    x = np.empty(n); x[0] = mu
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * eps[t]
    ipd = np.exp(x)
    # 突发：随机少量位置出现密集包（IPD 被压到很小）
    nb = int(n * p["burst"])
    if nb > 0:
        idx = r.integers(1, n, nb)
        ipd[idx] = ipd[idx] * r.uniform(0.30, 0.60, nb)
    return ipd


# ----------------------------------------------------------------------
# 在合法基线之上叠加各类隐蔽定时编码。
# 思路：在等间隔"符号位置"按比特/符号修改对应 IPD（乘一个倍率），
# 这会在相邻变化率 Δ 上留下有规律的跳变（论文 Table 1 针对的正是它）。
# ----------------------------------------------------------------------
def covert_encode(base, r, channel):
    ipd = base.copy()
    n = len(ipd)

    if channel == "IPCTC":
        # 经典 IPCTC：两种间隔代表 0/1，倍率差异大 -> Δ 出现 ±2/±3（sharp）
        T = 8
        bits = r.integers(0, 2, n // T)
        for j, b in enumerate(bits):
            ipd[j * T] = base[j * T] * (0.45 if b == 0 else 2.20)

    elif channel == "LNCTC":
        # LNCTC：多档（4 档）间隔查表，Δ 中等且符号集中
        T = 6
        mult = np.array([0.50, 0.75, 1.33, 2.00])
        sym = r.integers(0, 4, n // T)
        for j, s in enumerate(sym):
            ipd[j * T] = base[j * T] * mult[s]

    elif channel == "TRCTC":
        # TRCTC：周期性固定重放扰动 -> 强周期节奏
        T = 12
        for i in range(0, n, T):
            ipd[i] = base[i] * 2.00

    elif channel == "Jitterbug":
        # Jitterbug：把 IPD 向最近对齐点微调。这里建模为周期性微缩+下一点回摆，
        # 幅度小（Δ≈−0.09，平滑后稳定落在 ±1，窗口级仍高相似）、节奏固定（T=4），
        # 规律 ±1 让循环特征 L_cy 明显偏小、指纹相似度又高（对应论文"高相似却隐蔽"）。
        T = 4
        for i in range(0, n, T):
            ipd[i] = base[i] * 0.91      # 周期微缩（Δ≈−0.09），下一点回摆（+）

    elif channel == "EpsilonKlibur":
        # ε-klibur：保证与正常分布的统计距离很小（最难检测）。这里建模为比
        # Jitterbug 更弱、更稀疏（周期 T=8、幅度仅 ±5%）的确定性微调+回摆；
        # 平滑后只在边缘呈稀疏规律 ±1，窗口指纹几乎与合法无异（最高相似度），
        # 仅在 L_cy 上留下轻微、一致的偏稀疏节奏，供多尺度统计弱检出。
        T = 8
        for i in range(0, n, T):
            ipd[i] = base[i] * 0.95      # 稀疏微缩（Δ≈−0.05），下一点回摆

    return ipd


def build_dataset():
    ipd_list, labels, channels, protocols = [], [], [], []
    r = rng(SEED)

    # 合法：3 协议 × N_CONN_PER_KIND 条
    for proto in common.LEGIT_PROTOCOLS:
        for k in range(N_CONN_PER_KIND):
            base = legit_connection(CONN_LEN, r, proto)
            ipd_list.append(base)
            labels.append(0); channels.append(CH_LEGIT); protocols.append(proto)

    # 隐蔽：5 类 × N_CONN_PER_KIND 条（先抽一条合法基线，再编码）
    for ch_id, ch_name in [(CH_IPCTC, "IPCTC"), (CH_LNCTC, "LNCTC"),
                           (CH_TRCTC, "TRCTC"), (CH_JITTER, "Jitterbug"),
                           (CH_EPS, "EpsilonKlibur")]:
        for k in range(N_CONN_PER_KIND):
            proto = common.LEGIT_PROTOCOLS[k % 3]
            base = legit_connection(CONN_LEN, r, proto)
            ipd_list.append(covert_encode(base, r, ch_name))
            labels.append(1); channels.append(ch_id); protocols.append(proto)

    return (np.array(ipd_list, dtype=object),
            np.array(labels, dtype=np.int64),
            np.array(channels, dtype=np.int64),
            np.array(protocols, dtype=object))


if __name__ == "__main__":
    common.banner("01 生成合成 IPD 数据集（合法 + 5 类隐蔽定时信道）")
    ipd_list, labels, channels, protocols = build_dataset()
    print(f"连接总数 = {len(ipd_list)}，每条长度约 {CONN_LEN} IPD")

    sub = common.sub_banner
    sub("各类连接条数与示例 IPD（前 8 个）")
    for cid, name in enumerate(CHANNEL_NAMES):
        idx = np.where(channels == cid)[0]
        ex = ipd_list[idx[0]]
        head = ", ".join(f"{v:.4f}" for v in ex[:8])
        print(f"  {name:14s} n={len(idx):2d}  label={int(labels[idx[0]])}  示例IPD: [{head}, ...]")

    sub("编码后相邻变化率 |Δ| 的粗略对比（取每条连接前 2000 个 IPD）")
    for cid, name in enumerate(CHANNEL_NAMES):
        idx = np.where(channels == cid)[0]
        ipd = ipd_list[idx[0]]
        delta = np.diff(ipd[:2001]) / ipd[:2000]
        sharp = np.mean(np.abs(delta) >= 0.06)   # 落在 ±2 及以上的比例
        print(f"  {name:14s} |Δ|≥0.06(±2 及以上)占比 = {sharp*100:5.1f}%   "
              f"mean|Δ| = {np.mean(np.abs(delta)):.4f}")

    path = common.save_npz(
        "01_connections.npz",
        ipd_list=ipd_list, labels=labels, channel_id=channels,
        protocols=protocols, channel_names=np.array(CHANNEL_NAMES),
    )
    common.set_data_source("synthetic")
    print(f"\n已保存：{path}")
    print(f"数据来源标记：{common.out_path('data_source.txt')} = synthetic")
