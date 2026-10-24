# -*- coding: utf-8 -*-
"""
03_change_rate_symbol.py —— 相邻 IPD 变化率 Δ 与七级符号映射（Table 1）
================================================================
论文 §3.2.1 / Table 1：
  相邻 IPD 的相对变化率  Δ_i = (IPD_{i+1} − IPD_i) / IPD_i
  再按 Table 1 离散到 −3…+3 七个符号：
      +3  Δ ≥ 0.15          （sharp rise，多见于编码/极端网络事件）
      +2  0.06 ≤ Δ < 0.15   （clear rise）
      +1  0.02 ≤ Δ < 0.06   （slight rise，多见于合法流量）
       0  −0.02 < Δ < 0.02  （基本平稳）
      −1  −0.06 < Δ ≤ −0.02 （slight drop）
      −2  −0.15 < Δ ≤ −0.06 （clear drop）
      −3  Δ ≤ −0.15          （sharp drop）

每段 IPD 长度 2000，得到长度 1999 的 Δ 序列与符号序列。

输出：outputs/03_symbols.npz
  seg_delta   : (num_seg, 1999) 变化率
  seg_symbols : (num_seg, 1999) 七级符号
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)


def change_rate(ipd):
    """相邻 IPD 的相对变化率：Δ_i = (IPD_{i+1} − IPD_i)/IPD_i。"""
    ipd = np.asarray(ipd, dtype=np.float64)
    return np.diff(ipd) / ipd[:-1]


def map_symbol(delta):
    """按 Table 1 把 Δ 映射到 −3…+3。"""
    d = np.asarray(delta)
    s = np.zeros(d.shape, dtype=np.int64)
    s = np.where(d >= 0.15, 3, s)
    s = np.where((d >= 0.06) & (d < 0.15), 2, s)
    s = np.where((d >= 0.02) & (d < 0.06), 1, s)
    s = np.where((d > -0.02) & (d < 0.02), 0, s)
    s = np.where((d > -0.06) & (d <= -0.02), -1, s)
    s = np.where((d > -0.15) & (d <= -0.06), -2, s)
    s = np.where(d <= -0.15, -3, s)
    return s


def main():
    data = common.load_npz("02_segments.npz", allow_pickle=True)
    seg_ipd = data["seg_ipd"]; seg_channel = data["seg_channel"]
    channel_names = data["channel_names"]

    common.banner("03 相邻变化率 Δ 与七级符号映射（Table 1）")

    # 玩具：给定一串 IPD，手算 Δ 与符号
    toy = np.array([100.0, 103.0, 103.2, 120.0, 95.0, 100.0])
    td = change_rate(toy)
    ts = map_symbol(td)
    print("玩具 IPD:", toy)
    print("  Δ  :", " ".join(f"{v:+.3f}" for v in td))
    print("  符号:", " ".join(f"{v:+d}" for v in ts))
    print("  （例：120/103.2−1 = +0.163 → +3；95/120−1 = −0.208 → −3）")

    # 全量
    seg_delta = np.array([change_rate(s) for s in seg_ipd])
    seg_symbols = np.array([map_symbol(d) for d in seg_delta])

    common.sub_banner("各信道符号分布（−3…+3 占比 %）——合法分散、编码集中")
    header = "信道           " + "".join(f"{k:>7d}" for k in common.SYMBOL_LEVELS)
    print(header)
    for cid, name in enumerate(channel_names):
        idx = np.where(seg_channel == cid)[0]
        syms = seg_symbols[idx]
        freq = np.array([(syms == k).mean() for k in common.SYMBOL_LEVELS]) * 100
        print(f"{str(name):14s}" + "".join(f"{v:7.1f}" for v in freq))

    print(f"\n输出形状：seg_delta={seg_delta.shape}, seg_symbols={seg_symbols.shape}")
    path = common.save_npz(
        "03_symbols.npz",
        seg_delta=seg_delta, seg_symbols=seg_symbols,
        seg_label=data["seg_label"], seg_channel=seg_channel,
        seg_conn_id=data["seg_conn_id"], seg_index=data["seg_index"],
        channel_names=channel_names,
    )
    print(f"已保存：{path}")


if __name__ == "__main__":
    main()
