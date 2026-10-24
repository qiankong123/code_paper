# -*- coding: utf-8 -*-
"""
02_preprocess_segment.py —— 指数平滑 + L=2000、25% 重叠分段
================================================================
论文 §3.2.1：
  "the IPD sequence is segmented using a sliding window of length L=2000,
   with 25% overlap ... For each segment, exponential smoothing is applied
   to suppress the amplification of change rates due to link jitter and
   transient spikes."

做两件事：
  1) 对每条连接做指数平滑（EMA），抑制链路抖动/瞬态尖峰把变化率放大；
  2) 用 L=2000、步进 1500（25% 重叠）把连接切成"段"——段就是后面图的节点。

输出：outputs/02_segments.npz
  seg_ipd        : (num_seg, 2000) 平滑后的段
  seg_label/channel/conn_id/seg_index : 每个段的元信息
"""

import numpy as np
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

# 指数平滑系数（论文未给，取教学值 α=0.3：α 越大越贴近原始值）
ALPHA = 0.30


def exponential_smoothing(x, alpha=ALPHA):
    """一阶指数平滑：s_t = α·x_t + (1−α)·s_{t−1}。"""
    x = np.asarray(x, dtype=np.float64)
    s = np.empty_like(x)
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = alpha * x[t] + (1.0 - alpha) * s[t - 1]
    return s


def main():
    data = common.load_npz("01_connections.npz", allow_pickle=True)
    ipd_list = data["ipd_list"]; labels = data["labels"]; channels = data["channel_id"]

    common.banner("02 指数平滑 + 分段（L=2000，25% 重叠）")
    # 先演示平滑对尖峰的抑制
    demo = np.array([10, 10, 10, 10, 30, 10, 10, 10], dtype=float)  # 一个尖峰 30
    sm = exponential_smoothing(demo)
    print("平滑去尖峰演示（尖峰 30）:")
    print("  原始:", " ".join(f"{v:6.1f}" for v in demo))
    print("  平滑:", " ".join(f"{v:6.1f}" for v in sm))

    seg_ipd, seg_label, seg_channel, seg_conn, seg_idx = [], [], [], [], []
    seg_len = common.SEG_LEN; step = common.SEG_STEP

    for ci, ipd in enumerate(ipd_list):
        smoothed = exponential_smoothing(np.asarray(ipd, dtype=np.float64))
        segs = common.segment_series(smoothed, seg_len, step)
        for si, seg in enumerate(segs):
            seg_ipd.append(seg)
            seg_label.append(labels[ci]); seg_channel.append(channels[ci])
            seg_conn.append(ci); seg_idx.append(si)

    seg_ipd = np.stack(seg_ipd)
    seg_label = np.array(seg_label); seg_channel = np.array(seg_channel)
    seg_conn = np.array(seg_conn); seg_idx = np.array(seg_idx)

    common.sub_banner(f"分段结果：每连接长度 8000，L={seg_len}，步进={step}")
    n_per_conn = common.count_windows(8000, seg_len, step)
    print(f"  每连接段数 = floor((8000−{seg_len})/{step})+1 = {n_per_conn}")
    print(f"  连接数 = {len(ipd_list)}，段总数（图的节点数）= {seg_ipd.shape[0]}")
    print(f"  段张量形状 seg_ipd = {seg_ipd.shape}（每段 {seg_len} 个 IPD）")
    print(f"  合法段 = {int((seg_label==0).sum())}，隐蔽段 = {int((seg_label==1).sum())}")
    for cid, name in enumerate(data["channel_names"]):
        n = int((seg_channel == cid).sum())
        print(f"    {name:14s} {n} 段")
    print(f"  相邻段重叠 IPD 数 = {seg_len-step}（{int(common.SEG_OVERLAP*100)}%）")

    path = common.save_npz(
        "02_segments.npz",
        seg_ipd=seg_ipd, seg_label=seg_label, seg_channel=seg_channel,
        seg_conn_id=seg_conn, seg_index=seg_idx,
        channel_names=data["channel_names"],
    )
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
