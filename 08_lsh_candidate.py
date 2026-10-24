# -*- coding: utf-8 -*-
"""
08_lsh_candidate.py —— LSH 分 band 粗筛候选对 + 召回率推导
================================================================
论文 §3.2.2：
  "To control the number of candidate pairs, a low-overhead LSH bucketing
   step is first used to generate a candidate set."

窗口级指纹 + 位置对齐 + 多点碰撞证据（避免"段内任意窗口必然撞桶"）：
  每个段有上百个窗口指纹。若规定"段内任意窗口撞桶即候选"，上百个窗口会把
  碰撞概率放大到必然，无法压缩。这里桶键含【窗口位置】：
      key = (band, 16位签名, 位置 bin)
  两个段必须在【同位置区间、同一个 band、16 位完全相同】才记 1 次碰撞证据；
  允许 ±MAX_SHIFT 个位置的容差（隐蔽编码可能只造成小相位偏移）。
  段对累计碰撞证据数 ≥ MIN_EVIDENCE（成片相似，而非单点偶然）才进候选。
  这与 07 的"位移对齐 + top 小比例"思想一致，且在窗口级进行，能反映模式差异。

band 配置（教学，论文未给）：8 band × 16 位 = 128 位。

单点召回率理论（回答"一对对齐窗口在某 band 撞桶的概率"）：
      R(s) = 1 − (1 − s^16)^8
本脚本先验证该公式（蒙特卡洛），再统计实际段对的碰撞证据数。

输出：outputs/08_candidates.npz（candidates, evidence）
"""

import numpy as np
import importlib.util, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("common", os.path.join(HERE, "00_common.py"))
common = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(common)

N_BANDS = common.LSH_BANDS
BAND_BITS = common.LSH_BAND_BITS
MAX_SHIFT = 2        # 位置容差（窗口步长），教学值
# 证据阈值不硬编码：在【合法-合法对】证据数的 2% 分位上自适应标定（保证合法团召回）


def collect_evidence(H, n_bands=N_BANDS, band_bits=BAND_BITS, max_shift=MAX_SHIFT):
    """H:(num_seg,num_win,16) 打包指纹 -> 段对碰撞证据计数字典。"""
    num_seg, nw, _ = H.shape
    bits = np.unpackbits(H, axis=-1)                     # (num_seg,nw,128)
    weights = (1 << np.arange(band_bits))[::-1]
    sigs = np.empty((num_seg, nw, n_bands), dtype=np.int64)
    for b in range(n_bands):
        sigs[:, :, b] = bits[:, :, b * band_bits:(b + 1) * band_bits].dot(weights)

    bucket = defaultdict(set)
    for seg in range(num_seg):
        for w in range(nw):
            lo = max(0, w - max_shift); hi = min(nw, w + max_shift + 1)
            for b in range(n_bands):
                s = int(sigs[seg, w, b])
                for pbin in range(lo, hi):
                    bucket[(b, s, pbin)].add(seg)

    ev = defaultdict(int)
    for members in bucket.values():
        if len(members) > 1:
            m = sorted(members)
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    ev[(m[i], m[j])] += 1
    return ev


def theoretical_recall(s, b=BAND_BITS, B=N_BANDS):
    return 1.0 - (1.0 - s ** b) ** B


def monte_carlo_recall(s, n_pairs=20000, b=BAND_BITS, B=N_BANDS, seed=common.SEED):
    r = common.rng(seed)
    hit = 0
    for _ in range(n_pairs):
        same = r.random((B, b)) < s
        if same.all(axis=1).any():
            hit += 1
    return hit / n_pairs


def main():
    fp = common.load_npz("05_fingerprints.npz", allow_pickle=True)
    H64 = fp["H64"]; seg_channel = fp["seg_channel"]; seg_label = fp["seg_label"]
    num_seg = H64.shape[0]

    common.banner("08 LSH 位置对齐粗筛候选对（8 band × 16 位，需成片碰撞）")

    common.sub_banner("单点召回率 R(s)=1−(1−s^16)^8：理论 vs 蒙特卡洛(20000 对)")
    print(f"{'相似度 s':>9}{'理论召回':>11}{'蒙特卡洛':>11}")
    for s in [0.95, 0.90, 0.85, 0.70, 0.50]:
        print(f"{s:9.2f}{theoretical_recall(s)*100:10.2f}%"
              f"{monte_carlo_recall(s, n_pairs=20000)*100:10.2f}%")

    ev = collect_evidence(H64)
    total_pairs = num_seg * (num_seg - 1) // 2

    def ev_of(i, j):
        return ev.get((i, j), 0)

    common.sub_banner("段对碰撞证据数分布（按类别）")
    print(f"{'段对类型':18s}{'中位数':>8}{'均值':>9}{'最小':>6}{'最大':>6}")
    groups = defaultdict(list)
    for i in range(num_seg):
        for j in range(i + 1, num_seg):
            if seg_channel[i] == 0 and seg_channel[j] == 0:
                groups["合法-合法"].append(ev_of(i, j))
            elif seg_channel[i] == 0 or seg_channel[j] == 0:
                other = seg_channel[j] if seg_channel[i] == 0 else seg_channel[i]
                groups[f"合法-{str(fp['channel_names'][other])}"].append(ev_of(i, j))
            else:
                groups["隐蔽-隐蔽"].append(ev_of(i, j))
    for name, vals in groups.items():
        v = np.asarray(vals)
        print(f"{name:18s}{int(np.median(v)):8d}{v.mean():9.1f}{v.min():6d}{v.max():6d}")

    # 自适应证据阈值：合法-合法对证据数的 2% 分位（保证合法团约 98% 召回）
    MIN_EVIDENCE = int(np.quantile(np.asarray(groups["合法-合法"]), 0.02))
    cand = np.array(sorted([p for p, c in ev.items() if c >= MIN_EVIDENCE]), dtype=np.int64)
    common.sub_banner(f"候选对规模（证据数 ≥ {MIN_EVIDENCE}，= 合法-合法证据 2% 分位）")
    print(f"  全配对 {total_pairs} 个，LSH 保留 {len(cand)} 个"
          f"（压缩到 {len(cand)/total_pairs*100:.1f}%）")

    ll = sum(1 for (i, j) in cand if seg_channel[i] == 0 and seg_channel[j] == 0)
    n_legit = int((seg_channel == 0).sum()); ll_total = n_legit * (n_legit - 1) // 2
    print(f"  合法-合法对召回 = {ll}/{ll_total} = {ll/ll_total*100:.1f}%")
    for cid in range(1, len(fp["channel_names"])):
        tot = n_legit * int((seg_channel == cid).sum())
        hit = sum(1 for (i, j) in cand
                  if (seg_channel[i] == 0 and seg_channel[j] == cid) or
                  (seg_channel[j] == 0 and seg_channel[i] == cid))
        print(f"  合法-{str(fp['channel_names'][cid]):14s} 保留 {hit:5d}/{tot:<5d} = {hit/tot*100:5.1f}%")

    ev_arr = np.array([[i, j, c] for (i, j), c in ev.items() if c >= MIN_EVIDENCE], dtype=np.int64)
    path = common.save_npz("08_candidates.npz", candidates=cand, evidence=ev_arr,
                           seg_channel=seg_channel, seg_label=seg_label,
                           channel_names=fp["channel_names"])
    print(f"\n已保存：{path}（候选对数组 {cand.shape}）")


if __name__ == "__main__":
    main()
