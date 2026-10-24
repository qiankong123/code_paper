# QuadFP-Cascade 论文计算过程复现（Python）

论文：《多尺度指纹级联框架 QuadFP-Cascade》，*Computer Communications* 251 (2026) 108495
任务：检测网络隐蔽定时信道（Covert Timing Channel, CTC）。

本目录用 15 个带数字前缀的脚本（`00` 公共模块 + `01`–`14` 流水线），把论文
**从原始 IPD 到最终段级判决**的关键计算步骤逐个复现，每一步都能独立运行、
打印中间结果，并尽量给出可手算的小例子。

## 运行

依赖：Python 3.10+、numpy、scikit-learn（`pip install numpy scikit-learn`）。

```bash
# Windows PowerShell 建议先设输出编码，避免中文乱码：
$env:PYTHONIOENCODING="utf-8"
python run_all.py          # 顺序执行 01→14（约 8–10 分钟，09 建图最慢）
python run_all.py 09       # 从 09 跑到末尾（需上游 outputs 已生成）
python 06_hamming_similarity.py   # 也可单独运行某一步（脚本自带演示）
```

运行后所有中间产物写入 `outputs/`（`.npz` 压缩数据；模型存 `13_models.pkl`）。

## 数据规模

`01` 共生成 **64 条连接、320 个段**：合法 3 协议（HTTP/HTTPS/SSH）×8 条 = 24 条；
隐蔽 5 类（IPCTC、LNCTC、TRCTC、Jitterbug、ε-klibur）×8 条 = 40 条。每条连接
8000 个 IPD，按 L=2000、25% 重叠（步长 1500）切成 5 段，**段 = 图节点**。
测试集按连接级取 `conn_id % 3 == 0`（110 段），同连接的相邻段不跨训练/测试。

## 重要说明（务必先读）

- **数据是合成的**：论文使用公开 NCTC 数据集，无法随附。`01` 用统计模型
  合成结构相近的合法流量（log-IPD 一阶自相关 + 随机突发）与五类隐蔽信道，
  仅用于打通并验证整条计算链路。**指标数字不代表论文真实结果**，只证明流程正确。
- **严格遵循论文的部分**：段长 L=2000、25% 重叠；Table 1 七级 Δ 阈值；
  Table 2 三尺度 (32/8)(64/16)(128/32)；7 维计数向量；固定 R∈ℝ^{128×7}
  与 `h=sign(Rv)`（公式2）；汉明相似度（公式3）；有限位移对齐；三尺度 w 与
  `m=max`；LSH 粗筛；θ0/θ1 取合法-合法对相似度的 Q0.30/Q0.60；
  节点=段、节点特征仅 [度, 邻边均权]；两层 GCNConv、16 维 embedding、
  `φ=σ(z_u·z_v)`（公式4）、MSE 自监督（公式5）；边融合（公式6）；
  四特征 H_sym/L_cy/ρ/√S_flow（公式7–10）；两级阈值级联（公式11）；
  Recall=TP/(TP+FN)（公式12）；连接级划分训练/测试。
- **论文未给、由本复现按标准做法补充的教学参数**（代码中均已标注"教学值"）：
  指数平滑系数 α=0.30；合成数据的 AR/突发/编码参数；固定随机种子；
  LSH 配置（8 band × 16 位、位置对齐的多点碰撞证据阈值）；
  `07` 两级聚合（每个位移内先取重叠窗口相似度的中位数，再在 2τ+1 个位移中取
  最高的 3 个平均；位移范围 ±5）；`11` 中 α/β 的网格搜索目标；top-k 中 k=5；
  GNN 隐藏宽度 16、训练轮数、学习率与特征归一化；
  两级分类器用 sklearn `GradientBoostingClassifier` 替代 LightGBM（同为 GBDT），
  并在连接级 5 折交叉验证的 out-of-fold 概率上标定阈值。

## 检测流程（本复现的两级判定，对应论文 §3.2.3–3.2.4）

1. **图拓扑硬规则（大倍率编码）**：两级阈值 θ0/θ1 下，IPCTC/LNCTC/TRCTC
   与任何段都不构成边，在图中成为**孤立节点**（ρ=0、√S_flow=0），直接判隐蔽。
2. **团内两级 GBDT（高相似信道）**：合法/Jitterbug/ε 都在同一个大团（高相似），
   GBDT 只在这些节点上训练——第一级 τ1 召回导向（宁多勿漏），第二级 τ2 精度
   导向（取候选中合法 OOF 概率的 99% 分位，保护合法）。
3. **合成数据结果（固定种子，仅示意流程）**：IPCTC/LNCTC/TRCTC/Jitterbug
   测试召回均为 **1.00**；合法误报率 FPR≈**0.075**；Precision≈0.95、F1≈0.86。
   ε-klibur 的周期微调被指数平滑完全吸收、段特征与合法重叠，达到 ε-不可区分的
   理论极限，精度导向下**召回为 0**（宁可漏检也不误报合法）——这是合成数据的
   局限与教学上的诚实取舍；真实 NCTC 数据痕迹更丰富，论文 Table 4 五类 AUC≥0.997。

## 文件与对应论文步骤

| 脚本 | 对应章节/公式 | 做什么 |
|---|---|---|
| `00_common.py` | 全局常量 | 种子、路径、段长/尺度/符号阈值、读写工具 |
| `01_generate_ipd.py` | §4.1 数据 | 合成 64 条连接（合法 3 协议 + 5 类隐蔽）的 IPD |
| `02_preprocess_segment.py` | §3.2.1 | 指数平滑（α=0.30）；L=2000、25% 重叠分段 |
| `03_change_rate_symbol.py` | §3.2.1, Table 1 | 相邻变化率 Δ；映射到 −3…+3 七级符号 |
| `04_multiscale_windows.py` | §3.2.1, Table 2 | 三尺度滑窗；每窗口 7 维计数向量 v（247/122/59） |
| `05_random_matrix_fingerprint.py` | 公式2 | 固定 R；`h=sign(Rv)` 得 128 位 SimHash 指纹 |
| `06_hamming_similarity.py` | 公式3 | 打包字节异或 + popcount；汉明距离/相似度；6 位玩具手算 |
| `07_robust_aggregation.py` | §3.2.1 | 有限位移对齐 + 两级聚合（位移内中位数→位移间 top3 均值） |
| `08_lsh_candidate.py` | §3.2.2 | 8×16 band 分桶 + 位置对齐多点碰撞证据；理论召回 R=1−(1−s^16)^8 |
| `09_threshold_graph.py` | §3.2.1–2 | Q0.30/Q0.60 标定 θ0/θ1；边分流；建稀疏图 G=(V,E)、X、A |
| `10_gnn_gcn.py` | §3.2.2, 公式4/5 | 两层 GCNConv 前向 + 手写 Adam 自监督训练；含 3 节点链手算 |
| `11_edge_fusion.py` | 公式6 | `s=α1w1+α2w2+α3w3+βφ`；α/β 网格搜索 |
| `12_graph_features.py` | 公式7–10 | 每段 H_sym、L_cy、ρ、√S_flow 四特征 |
| `13_two_level_classifier.py` | §3.2.4, 公式11 | 孤立硬规则 + 团内两级 GBDT；连接级 OOF 标定 τ1/τ2 |
| `14_evaluation.py` | §4.2, 公式12 | 混淆矩阵、Recall/Precision/FPR/F1/AUC、分信道召回 |
| `run_all.py` | — | 顺序执行 01→14（可传起始步骤号） |

## 数据流（outputs/*.npz）

```
01_connections → 02_segments → 03_symbols → 04_count_vectors
→ 05_fingerprints → 06_pairwise_demo
                  → 07_pair_similarities
                  → 08_candidates → 09_graph → 10_gnn
→ 11_fusion → 12_features → 13_prediction(+13_models.pkl) → 14_metrics
```

## 关键数值锚点（合成数据 + 固定种子）

- 每段 2000 IPD；三尺度每段窗口数 247 / 122 / 59；每窗口 7 维计数之和 = 窗口大小。
- 指纹 128 位；相似度 `sim = 1 − Ham/128`；人为翻 6 位 → 相似度 0.9531（手算）。
- LSH 单 band 16 位、8 band 取并：s=0.95→约 99%、s=0.90→约 81%、s=0.85→约 46%、
  s=0.70→约 3%（理论与蒙特卡洛一致，公式 R=1−(1−s^16)^8）。
- 建图：大倍率三类 120 段 100% 孤立；合法/Jitterbug/ε 200 段在同一大团。
- GNN 参数量（输入 2 维 →16→16，含偏置）= 2×16+16 + 16×16+16 = 320；两层覆盖 2 跳。
