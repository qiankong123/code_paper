# QuadFP-Cascade 论文计算过程复现（Python）

论文：《多尺度指纹级联框架 QuadFP-Cascade：面向网络隐蔽定时信道检测》，
*Computer Communications* 251 (2026) 108495。
任务：检测网络隐蔽定时信道（Covert Timing Channel, CTC）——从一条流量的包间
时间间隔（Inter-Packet Delay, IPD）序列，判定每一小段是正常流量还是隐蔽信道。

本目录用 `00` 公共模块 + `01`–`14` 共 15 个带数字前缀的脚本，把论文**从原始 IPD
到最终段级判决**的关键计算步骤逐个复现，每一步都能独立运行、打印中间结果，并在
关键步骤给出可以手算的小例子。

> **本版本默认跑真实 CAIDA/NCTC 数据**（`data/bignet`、`data/labnet`）。
> 合成数据生成器 `01_generate_ipd.py` 完整保留，仅用于在没有数据时打通链路；
> 见下文“两种数据源”。

---

## 1. 两种数据源（真实数据为主，合成器保留）

| 数据源 | 第 01 步 | 运行命令 |
|---|---|---|
| **真实 CAIDA/NCTC**（默认、主结果） | `01_load_caida.py` | `python run_all.py caida bignet 16` |
| 合成教学数据（链路自测） | `01_generate_ipd.py` | `python run_all.py synthetic` |

第 02–14 步对两种数据源**完全相同**，因为第 01 步都输出同一份 schema
（`ipd_list / labels / channel_id / protocols / channel_names`）。切换数据源只改
第一步，后续无需任何改动。`01` 会写 `outputs/data_source.txt` 标记当前数据来源，
第 14 步据此显示标题。

---

## 2. 目录结构与数据准备

```
code/
├── data/
│   ├── bignet/                     # Backbone 广域网（大网）场景
│   │   ├── http.csv https.csv ssh.csv        # 合法（三协议）
│   │   ├── IPCTC_1.csv IPCTC_2.csv           # 隐蔽：IPCTC 两组参数
│   │   ├── LNCTC_1.csv LNCTC_2.csv           # 隐蔽：LNCTC
│   │   ├── TRCTC_1.csv TRCTC_2.csv           # 隐蔽：TRCTC
│   │   └── Jitterbug_1.csv Jitterbug_2.csv   # 隐蔽：Jitterbug
│   └── labnet/                     # Lab 实验室网场景（文件结构同上）
├── 00_common.py … 14_evaluation.py           # 15 步流水线
├── 01_load_caida.py                           # 真实数据加载器
├── 01_generate_ipd.py                         # 合成数据生成器（保留备用）
├── make_shared_constants.py                   # 生成 Python/C++ 共享随机常量
├── random_matrix_R.csv   gcn_W0_init.csv  gcn_W1_init.csv   # 跨语言共享常量
├── run_all.py
└── outputs/                                   # 每步中间产物（.npz/.pkl）
```

**CSV 格式**：单列 CSV，第一行表头 `IPDs`，其后每行一个 IPD（单位：秒）；一个文件
是一整条 trace（真实合法文件约 200 万行，隐蔽文件约 1000 万行）。

**第 01 步如何把长 trace 变成“连接”**（与合成版对齐，使 02–14 不用改）：

1. 按 `CONN_LEN = 8000` 个 IPD 把长 trace 切成**互不重叠的连接块**（真实数据每类
   有 129–250 个可用块）；
2. 每类用**确定性均匀抽样**取 `K` 个块（默认 `K=16`，纯算术取点、无随机数，保证
   Python/C++ 取到完全相同的块）：`idx_j = round(j·(n−1)/(K−1))`，覆盖整条 trace；
3. 两个配置文件（`_1/_2`）各分摊一半名额；
4. **清洗**：真实正常流量存在少量 0/负 IPD（抓包时间戳误差），统一裁剪到
   `MIN_IPD = 1e-6` s（1 微秒），避免后续变化率 Δ 除零。

随后第 02 步把每条连接 8000 个 IPD 按段长 `L=2000`、25% 重叠（步长 1500）切成
`floor((8000−2000)/1500)+1 = 5` 个**段**，**段 = 图节点**。

> **数据集不含 ε-xlibur**（论文 Table 4 的第 5 类信道）。`channel_id=5` 样本数为 0，
> 下游分信道评估自动跳过；`labnet/https.csv` 仅 1255 个 IPD（不够一个 8000 块），
> 被自动跳过。

---

## 3. 环境与运行

依赖：Python 3.10+、numpy、scikit-learn、pandas（`pip install numpy scikit-learn pandas`）。

```powershell
# Windows PowerShell 建议先设控制台编码，避免中文乱码：
$env:PYTHONIOENCODING="utf-8"

python run_all.py caida bignet 16   # 真实·大网（Backbone），每类抽 16 连接
python run_all.py caida labnet 16   # 真实·实验室网（Lab）
python run_all.py caida             # 等价于 caida bignet（默认 K=16）
python run_all.py synthetic         # 合成数据（链路自测，无需 data/）
python run_all.py 13                # 从第 13 步跑到末尾（需上游 outputs 已生成）
python 06_hamming_similarity.py     # 也可单独运行某一步（脚本自带演示）
```

运行耗时：真实 bignet 全流程约 3–4 分钟（第 09 步建图最慢），labnet 约 1–2 分钟。
所有中间产物写入 `outputs/`（`.npz`；模型存 `13_models.pkl`）。

---

## 4. 真实数据实测结果（每类 K=16，连接级划分，可复现）

训练/测试按**连接级**划分：`conn_id % 3 == 0` 的整条连接做测试，其余做训练
（同一连接的 5 个相邻段不跨训练/测试，杜绝泄漏）。

### 4.1 Lab 实验室网（labnet）

84 连接 / 420 段；训练 280 段、测试 140 段（测试合法 35、隐蔽 105）。

```
                    预测合法(0)   预测隐蔽(1)
  实际合法(0)        TN=35         FP=0
  实际隐蔽(1)        FN=6          TP=99
Accuracy=0.9571  Precision=1.0000  Recall=0.9429  FPR=0.0000  F1=0.9706  ROC-AUC=0.9986
分信道召回：IPCTC=1.00(25/25)  LNCTC=1.00(30/30)  TRCTC=0.96(24/25)  Jitterbug=0.80(20/25)
两级阈值：τ1=0.4268（召回导向）  τ2=0.8000（精度导向）
边融合：α=[1,0,0]，β=2.00（启用 GCN 边权 φ）
```

### 4.2 Backbone 大网（bignet）

112 连接 / 560 段；训练 370 段、测试 190 段（测试合法 80、隐蔽 110）。

```
                    预测合法(0)   预测隐蔽(1)
  实际合法(0)        TN=80         FP=0
  实际隐蔽(1)        FN=31         TP=79
Accuracy=0.8368  Precision=1.0000  Recall=0.7182  FPR=0.0000  F1=0.8360  ROC-AUC=0.8621
分信道召回：IPCTC=1.00(30/30)  LNCTC=1.00(25/25)  TRCTC=0.80(20/25)  Jitterbug=0.133(4/30)
两级阈值：τ1=0.0696（召回导向）  τ2=0.6700（精度导向）
边融合：α=[1,0,0]，β=0（大网下 GCN 边权不带来分离增益，网格搜索自动置 0）
```

### 4.3 结论与短板

- **Lab 远好于 Backbone**（Recall 0.94 vs 0.72、AUC 0.999 vs 0.862），与论文
  “实验室网络更易检测、骨干大网更难”的结论一致。
- **IPCTC/LNCTC 两类大倍率编码在两场景都 100% 召回**，TRCTC 也较高（0.80/0.96）。
- **Jitterbug 是主要短板**：它是最小嵌入（只对 IPD 做很小的取整扰动），在教学版
  4 特征 + 简化两层 GCN 下与正常流量高度重叠；大网尤其难（0.13），实验室网 0.80。
  论文使用完整特征工程与模型，Table 4 报告五类信道 AUC≥0.95、Recall 多在 0.95 以上；
  本复现的抽样规模、阈值标定、连接级划分为教学设置，数值不必与论文逐位相等。
- **两个场景都做到 0 误报（FP=0、FPR=0、Precision=1.0）**，这是第二级“精度导向”
  阈值的直接结果。

### 4.4 真实数据的判定路径（与合成版的一个重要差别）

真实数据的测试集里**没有孤立段**（bignet/labnet 测试孤立数均为 0，训练侧 bignet
孤立 0、labnet 仅 1）。也就是说，真实网络中大倍率信道段彼此、或与其它段之间仍然
形成边，**并不像合成数据那样必然成为孤立节点**。真实数据的主要判定力量是
**两级 GBDT 对四个图特征（H_sym/L_cy/ρ/√S）的打分**，而不是孤立硬规则：

1. 第一级 τ1 把少量明显合法段直接放行（labnet 34 段、bignet 5 段）；
2. 其余进入第二级，τ2 高置信度确认隐蔽（labnet 106 候选判 99，bignet 185 候选判 79）。

孤立硬规则（ρ≈0 直接判隐蔽）作为流水线的一部分仍然保留，在合成教学数据上负责
捕获大倍率编码（见第 5 节）；在真实数据上它很少触发，但不影响结果。

---

## 5. 合成数据（保留的链路自测，synthetic 模式）

`01_generate_ipd.py` 用统计模型合成结构相近的合法流量（log-IPD 一阶自相关 + 随机
突发）与五类隐蔽信道，共 64 连接 / 320 段（合法 3 协议×8、5 类隐蔽×8），用于在没有
真实数据时打通并验证整条计算链路。**其指标只证明流程正确，不代表论文真实结果。**

合成数据上的典型现象（教学演示）：

- 大倍率 IPCTC/LNCTC/TRCTC 在两级阈值下**与任何段都不成边，成为孤立节点**，由图
  拓扑硬规则 100% 判隐蔽；
- Jitterbug 在团内靠周期节奏特征（H_sym 偏高、L_cy 偏小）被第二级 GBDT 检出；
- **ε-xlibur 召回为 0**：其周期微调被指数平滑完全吸收、段特征与合法重叠，达到
  ε-不可区分的理论极限，精度导向阈值宁可漏检也不误报合法。

最新一次固定种子的 synthetic 测试结果（110 段）：大倍率三类与 Jitterbug 召回均为
1.00、ε 召回 0.00，Recall=0.7857。

> 注意：synthetic 模式下 Python 与 C++ 各自用本机 RNG 生成**不同的随机 trace**
> （第 01 步不共享），因此合成版只要求**同分布、同结论**（大倍率全中、ε 为 0、
> 召回一致），不要求逐位相同；**真实 CAIDA 数据两语言则逐位一致**（见第 7 节）。

---

## 6. 两级阈值如何标定（第 13 步，对应论文公式 11）

论文只规定“**第一级召回导向、第二级精度导向**”，未给具体阈值。本复现的做法：

- 两级都在**同一个团内训练集**上训练一棵 150 树、深度 2 的 GBDT（用 sklearn
  `GradientBoostingClassifier` 替代论文的 LightGBM，同为 GBDT）；
- 用**连接级 GroupKFold(5) 交叉验证**的 out-of-fold（OOF）概率标定，避免同连接段
  泄漏：
  - **τ1（召回导向）** = 团内正类 OOF 概率的 1% 分位，宁多勿漏；
  - **τ2（精度导向）** = 在 OOF 上扫描阈值，取“满足 Precision≥0.95 且召回最大”的
    工作点；达不到 0.95 放宽到 0.90；再不行退化为 τ1。
- **召回地板（防小样本坍塌）**：当 P≥0.95 工作点的召回低于 0.30 时，说明高阈值在
  小样本上被一两个边界样本“绑架”（经验精度虚高、第二级几乎全否），自动放宽到
  P≥0.90。真实 bignet/labnet 该工作点召回为 0.75/0.96，**不会触发**；仅小规模
  合成数据触发，用来保住 Jitterbug 的教学演示。

测试判定：孤立段直接判隐蔽；团内段 `pro1<τ1` 判合法，否则 `pro2≥τ2` 判隐蔽。

---

## 7. 与 C++ 版的一致性（“一模一样”是如何保证的）

`cpp/` 下有一份**纯 C++11、无第三方依赖**的等价实现（同一套 01–14 分步）。
真实 CAIDA 数据上，**Python 与 C++ 的混淆矩阵、召回率、τ2、各信道召回完全一致**。

三处跨语言随机/划分差异已通过**共享文件**或**算法对齐**消除：

| 环节 | 潜在差异 | 解决方式 |
|---|---|---|
| 05 SimHash 投影矩阵 R∈ℝ¹²⁸ˣ⁷ | numpy PCG64 与 C++ mt19937 生成不同 | 共享 `random_matrix_R.csv`（128×7） |
| 10 GCN 初始权重 W0(2×16)/W1(16×16) | 各自随机初始化 | 共享 `gcn_W0_init.csv` / `gcn_W1_init.csv` |
| 13 连接级 GroupKFold(5) | sklearn 与手写折划分可能不同 | C++ 精确复刻 sklearn `GroupKFold(shuffle=False)` 的贪心分组 |
| 13 GBDT | sklearn vs C++ 手写 | 相同分裂增益/叶值/排序规则下决策一致；仅概率值约 1e-3 微差 |

- 三个共享 CSV 由 `make_shared_constants.py` 用固定种子生成（`%.17g`、无表头、逗号
  分隔）。Python 直接读当前目录，C++ 从 `cpp/` 以 `../` 前缀读同一份。**若文件缺失，
  两语言都会回退到各自 RNG 并打印警告（此时真实数据不再逐位一致）**；要重新生成可
  先删除三个 CSV 再运行该脚本，脚本会校验新生成值与现有文件是否一致、防止误改种子。
- C++ 自实现的 GBDT、GCN（手写前向/反向/Adam）、分位数、汉明 popcount、Mann–Whitney
  AUC 与 Python 数学一致；真实数据上判定完全相同，仅 ROC-AUC 概率值有 ~1e-3 差异
  （手写梯度提升与 sklearn 的数值路径不同），不影响任何阈值判决。

---

## 8. 文件与对应论文步骤

| 脚本 | 对应章节/公式 | 做什么 |
|---|---|---|
| `00_common.py` | 全局常量 | 种子、路径、段长/尺度/符号阈值、数据源标记、读写工具 |
| `01_load_caida.py` | §4.1 数据 | **真实数据**：长 trace 切 8000 连接块、确定性均匀抽样、清洗 |
| `01_generate_ipd.py` | §4.1 数据 | **合成数据**（保留）：合法 3 协议 + 5 类隐蔽 IPD |
| `02_preprocess_segment.py` | §3.2.1 | 指数平滑（α=0.30）；L=2000、25% 重叠分段 |
| `03_change_rate_symbol.py` | §3.2.1, Table 1 | 相邻变化率 Δ；映射到 −3…+3 七级符号 |
| `04_multiscale_windows.py` | §3.2.1, Table 2 | 三尺度滑窗（32/8)(64/16)(128/32)；每窗口 7 维计数向量 |
| `05_random_matrix_fingerprint.py` | 公式2 | 加载共享 R；`h=sign(Rv)` 得 128 位 SimHash |
| `06_hamming_similarity.py` | 公式3 | 字节异或+popcount；汉明距离/相似度；翻 6 位手算 |
| `07_robust_aggregation.py` | §3.2.1 | 有限位移 ±5 对齐 + 两级聚合（位移内中位数→位移间 top3） |
| `08_lsh_candidate.py` | §3.2.2 | 8×16 band 分桶 + 位置对齐多点碰撞；理论召回 R=1−(1−s¹⁶)⁸ |
| `09_threshold_graph.py` | §3.2.1–2 | 合法-合法相似度 Q0.30/Q0.60 标定 θ0/θ1；建稀疏图 G、X、A |
| `10_gnn_gcn.py` | §3.2.2, 公式4/5 | 两层 GCNConv（共享 W0/W1）+ 手写 Adam 自监督；φ=σ(z_u·z_v) |
| `11_edge_fusion.py` | 公式6 | `s=α1w1+α2w2+α3w3+βφ`；α/β 网格搜索最大化两类边分离度 |
| `12_graph_features.py` | 公式7–10 | 每段 H_sym、L_cy、ρ、√S_flow 四特征（top-k=5） |
| `13_two_level_classifier.py` | §3.2.4, 公式11 | 孤立硬规则 + 团内两级 GBDT；连接级 GroupKFold OOF 标定 τ1/τ2 |
| `14_evaluation.py` | §4.2, 公式12 | 混淆矩阵、Acc/Precision/Recall/FPR/F1/AUC、分信道召回 |
| `make_shared_constants.py` | — | 生成/校验跨语言共享的 R、W0、W1 三个 CSV |
| `run_all.py` | — | 顺序执行 01→14（caida/synthetic，或起始步骤号） |

## 数据流（outputs/*.npz）

```
01_connections → 02_segments → 03_symbols → 04_count_vectors
→ 05_fingerprints → 06_pairwise_demo
                  → 07_pair_similarities
                  → 08_candidates → 09_graph → 10_gnn
→ 11_fusion → 12_features → 13_prediction(+13_models.pkl) → 14_metrics
```

---

## 9. 论文固定参数 vs 教学取值

**严格遵循论文**：段长 L=2000、25% 重叠；Table 1 七级 Δ 阈值；Table 2 三尺度
(32/8)(64/16)(128/32)；7 维计数向量；128 位 SimHash `h=sign(Rv)`（公式2）；汉明
相似度（公式3）；有限位移对齐；三尺度 w 与 m=max；LSH 粗筛；节点=段、节点特征仅
[度, 邻边均权]；两层 GCNConv、16 维 embedding、φ=σ(z_u·z_v)（公式4）、MSE 自监督
（公式5）；边融合（公式6）；四特征 H_sym/L_cy/ρ/√S（公式7–10）；两级阈值级联
（公式11）；Recall=TP/(TP+FN)（公式12）；连接级划分训练/测试。

**论文未给、按标准做法补充的教学值**（代码中均标注“教学值”）：指数平滑 α=0.30；
合成数据参数；LSH 8 band×16 位与多点碰撞证据阈值；位移范围 ±5、top-3；θ0/θ1 取
合法-合法对 Q0.30/Q0.60；top-k 中 k=5；GCN 隐藏宽度 16、300 轮、学习率 1e-2；
GBDT 用 sklearn GBDT（150 树、深度 2）替代 LightGBM；τ1/τ2 的 OOF 数据驱动标定与
0.30 召回地板；每类抽 K=16 连接。

**关键数值锚点**：每段三尺度窗口数 247/122/59，每窗口 7 维计数之和=窗口大小；
指纹 128 位，`sim=1−Ham/128`，人为翻 6 位 → sim=0.9531；GNN 参数量
2×16+16+16×16+16=**320**，两层覆盖 2 跳。
