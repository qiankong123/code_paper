# QuadFP-Cascade 论文复现 —— C++ 版（与 Python 版一一对应）

用**纯 C++11 标准库**（无任何第三方依赖）1:1 复现论文《多尺度指纹级联框架
QuadFP-Cascade：面向网络隐蔽定时信道（Covert Timing Channels, CTC）检测》
（*Computer Communications* 251 (2026) 108495）的**全部计算过程**。

本目录与上一级 Python 版**严格对应**：同样的数字前缀、同样的分步、同样的固定参数、
同样的算法与输出结构，可编译、可运行、可逐步对照。

> **默认跑真实 CAIDA/NCTC 数据**（`../data/bignet`、`../data/labnet`）。合成数据
> 生成器 `01_generate_ipd.cpp` 完整保留，用于在没有数据时打通链路。第 02–14 步对
> 两种数据源完全相同。

---

## 1. 编译与运行

### 环境
- 编译器：g++（已在本机 Strawberry Perl 自带的 `g++ 4.9.2 (i686)` 验证；任何支持
  C++11 的 g++/clang 均可）。
- 依赖：**无**。只用 `<vector> <random> <cmath> <fstream> <algorithm> …` 等标准库。
- 平台：Windows（`-static` 静态链接，无需额外 DLL）；Linux/macOS 去掉 `-static` 即可。

### 一键编译
```bat
build.bat
```
对每个 `??_*.cpp` 执行 `g++ -std=c++11 -O2 -static -o NN_*.exe NN_*.cpp`。

### 一键运行（真实数据 / 合成数据）
```bat
run_all.bat caida bignet 16     :: 真实·Backbone 大网，每类抽 16 连接（默认）
run_all.bat caida labnet 16     :: 真实·Lab 实验室网
run_all.bat                       :: 不带参数 = synthetic（合成数据，无需 ../data）
run_all.bat synthetic
```
中间产物写入 `outputs/NN_*.bin`（自定义二进制归档，等价于 Python 的 `.npz`）。
真实 CSV 从 `../data/<subset>/` 读取（路径在程序内以 `../` 前缀写死，必须在 `cpp/`
目录下运行）。

### 手动单步（示例）
```bat
g++ -std=c++11 -O2 -static -o 01_load_caida.exe 01_load_caida.cpp
01_load_caida.exe bignet 16
```
> 控制台中文：程序内已 `SetConsoleOutputCP(65001)`，`build.bat`/`run_all.bat` 也 `chcp 65001`。

### 类版本（第 10 步，真实 .npz + 线程池）
```bat
build_class.bat
10_gnn_gcn_class.exe
```
`10_gnn_gcn_class.cpp` 是第 10 步的**类实现版**（与函数式 `10_gnn_gcn.cpp` 并存、互不覆盖）：
- 直接**加载/保存真正的 .npz**（`npz_archive.hpp`：自实现 ZIP 容器 + NPY 头，raw-deflate 走 zlib），
  与 numpy 互通——读上一级 Python 的 `../outputs/09_graph.npz`，写本目录 `outputs/10_gnn.npz`；
- 用**线程池** `qpfc::ThreadPool` 并行加速矩阵乘、ReLU、边置信度；
- 需链接 zlib（本机 Strawberry 自带，路径已在 `build_class.bat` 写死）。

---

## 2. 14 个步骤与 Python 版对应表

| 文件 | 对应论文 | 计算内容 |
|---|---|---|
| `common.hpp` | 00_common.py | 固定参数、RNG、统计、矩阵、二进制 `Store` 读写、指纹/汉明/位移聚合公共工具 |
| `01_load_caida.cpp` | §4.1 数据 | **真实数据**：长 trace 切 8000 连接块、确定性均匀抽样、0/负 IPD 清洗 |
| `01_generate_ipd.cpp` | §4.1 数据 | **合成数据**（保留）：合法 HTTP/HTTPS/SSH 与 5 类隐蔽定时信道 |
| `02_preprocess_segment.cpp` | §3.2.1 | 指数平滑 α=0.30；L=2000、25% 重叠分段（**段=图节点**） |
| `03_change_rate_symbol.cpp` | Table 1 | 相邻变化率 Δ，按阈值映射为七级符号 −3…+3 |
| `04_multiscale_windows.cpp` | §3.2.1 / Table 2 | 三尺度（32/8)(64/16)(128/32)→ 每窗口 7 维计数向量 v（和=win） |
| `05_random_matrix_fingerprint.cpp` | 公式(2) | 加载共享 R∈ℝ¹²⁸ˣ⁷，h=sign(Rv) 得 128 位 SimHash，打包 16 字节 |
| `06_hamming_similarity.cpp` | 公式(3) | 汉明距离/相似度（异或+popcount）；含“翻 6 位→0.9531”手算 |
| `07_robust_aggregation.cpp` | §3.2.1 | 有限位移 ±5 对齐 + top-3 位移聚合，得段对三尺度 w1/w2/w3，m=max |
| `08_lsh_candidate.cpp` | §3.2.2 | 8 band×16 位 LSH：窗口级指纹+位置对齐+多点碰撞；理论召回 R(s)=1−(1−s¹⁶)⁸ |
| `09_threshold_graph.cpp` | §3.2.2 / Fig.3 | 合法-合法 m 的 Q30/Q60 标定 θ0/θ1；边分流；建稀疏图 A；节点特征[度,邻边均权] |
| `10_gnn_gcn.cpp` | 公式(4)(5) | 两层 GCNConv（H=16，加载共享 W0/W1）；手写前向/反向/Adam 自监督 MSE |
| `npz_archive.hpp` | — | 类基础设施：`ThreadPool` 线程池、`NpyArray`/`NpzArchive` 真实 NPZ（zlib）、`DMat` 并行矩阵 |
| `10_gnn_gcn_class.cpp` | 公式(4)(5) | 第 10 步**类实现版**：GCNModel 类封装；真实 .npz 读写；线程池加速（与函数式并存） |
| `11_edge_fusion.cpp` | 公式(6) | 边强度 s=α1w1+α2w2+α3w3+β·φ；网格搜索最大化两类边分离度 |
| `12_graph_features.cpp` | 公式(7)–(10) | 每段 4 特征：符号熵 H_sym、周期节奏 L_cy、图累加 ρ、图流量 √S_flow（top-k=5） |
| `13_two_level_classifier.cpp` | 公式(11) | 孤立硬规则 + 两级 GBDT；连接级 GroupKFold OOF 标定 τ1/τ2（含召回地板） |
| `14_evaluation.cpp` | 公式(12)/Table 4 | 混淆矩阵、Acc/Precision/Recall/FPR/F1、ROC-AUC（Mann–Whitney）、分信道召回 |

---

## 3. “一模一样”的准确含义

### 真实 CAIDA 数据：逐位一致（刻意对齐）

真实数据上 Python 与 C++ 的**混淆矩阵、召回率、τ2、各信道召回完全相同**。三处
跨语言随机/划分差异已消除：

| 环节 | 解决方式 |
|---|---|
| 05 SimHash 投影矩阵 R∈ℝ¹²⁸ˣ⁷ | 两语言都读上一级目录的 `../random_matrix_R.csv`（Python 由 `make_shared_constants.py` 生成） |
| 10 GCN 初始权重 W0(2×16)/W1(16×16) | 两语言都读 `../gcn_W0_init.csv`、`../gcn_W1_init.csv` |
| 13 连接级 5 折交叉验证 | C++ `group_kfold_assign` 精确复刻 sklearn `GroupKFold(shuffle=False)`：unique 组按样本数降序、贪心放入当前最轻折 |
| 01 连接块均匀抽样 | 纯算术取点 `idx_j=round(j·(n−1)/(K−1))`，无随机数，两语言取到相同块 |
| 13 GBDT | 相同 F0、负梯度、分裂增益 SSE、叶值、深度/叶约束、排序 tie-break 下决策一致 |

**真实数据仍存在的唯一数值差异**：手写 GBDT（C++）与 sklearn `GradientBoostingClassifier`
（Python）的梯度累加数值路径不同，使 ROC-AUC 这种依赖全段概率排序的指标有约 1e-3
差异（bignet 0.8654 vs 0.8621；labnet 0.9992 vs 0.9986），第一级 τ1 在 labnet 有
0.0034 微差（0.4234 vs 0.4268，不影响候选集合）；**所有阈值判决、混淆矩阵、分信道
召回完全一致**。若共享 CSV 缺失，程序会回退本机 RNG 并打印警告（此时不再逐位一致）。

### 合成数据：同分布、同结论（不逐位相同）

`synthetic` 模式下第 01 步由 Python（numpy PCG64）与 C++（mt19937 + 极坐标
Box-Muller）**各自生成不同的随机 trace**，因此合成版的具体数值不逐位相同，但
算法、参数、结构、定性结论一致（大倍率三类全中、Jitterbug 全中、ε 为 0、召回 0.7857）。

中间归档格式：Python 用 `.npz`，C++ 用自定义二进制 `.bin`（`common.hpp` 中 `Store`：
数组名/类型/shape/字节；小端）。

---

## 4. 真实数据本机实测结果（每类 K=16，连接级划分，可复现）

连接级划分：`conn_id % 3 == 0` 的整条连接做测试，同连接的相邻段不跨训练/测试。

**Lab 实验室网（84 连接 / 420 段；训练 280、测试 140，测试合法 35 / 隐蔽 105）：**
```
                    预测合法   预测隐蔽
  实际合法          TN=35      FP=0
  实际隐蔽          FN=6       TP=99
Accuracy=0.9571  Precision=1.0000  Recall=0.9429  FPR=0.0000  F1=0.9706  ROC-AUC=0.9992
分信道召回：IPCTC=1.00(25/25)  LNCTC=1.00(30/30)  TRCTC=0.96(24/25)  Jitterbug=0.80(20/25)
τ1=0.4234  τ2=0.8000；边融合 α=[1,0,0]，β=2.00（启用 GCN 边权）
```

**Backbone 大网（112 连接 / 560 段；训练 370、测试 190，测试合法 80 / 隐蔽 110）：**
```
                    预测合法   预测隐蔽
  实际合法          TN=80      FP=0
  实际隐蔽          FN=31      TP=79
Accuracy=0.8368  Precision=1.0000  Recall=0.7182  FPR=0.0000  F1=0.8360  ROC-AUC=0.8654
分信道召回：IPCTC=1.00(30/30)  LNCTC=1.00(25/25)  TRCTC=0.80(20/25)  Jitterbug=0.133(4/30)
τ1=0.0696  τ2=0.6700；边融合 α=[1,0,0]，β=0（大网下 GCN 边权无增益，自动置 0）
```

**结论**

1. Lab 远好于 Backbone（Recall 0.94 vs 0.72、AUC 0.999 vs 0.865），与论文
   “实验室网络更易、骨干大网更难”一致；两场景均 0 误报（第二级精度导向）。
2. IPCTC/LNCTC 大倍率编码两场景均 100% 召回；TRCTC 较高（0.80/0.96）。
3. Jitterbug 嵌入最小、与合法重叠，是主要短板（大网 0.13、实验室 0.80）。
4. 数据集不含 ε-xlibur，故无该类召回。
5. **真实数据测试集无孤立段**——大倍率信道在真实网络下仍成边，主要靠两级 GBDT 对
   四特征的打分识别（孤立硬规则主要在合成数据上触发，本版予以保留）。

---

## 5. 两级阈值标定（第 13 步）

- 两级 GBDT 都在**同一个团内训练集**上训练：第一级配置 `{150 树, 深度2, min_leaf3,
  lr0.05}`，第二级 `{150, 2, 5, 0.05}`。
- 用连接级 GroupKFold(5) 的 OOF 概率标定：
  - **τ1 召回导向** = 团内正类 OOF 概率 1% 分位；
  - **τ2 精度导向** = 扫描 0.01…0.99，取“Precision≥0.95 且召回最大”的工作点，达不到
    降 0.90，再不行退 τ1。
- **召回地板 0.30**：若 P≥0.95 工作点召回 <0.30（小样本上经验精度被个别边界样本
  绑架、第二级坍塌），自动放宽到 P≥0.90。真实 bignet/labnet 该工作点召回 0.75/0.96
  不触发，仅小规模合成数据触发。

---

## 6. 无第三方依赖的关键自实现（便于对照/手算）

- **正态随机数**：`common.hpp::RNG::standard_normal()` 极坐标 Box-Muller（仅合成数据
  01 用；真实数据的 R/W 由共享 CSV 给定）。
- **分位数**：`quantile_vec` 排序后线性插值（与 numpy `linear` 一致）。
- **128 位指纹**：`pack_bits` 按 MSB first 打包（等价 numpy `packbits`）；汉明距离
  = 16 字节异或后的 popcount 之和。
- **位移聚合**：每个位移取重叠窗口汉明相似度的中位数，再在 2·5+1 个位移里取最高 3 个平均。
- **GCN**：`build_P()` 构造对称归一化邻接 P=D̂⁻¹ᐟ²(A+I)D̂⁻¹ᐟ²；前向 M1=P·X·W0+b0、
  H1=ReLU、Z=P·H1·W1+b1；边 φ=σ(z_u·z_v)；对 MSE 反传（dZ 对边两端 scatter、
  dW1=H1ᵀP·dZ、dH1=P·dZ·W1ᵀ、dW0=XᵀP·dM1），手写 Adam（0.9,0.999,1e-8）。
  参数量 2×16+16+16×16+16=**320**。
- **GBDT**：`13_*.cpp` 内 `GBDT` 类——二分类 logistic，F0=log(pos/neg)，每轮以负梯度
  y−p 拟合深度 2 的回归树（按特征排序中点枚举分裂、最小化 SSE、min_leaf 约束、
  相等特征值不分裂、严格正增益），叶值 γ=Σg/Σp(1−p)，F+=lr·γ。
- **ROC-AUC**：Mann–Whitney 统计量（正样本分数高于负样本的概率），可直接手算核对。
- **真实 NPZ（类版）**：`NpyArray` 解析/生成 NPY（magic `\x93NUMPY`+版本+头长度+dict 头，
  支持 `<f8 / <i8 / <i4 / |u1`，头按 64 字节对齐）；`NpzArchive` 实现 ZIP（local header、
  central directory、EOCD，CRC32），压缩方式 8=deflate，用 zlib `inflateInit2/deflateInit2`
  的 raw 模式（windowBits=−15）。numpy 可直接打开，实测与 Python 输出 phi 的 RMSE=0、
  Z 最大偏差 4e-15（浮点级）。
- **线程池（类版）**：`ThreadPool` 用固定数量 `std::thread` + 互斥/条件变量任务队列，
  `parallel_for` 按连续块分块并经 Latch 等待；本机 560 节点全流程单线程 4.96 s → 16 线程
  2.75 s（1.81×；非线性受内存带宽、dZ 串行累加与小矩阵调度开销限制）。

---

## 7. 论文固定参数 vs 教学取值

**严格遵循论文**：L=2000、25% 重叠；七级 Δ 阈值；三尺度 (32/8)(64/16)(128/32)；
7 维计数；128 位 SimHash（公式2）；汉明相似度（公式3）；有限位移对齐；LSH 粗筛；
节点=段、节点特征[度,邻边均权]；两层 GCNConv、16 维、φ=σ(z_u·z_v)（公式4）、
MSE 自监督（公式5）；边融合（公式6）；四特征（公式7–10）；两级级联（公式11）；
Recall（公式12）；连接级划分。

**教学取值**（代码注释与输出均标注）：指数平滑 α=0.30；合成数据参数；LSH 8 band×16
位与多点碰撞证据阈值；位移 ±5、top-3；θ0/θ1 用 Q30/Q60；top-k=5；GCN H=16、
300 轮、lr=1e-2；手写 GBDT 替代 LightGBM；τ1/τ2 的 OOF 数据驱动标定与召回地板；
每类抽 K=16 连接、每连接 8000 IPD。

---

## 8. 输出文件（`outputs/`）

`01_connections.bin` … `14_metrics.bin`，每个文件由若干具名数组构成（`Store::save/load`），
可在任意步骤重新运行而不影响其它步骤（后一步只读前一步的 `.bin`）。真实/合成模式还会
写一个 `data_source` 标记，供第 14 步显示标题。
