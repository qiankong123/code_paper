# QuadFP-Cascade 论文复现 —— C++ 版（与 Python 版一一对应）

用**纯 C++11 标准库**（无任何第三方依赖）1:1 复现论文
《多尺度指纹级联框架 QuadFP-Cascade：…网络隐蔽定时信道（Covert Timing Channels, CTC）检测》
（*Computer Communications* 251 (2026) 108495）的**全部计算过程**。

本目录与上一级 Python 版**严格对应**：同样的数字前缀、同样的分步、同样的固定参数、同样的算法与
输出结构，可编译、可运行、可逐步对照。

---

## 1. 编译与运行

### 环境
- 编译器：g++（已在本机 Strawberry Perl 自带的 `g++ 4.9.2 (i686)` 验证；任何支持 C++11 的 g++/clang 均可）。
- 依赖：**无**。只用到 `<vector> <random> <cmath> <fstream> <algorithm> …` 等标准库头文件。
- 平台：Windows（`-static` 静态链接，无需额外 DLL）；Linux/macOS 去掉 `-static` 即可。

### 一键编译（Windows）
```bat
build.bat
```
它会对每个 `??_*.cpp` 执行：
```
g++ -std=c++11 -O2 -static -o NN_*.exe NN_*.cpp
```

### 一键运行（按顺序 01→14）
```bat
run_all.bat
```
中间产物写入 `outputs/NN_*.bin`（自定义二进制归档，等价于 Python 版的 `.npz`）。

### 手动单步（示例）
```bat
g++ -std=c++11 -O2 -static -o 01_generate_ipd.exe 01_generate_ipd.cpp
01_generate_ipd.exe
```
> 控制台中文：程序内已调用 `SetConsoleOutputCP(65001)`，`build.bat`/`run_all.bat` 也会 `chcp 65001`。

---

## 2. 14 个步骤与 Python 版对应表

| 文件 | 对应论文 | 计算内容 |
|---|---|---|
| `common.hpp` | 00_common.py | 固定参数、RNG、统计、矩阵、二进制序列化、指纹/汉明/位移聚合公共工具 |
| `01_generate_ipd.cpp` | §3.1 数据 | 合成合法 HTTP/HTTPS/SSH（log-AR1+突发）与 5 类隐蔽定时信道（IPCTC/LNCTC/TRCTC/Jitterbug/ε-klibur） |
| `02_preprocess_segment.cpp` | §3.2.1 | 指数平滑 α=0.30；L=2000、25% 重叠分段（**段=图节点**） |
| `03_change_rate_symbol.cpp` | Table 1 | 相邻变化率 Δ=(xₜ₊₁−xₜ)/xₜ，按阈值映射为七级符号 −3…+3 |
| `04_multiscale_windows.cpp` | §3.2.1 / Table 2 | 三尺度滑窗（win/step=32/8、64/16、128/32）→ 每窗口 7 维计数向量 v（和=win） |
| `05_random_matrix_fingerprint.cpp` | 公式(2) | 固定随机矩阵 R∈ℝ^{128×7}，h=sign(Rv) 得 128 位 SimHash，打包成 16 字节 |
| `06_hamming_similarity.cpp` | 公式(3) | 汉明距离/相似度（异或+popcount）；含"翻 6 位→0.9531"手算 |
| `07_robust_aggregation.cpp` | §3.2.1 | 有限位移 ±5 对齐 + top-3 位移聚合，得段对三尺度 w1/w2/w3，m=max |
| `08_lsh_candidate.cpp` | §3.2.2 | 8 band×16 位 LSH：窗口级指纹+位置对齐+多点碰撞证据；理论召回 R(s)=1−(1−s¹⁶)⁸ 与蒙特卡洛 |
| `09_threshold_graph.cpp` | §3.2.2 / Fig.3 | 用合法-合法 m 的 Q30/Q60 标定 θ0/θ1；边分流；建稀疏相似图 A；节点特征[度,邻边均权] |
| `10_gnn_gcn.cpp` | 公式(4)(5) | 两层 GCNConv（H=16）自监督 MSE：手写前向/反向/Adam；含 3 节点链手算 |
| `11_edge_fusion.cpp` | 公式(6) | 边强度 s=α1w1+α2w2+α3w3+β·φ，网格搜索（α 单纯形、β 集合）最大化两类分离度 |
| `12_graph_features.cpp` | 公式(7)(8) | 每段 4 维特征：符号熵 H_sym、周期节奏 L_cy、图累加 ρ、图流量 √S_flow（top-k） |
| `13_two_level_classifier.cpp` | 公式(11) | 孤立硬规则（ρ=0→隐蔽）+ 两级 GBDT（τ1 召回导向、τ2 精度导向）；连接级 OOF 标定 |
| `14_evaluation.cpp` | 公式(12)/Table 4 | 混淆矩阵、Accuracy/Precision/Recall/FPR/F1、ROC-AUC（Mann–Whitney）、分信道召回 |

---

## 3. "一模一样"的准确含义

**完全一致（刻意对齐）**
- 全部固定参数：种子 `SEED=20261014`、`R_SEED=SEED+7`；L=2000、步长 1500；三尺度 (32/8)(64/16)(128/32)；
  七级阈值；128 位指纹=16 字节；8 band×16 bit；α=0.30；位移 ±5、top-3；θ 用 Q30/Q60；
  GCN H=16/EPOCHS=300/LR=0.01/Adam(0.9,0.999)；两级 GBDT 150/40 棵、深度 2、叶 3/8、lr 0.05/0.10；
  连接级划分 `conn%3==0` 测试、`conn%5` 折 OOF；孤立阈值 1e-6；τ1=正类 OOF 1% 分位、τ2=合法 OOF 99% 分位。
- 判定逻辑与数据流：**孤立硬规则 → 第一级召回阈值 → 第二级精度阈值**，完全一致。
- 数据规模：64 连接 × 8000 IPD → 320 段（合法 120 + 5 类隐蔽各 40）。

**有意不同、但等价（已在代码注释标注）**
- 随机数：Python 用 numpy `PCG64`（`default_rng`），C++ 用 `std::mt19937` + **极坐标 Box-Muller**。
  两者**同分布、同种子、同调用顺序**，但不是同一个数，因此合成数据的**具体数值不逐位相同**；
  算法、参数、结构、结论保持一致（见第 4 节实测）。
- 中间归档：Python 用 `.npz`，C++ 用自定义二进制 `.bin`（`common.hpp` 中 `Store`：数组名/类型/shape/字节）。
- GBDT：Python 用 sklearn `GradientBoostingClassifier`，C++ **纯标准库自实现**等价的
  二分类 logistic 梯度提升（F0=log(pos/neg)，负梯度 y−p，深度受限回归树，叶值 Σg/Σp(1−p)）；
  树的具体分裂点与 sklearn 不逐位相同，但模型类型与超参一致。
- 交叉验证：Python 用 sklearn `GroupKFold(5)`，C++ 用确定性等价的 `fold = conn_id % 5`
  （同样保证同连接的段不跨训练/验证）。

---

## 4. 本机实测结果（固定种子，可复现）

**C++（本目录，mt19937；`build.bat` 后 `run_all.bat` 干净运行的测试集 110 段）：**
```
混淆矩阵          预测合法   预测隐蔽
  实际合法          TN=40      FP=0
  实际隐蔽          FN=15      TP=55
Accuracy=0.8636  Precision=1.0000  Recall=0.7857  FPR=0.0000  F1=0.8800  ROC-AUC=0.8741
分信道召回：IPCTC=1.0000  LNCTC=1.0000  TRCTC=1.0000  Jitterbug=1.0000  EpsilonKlibur=0.0000
两级阈值：τ1=0.0281（召回导向）  τ2=0.8358（精度导向）
```

**Python 版（同参数，numpy PCG64）对应结果：**
```
TN=37 FP=3 FN=15 TP=55
Accuracy=0.8364  Precision=0.9483  Recall=0.7857  FPR=0.0750  F1=0.8594  ROC-AUC=0.8586
大倍率三类=1.0000  Jitterbug=1.0000  EpsilonKlibur=0.0000
```

**两版共同结论（这才是复现要验证的东西）**
1. 大倍率三类 **IPCTC/LNCTC/TRCTC 在两级阈值下与任何段都不成边，成为孤立节点（ρ=0），
   由图拓扑硬规则 100% 判隐蔽**：这三类共 120 段在图中 100% 孤立（干净运行另有 22 个
   高相似但连接稀疏的 Jitter 段也成为孤立节点、被硬规则一并捕获，故孤立节点共 142 个）。
2. 其余进入团内的 **Jitterbug 靠周期节奏特征**（H_sym 偏高、L_cy 偏小）被 GBDT 检出，
   与孤立硬规则合并后 Jitter 测试召回同样达到 1.0000。
3. **ε-klibur 最接近合法、是五类中最难的**：干净运行中其团内段特征与合法高度重叠，
   C++ 与 Python 两版 ε 召回都为 0.0000，第二级精度阈值优先保护合法（C++ 合法 FPR=0，
   Python 合法 FPR=0.075）。
4. 两版**测试集 TP=55、FN=15、Recall=0.7857 完全相同**，AUC（0.8741 vs 0.8586）、F1
   （0.88 vs 0.86）接近；差异只在合法误报个数（0 vs 3），来自自实现 GBDT/树分裂与 RNG
   噪声序列的不同，不影响任何一条定性结论。

> 本结果均来自**合成数据**，仅用于验证计算链路，不代表论文在真实 NCTC 数据集上的数值；
> 论文 Table 4 中五类信道 AUC≥0.997、Recall 都很高。

---

## 5. 无第三方依赖的关键自实现（便于对照/手算）

- **正态随机数**：`common.hpp::RNG::standard_normal()` 极坐标 Box-Muller，u,v 取 2·U−1 ∈[−1,1]。
- **分位数**：`quantile_vec` 排序后线性插值（与 numpy `linear` 一致）。
- **128 位指纹**：`pack_bits` 按 MSB first 打包（等价 numpy `packbits`）；
  汉明距离 = 16 字节异或后的 popcount 之和。
- **位移聚合**：每个位移取重叠窗口汉明相似度的**中位数**，再在 2·5+1 个位移里取最高 3 个平均。
- **GCN**：`build_P()` 构造对称归一化邻接 P=D̂⁻¹ᐟ²(A+I)D̂⁻¹ᐟ²；
  前向 M1=P·X·W0+b0、H1=ReLU、Z=P·H1·W1+b1；边 φ=σ(z_u·z_v)；
  对 MSE 反传（dZ 对边两端 scatter、dW1=H1ᵀP·dZ、dH1=P·dZ·W1ᵀ、dW0=XᵀP·dM1），
  手写 Adam（b1=0.9,b2=0.999,ε=1e-8）。参数量 2×16+16+16×16+16=**320**。
- **GBDT**：`13_*.cpp` 内 `GBDT` 类——二分类 logistic，F0=log(pos/neg)，
  每轮以负梯度 y−p 拟合深度 2 的回归树（按特征排序中点枚举分裂、最小化 SSE、min_samples_leaf），
  叶值 γ=Σg/Σp(1−p)，F+=lr·γ。
- **ROC-AUC**：Mann–Whitney 统计量（正样本分数高于负样本的概率），可直接手算核对。

---

## 6. 论文固定参数 vs 教学取值

论文只给出框架与部分参数；为把链路跑通，下列为**教学取值**（已在对应 `.cpp` 注释与输出中明确标注）：
合成数据规模与 log-AR1 参数、突发率、5 类编码倍率/周期、LSH 的 band 数/位置容差/证据阈值、
位移范围与 top 数、GCN 维度/训练轮数、边融合权重网格、GBDT 树数、连接级 OOF 折法等。
真实复现论文指标须替换为 NCTC 真实数据与论文实验设置。

---

## 7. 输出文件（`outputs/`）

`01_connections.bin` … `14_metrics.bin`，每个文件由若干具名数组构成（`Store::save/load`），
可在任意步骤重新运行而不影响其它步骤（后一步只读前一步的 `.bin`）。
