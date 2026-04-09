---
title: "模型效率与量化：面试问题与推荐回答"
author: yo3nglau
date: '2026-04-08'
categories:
  - Interview
tags:
  - Deep Learning
  - Quantization
  - Model Compression
toc: true
---

## 量化基础

### Q1 [基础] 解释均匀量化的关键设计选择

**Q:** 哪些数值决策决定了浮点权重张量到整数的映射方式？对称方案与非对称方案有何区别？

**A:** **均匀量化**（Uniform quantization）使用两个参数——**缩放因子**（scale factor）$s$ 和**零点**（zero-point）$z$——将浮点范围 $[x_{\min}, x_{\max}]$ 映射到 $2^b$ 个均匀分布的整数值。位宽 $b$ 决定可表示的级别数量：INT8 提供 256 级，INT4 仅提供 16 级。缩放因子由张量范围决定，截断操作会对超出选定范围的值引入**裁剪误差**；校准时如何设置 $x_{\min}$ 和 $x_{\max}$ 是裁剪误差与舍入误差之间的权衡（Nagel et al., 2021）。

**对称**量化令 $z = 0$ 并对称裁剪（$x_{\min} = -x_{\max}$），从而简化计算：零恰好映射到整数 0，零点可以从乘加运算路径中省略。**非对称**量化允许 $z \neq 0$，通过平移量化网格以更高效地覆盖实际数据范围——例如，ReLU 后的激活值始终非负，非对称网格可将所有级别集中于 $[0, x_{\max}]$。代价是每次矩阵乘法都需要额外的零点修正项。

粒度是第三个维度：**逐张量**（per-tensor）量化对整个权重矩阵使用单一的 $s$ 和 $z$，而**逐通道**（per-channel）量化对每个输出通道使用独立值。不同输出通道的动态范围可能差异很大，因此逐通道校准能够以可忽略不计的推理开销显著降低量化误差。Nagel et al. (2021) 对这些选择在 CNN 和 Transformer 架构中的相互作用进行了全面分析。

---

### Q2 [基础] 区分训练后量化与量化感知训练

**Q:** PTQ 和 QAT 在量化相对于训练的时机上有何不同？如何判断哪种方案更合适？

**A:** **训练后量化**（Post-training quantization，PTQ）在不进行任何基于梯度的优化的情况下，对已经完全训练好的浮点模型应用量化。通过将少量校准数据集（通常数百至数千个样本）送入模型，收集激活统计信息以逐层设定缩放因子和零点。PTQ 无需访问原始训练流程，可在数分钟内完成量化。其局限在于量化误差无法通过权重更新来补偿，因此精度下降幅度大于 QAT——尤其在低于 8 位时——敏感层可能需要更精细的校准（Nagel et al., 2021）。

**量化感知训练**（Quantization-aware training，QAT）在训练时向前向传播中插入**伪量化**（fake quantization）节点：每个张量被量化后立即反量化，将量化误差注入损失函数，同时以浮点形式保留用于梯度传播的计算。由于舍入操作的梯度为零，**直通估计器**（straight-through estimator，STE）将梯度当作恒等函数直接通过截断操作传递（Jacob et al., 2018）。这使优化器能够学习对量化更鲁棒的权重，通常能在 INT8 下弥补大部分 PTQ 的精度损失，并在 INT4 下达到可用精度。

实践中，当训练数据不可用、目标精度为 INT8（PTQ 精度损失不大）或需要快速部署时，优先选用 PTQ。当目标精度为 INT4 或更低、或任务特定精度要求严格时，QAT 的基础设施成本是值得的。

---

### Q3 [进阶] 解释为何激活量化比权重量化更困难

**Q:** 神经网络激活值的哪些特性给量化带来了权重量化中不存在的挑战？这一问题在大型语言模型中如何加剧？

**A:** 权重在训练后是**静态**的：其分布在量化时完全已知，可根据保存的检查点精确计算最优的逐通道缩放因子。激活值是**动态**的：其分布随每个输入而变化。在代表性数据集上校准的缩放因子可能与分布外输入不匹配，导致过度裁剪（大值饱和）或过度舍入误差（缩放过于保守，浪费位宽于正常范围）。这种输入依赖性迫使选择保守的静态校准或推理时昂贵的逐 token 动态量化，两者均存在精度或延迟代价。

更根本的问题是，Transformer 模型中的激活分布可能高度非高斯。Dettmers et al. (2022) 发现，在参数量约 6B 以上的 OPT 和 BLOOM 模型中，极少数（约 $0.1\%$）的激活维度在所有输入上持续呈现比其他维度大 100 倍的值——即**系统性离群通道**（systematic outlier channels）。将整个张量量化到这些离群值的尺度会浪费几乎所有可表示的整数；量化到正常范围则使离群值饱和，两者均无法在 INT8 下接受。

第三个挑战是，逐通道量化对权重非常有效，但难以应用于按（batch, token, channel）索引的激活值：同一 batch 中不同 token 的逐通道统计特性差异很大，使静态逐通道校准不可靠。动态范围、输入依赖性与系统性离群值的组合，解释了为何激活值需要专用解决方案（SmoothQuant、LLM.int8()），而不能像权重那样直接使用逐通道 PTQ。

---

### Q4 [进阶] 描述 GPTQ 如何实现千亿参数模型的精确一次性权重量化

**Q:** GPTQ 应用何种优化原理以最小化逐层量化误差？哪些算法改进使其在 1750 亿参数规模下可行？

**A:** GPTQ 是一种基于**最优脑量化**（Optimal Brain Quantization，OBQ）框架的一次性 PTQ 方法（Frantar et al., 2022）。OBQ 将最优脑外科手术（Optimal Brain Surgeon，OBS）方法从剪枝扩展到量化：当权重 $w_q$ 被舍入为 $\hat{w}_q$ 时，同层中其余未量化的权重将被更新以补偿引入的误差。对剩余权重集合 $F$ 量化索引 $q$ 时的最优更新为：

$$\delta_F = -\frac{w_q - \hat{w}_q}{[\mathbf{H}_F^{-1}]_{qq}} \cdot (\mathbf{H}_F^{-1})_{:,q}$$

其中 $\mathbf{H}_F = 2\mathbf{X}\mathbf{X}^T$ 是该层输出重建误差关于权重的 Hessian（对于校准输入 $\mathbf{X}$ 的线性层）。这种逐权重的贪婪补偿利用二阶信息选择更优的量化目标。

朴素的 OBQ 每层的时间复杂度为 $O(d^3)$（来自 Hessian 求逆），且逐权重处理——对于数百万参数的层而言不可行。GPTQ 引入两项改进：第一，宽度为 128 的列块内所有权重同时量化，以**延迟批量**方式累积逆 Hessian 更新：预先计算后续更新所需的逆 Hessian 行并复用，减少冗余计算。第二，GPTQ 按列顺序处理权重，而非像 OBQ 那样动态重排，这对硬件更友好且具有更好的缓存效率。Frantar et al. (2022) 在单张 A100 上约 4 GPU 小时内将 OPT-175B 量化至 INT4，WikiText-2 上的困惑度下降不足 1 点。

---

## 大型语言模型量化

### Q5 [进阶] 解释 AWQ 如何识别并保护量化中的显著权重

**Q:** 是什么观察驱动了 AWQ 对权重通道的选择性处理？在不重新训练的情况下如何确定最优保护缩放比？

**A:** **AWQ**（Activation-aware Weight Quantization，激活感知权重量化）基于一项发现：对应于大幅值激活的权重通道对量化误差的贡献不成比例——量化至 INT4 时，约 $1\%$ 的权重通道主导了精度损失（Lin et al., 2023）。朴素的解决方法——保留这些显著通道为 FP16 同时量化其余通道——硬件效率低下，因为细粒度混合精度会引入不规则的内存访问。

AWQ 改为在量化前对每个通道应用**激活感知缩放**。将权重通道乘以 $s > 1$ 可降低其相对量化误差（$w$ 增大时 $\Delta w / w$ 减小），相当于在不改变整数格式的情况下为其分配更多精度。相应的激活维度除以 $s$ 以保持矩阵乘积不变。缩放离线应用于权重矩阵，并吸收到前一个归一化层的仿射参数中，推理开销为零。

最优逐通道缩放比通过在校准集上最小化输出重建误差确定：

$$s^* = \arg\min_{s} \|\mathbf{W}_q(\mathbf{s})\mathbf{x} - \mathbf{W}\mathbf{x}\|^2$$

其中 $\mathbf{W}_q(\mathbf{s})$ 表示 $\mathbf{W}$ 在按 $s$ 缩放后量化的结果。Lin et al. (2023) 将 $s_j = \bar{a}_j^\alpha$ 参数化，其中 $\bar{a}_j$ 是通道 $j$ 的平均激活幅值，$\alpha \in [0, 1]$ 在约 20 个值上网格搜索——使搜索高效且仅需小型校准集。在 LLaMA-7B INT4 上，AWQ 以相当的硬件吞吐量实现了低于 GPTQ 的困惑度，表明激活引导的缩放比纯二阶权重信息是更有效的量化敏感性代理。

---

### Q6 [基础] 对比仅权重量化与权重-激活联合量化的部署权衡

**Q:** 实践者何时应优先量化仅权重而非同时量化权重和激活？什么硬件特性驱动这一选择？

**A:** **仅权重量化**（W4A16 或 W8A16）以低精度存储模型权重，但以 FP16 执行矩阵乘法，在运算前即时反量化权重。主要收益是**内存带宽降低**：在批量大小为 1 的 LLM 自回归解码中，瓶颈在于从 GPU HBM 将权重矩阵加载到片上计算单元——INT4 的仅权重量化将此传输量减少 $4\times$，直接提升 token 吞吐量。GPTQ（Frantar et al., 2022）和 AWQ（Lin et al., 2023）等方法采用此策略，以最小精度损失在同等硬件上运行更大的模型。

**权重-激活联合量化**（W8A8）同时量化两个张量，从而能够使用整数算术单元——NVIDIA GPU 上的 INT8 张量核心提供比 FP16 单元更高的峰值吞吐量，在运算**受计算限制**时有益。这一情形发生于大批量推理，此时算术强度（每字节的 FLOP 数）足以让 ALU 而非内存总线成为瓶颈。SmoothQuant（Xiao et al., 2022）和 LLM.int8()（Dettmers et al., 2022）等方法面向此场景，实现多并发请求的高吞吐量服务。代价是激活量化的额外难度（参见 Q3）。

实践中，单用户 LLM 推理通常强烈受内存带宽限制（优选 W4A16），而以大批量服务多并发请求通常受计算限制（优选 W8A8）。转折点取决于模型大小、硬件代次和服务批量大小。

---

### Q7 [进阶] 描述 SmoothQuant 如何将量化难度从激活转移到权重

**Q:** SmoothQuant 核心的数学恒等式是什么？为何将缩放从激活转移到权重能解决离群值问题？

**A:** **SmoothQuant** 利用一项关键结构性观察：Transformer 激活中的离群通道在 token 之间**保持一致**——相同的特征维度对所有输入都表现为大值（Xiao et al., 2022）。这种一致性意味着离群模式可以离线表征，并通过固定的逐通道变换加以抵消。对于任意逐通道缩放向量 $\mathbf{s} \in \mathbb{R}^{C_{\text{in}}}$，线性层输出通过以下变换精确保持：

$$\mathbf{Y} = \mathbf{X}\mathbf{W} = (\mathbf{X}\operatorname{diag}(\mathbf{s})^{-1}) \cdot (\operatorname{diag}(\mathbf{s})\mathbf{W})$$

将 $s_j$ 设置为与激活中通道 $j$ 的观测幅值成比例，变换后的激活 $\hat{\mathbf{X}} = \mathbf{X}\operatorname{diag}(\mathbf{s})^{-1}$ 将离群通道缩减至与其他通道相同的范围，而变换后的权重 $\hat{\mathbf{W}} = \operatorname{diag}(\mathbf{s})\mathbf{W}$ 则将对应列放大。权重本身已易于量化，且权重的逐通道缩放可直接吸收到量化缩放因子中——难度迁移到了能够很好处理它的介质。

迁移强度由逐通道参数 $\alpha$ 控制：

$$s_j = \frac{\max(|\mathbf{X}_{:,j}|)^\alpha}{\max(|\mathbf{W}_{j,:}|)^{1-\alpha}}$$

$\alpha = 0.5$ 是鲁棒的默认值。缩放离线折入前一个 LayerNorm 的仿射参数，推理开销为零。Xiao et al. (2022) 在困惑度和下游零样本基准上均展示了 OPT-175B 和 BLOOM-176B 接近无损的 INT8 推理，实现了 176B 规模模型的首个实用 W8A8 量化。

---

### Q8 [进阶] 解释 LLM.int8() 如何处理 Transformer 模型中的激活离群值

**Q:** LLM.int8() 使用什么计算分解来实现精确的 INT8 推理？为何这种方法专门在大型模型中才变得必要？

**A:** **LLM.int8()**（Dettmers et al., 2022）通过**混合精度分解**解决离群值问题：不是量化整个激活张量，而是根据每个输入特征维度是否包含离群值来拆分矩阵乘法。激活值超过阈值（通常为 $|x| > 6$）的特征维度被提取到**离群矩阵** $\mathbf{X}_o \in \mathbb{R}^{B \times C_o}$；其余部分构成 $\mathbf{X}_r \in \mathbb{R}^{B \times C_r}$。权重矩阵对应地划分为 $\mathbf{W}_o$ 和 $\mathbf{W}_r$：

$$\mathbf{Y} = \mathbf{X}_o \mathbf{W}_o^T + \operatorname{Int8}(\mathbf{X}_r) \cdot \operatorname{Int8}(\mathbf{W}_r)^T$$

离群部分（约 $0.1\%$ 的特征维度）以 FP16 计算；其余约 $99.9\%$ 使用 INT8 张量核心。INT8 路径主导总开销；FP16 路径仅增加少量额外开销。由于离群通道具有持续性——Dettmers et al. (2022) 表明它们在模型规模（$\gtrsim 6$B 参数）处以相变形式出现，且对所有输入保持一致——离群集合可从校准数据中一次性确定，并在推理时复用。

实际动机是内存：FP16 175B 参数模型需要约 350 GB GPU 内存；LLM.int8() 将权重存储降至约 175 GB（INT8），使 176B 模型可在 4× 48 GB GPU 上推理，而 FP16 需要 8× GPU。Dettmers et al. (2022) 报告在标准零样本基准上对 OPT 和 BLOOM 模型的性能下降不足 $1\%$。

---

## 剪枝与稀疏性

### Q9 [基础] 对比结构化剪枝与非结构化剪枝对硬件利用率的影响

**Q:** 为何通过非结构化剪枝实现 90% 稀疏度往往无法带来实际推理加速，而 50% 稀疏度的结构化剪枝却可以？

**A:** **非结构化剪枝**（Unstructured pruning）无论位置如何地移除单个权重，在权重矩阵中产生散落各处的不规则零值模式。尽管稀疏率很高，现代 GPU 和 TPU 仍在密集的矩形块上执行矩阵乘法——除非零值恰好符合特定硬件支持的模式（如 NVIDIA A100 的 2:4 结构化稀疏），否则不会跳过对单个零值的计算。没有专用稀疏算子时，90% 稀疏的密集矩阵仍占用与原始矩阵相同的内存布局，执行时间也大致相同（Han et al., 2016）。即使使用稀疏格式（CSR、CSC），不规则的索引开销和较差的向量化也往往使实际加速比远低于稀疏率所示。

**结构化剪枝**（Structured pruning）移除整个计算单元——输出通道、注意力头、FFN 神经元或整个 Transformer 层——产生更小的**密集**模型。移除了 $50\%$ 输出通道的模型在任何支持密集运算的硬件上，对应矩阵乘法的开销直接降为原来的 $50\%$，无任何稀疏格式开销。被剪枝的维度直接从权重张量形状中消失。

实践意义在于：非结构化剪枝主要是**存储压缩**技术——减少磁盘和内存中的模型大小——而结构化剪枝在推理时直接降低延迟和内存占用。对于在普通硬件（CPU、标准 GPU）上部署，结构化剪枝是实现可测量加速的途径。

---

### Q10 [进阶] 解释彩票假设及其对神经网络过参数化的揭示

**Q:** Frankle 和 Carlin 在密集网络中的稀疏子网络上发现了什么？子网络的初始化为何重要？

**A:** **彩票假设**（Lottery Ticket Hypothesis，Frankle & Carlin, 2019）指出：随机初始化的密集网络中包含一个稀疏子网络——即"中奖彩票"——当从其**原始初始化**（而非重新随机初始化）开始单独训练时，能以相当的步数收敛到与完整网络相似甚至更好的精度。Frankle & Carlin (2019) 通过**迭代幅值剪枝**找到这些彩票：训练网络，剪除最终幅值最小的 $p\%$ 权重，将剩余权重重置为其初始化值，重复此过程。经过数轮后，在 MNIST 和 CIFAR-10 上，原始大小 $10$–$20\%$ 的稀疏子网络匹配了完整网络的精度。

关键且令人惊讶的发现是**初始权重值至关重要**：取中奖子网络但随机重新初始化而非使用原始初始化，会破坏其性能优势。中奖彩票的初始值似乎编码了一条有利的优化轨迹——也许是训练早期更好的梯度信号。这意味着密集网络的作用部分是搜索过程：隐式地识别应保留哪个稀疏子结构以及哪种初始化对其有效。

对于更大的网络（ResNet-50、语言模型），Frankle & Carlin (2019) 发现纯粹的初始化重置不再有效：权重必须重置到早期少量训练步骤时的值（**权重回溯**，rewinding），而非初始化值。这种"延迟重置"要求表明，大型网络的有效归纳偏置是在早期训练动态中建立的，而非在初始化时。该假设对理解过参数化为何有助于优化——它增加了包含良好初始化中奖彩票的概率——以及激励在训练前而非训练后寻找稀疏子网络的研究具有重要意义。

---

### Q11 [进阶] 描述 SparseGPT 如何实现大型语言模型的一次性剪枝

**Q:** 为何迭代剪枝对 LLM 不可行？SparseGPT 使用何种算法原理在不进行任何梯度更新的情况下完成剪枝？

**A:** 迭代剪枝在每轮剪枝后需要基于梯度的微调以恢复精度。对于 ResNet，这是标准做法，但对于 1000 亿以上参数的 LLM，即使一个 epoch 的微调也需要对完整预训练语料库进行反向传播——在大型研究集群之外计算上不可行，即便在集群内也往往不切实际。**SparseGPT**（Frantar & Alistarh, 2023）通过为每个线性层求解权重重建问题，消除了对任何梯度更新的需求：寻找稀疏权重 $\hat{\mathbf{W}}$ 以最小化在小型校准集上的输出重建误差：

$$\min_{\hat{\mathbf{W}}} \|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_F^2 \quad \text{s.t.} \quad \|\hat{\mathbf{W}}\|_0 \leq k$$

使用与 GPTQ（Q4）相同的 OBS/OBQ 二阶框架求解：当一个权重被剪枝（置零）时，行内其余权重通过逆 Hessian 修正量 $\delta_F$ 更新以补偿引入的误差。处理逐行进行，可并行化，并使用延迟批量更新技巧跨权重块摊销 Hessian 运算。

Frantar & Alistarh (2023) 证明，OPT-175B 可以在 WikiText-2 上仅损失不足 1 点困惑度的情况下被剪枝至 $50\%$ 非结构化稀疏度，在单张 A100 上约 4 GPU 小时内完成。在 $2{:}4$ 结构化稀疏（支持硬件加速的 NVIDIA 原生格式）下，SparseGPT 实现了约 2 点困惑度的下降——达到可用的精度水平。他们还展示了将 SparseGPT 与量化结合（稀疏-量化模型）在更高压缩比下仍能保持竞争性精度，表明稀疏性和量化沿不同压缩轴互补。

---

### Q12 [进阶] 对比静态与动态稀疏训练，以及 RigL 如何推进技术前沿

**Q:** 为何从随机固定掩码开始训练稀疏网络的性能不及先密集训练再剪枝的流程？RigL 如何在不经过密集训练阶段的情况下克服这一问题？

**A:** **静态稀疏训练**——在初始化时固定随机稀疏掩码，仅训练选定权重——在相同最终稀疏度下通常不及先密集训练再剪枝的流程。彩票假设解释了原因：随机稀疏初始化缺乏幅值剪枝在密集训练后发现的有利结构。中奖彩票的价值来自其拓扑结构和初始权重两方面；随机掩码加重新初始化的权重两者皆不具备。静态稀疏训练因此从结构上较差的子图出发，即使在高学习率下也难以优化。

**动态稀疏训练**交替进行稀疏训练步骤和掩码更新步骤，后者在修剪已有连接的同时生长新连接。**RigL**（Rigging the Lottery，Evci et al., 2020）通过在稀疏前向传播中计算损失相对于当前零权重的**瞬时梯度**来生长连接——梯度大的权重若被激活将显著降低损失。连接按**权重幅值**修剪，因为小的有效权重对输出贡献甚微。掩码每 $\Delta T$ 步更新一次，更新比例按余弦衰减——从大（积极的拓扑探索）到小（随训练收敛而减少）。

Evci et al. (2020) 表明，RigL 在与标准密集训练相同训练 FLOP 下，在 ImageNet ResNet 的高稀疏度（$80$–$99\%$）上匹配或超越带微调的迭代幅值剪枝。在 $2\times$ FLOP 预算下，RigL 在 $90\%$ 稀疏度上超越了密集基线——首个在匹配 FLOP 成本下超越密集训练质量的稀疏训练方法。关键优势在于在训练过程中发现任务自适应的稀疏模式，而非依赖剪枝流程事后发现的幅值排序。

---

## 知识蒸馏与低秩压缩

### Q13 [基础] 解释软目标知识蒸馏的工作原理及教师模型超越硬标签的提供

**Q:** 教师模型的输出分布编码了独热标签丢弃的什么信息？温度缩放如何影响蒸馏信号的质量？

**A:** **知识蒸馏**（Knowledge distillation，Hinton et al., 2015）训练较小的学生模型匹配较大教师模型的完整输出分布，而非仅仅预测真实硬标签。训练良好的教师会为错误类别分配非零概率——例如，对于一张汽车图像，可能输出 $p(\text{car}) = 0.7$，$p(\text{truck}) = 0.2$，$p(\text{bus}) = 0.08$。这些**软目标**（soft targets）编码了教师关于**类间相似度**的知识：较高的汽车/卡车概率反映了这两个类别在视觉上相似且易混淆。硬标签（car=1，其余=0）完全丢弃了这种关系信息。基于软目标训练的学生不仅学到"这是汽车"，还学到"汽车比鸟更像卡车"，提供了更丰富的监督信号。

学生最小化硬标签交叉熵与教师软化分布 KL 散度的加权组合：

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{\text{CE}}(y,\, \sigma(\mathbf{z}_s)) + \lambda T^2\,\mathcal{L}_{\text{KL}}(\sigma(\mathbf{z}_t/T),\, \sigma(\mathbf{z}_s/T))$$

其中 $T$ 是**温度**参数。$T = 1$ 时，教师分布尖锐，对次要类别提供的信号很少。提高 $T$ 使分布平坦，放大非主导类别的相对概率。$T^2$ 因子补偿了高温时梯度幅值的减小（Hinton et al., 2015）。典型值为 $T \in [2, 20]$，由验证集选定。

---

### Q14 [进阶] 描述特征级蒸馏及对齐中间表示的挑战

**Q:** 特征级蒸馏传递了输出蒸馏无法传递的什么信息？在不同容量网络间匹配中间表示为何困难？

**A:** 输出蒸馏将教师的知识压缩为 $K$ 类概率向量。特征级蒸馏还额外传递中间层学习到的**内部表示**——空间特征图、注意力模式或隐藏状态——其中包含更丰富的结构信息。**FitNets**（Romero et al., 2015）提出训练学生最小化其中间激活与教师指定"提示"层之间的 MSE：

$$\mathcal{L}_{\text{hint}} = \frac{1}{2}\|f_t(\mathbf{x}) - W_r f_s(\mathbf{x})\|_2^2$$

其中 $W_r$ 是将学生特征维度投影到与教师匹配的可学习**回归器**（regressor）。该投影是必要的，因为教师与学生层通常宽度不同；若无投影，学生在每个中间层都需要精确复制教师的维度，这在架构上非常受限。

更深层的困难在于，不同容量的网络即使在解决相同任务时也会学习到本质上不同的内部表示。大型教师可能将信息分散于众多通道；小型学生可能将相同信息集中于更少但更密集的特征中。强迫学生精确模仿教师的高维特征图是过于刚性的约束，会阻止其找到自身的高效表示。激活值上的 MSE 也会被高方差通道主导，而这些通道未必是信息量最大的。**关系知识蒸馏**（Relational Knowledge Distillation，RKD，Park et al., 2019）通过传递关系结构——样本嵌入间的成对距离和角度——而非绝对激活值来解决这一问题，使学生能够自由寻找自身的表示几何结构，同时保留教师学到的相对结构。

---

### Q15 [基础] 解释 LoRA 如何在不修改原始权重的情况下减少微调的可训练参数

**Q:** LoRA 引入了何种低秩结构？它如何在保留预训练模型的同时实现任务特定适应？

**A:** **LoRA**（Low-Rank Adaptation，低秩适应，Hu et al., 2022）的动机是：将大型预训练模型适应下游任务所需的权重更新 $\Delta W$ 具有低内在秩——任务特定的适应存在于远低于完整权重矩阵的低维子空间中。LoRA 不更新完整的 $d \times k$ 权重矩阵，而是将更新参数化为两个小矩阵的乘积：

$$W' = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$。原始权重 $W_0$ 被冻结；仅训练 $A$ 和 $B$。对于 $d = k = 12{,}288$ 且 $r = 8$ 的 GPT-3 规模注意力投影，每个矩阵的可训练参数从约 $151\text{M}$ 降至约 $196\text{K}$——减少 $770\times$。

$A$ 从小高斯分布初始化，$B$ 初始化为零，使得初始时 $\Delta W = BA = 0$——训练从精确的预训练输出开始。在推理时，LoRA 权重通过一次性计算 $W' = W_0 + BA$ 合并，不产生任何推理开销。Hu et al. (2022) 表明，将 LoRA 应用于 GPT-3 的注意力权重矩阵，在下游 NLP 基准上与完整微调性能相当或更优，同时将可训练参数减少 $10{,}000\times$。针对单一基础模型可同时维护多个任务特定的 LoRA 适配器，每个适配器的存储开销可忽略不计。

---

### Q16 [进阶] 描述截断 SVD 如何近似并压缩线性层与卷积层

**Q:** 低秩矩阵分解如何减少层的参数量和 FLOP？训练好的权重矩阵的哪些特性限制了其有效性？

**A:** 给定权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$，其奇异值分解（SVD）为 $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，奇异值满足 $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$。**截断秩-$r$ 近似**仅保留前 $r$ 个奇异值三元组：

$$\hat{\mathbf{W}}_r = \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^T = (\mathbf{U}_r \boldsymbol{\Sigma}_r^{1/2})(\boldsymbol{\Sigma}_r^{1/2}\mathbf{V}_r^T)$$

这将一次开销为 $O(mn)$ 的矩阵乘法替换为两次依次开销为 $O(mr)$ 和 $O(rn)$ 的乘法，总计 $O(r(m+n))$——FLOP 减少系数为 $mn / (r(m+n))$。存储同样从 $mn$ 减少至 $r(m+n)$。

对于卷积网络，Denton et al. (2014) 将 SVD 应用于重塑后的滤波器组，在 VGG 风格网络上实现了约 $2\times$ 加速，ImageNet top-5 精度下降约 $1\%$。根本局限在于，训练网络权重矩阵的奇异值谱往往**衰减缓慢**：许多奇异值幅值相近，因此截断秩-$r$ 需要较大的 $r$ 才能避免显著的近似误差。靠近输入的层和方阵（$m \approx n$）往往具有更平坦的谱，难以压缩。压缩比因层而异——某些层的谱质量集中于前几个奇异值，其他层则广泛分布——因此非均匀秩分配（对关键层使用更大的 $r$）对于在精度-压缩比之间取得良好权衡至关重要。

---

## 高效架构与推理

### Q17 [基础] 解释深度可分离卷积相比标准卷积如何减少计算量

**Q:** 将卷积分解为深度卷积和逐点卷积两个阶段如何改变总运算量？计算节省来自哪里？

**A:** **标准卷积**对 $M$ 个输入通道、$N$ 个输出通道、核大小 $D_K \times D_K$ 和输出空间大小 $D_F \times D_F$ 执行 $D_K^2 \cdot M \cdot N \cdot D_F^2$ 次乘加（MAC）运算：$N$ 个输出滤波器中的每一个都在所有 $M$ 个输入通道上滑动。

**深度可分离卷积**（Depthwise separable convolution，Howard et al., 2017）将其分解为两个阶段。首先，**深度卷积**（depthwise convolution）对每个输入通道独立应用一个 $D_K \times D_K$ 滤波器，开销为 $D_K^2 \cdot M \cdot D_F^2$ MAC（捕获空间模式，无通道混合）。然后，**逐点卷积**（pointwise convolution，$1 \times 1$）混合通道信息以产生 $N$ 个输出特征，开销为 $M \cdot N \cdot D_F^2$ MAC。总开销：$(D_K^2 + N) \cdot M \cdot D_F^2$。相对于标准卷积的减少比例为：

$$\frac{D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

对于 $D_K = 3$ 和较大的 $N$，这趋近于 $1/9$，约 $8$–$9\times$ 的减少。Howard et al. (2017) 证明，完全由深度可分离卷积构建的 MobileNets 在 ImageNet 上比 VGG-16 的 top-1 精度低约 $1\%$，同时使用约 $27\times$ 更少的 MAC 和 $32\times$ 更少的参数——确立了深度可分离卷积作为移动端视觉模型关键基础运算的地位。

---

### Q18 [进阶] 描述 FlashAttention 如何在 Transformer 计算中降低内存并提升吞吐量

**Q:** 标准缩放点积注意力的内存与带宽瓶颈是什么？FlashAttention 的分块方法如何在不改变数学输出的情况下消除该瓶颈？

**A:** 标准缩放点积注意力 $\text{softmax}(\mathbf{QK}^T/\sqrt{d})\mathbf{V}$ 需要在 GPU HBM（高带宽内存）中实化 $N \times N$ 的注意力分数矩阵，其中 $N$ 为序列长度。这是 $O(N^2)$ 的内存占用，对长序列非常大。更关键的是，注意力是**IO 受限**而非计算受限的：现代 GPU 的算术吞吐量远超内存带宽，因此对大型激活张量的反复 HBM 读写主导了实际耗时。Dao et al. (2022) 估计，朴素 PyTorch 注意力的大部分时间花费在内存读写而非 FLOP 上。

**FlashAttention** 通过以适合片上 SRAM 的**分块**（tile）方式计算注意力来消除 $N \times N$ 矩阵的 HBM 实化。对于每个查询分块 $\mathbf{Q}_i$，它迭代所有键值分块 $\mathbf{K}_j$ 和 $\mathbf{V}_j$，使用**在线 softmax**（online softmax）算法维护运行统计信息：增量更新运行最大值 $m_i$ 和归一化常数 $\ell_i$，使得可以在不存储所有分数的情况下修正部分注意力输出：

$$m_i^{(j)} = \max\!\left(m_i^{(j-1)},\, \operatorname{rowmax}\!\left(\mathbf{Q}_i \mathbf{K}_j^T / \sqrt{d}\right)\right)$$

最终输出逐分块组装，仅需一次扫描，从不将 $N \times N$ 矩阵写入 HBM。数学结果与标准注意力完全相同（Dao et al., 2022）。

FlashAttention 将 HBM 内存从 $O(N^2)$ 降至 $O(N)$，在 A100 GPU 上对序列长度 1K–16K 实现了比 PyTorch 注意力快 2–4× 的实际加速。FlashAttention-2（Dao, 2023）通过在序列维度上更好的并行化和减少注意力算子中的非矩阵乘 FLOP，进一步将吞吐量提升约 $2\times$。

---

### Q19 [进阶] 解释推测解码如何加速自回归 token 生成

**Q:** 推测解码利用了标准逐 token 生成的何种低效之处？它如何保证输出分布不变？

**A:** 在标准自回归生成中，目标 LLM 每次前向传播生成一个 token。对于大型模型，每次前向传播都是**内存带宽受限**的：每生成一个 token 都需要从 HBM 加载权重矩阵，在批量大小为 1 时 GPU 算术单元大部分时间在等待数据。生成长度为 $T$ 的序列需要 $T$ 次顺序前向传播，每次都未充分利用可用的计算能力（Leviathan et al., 2022）。

**推测解码**（Speculative decoding）使用小型快速的**草稿模型**（draft model）通过 $k$ 次廉价的顺序步骤提出 $k$ 个候选 token。这 $k$ 个 token 随后在目标模型的**单次并行前向传播**中验证。由于目标模型对 $k$ 个 token 的一次传播只加载一次权重且同时处理所有位置，其开销约等于一次标准 token 生成步骤——但可能接受多个 token。当所有 $k$ 个草稿 token 均被接受时，算法实现 $k\times$ 吞吐量；若有拒绝，至少始终产生一个新 token。

正确性由拒绝采样方案保证：位置 $i$ 处的草稿 token $x_i$ 以概率 $\min(1,\, p(x_i|x_{<i}) / q(x_i|x_{<i}))$ 被接受，其中 $p$ 是目标分布，$q$ 是草稿分布。若被拒绝，则从修正分布 $\text{norm}(\max(0,\, p(\cdot) - q(\cdot)))$ 采样新 token。所得 token 序列与**目标模型单独采样的分布完全相同**——不引入任何近似（Leviathan et al., 2022）。

期望加速约为 $\frac{1-\alpha^{k+1}}{(1-\alpha)(1 + k \cdot c)}$，其中 $\alpha$ 是逐 token 接受概率，$c$ 是一次草稿步骤与一次目标步骤的开销比。Leviathan et al. (2022) 以 T5-Small 作为草稿模型，在 T5-XXL 上报告了 2–3× 的加速，输出质量完全保持。

---

### Q20 [基础] 描述可微分架构搜索如何找到高效神经网络架构

**Q:** 早期 NAS 方法为何代价极高？DARTS 如何通过连续松弛使架构搜索变得可行？

**A:** 经典神经架构搜索（Zoph & Le, 2017）使用强化学习控制器采样候选架构，在目标数据集上从头训练每个架构以评估验证精度，并根据奖励更新控制器。Zoph & Le (2017) 需要 800 个 GPU 运行 28 天才能在 CIFAR-10 上发现有竞争力的架构——独立评估每个架构的代价使得在大搜索空间上进行穷举或强化学习引导的搜索对大多数研究场景不可行。

**DARTS**（Differentiable Architecture Search，可微分架构搜索，Liu et al., 2019）通过**连续松弛**使搜索变得可行：架构单元（DAG，节点为特征图）中每条边 $(i, j)$ 保存 $|\mathcal{O}|$ 个候选操作的混合，由 softmax 归一化的架构参数 $\boldsymbol{\alpha}$ 加权：

$$\bar{o}^{(i,j)}(\mathbf{x}) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i,j)})} \cdot o(\mathbf{x})$$

架构参数 $\boldsymbol{\alpha}$ 和网络权重 $\mathbf{w}$ 通过**双层优化**（bilevel optimization）联合优化：$\mathbf{w}$ 在训练数据上更新，$\boldsymbol{\alpha}$ 在验证数据上更新，在完整数据集的单次训练过程中交替进行。搜索完成后，通过选择每条边上权重最高的操作得到离散架构。DARTS 将搜索开销从数千 GPU 天降至在 CIFAR-10 上约 4 GPU 天，所发现的架构无需重新训练即可竞争性地迁移至 ImageNet（Liu et al., 2019）。

---

## 快速参考

| # | 难度 | 主题 | 章节 |
|---|------|------|------|
| Q1 | 基础 | 均匀量化：缩放因子、零点、粒度 | 量化基础 |
| Q2 | 基础 | PTQ 与 QAT | 量化基础 |
| Q3 | 进阶 | 激活量化比权重量化更难的原因 | 量化基础 |
| Q4 | 进阶 | GPTQ 一次性权重量化 | 量化基础 |
| Q5 | 进阶 | AWQ 显著权重感知量化 | 大型语言模型量化 |
| Q6 | 基础 | 仅权重量化与权重-激活联合量化 | 大型语言模型量化 |
| Q7 | 进阶 | SmoothQuant 激活到权重的难度迁移 | 大型语言模型量化 |
| Q8 | 进阶 | LLM.int8() 混合精度分解 | 大型语言模型量化 |
| Q9 | 基础 | 结构化剪枝与非结构化剪枝 | 剪枝与稀疏性 |
| Q10 | 进阶 | 彩票假设 | 剪枝与稀疏性 |
| Q11 | 进阶 | SparseGPT 一次性 LLM 剪枝 | 剪枝与稀疏性 |
| Q12 | 进阶 | 静态与动态稀疏训练及 RigL | 剪枝与稀疏性 |
| Q13 | 基础 | 软目标知识蒸馏 | 知识蒸馏与低秩压缩 |
| Q14 | 进阶 | 特征级蒸馏与关系知识蒸馏 | 知识蒸馏与低秩压缩 |
| Q15 | 基础 | LoRA 低秩微调 | 知识蒸馏与低秩压缩 |
| Q16 | 进阶 | 截断 SVD 层压缩 | 知识蒸馏与低秩压缩 |
| Q17 | 基础 | 深度可分离卷积 | 高效架构与推理 |
| Q18 | 进阶 | FlashAttention IO 感知分块 | 高效架构与推理 |
| Q19 | 进阶 | 推测解码 | 高效架构与推理 |
| Q20 | 基础 | DARTS 可微分架构搜索 | 高效架构与推理 |

## 参考文献

- Nagel et al., [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) (2021)
- Jacob et al., [Quantization and Training of Neural Networks for Inference at Integer Arithmetic](https://arxiv.org/abs/1712.05877) (2018)
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2022)
- Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (2023)
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) (2022)
- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (2022)
- Han et al., [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) (2016)
- Frankle & Carlin, [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) (2019)
- Frantar & Alistarh, [SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot](https://arxiv.org/abs/2301.00774) (2023)
- Evci et al., [Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/1911.11134) (2020)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015)
- Romero et al., [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) (2015)
- Park et al., [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) (2019)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2022)
- Denton et al., [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736) (2014)
- Howard et al., [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (2017)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022)
- Zoph & Le, [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) (2017)
- Liu et al., [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (2019)
