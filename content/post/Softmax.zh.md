---
title: "Softmax：技术入门"
author: yo3nglau
date: '2026-05-15'
categories:
  - Deep Learning
tags:
  - Neural Networks
  - Optimization
  - Mathematics
toc: true
---

## 引言

**Softmax** 是一个向量值函数，将实值 logit 向量映射为有限结果集上的概率分布。给定输入 $\mathbf{z} \in \mathbb{R}^K$，softmax 产生一个各分量非负且之和为一的向量——将输出置于 $K$ 维概率单纯形 $\Delta^{K-1}$ 的内部。该函数的名称反映了它作为 argmax 操作的光滑可微替代的角色：argmax 对最大 logit 返回独热指示向量，而 softmax 则将概率质量连续地分布在所有类别上，保留梯度流动，从而支持通过反向传播进行端到端训练。

该函数的起源可追溯至统计物理学，其中**玻尔兹曼分布**（Boltzmann distribution）将系统处于能量状态 $E_i$、温度为 $T$ 时的概率描述为正比于 $e^{-E_i/T}$。将 logit $z_i$ 与负能量 $-E_i$ 对应，并令 $T = 1$，即可还原出 softmax 公式。同一函数在计量经济学中以**多项 logit 模型**（multinomial logit model）的形式出现（McFadden, 1974），并由 Bridle (1990) 作为多分类问题的原则性概率输出层引入神经网络文献。

Softmax 如今是深度学习中应用最广泛的操作之一。它作为分类器的输出层、Transformer 中注意力机制的归一化算子、强化学习中的策略参数化以及事后可靠性校正中的温度标定目标出现。本文从研究生层次出发，系统阐述其数学性质，并深入考察以上四个应用领域。

## 数学基础

对于输入向量 $\mathbf{z} = (z_1, \ldots, z_K)^\top \in \mathbb{R}^K$，其中 $K \geq 2$ 为类别数，softmax 函数 $\sigma: \mathbb{R}^K \to \Delta^{K-1}$ 按分量定义为

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K$$

每个输出 $\sigma(\mathbf{z})_i$ 严格为正，且由构造知 $\sum_{i=1}^K \sigma(\mathbf{z})_i = 1$，因此输出向量位于开概率单纯形内。一个关键的代数性质是**平移不变性**（translation invariance）：对任意标量 $c \in \mathbb{R}$，$\sigma(\mathbf{z} + c\mathbf{1}) = \sigma(\mathbf{z})$，其中 $\mathbf{1} \in \mathbb{R}^K$ 为全一向量。这是因为对每个指数加 $c$ 会将分子分母同乘 $e^c$，从而相消。平移不变性意味着 softmax 是过参数化的：$K$ 维输入实际上只有 $K - 1$ 个有效自由度。在数值实现中，计算指数前令 $c = -\max_j z_j$ 可防止溢出，且不改变输出——这是标准的数值稳定性技巧。

反向传播需要 softmax 的雅可比矩阵（Jacobian）。令 $S = \sum_{k=1}^K e^{z_k}$ 为归一化求和项，对 $\sigma_i = e^{z_i}/S$ 关于 $z_j$ 求导：当 $i = j$ 时，由商法则得 $\sigma_i(1 - \sigma_i)$；当 $i \neq j$ 时，得 $-\sigma_i \sigma_j$。两种情况统一为

$$\frac{\partial \sigma(\mathbf{z})_i}{\partial z_j} = \sigma(\mathbf{z})_i \bigl(\delta_{ij} - \sigma(\mathbf{z})_j\bigr)$$

其中 $\delta_{ij}$ 为 Kronecker delta，当 $i = j$ 时等于 1，否则等于 0。完整的雅可比矩阵 $J \in \mathbb{R}^{K \times K}$ 可写为 $J = \mathrm{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^\top$，其中 $\boldsymbol{\sigma} = \sigma(\mathbf{z})$ 为输出向量。这一结构使雅可比-向量积可在 $O(K)$ 而非 $O(K^2)$ 时间内完成。

Softmax 与**log-sum-exp** 函数 $\mathrm{LSE}(\mathbf{z}) = \log \sum_{j=1}^K e^{z_j}$ 密切相关，后者是 $\max_j z_j$ 的光滑凸上界。Softmax 的各分量输出恰好是 log-sum-exp 的偏导数：$\sigma(\mathbf{z})_i = \partial \,\mathrm{LSE}(\mathbf{z}) / \partial z_i$；等价地，$\sigma(\mathbf{z})_i = \exp(z_i - \mathrm{LSE}(\mathbf{z}))$，表明每个 softmax 输出是对应 logit 减去对数配分函数后的指数。引入**温度**（temperature）参数 $T > 0$，带温度的 softmax 为

$$\sigma(\mathbf{z};\, T)_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$

其中将每个 logit 除以 $T$ 控制了输出分布的尖锐程度。

## 核心直觉

理解 softmax 最直接的方式是将其视为 argmax 的可微松弛。$\mathbf{z}$ 的 argmax 对最大项赋予概率 1，其余项为 0——这是一个确定性的、不可微的操作，会阻断梯度流动。Softmax 以软性分配取代了这种硬性选择：它对最大 logit 赋予最高概率，但按照 $e^{z_j}$ 的比例将剩余概率质量分布到其他项上。Logit 之间的差距越大，输出越集中于领先项；在单一主导 logit 的极限情况下，softmax 输出趋近于独热向量。

温度参数 $T$ 连续地控制这种尖锐程度。当 $T \to 0^+$ 时，比值 $z_i/T$ 无界增大，softmax 收敛至 argmax：$\sigma(\mathbf{z};\, T) \to \mathbf{e}_c$，其中 $c = \arg\max_j z_j$，$\mathbf{e}_c$ 为第 $c$ 方向的单位向量。当 $T \to \infty$ 时，所有比值 $z_i/T \to 0$，$\sigma(\mathbf{z};\, T) \to \frac{1}{K}\mathbf{1}$，即所有 $K$ 个类别上的均匀分布。温度由此提供了一个单一的标量旋钮，在确定性与最大熵之间平滑插值——这一性质在知识蒸馏、校正以及语言模型采样中均被加以利用。

玻尔兹曼分布的类比阐明了 softmax 行为的原因。在统计力学中，处于热平衡、温度为 $T$ 的系统占据能量状态 $E_i$ 的概率正比于 $e^{-E_i/T}$。将负 logit $-z_i$ 对应于能量 $E_i$，即可还原出带温度的 softmax。低温对应于系统强烈偏向其最低能量（最高 logit）状态；高温对应于系统不论能量如何均匀采样所有状态。Logit 扮演负能量的角色，归一化分母 $\sum_j e^{z_j}$ 即为配分函数——其对数为系统的自由能。

## 在人工智能中的应用

### 多分类任务

在神经网络分类器中，softmax 将最后一个线性层的非归一化输出——**logit 向量** $\mathbf{z} = W\mathbf{h} + \mathbf{b}$，其中 $\mathbf{h} \in \mathbb{R}^d$ 为倒数第二层的隐藏表示，$W \in \mathbb{R}^{K \times d}$ 为权重矩阵，$\mathbf{b} \in \mathbb{R}^K$ 为偏置——转换为 $K$ 个类别上的分类分布 $\hat{\mathbf{y}} = \sigma(\mathbf{z})$。训练最小化交叉熵损失 $\mathcal{L} = -\log \hat{y}_c = -\log \sigma(\mathbf{z})_c$，其中 $c \in \{1, \ldots, K\}$ 为真实类别的索引。利用 log-sum-exp 展开得 $\mathcal{L} = -z_c + \mathrm{LSE}(\mathbf{z})$，对其求导得到梯度

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sigma(\mathbf{z})_j - \mathbf{1}[j = c]$$

其中 $\mathbf{1}[j = c]$ 当 $j = c$ 时等于 1，否则等于 0。该梯度具有简洁的解释：它是模型对第 $j$ 类的预测概率与真实概率（0 或 1）之差。当模型以概率 1 预测正确类别时梯度为零；当模型对错误类别过于自信时，梯度最强烈地指向降低该类概率的方向。BERT（Devlin et al., 2018）在其分类头中采用了这一精确形式，对特殊分类词元的汇聚表示 $\mathbf{h}_\mathrm{[CLS]}$ 计算 $\sigma(W\mathbf{h}_\mathrm{[CLS]})$。

### 缩放点积注意力

**缩放点积注意力**（scaled dot-product attention）机制（Vaswani et al., 2017）利用 softmax 将查询与键向量之间的原始相似度分数转换为对值向量的概率分布。给定查询矩阵 $Q \in \mathbb{R}^{n \times d_k}$（包含 $n$ 个维度为 $d_k$ 的查询向量）、键矩阵 $K \in \mathbb{R}^{m \times d_k}$（包含 $m$ 个维度为 $d_k$ 的键向量）以及值矩阵 $V \in \mathbb{R}^{m \times d_v}$（包含 $m$ 个维度为 $d_v$ 的值向量），注意力输出为

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

其中 softmax 按行应用于 $n \times m$ 的分数矩阵 $QK^\top / \sqrt{d_k}$，产生 $n$ 个各覆盖 $m$ 个键位置的概率分布，所得 $n \times m$ 权重矩阵再与 $V$ 相乘，输出维度为 $n \times d_v$。输出的每一行是值向量的凸组合，权重由对点积相似度的 softmax 决定。缩放因子 $1/\sqrt{d_k}$ 抵消了点积随 $d_k$ 增大而增大的趋势——Vaswani 等人指出，对于较大的 $d_k$，未缩放的点积会将 softmax 推入梯度极小的区域，从而减慢学习速度。Softmax 在此不仅起归一化作用，更产生了可微的加权平均，使反向传播的端到端训练成为可能。

### 强化学习中的 Softmax 策略

在离散动作空间的强化学习中，softmax 提供了随机策略的自然参数化方式。给定状态 $s$ 与可能动作集 $\mathcal{A} = \{a_1, \ldots, a_K\}$，参数为 $\theta$ 的神经网络对每个动作 $a$ 产生偏好分数 $h_\theta(s, a) \in \mathbb{R}$，策略定义为

$$\pi_\theta(a \mid s) = \frac{\exp(h_\theta(s, a))}{\sum_{a' \in \mathcal{A}} \exp(h_\theta(s, a'))}$$

这种参数化确保了对任意状态 $s$ 和任意 $\theta$ 值，$\pi_\theta(\cdot \mid s)$ 均为有效的概率分布，且关于 $\theta$ 可微，从而支持策略梯度更新。Softmax 策略避免了贪心动作选择的硬性承诺，同时可端到端训练；其熵 $H(\pi_\theta(\cdot \mid s)) = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$ 直接由 $h_\theta$ 的尺度控制，在奖励中加入熵正则项可鼓励探索。Mnih et al. (2016) 在 A3C（Asynchronous Advantage Actor-Critic）的演员网络中采用了这一 softmax 策略参数化，用于离散动作 Atari 游戏，演员输出游戏动作集上的 softmax 分布。

### 温度标定与模型校正

**校正良好**（well-calibrated）的分类器，其对某事件预测的概率 $p$ 与模型在所有赋予该概率 $p$ 的实例中该事件实际发生的频率相吻合。以交叉熵训练的现代深度神经网络存在系统性的校正不足：即使对不确定的输入，置信度也往往接近 1，这种现象被称为**过度自信**（overconfidence）。Guo et al. (2017) 通过实验验证了这一现象，并证明**温度标定**（temperature scaling）——在应用 softmax 前用单一可学习标量 $T > 1$ 除以所有 logit——是最有效的单参数事后校正方法。类别 $i$ 的校正概率为 $\hat{p}_i = \sigma(\mathbf{z}/T)_i$，其中 $\mathbf{z}$ 为预训练模型输出的 logit 向量，$T > 0$ 通过在留出验证集上最小化负对数似然来拟合。由于 $T$ 是均匀作用于所有 logit 的标量，它不改变 argmax 预测——类别排序保持不变——但会软化概率分布，降低过度自信。温度标定在知识蒸馏（Hinton et al., 2015）中也是核心操作，其中 $T > 1$ 在训练期间用于向学生网络展示教师网络的软概率分配，这些分配比独热标签携带更多关于类间相似性的信息。

## 核心要点

Softmax 是将实值 logit 转换为离散集合上概率分布的标准操作，其可微性使多分类神经网络可以通过梯度下降训练；交叉熵损失关于 logit 的梯度化简为预测概率与真实概率之差，这一形式使数学推导和优化过程均清晰透明。它与 log-sum-exp 的联系确立了 softmax 作为 argmax 光滑松弛的地位，温度参数提供了对尖锐程度的连续控制——从低温下的近似确定性选择到高温下的近均匀分布。在注意力机制中，这种尖锐程度控制决定了查询从上下文中聚合信息的广度；在强化学习中，它通过策略熵平衡探索与利用；在校正中，它修正了大模型因交叉熵训练而产生的系统性过度自信。贯穿所有这些角色，玻尔兹曼分布的解释始终是统一的线索：softmax 本质上是一个配分函数，而调节其温度即调节了底层概率模型的有效能量尺度。

## 参考文献

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (2018)
- Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (2016)
- Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (2017)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015)
