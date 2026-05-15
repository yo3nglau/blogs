---
title: "KL 散度：技术入门"
author: yo3nglau
date: '2026-05-15'
categories:
  - Deep Learning
tags:
  - Information Theory
  - Variational Inference
  - Mathematics
toc: true
---

## 引言

**Kullback-Leibler 散度**（KL 散度），由 Solomon Kullback 与 Richard Leibler 于 1951 年在其关于信息充分性的论文中提出（Kullback & Leibler, 1951），是衡量概率分布 $P$ 与参考分布 $Q$ 之间差异的度量。与距离度量不同，KL 散度具有方向性：$D_{KL}(P \| Q)$ 量化了用 $Q$ 近似 $P$ 时所损失的信息量，因此从根本上是不对称的。这种不对称性并非缺陷，而是一种特性——它在"真实"分布与"近似"分布之间编码了有意义的区别，这一区别直接对应于统计推断的结构。

KL 散度处于信息论、统计学与机器学习的交汇处。它与香农熵的联系使其成为表达模型信念与现实偏差的自然工具，其变分性质则支撑着现代 AI 中一些最具影响力的训练目标。变分自编码器、信任域策略优化、基于人类反馈的强化学习以及知识蒸馏，都将 KL 散度作为损失函数的核心要素——并非偶然，而是结构性的。

本文从研究生层次出发，系统阐述 KL 散度的数学基础，建立对其行为的直觉认识，并考察其在四个主要 AI 应用领域中的角色。

## 数学基础

对于定义在同一支撑集 $\mathcal{X}$ 上的离散分布 $P$ 与 $Q$，从 $Q$ 到 $P$ 的 KL 散度定义为

$$D_{KL}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$

约定 $0 \log 0 = 0$，若存在某 $x$ 使得 $Q(x) = 0$ 而 $P(x) > 0$，则上式等于 $+\infty$。对于连续分布，求和变为积分：$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$。本文中对数取自然对数，单位为奈特（nats）；取以 2 为底的对数则单位为比特（bits）。

KL 散度与香农熵及交叉熵直接相关。回顾 $P$ 的**熵**（entropy）为 $H(P) = -\sum_x P(x) \log P(x)$，$Q$ 相对于 $P$ 的**交叉熵**（cross-entropy）为 $H(P, Q) = -\sum_x P(x) \log Q(x)$，则

$$D_{KL}(P \| Q) = H(P, Q) - H(P)$$

这使得信息论的解释更加精确：KL 散度是当从 $P$ 中采样的事件使用针对 $Q$ 而非 $P$ 优化的编码方案时，所产生的期望额外编码长度。

非负性，即 $D_{KL}(P \| Q) \geq 0$，当且仅当 $P = Q$（几乎处处）时等号成立，由 **Gibbs 不等式**推出，而后者本身是 Jensen 不等式应用于凸函数 $-\log$ 的结果。由于 $\log$ 严格凹，有 $\mathbb{E}_P[\log(Q/P)] \leq \log(\mathbb{E}_P[Q/P]) = \log(1) = 0$，故 $\mathbb{E}_P[\log(P/Q)] \geq 0$。尽管具有非负性，KL 散度并不是度量：它是不对称的（一般情况下 $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$），且不满足三角不等式。

对于两个对角高斯分布这一在变分方法中反复出现的特殊情形，KL 散度具有解析闭式表达。设 $q = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$，$p = \mathcal{N}(0, I)$ 为标准正态先验，则

$$D_{KL}(q \| p) = \frac{1}{2} \sum_{j=1}^{d} \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)$$

该表达式关于 $\mu$ 和 $\sigma^2$ 可微，可直接用于基于梯度的优化——这一性质在变分自编码器中至关重要。

## 核心直觉

理解 KL 散度最直观的方式是通过编码理论。假设 $P$ 是消息的真实分布，$Q$ 是你所假设的模型。香农信源编码定理保证，针对 $Q$ 的最优编码为消息 $x$ 分配约 $-\log Q(x)$ 比特。若使用此编码但消息实际从 $P$ 中采样，期望编码长度为 $H(P, Q) = \mathbb{E}_P[-\log Q(x)]$。针对 $P$ 的最优编码可实现 $H(P) = \mathbb{E}_P[-\log P(x)]$。两者之差——浪费的比特数——恰好是 $D_{KL}(P \| Q)$。零浪费意味着 $P = Q$；KL 散度越大，说明模型与现实的匹配越差。

这一图景中的危险地带在于：当 $Q(x) \approx 0$ 而 $P$ 对事件 $x$ 赋予有意义的概率时，$\log(P(x)/Q(x))$ 趋向无穷大，即便 $P$ 在此处的概率质量很小，也会使 $D_{KL}(P \| Q)$ 极大。这正是为何最小化 $D_{KL}(P \| Q)$——其中 $P$ 固定，$Q$ 为待优化的近似——会迫使 $Q$ 覆盖 $P$ 的所有模式：遗漏 $P$ 赋予概率的任何区域，代价实际上是无穷大的。这种行为被称为**零规避**（zero-avoiding）或**均值寻求**（mean-seeking）。

反向目标 $D_{KL}(Q \| P)$ 则具有相反的特性。此时 $Q$ 为近似分布，对数比值为 $\log(Q(x)/P(x))$；当 $Q(x) > 0$ 而 $P(x) \approx 0$ 时，比值趋向无穷大，因此优化器会受到惩罚，不敢在 $P$ 未赋予概率的地方放置质量。这驱使 $Q$ 集中于 $P$ 的单一模式，而非覆盖所有模式，这种行为被称为**零强制**（zero-forcing）或**模式寻求**（mode-seeking）。因此，选择最小化正向 KL（$D_{KL}(P \| Q)$）还是反向 KL（$D_{KL}(Q \| P)$），编码了一个根本性的建模决策，对生成模型如何学习表示多模态分布具有直接影响。

## 在人工智能中的应用

### 变分推断与变分自编码器

**变分推断**（variational inference）将难以处理的后验计算问题 $p_\theta(z|x)$ 重新表述为优化问题：在一个易处理的分布族 $q_\phi(z|x)$ 中，找到 KL 散度意义下与真实后验最接近的成员。对数似然的分解

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right] - D_{KL}(q_\phi(z|x) \| p(z))}_{\mathcal{L}(\theta,\phi;\,x)} + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

表明 $\mathcal{L}(\theta, \phi; x)$——**证据下界**（evidence lower bound，ELBO）——等于对数似然减去一个非负的 KL 项。最大化 ELBO 既能收紧该下界，使 $q_\phi$ 趋近真实后验，也能推动 $p_\theta$ 生成与 $x$ 相似的数据。

**变分自编码器**（Kingma & Welling, 2013）通过神经网络编码器 $q_\phi(z|x)$（参数化为对角高斯）和神经网络解码器 $p_\theta(x|z)$ 来实现这一框架。ELBO 中的 KL 项 $D_{KL}(q_\phi(z|x) \| p(z))$ 作为正则化项，防止编码器退化为点质量，并在潜在空间中施加平滑结构。由于 $q_\phi$ 和 $p$ 均为高斯分布，该项可利用上文推导的闭式表达式精确计算，从而通过重参数化技巧对整个训练目标进行精确梯度计算。

### 信任域策略优化

在强化学习中，采用大梯度步长的策略梯度方法可能因单次更新改变未来状态的分布而导致性能灾难性下降。**信任域策略优化**（Trust Region Policy Optimization，TRPO）（Schulman et al., 2015）通过将每次策略更新约束在由 KL 散度定义的信任域内来解决这一问题。形式化地，优化问题为

$$\max_\theta \; \mathbb{E}_{s \sim \rho^{\pi_\mathrm{old}},\, a \sim \pi_\mathrm{old}}\!\left[\frac{\pi_\theta(a|s)}{\pi_\mathrm{old}(a|s)} A^{\pi_\mathrm{old}}(s, a)\right] \quad \text{s.t.} \quad \mathbb{E}_{s \sim \rho^{\pi_\mathrm{old}}}\!\left[D_{KL}(\pi_\mathrm{old}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta$$

其中 $A^{\pi_\mathrm{old}}$ 为优势函数，$\delta$ 为信任域半径。KL 约束确保新策略保持在旧策略的邻域内，在该邻域中，重要性加权目标是真实性能提升的可靠估计。Schulman 等人在此约束下证明了单调改进保证，使 TRPO 成为首个具有可证明安全更新规则的策略梯度方法。

### 基于人类反馈的强化学习

训练大语言模型遵循指令，需要优化从人类偏好判断中导出的奖励模型，但对此类奖励信号的无约束最大化会导致**奖励欺骗**（reward hacking）——模型会找到能获得高奖励但无法产生有用语言的退化输出。标准解决方案由 Ziegler 等人（2019）在微调框架中引入，并在 InstructGPT（Ouyang et al., 2022）中扩展规模，即引入 KL 惩罚项，使微调策略 $\pi$ 保持靠近冻结的参考模型 $\pi_\mathrm{ref}$：

$$\max_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)}\!\left[r(x, y)\right] - \beta \, D_{KL}(\pi(\cdot|x) \| \pi_\mathrm{ref}(\cdot|x))$$

系数 $\beta$ 在奖励最大化与分布保真度之间权衡。当 $\beta = 0$ 时策略无约束并退化；随着 $\beta$ 增大，策略更接近参考模型，但可能对奖励信号欠拟合。实践中通过调整 $\beta$ 找到一个区间，使模型在提升人类偏好指标的同时仍保持连贯的语言生成能力。该目标中的 KL 项在形式上与 TRPO 中的相同，但作用不同：它不是保证策略安全改进，而是防止不完美奖励建模所诱发的分布偏移。

### 知识蒸馏

**知识蒸馏**（knowledge distillation）（Hinton et al., 2015）通过训练学生网络复现教师网络的输出分布（而非硬标签），将大型教师网络的学习表示迁移到较小的学生网络中。对于分类任务，设教师输出 logits 为 $z_t$，学生输出 logits 为 $z_s$，蒸馏损失为

$$\mathcal{L}_\mathrm{KD} = T^2 \cdot D_{KL}\!\left(\sigma(z_t / T) \,\|\, \sigma(z_s / T)\right)$$

其中 $\sigma$ 为 softmax 函数，$T > 1$ 为**温度**（temperature）参数。当 $T = 1$ 时，softmax 将概率质量集中在 argmax 类上；当 $T$ 较大时，概率分布更均匀，揭示出教师的相对置信度结构——例如，某个"2"比"1"更像"7"。在这些软化分布上训练的学生获得了比 one-hot 标签更丰富的监督信号，在相同参数量下比从头训练取得更好的泛化效果。$T^2$ 因子补偿了高温下软目标梯度幅度的减小。最小化该正向 KL（教师分布固定）迫使学生覆盖教师输出的所有模式，防止学生忽略编码了语义关系的低概率类别。

## 核心要点

KL 散度度量的是用一个概率分布近似另一个概率分布时的信息代价，其不对称性——最小化哪个方向——产生了性质截然不同的学习分布：最小化正向 KL 时表现为均值寻求，最小化反向 KL 时表现为模式寻求。这一性质并非数学上的偶然，它直接决定了变分自编码器、信任域方法、RLHF 奖励塑形以及知识蒸馏的行为，每种方法都嵌入了经过选择的 KL 项以施加特定的归纳偏置。在所有这些场景中，高斯 KL 的解析可处理性与非负性保证，使其在计算上和理论上都十分便利，从而确立了它作为现代机器学习基本量之一的地位。

## 参考文献

- Kullback & Leibler, [On Information and Sufficiency](https://www.jstor.org/stable/2236703) (1951)
- Kingma & Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (2013)
- Schulman et al., [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (2015)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015)
- Ziegler et al., [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (2019)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022)
