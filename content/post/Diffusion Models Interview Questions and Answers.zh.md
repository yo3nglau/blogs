---
title: "扩散模型：面试问题与推荐回答"
author: yo3nglau
date: '2026-04-08'
categories:
  - Interview
tags:
  - Deep Learning
  - Diffusion Models
  - Generative Models
toc: true
---

## 扩散模型基础

### Q1 [基础] 描述DDPM的前向过程与反向过程

**Q:** DDPM的前向过程与反向过程如何定义一个生成模型？什么数学结构使反向过程易于处理？

**A:** **DDPM**（Ho et al., 2020）通过两条马尔可夫链定义生成模型。**前向过程** $q(x_{1:T}|x_0)$ 在 $T$ 步内逐步向数据样本 $x_0$ 添加少量高斯噪声：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\right)$$

其中 $\{\beta_t\}_{t=1}^T$ 是固定的方差调度表。一个关键性质是闭式边缘分布：令 $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$，可得

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t) I\right)$$

这意味着任意时间步的噪声样本都可以直接采样，无需逐步迭代。在 $T = 1000$、线性调度从 $\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$ 的设置下，$x_T$ 近似为各向同性高斯分布（Ho et al., 2020）。

**反向过程** $p_\theta(x_{0:T-1}|x_T)$ 从高斯噪声出发逐步去噪。真实反向后验 $q(x_{t-1}|x_t, x_0)$ 具有解析形式且本身是高斯分布。模型学习 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$ 来近似该后验。生成时先采样 $x_T \sim \mathcal{N}(0, I)$，再反复应用已学习的去噪步骤——这一过程在概念上植根于 Sohl-Dickstein et al.（2015）的非平衡热力学框架。

---

### Q2 [基础] 解释得分匹配与去噪之间的联系

**Q:** 什么是得分函数？去噪得分匹配如何为基于得分的生成模型提供可行的训练目标？

**A:** 分布 $p(x)$ 的**得分函数**（score function）是其对数密度梯度 $\nabla_x \log p(x)$。基于得分的生成模型学习该梯度场，再通过**朗之万动力学**（Langevin dynamics）——在得分方向上迭代移动并叠加噪声——来生成样本。Song & Ermon（2019）证明，训练神经网络来估计 $\nabla_x \log p(x)$ 可以生成高质量样本，但标准得分匹配目标需要计算 Hessian 矩阵的迹，在高维数据上不可行。

**去噪得分匹配**（denoising score matching）通过训练网络对加噪数据进行去噪来绕过这一难题。给定干净样本 $x_0$ 和噪声 $\epsilon \sim \mathcal{N}(0, I)$，将其加噪为 $x_t = x_0 + \sigma_t \epsilon$，并训练 $s_\theta(x_t, \sigma_t)$ 来预测 $\nabla_{x_t} \log p_{\sigma_t}(x_t)$。Vincent（2011）证明该去噪目标与不可行的得分匹配目标具有相同的最优解。其内在联系在于：最优去噪器预测的是干净数据，而其残差正指向得分方向。

Song & Ermon（2019）将其推广为多尺度框架（**NCSN**）：训练一个以噪声序列 $\sigma_1 < \sigma_2 < \cdots < \sigma_L$ 为条件的单一得分网络，并在推理时使用退火朗之万动力学。这直接推动了 DDPM 的诞生——DDPM 中的噪声预测器 $\epsilon_\theta(x_t, t)$ 等价于一种经过缩放的得分估计：$s_\theta(x_t, t) \approx -\epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}$。

---

### Q3 [进阶] 解释SDE框架及其如何统一基于得分的模型与DDPM

**Q:** Song et al.的SDE视角如何统一DDPM与NCSN，以及连续时间形式化带来了哪些新能力？

**A:** Song et al.（2021）表明，DDPM 和 NCSN 都是具有相同数学结构的两个连续时间随机微分方程（SDE）的离散化：

$$dx = f(x, t)\,dt + g(t)\,dW$$

其中 $W$ 是标准维纳过程。**VP-SDE**（方差保持）是 DDPM 前向过程的连续极限，$f(x,t) = -\frac{1}{2}\beta(t) x$，$g(t) = \sqrt{\beta(t)}$。**VE-SDE**（方差爆炸）是 NCSN 的连续极限，$f = 0$，$g(t) = \sqrt{d[\sigma^2(t)]/dt}$。

两种 SDE 均有精确的反向 SDE：

$$dx = \left[f(x,t) - g^2(t)\nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{W}$$

以及对应的**概率流 ODE**——其边缘分布与前向 SDE 一致，但演化过程是确定性的：

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g^2(t) \nabla_x \log p_t(x)$$

连续时间形式化带来了离散 DDPM 所不具备的三项新能力。第一，概率流 ODE 可以用现成的数值 ODE 求解器积分，通过瞬时变量替换公式实现精确的对数似然计算。第二，ODE 轨迹定义了噪声与数据之间的确定性映射，支持潜空间插值和编辑。第三，灵活的噪声调度成为第一类支持对象——任何从 $p_T \approx \mathcal{N}(0, I)$ 到 $p_0 = p_\text{data}$ 的平滑调度均可定义有效模型，且已训练的得分网络可在整个噪声连续体上泛化。

---

### Q4 [进阶] 分析ELBO分解与简化训练目标

**Q:** DDPM 训练目标与 ELBO 有什么关系？为什么简化的噪声预测损失在经验上优于完整的变分下界？

**A:** DDPM 生成模型定义了对数似然的变分下界（ELBO）：

$$\log p_\theta(x_0) \geq -\mathcal{L}_\text{VLB} = \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

展开 ELBO 可得每个时间步上已学习反向步骤 $p_\theta(x_{t-1}|x_t)$ 与解析后验 $q(x_{t-1}|x_t, x_0)$ 之间的 KL 散度之和。由于 $q(x_{t-1}|x_t, x_0)$ 是均值为 $\tilde{\mu}_t(x_t, x_0)$ 的高斯分布，逐步 KL 散度退化为已学习均值与真实后验均值之间的平方误差，而均值可通过噪声预测重参数化：

$$\mathcal{L}_t \propto \mathbb{E}_{x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, t)\|^2\right]$$

Ho et al.（2020）提出了舍弃时间步相关权重的**简化目标**：

$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\, t)\|^2\right]$$

在经验上，简化损失在样本质量上优于加权 ELBO，即使它并不优化显式的似然下界。其直觉是：VLB 因小 $t$ 时的后验方差很小，会对这些几乎干净的图像时间步赋予更大权重；简化损失对所有时间步一视同仁，在高噪声区域提供更多梯度信号——而这正是模型需要学习更多的区域。Nichol & Dhariwal（2021）随后表明，学习方差调度可以优化简化损失与 VLB 的混合目标，从而恢复更好的似然估计。

---

## 加速采样

### Q5 [基础] 解释DDIM如何实现确定性与加速采样

**Q:** DDIM 的关键洞察是什么，使其能够在不重新训练噪声预测器的情况下实现确定性生成和步骤跳过？

**A:** **DDIM**（Song et al., 2020）的出发点是：DDPM 的前向过程 $q(x_{1:T}|x_0)$ 无需是马尔可夫的。Song et al.（2020）推导出一种非马尔可夫前向过程，其边缘分布 $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$ 与 DDPM 相同——意味着同一个噪声预测器 $\epsilon_\theta$ 可以直接使用——但反向更新规则不同：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0(x_t) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t\,\epsilon$$

其中 $\hat{x}_0(x_t) = (x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t))/\sqrt{\bar{\alpha}_t}$ 是预测的干净图像，$\sigma_t$ 控制随机性。令 $\sigma_t = 0$ 使更新过程变为确定性：相同的噪声输入 $x_T$ 始终产生相同的输出 $x_0$。

确定性 ODE 形式化支持**步骤跳过**：DDIM 可以只在 $S \ll T$ 个时间步的子序列处评估噪声预测器，而无需逐步从 $t = T, T-1, \ldots, 1$ 迭代。在 $S = 50$ 步时，DDIM 实现了与 DDPM $T = 1000$ 步相当的生成质量——提速 $20$ 倍——这是因为 ODE 轨迹平滑，可以粗略积分。使用 DDPM 目标训练的同一个噪声预测器可直接使用，无需重新训练。

---

### Q6 [进阶] 解释DPM-Solver如何利用ODE理论加速采样

**Q:** DPM-Solver 利用了扩散 ODE 的哪种数学结构？它如何在极少步数内实现高质量生成？

**A:** 来自 SDE 框架（Song et al., 2021）的概率流 ODE 可以改写为揭示**半线性结构**的形式。Lu et al.（2022）用对数信噪比 $\lambda_t = \log(\sqrt{\bar{\alpha}_t} / \sqrt{1-\bar{\alpha}_t})$ 重写了该 ODE：

$$\frac{dx_\lambda}{d\lambda} = x_\lambda - \frac{\sqrt{1-e^{-2\lambda}}}{e^{-\lambda}} \epsilon_\theta(x_\lambda, \lambda)$$

该 ODE 的线性部分（$x_\lambda$ 项）有精确解。非线性部分（得分项）用 $\epsilon_\theta$ 关于 $\lambda$ 的泰勒展开来近似，使用 $\epsilon_\theta$ 在少量点处的函数求值来计算。DPM-Solver-2 使用二阶展开，DPM-Solver-3 使用三阶展开。

相对于 DDIM（在此框架中本质上是一阶 ODE 求解器）的关键优势在于：高阶方法在相同近似误差下可以迈出更大的步长。DPM-Solver-2 在 20 步内即可达到接近完美的样本质量（DDIM 需要 50–100 步），DPM-Solver-3 在 CIFAR-10 和 ImageNet 上仅需 10 步（Lu et al., 2022）。该方法无需重新训练即可与任何 DDPM 训练的模型配合使用，且兼容噪声预测和数据预测两种参数化方式，可作为 DDIM 的即插即用替代品。

---

### Q7 [进阶] 描述一致性模型及其与扩散的关系

**Q:** 一致性模型如何实现单步生成？一致性蒸馏与一致性训练有何区别？

**A:** **一致性模型**（consistency models；Song et al., 2023）定义了一个一致性函数 $f_\theta(x_t, t)$，将 PF-ODE 轨迹上任意噪声水平 $t$ 处的点映射回同一个原点 $x_0$。自洽性要求：

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \text{对同一 ODE 轨迹上的所有 } t, t'$$

边界条件为 $f_\theta(x_0, 0) = x_0$（零噪声时函数为恒等映射）。生成因此只需单步：采样 $x_T \sim \mathcal{N}(0, \sigma_T^2 I)$，计算 $f_\theta(x_T, T)$ 即可。

**一致性蒸馏**（CD）通过从预训练扩散模型中蒸馏来训练 $f_\theta$。给定 ODE 轨迹上的相邻点 $(x_{t_{n+1}}, x_{t_n})$——其中 $x_{t_n}$ 由数值 ODE 求解器从 $x_{t_{n+1}}$ 迈一步得到——损失最小化：

$$\mathcal{L}_\text{CD} = \mathbb{E}\!\left[d\!\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\, f_{\theta^-}(x_{t_n}, t_n)\right)\right]$$

其中 $d(\cdot, \cdot)$ 是某种度量（实践中为 LPIPS），$\theta^-$ 是 $\theta$ 的指数移动平均。

**一致性训练**（CT）无需预训练扩散模型，完全从数据分布本身自举估计 ODE 相邻点，但样本质量略低于 CD。Song et al.（2023）在 CIFAR-10 上报告 CD 单步 FID 为 3.55，接近 DDIM 10 步的质量；多步一致性采样（2–3 次函数求值）可进一步逼近完整扩散的质量。

---

### Q8 [进阶] 比较流匹配与扩散模型

**Q:** 流匹配与扩散模型在训练目标和采样路径上有何不同？这带来了哪些实际优势？

**A:** 扩散模型通过预测得分 $\nabla_x \log p_t(x)$ 或等价地预测噪声 $\epsilon_t$ 来学习逆转加噪过程。前向过程在数据空间中定义了弯曲轨迹（由信噪比调度控制的椭球路径），反向 ODE 必须精确追踪这些弯曲路径。这种曲率需要大量的函数求值（NFE）才能精确积分。

**流匹配**（flow matching；Lipman et al., 2022）直接学习向量场 $v_\theta(x, t)$，通过 ODE $dx/dt = v_\theta(x, t)$ 将噪声 $x_0 \sim \mathcal{N}(0, I)$ 传输到数据 $x_1 \sim p_\text{data}$，从而绕过上述问题。条件流匹配（CFM）目标以单个数据点为条件：

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t,\, q(x_1),\, p_t(x | x_1)}\!\left[\|v_\theta(x, t) - u_t(x|x_1)\|^2\right]$$

选择**直线路径** $x_t = (1-t)x_0 + tx_1$ 给出常数条件向量场 $u_t(x_t|x_1) = x_1 - x_0$，使训练极为简单。直线路径需要更少的积分步数，因为 ODE 轨迹曲率为零——同样的步长对弯曲扩散路径会失败，但对直线流可以精确积分。

**Rectified Flow**（Liu et al., 2022）是同期独立提出的相同线性插值思路。流匹配和 Rectified Flow 均已被生产模型采用：Stable Diffusion 3（Esser et al., 2024）使用流匹配目标配合多模态 DiT 架构，在相当计算量下报告了更高的训练效率和样本质量。此外，不需要预定义噪声调度也简化了模型设计的消融分析。

---

## 条件生成

### Q9 [基础] 解释无分类器引导及其如何控制生成

**Q:** 无分类器引导（CFG）如何在不使用单独分类器的情况下引导扩散模型朝向目标条件生成？

**A:** **无分类器引导**（CFG；Ho & Salimans, 2022）训练一个单一的条件扩散模型，该模型也能以无条件方式运行——通过在训练时随机丢弃条件 $c$（以一定概率将其替换为空标记 $\varnothing$，通常为 10–20%）。推理时，每个去噪步骤运行两次前向传播——一次带条件，一次不带——并在得分空间中进行线性外插：

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w\,\bigl(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\bigr)$$

其中 $w \geq 1$ 是引导尺度。这等价于放大得分中指向与条件一致样本方向的分量。引导尺度权衡保真度与多样性：$w = 1$ 恢复标准条件采样；$w \gg 1$ 将分布集中于高概率模式，生成更锐利、更符合条件的图像，但多样性降低。

CFG 无需单独的分类器，且比基于分类器的替代方案产生更干净的引导梯度。它已成为 Stable Diffusion（典型 $w = 7.5$）、DALL-E 2 和 Imagen 等系统中的标准调制机制，显著提升了文本-图像对齐质量。

---

### Q10 [基础] 描述ControlNet及其如何添加空间条件

**Q:** ControlNet 如何在不降低现有生成质量的情况下扩展预训练扩散模型以接受新的空间条件信号？

**A:** **ControlNet**（Zhang & Agrawala, 2023）解决的问题是：在不造成灾难性遗忘的前提下，将大型预训练扩散 U-Net 适配到新的空间输入——边缘图、深度图、人体姿态关键点、分割掩码等。

该架构创建了 U-Net 编码块的**可训练副本**，同时保持原始模型冻结。空间条件输入（如 Canny 边缘图）与噪声图像一起输入可训练副本。可训练块的输出通过**零卷积**——权重和偏置均初始化为零的 $1\times 1$ 卷积层——叠加回冻结的解码器。由于这些层的初始输出恰好为零，ControlNet 最初对模型输出毫无影响，完整保留了预训练能力。随着训练推进，零卷积学会将条件信息路由进冻结的解码器。

核心洞察在于：零初始化提供了安全的起始点——由于输出对条件信号的初始梯度恰好为零，微调可以使用比其他情况更大的学习率。Zhang & Agrawala（2023）证明，针对单一空间条件类型训练的 ControlNet 泛化能力良好，且多个 ControlNet 可以在推理时通过将各自对解码器的贡献相加来组合使用。

---

### Q11 [进阶] 分析分类器引导与无分类器引导的权衡

**Q:** 分类器引导与 CFG 在实践和理论上有何区别？各自在何时失效？

**A:** **分类器引导**（Dhariwal & Nichol, 2021）将单独训练的噪声分类器 $p_\phi(y|x_t)$ 的梯度加入得分：

$$\tilde{\nabla}_{x_t} \log p(x_t) = \nabla_{x_t} \log p_t(x_t) + w\,\nabla_{x_t} \log p_\phi(y|x_t)$$

这需要专门在所有时间步的噪声数据上训练一个分类器，与扩散模型分开训练。在 ImageNet 256$\times$256 上，Dhariwal & Nichol（2021）使用改进的 U-Net 骨干 ADM 配合分类器引导，以 $w = 1.0$ 实现了 FID 4.59，首次证明扩散模型在类别条件 ImageNet 上可以超越最先进的 GAN。

**CFG** 避免了单独的分类器，但产生了本质上不同类型的引导。关键理论区别在于：分类器引导计算的是判别分类器的梯度，可能包含对抗性伪影——分类器学会利用真实数据中不存在的低级纹理统计，导致引导样本出现过度锐化，看起来对分类器真实但对人类而言不自然。CFG 在得分空间中的外插不会产生此类伪影，因为两个得分估计都来自同一个生成模型。

两种方法都存在根本性的**多样性-保真度权衡**：更高的 $w$ 提高了精度（样本更接近条件分布的模式），代价是召回率降低（采样的模式更少，多样性下降）。这种权衡在精度/召回率框架中得到了很好的刻画：引导随 $w$ 单调地推高精度、压低召回率。在极高的引导尺度下，由于分布坍缩到少量高度典型的样本，偏离训练分布的完整多样性，FID 反而开始上升。

---

### Q12 [进阶] 分析文本编码器选择对文本到图像扩散模型的影响

**Q:** 文本编码器的选择如何影响扩散模型中的文本-图像对齐？Imagen 的发现揭示了语言模型规模的什么重要性？

**A:** 早期文本到图像扩散模型以 CLIP 文本嵌入为条件（Rombach et al., 2022），因为 CLIP 表示已通过图像-文本对的对比训练与视觉概念对齐。CLIP 的 77 个 token 上下文窗口及其训练目标（最大化图像-文本相似度）产生的嵌入非常适合粗粒度语义对齐，但对于组合性指令或图像-文本语料库中稀有视觉概念的处理能力有限。

**Imagen**（Saharia et al., 2022）有一个令人惊讶的发现：将 CLIP 替换为仅在文本上训练的大型冻结语言模型（**T5-XXL**，46亿参数），显著改善了文本-图像对齐，尤其是对于组合性提示、不寻常的词序以及图像-文本语料库中表示不足的领域专业术语。这一发现是反直觉的，因为 T5 从未接受过视觉监督；然而，其从文本完整分布中学到的更丰富语言表示编码了 CLIP 的对比目标所无法捕获的句法和语义结构。

其机制在于：T5 产生独立的逐 token 嵌入，扩散 U-Net 的交叉注意力层可以将其在空间上路由到对应的图像区域。CLIP 产生单一的全局嵌入或相对浅层的逐 token 表示，限制了其表达"红色立方体在蓝色球体上方，旁边有绿色圆柱体"等组合性描述的能力。Saharia et al.（2022）表明，在固定图像解码器规模的情况下，扩大文本编码器规模（从 T5-Small 到 T5-XXL）可以提高 DrawBench 人类偏好得分——这说明对于复杂提示，文本理解而非图像生成能力才是主要瓶颈。

---

### Q13 [基础] 描述噪声调度设计如何影响扩散模型的训练与采样

**Q:** 设计噪声调度的关键选择是什么？为什么余弦调度对图像生成优于线性调度？

**A:** **噪声调度** $\{\beta_t\}$（或等价地 $\{\bar{\alpha}_t\}$）控制每个时间步的信噪比（SNR），并决定训练损失如何分配在去噪难度谱上。Ho et al.（2020）使用线性调度，从 $\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$。对于 256$\times$256 的图像此法有效，但该调度加噪过于激进：在低分辨率或较晚的时间步，图像已几乎是纯噪声，模型的容量被浪费在几乎没有特征的输入上。

Nichol & Dhariwal（2021）提出了**余弦调度**：

$$\bar{\alpha}_t = \cos^2\!\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

其中小偏移量 $s = 0.008$ 防止 $\bar{\alpha}_t$ 恰好达到零。这使 SNR 轨迹在各时间步上更平滑、更均匀，在模型必须学习区分连贯结构的中等噪声水平上花费更多步骤。余弦调度在 CIFAR-10 和 ImageNet 上显著改善了对数似然和样本质量（Nichol & Dhariwal, 2021）。

更深层的原则是：最优调度取决于数据分辨率和内容——低分辨率图像结构更简单，所需的破坏噪声更少，因此给定的 $\bar{\alpha}_t$ 在低分辨率时对应更高的有效 SNR。这种**分辨率-SNR 不匹配**是级联模型（如 Imagen）为每个分辨率阶段使用独立调度的原因之一。

---

## 大规模架构

### Q14 [基础] 解释潜在扩散模型如何降低计算成本

**Q:** 在学习到的潜在空间而非像素空间中运行扩散，如何降低计算量？编码器-解码器架构贡献了什么？

**A:** 直接在高分辨率像素空间中运行扩散模型的扩展性很差：512$\times$512$\times$3 的图像需要 U-Net 注意力层在 $512^2 = 262,144$ 个空间位置上操作，训练和采样都极其昂贵。**潜在扩散模型**（LDM；Rombach et al., 2022）通过先训练感知压缩模型——VQ-VAE 或 KL 正则化 VAE——将图像编码到紧凑的潜在空间，再在该潜在空间中运行扩散过程来解决这一问题。

编码器将图像压缩为潜在表示 $z \in \mathbb{R}^{h \times w \times c}$，空间下采样倍数 $f \in \{4, 8, 16\}$（512$\times$512 的图像映射到 $64 \times 64$ 或 $32 \times 32$ 的潜变量）。扩散 U-Net 在此潜在分辨率下操作，注意力复杂度降低 $f^2$ 到 $f^4$ 倍。解码器再将生成的潜变量映射回像素空间。

编码器-解码器的贡献不仅仅是压缩：VAE 被训练成在结构上相似的图像彼此靠近的潜在空间，KL 或 VQ 正则化防止潜在空间坍缩或爆炸。扩散模型因此在语义上有意义的特征空间中操作——全局结构（物体布局、场景构成）和局部纹理在不同潜在维度上自然分离，使去噪任务比像素空间去噪更易处理，因为后者需要同时推理两种尺度。Rombach et al.（2022）证明，使用 $f = 4$ 下采样倍数时，LDM 以像素空间模型的极少计算量实现了具有竞争力的 FID 分数。

---

### Q15 [进阶] 分析DiT架构及其扩展特性

**Q:** DiT 替换了标准扩散 U-Net 中的什么？哪些证据表明基于 Transformer 的扩散比基于 U-Net 的扩散具有更好的扩展性？

**A:** **DiT**（扩散 Transformer；Peebles & Xie, 2023）用纯视觉 Transformer 替代了大多数潜在扩散模型中使用的 U-Net 骨干。输入潜在块被展平为 token 序列，经过 $N$ 个带有改进条件的标准 Transformer 块处理。关键架构选择是 **adaLN-Zero** 条件：时间步 $t$ 和类别标签 $y$ 用于预测自适应层归一化的缩放和平移参数——类似于 FiLM 条件——输出投影初始化为零，使每个 Transformer 块的初始输出为零残差。

DiT 摒弃了 U-Net 的多尺度结构（编码器、瓶颈、跳跃连接解码器），转而使用在单一空间分辨率上操作的扁平相同 Transformer 块序列。这使得扩展变得直接：增加深度 $N$、宽度 $d$ 或块尺寸 $p$ 均可转化为 GFLOPs 和参数量的可预测增长。

Peebles & Xie（2023）在类别条件 ImageNet 256$\times$256 生成上进行了系统性扩展实验，发现计算量与 FID 之间存在简洁的幂律关系。**DiT-XL/2**（6.75亿参数，块尺寸 2）实现了 FID 2.27，在同一基准上优于包括 ADM 在内的所有先前扩散模型。关键在于，与 U-Net 模型在某些规模上 FID 改善饱和不同，DiT 的性能随计算量单调提升，表明 ViT 风格架构——在判别任务中已展现出类似的有利扩展性——将其扩展效率迁移到了生成建模。该架构此后已被大规模视频生成所采用。

---

### Q16 [进阶] 描述DALL-E 2的两阶段设计

**Q:** DALL-E 2 的层次化架构如何分离语义生成与视觉生成？这对生成多样性和可编辑性有什么影响？

**A:** **DALL-E 2**（Ramesh et al., 2022）使用两阶段架构，明确地将语义理解与视觉合成分开。第一阶段是**先验**（prior）：给定 CLIP 文本嵌入 $z_t$，扩散模型（或自回归模型）生成 CLIP 图像嵌入 $z_i$。第二阶段是**解码器**（也称 UNCLIP）：给定 $z_i$，以 CLIP 图像嵌入为条件的扩散模型生成全分辨率图像。

其动机在于 CLIP 嵌入空间的结构：CLIP 图像嵌入捕获图像的语义内容（物体、场景类型、风格），同时对精确的像素级细节（光照、纹理、摄像机角度）基本不变。让图像解码器以 $z_i$ 而非直接以文本为条件，允许模型生成许多在视觉上不同但语义内容相同的图像——同一概念在 $z_i$ 中编码的不同有效"渲染"。

这种分离有两个实际后果。第一，**图像变体**：通过从同一 $z_i$ 采样不同的解码，DALL-E 2 可以从单个文本提示生成多个语义一致但视觉上多样的图像。第二，**通过嵌入插值进行图像编辑**：在两幅图像的 CLIP 嵌入 $z_i^{(1)}$ 和 $z_i^{(2)}$ 之间插值，可以在它们之间产生平滑的语义轨迹，支持难以用文本提示表达的风格迁移等编辑操作。Ramesh et al.（2022）展示了这些能力，尽管两阶段设计也带来了误差累积：先验必须生成既在语义上忠实于文本、又在解码器可处理的分布范围内的嵌入。

---

### Q17 [进阶] 分析Imagen中的级联扩散

**Q:** Imagen 的级联设计如何实现 1024×1024 的生成？动态阈值化和文本编码器选择各自发挥了什么作用？

**A:** **Imagen**（Saharia et al., 2022）通过三阶段级联生成 1024$\times$1024 图像：基础模型从文本生成 64$\times$64 图像，第一个超分辨率模型上采样至 256$\times$256，第二个超分辨率模型上采样至 1024$\times$1024。每个阶段都以较低分辨率输入（对超分辨率阶段）和 T5-XXL 文本嵌入为条件。所有阶段均使用像素空间 U-Net 而非潜在扩散。

一项关键技术贡献是**动态阈值化**（dynamic thresholding）。标准扩散在采样时将像素值截断到 $[-1, 1]$（静态阈值化）。在高引导尺度（$w \gg 1$）下，预测的 $\hat{x}_0$ 经常超出此范围，导致颜色饱和和伪影。动态阈值化改为截断到预测绝对值的第 $s$ 百分位，然后重新缩放：若任意坐标 $j$ 的 $|\hat{x}_{0,j}| > s$，则将整个张量缩放 $s / \max_j |\hat{x}_{0,j}|$。这在防止离群坐标主导的同时保留了预测的相对结构。动态阈值化使得可以用大引导尺度训练，否则会产生饱和图像，这也是 Imagen 能在所有级联阶段以 $w \geq 7$ 有效运行的原因。

Saharia et al.（2022）发现，文本编码器规模是文本-图像对齐最重要的单一因素——比扩散模型大小更重要——而级联设计合理地分配了生成任务：基础模型处理全局构图，超分辨率模型添加细粒度纹理和细节。Imagen 在 COCO 零样本评估上报告了 FID 7.27（Saharia et al., 2022）。

---

## 视频、逆问题与评估

### Q18 [基础] 描述视频扩散模型如何扩展图像扩散

**Q:** 视频扩散需要哪些架构扩展？区分视频与图像生成的核心挑战是什么？

**A:** **视频扩散模型**（Video Diffusion Models；Ho et al., 2022）通过将 2D 空间卷积替换为**3D 时空卷积**并添加**时序注意力**层（在每个空间位置上跨帧进行注意力计算）来扩展 U-Net 骨干。前向过程对每一帧独立添加噪声，去噪 U-Net 学习对完整视频片段联合去噪。

核心计算挑战在于规模：$F$ 帧、分辨率 $H \times W$ 的视频需要在 $F \cdot H \cdot W$ 个位置上进行注意力计算。对于 16 帧 256$\times$256 的视频，这是 $1,048,576$ 个位置——完整的 3D 注意力完全不可行。Ho et al.（2022）通过**因式分解时空注意力**来解决这一问题：独立的空间注意力（在每帧内进行注意力计算）和时序注意力（在固定空间位置上跨帧计算），将复杂度从每个 token 的 $O(F^2 H^2 W^2)$ 降至 $O(F^2 + H^2 W^2)$。

更深层的挑战是**时序一致性**：模型必须跨帧生成语义连贯的内容——物体应保持其外观，物理规律应合理，摄像机运动应平滑。这要求模型内化时空动态，而不仅仅是逐帧外观。大规模时序一致的视频数据是主要瓶颈，因为相比图像，时序一致的视频收集和标注要昂贵得多。长视频生成带来了额外困难：在长达数秒的视频中保持一致性需要推理叙事或事件结构，超出了固定长度上下文窗口所能捕获的范围。

---

### Q19 [进阶] 解释扩散模型如何求解线性逆问题

**Q:** 如何将预训练的无条件扩散模型应用于求解测量条件重建问题，而无需针对特定任务训练？

**A:** 线性逆问题的形式为：给定测量 $y = Ax + \eta$，其中 $A$ 是已知的退化算子（下采样、掩码、模糊），$\eta$ 是噪声，重建干净信号 $x$。目标是从后验 $p(x|y) \propto p(y|x)\,p(x)$ 中采样，其中 $p(x)$ 是数据先验。预训练扩散模型通过其得分函数 $\nabla_x \log p_t(x)$ 隐式编码了该先验。

**DDRM**（Kawar et al., 2022）利用 $A = U\Sigma V^\top$ 的奇异值分解来解耦问题。在每个去噪步骤中，解在谱域中被投影到与测量一致的子空间，将扩散先验与未测量方向混合，将测量保真度与已测量方向混合。这给出了精确的测量一致性，且每步只需一次噪声预测器的前向传播。

**DPS**（扩散后验采样；Chung et al., 2022）通过近似似然梯度来处理非线性和非均匀退化。在每一步，噪声预测器提供干净图像的估计 $\hat{x}_0 = \mu_\theta(x_t, t)$。似然项近似为：

$$\nabla_{x_t} \log p(y|x_t) \approx \nabla_{x_t} \log p(y|\hat{x}_0(x_t))$$

并作为额外引导项加入标准得分更新。该近似由 Tweedie 估计器的自洽性所保证：$\hat{x}_0$ 是给定 $x_t$ 时 $x_0$ 的最小均方误差估计。DPS 处理不存在奇异值分解的退化（非线性前向模型、相位恢复），以每步额外计算开销为代价，泛化能力远超 DDRM 的假设。

---

### Q20 [进阶] 分析FID及其作为生成模型评估指标的局限性

**Q:** FID 衡量什么？哪些系统性局限性使其无法完整刻画生成模型的质量？

**A:** **FID**（Fréchet 初始距离；Heusel et al., 2017）衡量真实图像与生成图像的 Inception-v3 特征分布之间的 Fréchet 距离：

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)$$

其中 $(\mu_r, \Sigma_r)$ 和 $(\mu_g, \Sigma_g)$ 分别是真实和生成图像的 Inception 特征的均值和协方差。FID 越低，说明生成分布在该特征空间中越接近真实分布。

FID 存在四个系统性局限性。第一，**Inception 特征不足**：该特征空间针对 ImageNet 分类而非感知相似度进行了优化。欺骗分类器的生成伪影（错误纹理、幻觉物体）可能不被惩罚，而有效但不寻常的构图可能使 FID 虚高。第二，**样本量敏感性**：稳定的 FID 估计需要 $\geq 50,000$ 个生成和参考样本；样本更少时方差很大，掩盖了真实的模型差异。第三，**精度与召回率的混淆**：FID 是一个标量，当生成分布很好地覆盖真实分布时和当生成高保真样本时都会改善。一个只生成少量完美图像的模型，如果这些图像与训练分布的模式高度匹配，也可以获得很低的 FID。

精度/召回率框架直接解决了这一问题，通过分别测量生成样本是否落在真实分布流形内（精度）以及真实分布流形是否被生成样本覆盖（召回率）来评估。对于文本到图像模型，**CLIP 分数**衡量生成图像与文本提示之间的语义对齐——这是 FID 无法捕获的维度。来自成对比较（通过 ELO 评分等系统）的人类偏好分数仍是评估感知质量和提示忠实度的金标准，但大规模收集成本昂贵。

---

## 快速参考

| # | 难度 | 主题 | 章节 |
|---|------|------|------|
| Q1 | 基础 | DDPM前向与反向过程 | 扩散模型基础 |
| Q2 | 基础 | 得分匹配与去噪 | 扩散模型基础 |
| Q3 | 进阶 | 统一扩散模型的SDE框架 | 扩散模型基础 |
| Q4 | 进阶 | ELBO与简化训练目标 | 扩散模型基础 |
| Q5 | 基础 | DDIM确定性采样 | 加速采样 |
| Q6 | 进阶 | DPM-Solver基于ODE的加速 | 加速采样 |
| Q7 | 进阶 | 一致性模型 | 加速采样 |
| Q8 | 进阶 | 流匹配与扩散的比较 | 加速采样 |
| Q9 | 基础 | 无分类器引导 | 条件生成 |
| Q10 | 基础 | ControlNet空间条件 | 条件生成 |
| Q11 | 进阶 | 分类器引导与CFG的权衡 | 条件生成 |
| Q12 | 进阶 | 文本编码器选择：T5与CLIP | 条件生成 |
| Q13 | 基础 | 噪声调度设计 | 条件生成 |
| Q14 | 基础 | 潜在扩散模型 | 大规模架构 |
| Q15 | 进阶 | DiT架构与扩展性 | 大规模架构 |
| Q16 | 进阶 | DALL-E 2两阶段设计 | 大规模架构 |
| Q17 | 进阶 | Imagen中的级联扩散 | 大规模架构 |
| Q18 | 基础 | 视频扩散模型 | 视频、逆问题与评估 |
| Q19 | 进阶 | 扩散模型求解线性逆问题 | 视频、逆问题与评估 |
| Q20 | 进阶 | FID的局限性与评估指标 | 视频、逆问题与评估 |

## 参考文献

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Sohl-Dickstein et al., [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (2015)
- Song & Ermon, [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) (2019)
- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (2021)
- Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (2020)
- Nichol & Dhariwal, [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (2021)
- Lu et al., [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) (2022)
- Song et al., [Consistency Models](https://arxiv.org/abs/2303.01469) (2023)
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (2022)
- Liu et al., [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) (2022)
- Esser et al., [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) (2024)
- Ho & Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) (2022)
- Dhariwal & Nichol, [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) (2021)
- Zhang & Agrawala, [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (2023)
- Saharia et al., [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) (2022)
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (2022)
- Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (2023)
- Ramesh et al., [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) (2022)
- Ho et al., [Video Diffusion Models](https://arxiv.org/abs/2204.03458) (2022)
- Kawar et al., [Denoising Diffusion Restoration Models](https://arxiv.org/abs/2201.11793) (2022)
- Chung et al., [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687) (2022)
- Heusel et al., [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) (2017)
