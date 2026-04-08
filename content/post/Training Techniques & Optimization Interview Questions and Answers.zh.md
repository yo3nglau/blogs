---
title: "训练技术与优化：面试问题与推荐回答"
author: yo3nglau
date: '2026-04-08'
categories:
  - Interview
tags:
  - Deep Learning
  - Optimization
  - Training
toc: true
---

## 优化算法

### Q1 [基础] 解释 Adam 如何利用一阶与二阶矩估计优化参数

**Q:** Adam 优化器如何利用梯度历史为每个参数自适应调整学习率？

**A:** **Adam**（Adaptive Moment Estimation，自适应矩估计）为每个参数维护两个运行统计量：一阶矩 $m_t$（梯度的指数移动平均）和二阶矩 $v_t$（梯度平方的指数移动平均）(Kingma & Ba, 2015)。在每一步 $t$，这两个统计量的更新方式如下：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

其中 $g_t$ 是梯度，$\beta_1 = 0.9$、$\beta_2 = 0.999$ 为衰减率。由于 $m_t$ 和 $v_t$ 初始化为零，Adam 施加**偏差修正**以抵消初始化偏差：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

参数更新为 $\theta_t = \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$，其中 $\epsilon$ 是用于数值稳定性的小常数。其效果是：历史梯度较大的参数获得较小的有效学习率，而梯度较小或更新稀疏的参数则得到更大幅度的更新。这种自适应缩放通常比固定学习率的 SGD 收敛更快，尤其适用于稀疏或嘈杂的梯度场景。

---

### Q2 [进阶] 分析 Adam 中 L2 正则化失效的原因以及 AdamW 的修正方式

**Q:** 向损失函数添加 L2 正则化与在 Adam 中直接解耦权重衰减，二者在数学上有何区别？

**A:** 在标准 SGD 中，向损失函数添加 L2 惩罚项 $\frac{\lambda}{2}\|\theta\|^2$ 与**权重衰减**完全等价，因为此时梯度变为 $g_t + \lambda\theta$，更新步骤从参数中减去 $\eta\lambda\theta$——以固定比例对参数进行收缩。然而，这种等价性在 Adam 中不成立 (Loshchilov & Hutter, 2019)。

当 L2 正则化被加入损失函数时，正则化梯度 $\lambda\theta$ 会被吸收进 $g_t$，随后被 $\sqrt{\hat{v}_t} + \epsilon$ 除。对于历史梯度较大的参数，$\sqrt{\hat{v}_t}$ 较大，导致有效权重衰减不成比例地缩小——更新频繁的参数受到的正则化反而少于更新稀疏的参数，从而破坏了原本均匀收缩的意图。

**AdamW** 通过在自适应更新之外直接对参数施加权重衰减来修正这一问题：

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_{t-1}\right)$$

衰减项 $\lambda\theta_{t-1}$ 不再受二阶矩的缩放影响，因此无论梯度历史如何，它都能作为真正的权重衰减发挥作用。Loshchilov & Hutter (2019) 的实验表明，AdamW 在图像分类和语言建模基准上始终优于带 L2 正则化的 Adam，并已成为训练大型语言模型的默认优化器。

---

### Q3 [进阶] 识别 Adam 在训练初期遇到的方差问题及 RAdam 的解决方案

**Q:** 为何 Adam 有时会在训练初期出现发散或不稳定的更新，RAdam 采用什么机制来解决这一问题？

**A:** 在训练开始时，由于小 $t$ 时 $\beta_2^t \approx 1$，且指数移动平均尚未积累足够多的更新，二阶矩 $v_t$ 接近于零。偏差修正后，$\hat{v}_t$ 的方差仍可能很高——分母 $\sqrt{\hat{v}_t}$ 不可靠，导致有效学习率忽大忽小。这种初始化不稳定性可能引发早期发散，或在估计稳定之前将优化器锁定在损失曲面的不良区域 (Liu et al., 2020)。

**RAdam**（Rectified Adam，修正自适应矩估计）通过计算 $\hat{v}_t$ 方差的近似最大长度 $\rho_t$ 来解决这一问题：

$$\rho_t = \rho_\infty - \frac{2t\beta_2^t}{1 - \beta_2^t}, \quad \rho_\infty = \frac{2}{1-\beta_2} - 1$$

当 $\rho_t$ 超过某个阈值（约为 4）时，认为二阶矩估计已足够稳定，此时应用带方差修正项 $r_t$ 的自适应更新。当 $\rho_t$ 过小时，RAdam 退回到带动量的 SGD，从而完全避免不可靠的自适应缩放。

在实践中，RAdam 在许多场景下消除了预热调度的需求。Liu et al. (2020) 证明，RAdam 在机器翻译和语言模型微调任务上与带预热的 Adam 效果相当甚至更优，且无需调整预热超参数。

---

### Q4 [进阶] 比较深度学习中的一阶与二阶优化方法

**Q:** 二阶优化器利用了哪些一阶方法所忽视的信息，为何二阶方法至今未能在大规模训练中取代 Adam？

**A:** SGD、Adam 等一阶方法仅使用梯度 $\nabla_\theta \mathcal{L}$，它指向最陡上升方向。二阶方法则额外利用**海森矩阵**（Hessian）$H = \nabla^2_\theta \mathcal{L}$（或其近似），该矩阵编码了损失曲面的曲率信息。牛顿更新 $\theta \leftarrow \theta - H^{-1}\nabla_\theta\mathcal{L}$ 通过逆曲率对梯度进行重缩放，在平坦方向迈出更大步伐，在尖锐方向迈出更小步伐，从而在极小值附近实现二次收敛，而梯度下降只能达到线性收敛。

根本障碍在于：对于具有 $d$ 个参数的模型，存储和求逆完整海森矩阵的代价分别为 $O(d^2)$ 内存和 $O(d^3)$ 计算——对于数十亿参数的模型完全不可行。实用的近似方法包括对角海森估计和 **K-FAC**（Kronecker-Factored Approximate Curvature，克罗内克因子近似曲率），后者利用较小矩阵的克罗内克积来近似 Fisher 信息矩阵 (Martens & Grosse, 2015)。K-FAC 已成功应用于卷积和循环模型，在每个 epoch 上收敛更快，但每步开销较高。

更深层的问题在于，现代深度网络并非局部凸的：损失曲面包含鞍点、平坦区域和尖锐极小值 (Dauphin et al., 2014)。为凸优化设计的二阶方法可能被曲率吸引至鞍点，并收敛到泛化能力较差的尖锐极小值。经验上，带噪声的 SGD 所找到的平坦极小值往往具有更好的泛化性能。内存代价、计算开销与泛化方面的顾虑共同解释了为何一阶方法在大规模深度学习中仍居主导地位。

---

## 学习率调度

### Q5 [基础] 描述为何在训练开始时使用学习率预热

**Q:** 若没有预热阶段，优化动态会发生什么，从较小的学习率起步为何有所帮助？

**A:** 在训练最开始阶段，模型参数随机初始化，梯度估计不可靠——既因为模型输出近乎随机，也因为 Adam 的二阶矩累加器为零（参见 Q3）。在这种情况下使用较大学习率，可能将参数推入难以逃脱的损失曲面区域、破坏批归一化统计量，或在混合精度训练中引发 NaN 损失。

**预热**（Warmup）在固定步数内（大型模型通常为 1,000–10,000 步）将学习率从接近零线性或指数地增加至目标学习率。在此阶段，优化器以谨慎的小步幅推进，同时积累可靠的梯度统计量。一旦二阶矩估计趋于稳定，便可安全地应用完整学习率。

预热随 Transformer 架构的出现成为标准做法 (Vaswani et al., 2017)，该架构使用调度公式 $\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5},\, t \cdot T_w^{-1.5})$，其中 $T_w$ 为预热周期长度。在实践中，线性预热加余弦衰减已成为大型语言模型预训练中最常用的学习率调度方案。

---

### Q6 [进阶] 解释带热重启的余弦退火如何鼓励对损失曲面的探索

**Q:** SGDR 中周期性重置学习率相比单调衰减调度，如何改善模型泛化性能？

**A:** 标准的步进衰减或多项式衰减会单调地降低学习率，导致优化器逐渐收敛到某个局部极小值。**SGDR**（Stochastic Gradient Descent with Warm Restarts，带热重启的随机梯度下降）在每个周期内按余弦曲线衰减学习率，并在周期结束后将其重置为最大值 (Loshchilov & Hutter, 2017)：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)$$

其中 $T_{\text{cur}}$ 是自上次重启以来的步数，$T_i$ 是当前周期的总步数。

重启机制有两层作用。第一，在每次冷却至 $\eta_{\min}$ 后，模型处于损失曲面的某个局部区域；重启带来的大步长能够逃离当前盆地，探索新的区域。第二，集成效应：通过在每个周期极小值处保存模型权重快照，可以几乎零额外训练代价地集成多个模型——**Snapshot Ensembles**（快照集成）(Huang et al., 2017) 正是利用这一特性，在无需多次独立训练的情况下实现了有竞争力的准确率。

带热重启在仅使用单个周期（无重启）时自然退化为余弦退火调度，后者如今已成为标准做法。相比步进衰减，其核心优势在于平滑的连续调度，避免了离散学习率骤降所可能引发的损失突增。

---

### Q7 [进阶] 描述超收敛及其出现条件

**Q:** 在何种训练机制下，网络能在远少于标准调度所需的迭代次数内完成收敛，这一现象的背后原因是什么？

**A:** **超收敛**（Super-convergence）是指在使用循环调度配合较大最大学习率时，模型训练迭代次数比标准训练少一个数量级，但仍能达到相近准确率的现象 (Smith & Topin, 2019)。相比于需要数十个 epoch 并精心衰减学习率的标准训练，超收敛在单个周期内以比常规高 10–100 倍的学习率完成训练。

Smith & Topin (2019) 将超收敛归因于大学习率的正则化效果：大步长阻止优化器收敛到尖锐极小值，从而起到正则化作用。这减少了对其他正则化手段（dropout、权重衰减）的依赖，他们发现在大学习率阶段适当降低这些正则化强度能进一步促成超收敛。

该现象在特定条件下更容易观察到：相对较小的数据集、带批归一化的网络以及残差连接。在 ImageNet 等大规模数据集上，超收敛难以稳定复现。Smith & Topin (2019) 在 CIFAR-10 的 ResNet 上进行了演示，仅用 10,000 次迭代便达到超过 $93\%$ 的测试准确率，而标准训练通常需要约 100,000 次迭代。**学习率范围测试**——在短时运行中将学习率从小到大线性增加并观察损失变化——是他们提出的用于确定合适最大学习率的诊断方法。

---

### Q8 [进阶] 分析大规模训练中批量大小与学习率的关系

**Q:** 为什么增大批量大小需要相应调整学习率，这种缩放在实践中有哪些局限性？

**A:** 使用**大批量**训练时，梯度估计的方差低于小批量：小批量梯度的标准差以 $\sigma / \sqrt{B}$ 的比例随批量大小 $B$ 缩放。这意味着大批量的梯度步骤更准确，但在单位挂钟时间内覆盖的多样性更少。Goyal et al. (2017) 通过实验确立了**线性缩放规则**：将批量大小乘以 $k$ 时，学习率也应乘以 $k$。直觉上，$k$ 次小批量步骤覆盖的参数空间区域与一次 $k\times$ 学习率的大批量步骤大致相当。利用这一规则及 5 个 epoch 的线性预热，他们以 8,192 的批量大小在一小时内完成了 ImageNet 上 ResNet-50 的训练，与标准 256 批量 90 epoch 训练的准确率相当。

线性缩放规则在**噪声主导区间**（梯度信号远小于噪声）近似成立。当进入**曲率主导区间**（超大批量）时，步长受损失曲面最陡峭方向的限制，而非梯度噪声，此时进一步增大 $B$ 和 $\eta$ 会导致发散。该转变发生时的临界批量大小因数据集和架构而异；对于 ImageNet 上的 ResNet-50，降级在 $B = 16{,}384$ 以上开始显现。

因此，实际大批量训练需要：线性预热 (Goyal et al., 2017)、避免过度减小每设备有效批量大小的批归一化，以及在分布式场景下每设备批量过小时改用 ghost 批归一化或层归一化作为替代。

---

## 正则化与泛化

### Q9 [基础] 描述 Dropout 如何防止神经网络过拟合

**Q:** Dropout 在训练过程中对网络做了什么，为何这能减少过拟合？

**A:** **Dropout**（随机失活）在每次训练前向传播时，以概率 $p$ 随机将神经元激活置零 (Srivastava et al., 2014)。神经元输出随后被 $\frac{1}{1-p}$ 缩放以保持期望激活幅度（即逆 dropout）。推理时所有神经元均处于激活状态，不进行缩放。

核心直觉在于：dropout 防止神经元之间的协同适应（co-adaptation）——某个神经元不能依赖其他特定神经元始终存在，因此必须学习对自身独立有用的特征。Srivastava et al. (2014) 证明这等价于训练指数数量级的不同网络架构并对其预测取平均——一种隐式模型集成形式。他们在视觉、语音和文本任务上均验证了测试误差的降低，其中 $p = 0.5$ 是隐藏层的鲁棒默认值，$p = 0.8$ 适用于输入层。

在现代实践中，dropout 在卷积网络中使用较少（更倾向于空间 dropout 或权重衰减），但在 Transformer 的注意力层和前馈层中仍被广泛使用。最优 dropout 率取决于模型规模：具有更强过拟合能力的大型模型从较高的 $p$ 中受益更多。

---

### Q10 [进阶] 解释为何适当的权重衰减比 L2 正则化更适合自适应优化器

**Q:** 除 AdamW 的修正之外，从理论层面看，权重衰减为何是自适应方法训练神经网络时更合理的正则化方式？

**A:** 理论基础来自正则化的**最大后验估计**（MAP，Maximum A Posteriori）解释。向损失函数添加 L2 惩罚项 $\frac{\lambda}{2}\|\theta\|^2$，等价于在权重上施加各向同性高斯先验 $\mathcal{N}(0, 1/\lambda)$ 并求解 MAP 估计。对于 SGD，所得更新等价于权重衰减，正则化相对于梯度信号被正确施加。

使用自适应优化器时，这种贝叶斯解释失效，因为每个参数的有效学习率被 $1/(\sqrt{\hat{v}_t} + \epsilon)$ 缩放。Adam 加 L2 下隐式定义的先验不再是各向同性的——历史梯度较大的参数其先验实际上被削弱了。这意味着正则化对更新稀疏的参数（往往不需要太多正则化）最强，对更新频繁的参数（最容易过拟合）反而最弱。

如 AdamW 中的解耦权重衰减确保先验保持各向同性：无论梯度历史如何，每个参数每步都以相同的比例 $\lambda$ 向零收缩 (Loshchilov & Hutter, 2019)。从优化几何的角度，这对应于在球约束内进行梯度下降，而非向梯度添加惩罚项，是更具原则性的正则化方式。这种差异在大规模模型中最为显著——某些参数更新极为频繁（注意力投影），而另一些则更新稀疏（低频词元对应的嵌入行）。

---

### Q11 [基础] 识别梯度裁剪的必要场景及阈值设定方法

**Q:** 哪些训练异常促使了梯度裁剪的使用，应如何合理校准裁剪范数阈值？

**A:** **梯度裁剪**（Gradient Clipping）在应用优化器更新之前对梯度向量的范数进行限制。最常见的形式是全局范数裁剪：若 $\|\nabla_\theta\mathcal{L}\|_2 > \tau$，则将梯度重新缩放为 $\tau \cdot \nabla_\theta\mathcal{L} / \|\nabla_\theta\mathcal{L}\|_2$。Pascanu et al. (2013) 在循环神经网络的背景下提出了这一方法，其中梯度会随时间步指数级增长，形成**梯度爆炸**（exploding gradients）——对展开的多个时间步进行反复矩阵乘法，使梯度范数趋于无穷大，导致灾难性的大步参数更新。

在 RNN 之外，梯度裁剪还适用于以下场景：使用大学习率训练的深层架构、微调时损失曲面曲率较大，以及混合精度训练中 FP16 可能产生不正确梯度缩放的情形。裁剪阈值 $\tau$ 通常通过监控训练初期的梯度范数来设定：1.0 是常用默认值，但将其设置在早期训练梯度范数的第 95 百分位处是更具原则性的做法。若裁剪在超过少量步骤中被触发，则表明学习率过大或梯度累积步数过长。

---

### Q12 [基础] 解释标签平滑对训练动态和模型校准的影响

**Q:** 标签平滑改变了交叉熵损失中的哪些内容，它如何影响模型预测的置信度？

**A:** 标准交叉熵训练使用**硬标签**（hard labels）——将概率 1 分配给正确类别、0 分配给所有其他类别的独热向量。这鼓励模型输出使正确类别的 softmax 概率趋近于 1、其他类别趋近于 0 的 logits，在损失曲面上形成无限深的凹坑。**标签平滑**（Label Smoothing，Müller et al., 2019）将硬目标替换为软分布：

$$q_i = (1 - \epsilon)\cdot\mathbf{1}[i = y] + \frac{\epsilon}{K}$$

其中 $\epsilon$ 是平滑系数（通常为 0.1），$K$ 是类别数。正确类别的目标变为 $1 - \epsilon + \epsilon/K$，其他类别则获得 $\epsilon/K$ 的概率质量。

实际效果是模型被阻止过度自信：它无法通过使 logit 差距无限大来实现零损失。Müller et al. (2019) 表明，标签平滑能产生校准更好的模型——softmax 概率与经验准确率更为吻合——并改善图像分类上的泛化性能。然而，他们也发现标签平滑会损害知识蒸馏（knowledge distillation）：经过标签平滑训练的教师模型输出的概率分布更软、信息量更少，从而减少了传递给学生的信息量。这说明标签平滑是一种正则化手段，在模型用作教师时应当去除或降低其强度。

---

## 训练效率

### Q13 [基础] 描述混合精度训练及其在精度降低情况下保持准确率的原因

**Q:** 混合精度训练使用哪些数值格式，它如何避免低精度通常带来的准确率损失？

**A:** **混合精度训练**（Mixed Precision Training，Micikevicius et al., 2018）在前向传播和反向传播中使用 16 位浮点数（FP16），在权重更新时使用 32 位浮点数（FP32）。现代硬件——NVIDIA Tensor Cores、Google TPUs——在 FP16 矩阵运算上可实现 2–8 倍的吞吐量提升，内存消耗减半，从而支持更大的批量或更大的模型。

三种技术在精度降低的情况下保持了准确率。第一，维护一份 FP32 的**主权重副本**；每次梯度更新后，FP16 权重从中派生。在 FP16 中会下溢为零的小梯度更新在 FP32 中能正确累积。第二，**损失缩放**（loss scaling）在反向传播之前将损失乘以一个大常数（如 $2^{15}$），将梯度值移出 FP16 次正规数范围以避免下溢，再在权重更新前将梯度反向缩放回原始幅度。第三，部分操作（批归一化统计量、损失计算）保持在 FP32 以维护数值稳定性。

Micikevicius et al. (2018) 证明混合精度在图像分类（ResNet-50，ImageNet）、语音识别和语言建模上均与 FP32 准确率相当，且无需任何架构改动。现代训练框架（PyTorch AMP、JAX）自动化了损失缩放和 FP16/FP32 转换边界，使混合精度训练成为实践中的默认选项。

---

### Q14 [进阶] 分析梯度检查点的计算权衡

**Q:** 梯度检查点如何降低反向传播期间的峰值内存占用，额外的计算代价在何时是值得的？

**A:** 在标准反向传播中，前向传播的所有中间激活值必须保留在内存中，直至对应的反向传播计算完成——对于具有 $L$ 层的网络，内存代价为 $O(L)$。**梯度检查点**（Gradient Checkpointing，Chen et al., 2016）通过仅存储一部分激活值（**检查点**），并在反向传播时从最近的检查点重新计算中间激活值来降低这一代价。

最优检查点策略以 $\sqrt{L}$ 步为间隔存储激活值，所需内存为 $O(\sqrt{L})$，代价是对每个片段额外进行一次前向传播。Chen et al. (2016) 证明这将激活内存从 $O(L)$ 降至 $O(\sqrt{L})$，同时计算量增加约 $33\%$（因为每个片段被计算两次）。对于极深的网络或 Transformer 中较长的序列，内存节省远超计算开销。

成本收益分析依赖于具体场景。对于处理长序列的 Transformer，激活内存以 $O(B \cdot T \cdot d)$ 的比例增长（$B$ 为批量大小，$T$ 为序列长度，$d$ 为模型维度）——梯度检查点可能是模型能否放入 GPU 内存的决定性因素。当内存不是瓶颈时（如高内存硬件上的小模型），由于 $33\%$ 的计算开销直接降低训练吞吐量，其使用则不那么合理。在实践中，PyTorch 等框架提供逐层检查点控制，允许从业者选择性地对最耗内存的层（注意力块）进行检查点处理，同时保留较轻层的标准计算。

---

### Q15 [进阶] 评估梯度累积作为大批量训练替代方案的效果

**Q:** 梯度累积如何模拟更大批量的训练，哪些效果无法被复现？

**A:** **梯度累积**（Gradient Accumulation）在应用单次优化器更新之前，对连续的 $k$ 个大小为 $B_\mu$ 的微批次（micro-batch）执行前向-反向传播，从而在不将所有样本同时存入内存的情况下有效模拟大小为 $k \cdot B_\mu$ 的批量。每个微批次的梯度在更新步骤前进行求和（或求平均）。这使得在内存有限的硬件上进行大批量训练成为可能，或在所需批量大小超过单个 GPU 容量时提供解决方案。

梯度更新本身的模拟是精确的：$\nabla\mathcal{L}(B_\mu \cup \ldots \cup B_\mu^{(k)}) = \frac{1}{k}\sum_{i=1}^k \nabla\mathcal{L}(B_\mu^{(i)})$。然而，**批归一化**无法被忠实复现：BatchNorm 独立地在每个微批次上计算均值和方差统计量，而非在完整的有效批量上计算。这意味着归一化统计量与真实大批量训练存在差异，且推理时使用的运行统计量也是基于微批次计算的。对于使用 LayerNorm 或 RMSNorm 的模型（多数 Transformer），这一问题不存在。

一个更微妙的差异在于：梯度累积无法按比例缩短挂钟时间——它需要执行 $k$ 次顺序的前向-反向传播，吞吐量随累积步数线性下降。而通过数据并行实现的真实大批量训练可跨设备并行化来缩短挂钟时间。因此，梯度累积是一种内存解决方案，而非吞吐量解决方案，不应与分布式大批量训练在挂钟效率方面混为一谈。

---

### Q16 [进阶] 解释激活函数选择对梯度流动和训练稳定性的影响

**Q:** 激活函数的哪些性质决定了梯度能否在深层网络中有效传播，现代选择如何解决 ReLU 的局限性？

**A:** 理想的激活函数在梯度流动方面应具备以下特性：非饱和性（大输入时梯度不消失）、平滑性（在过渡区域提供有用的梯度信号）以及计算高效性。**ReLU**（$f(x) = \max(0, x)$）解决了 sigmoid 和 tanh 的饱和问题：其梯度要么为 0 要么为 1，对于正激活不会饱和。然而，ReLU 存在**神经元死亡**（dying ReLU）问题：若某个神经元的预激活始终为负（例如因某次错误更新导致较大负偏置），其梯度将永久为零，神经元再也无法恢复。

**GELU**（Gaussian Error Linear Unit，高斯误差线性单元）$f(x) = x\Phi(x)$（其中 $\Phi$ 为标准正态分布的累积分布函数）通过根据输入幅度进行软门控来避免 ReLU 的硬零门控 (Hendrycks & Gimpel, 2016)。这使得小的负激活值能够通过，从而防止神经元死亡问题，并产生更平滑的梯度。GELU 已成为 Transformer 架构中占主导地位的激活函数。

**Swish** $f(x) = x \cdot \sigma(\beta x)$ (Ramachandran et al., 2017) 是一种自门控变体，在线性（$\beta \to 0$）和类 ReLU（$\beta \to \infty$）行为之间平滑过渡。GELU 和 Swish 都具有以下共同特性：无上界（防止大正输入时的梯度消失）、平滑（在过渡区域提供有用梯度信号），以及非单调性——小负输入时的轻微下凹起到软门控的作用。对于极深的网络，激活函数与初始化方案和归一化放置方式相互影响；残差连接和 LayerNorm 在很大程度上降低了对激活函数选择的敏感性，这也解释了为何 GELU 和 ReLU 在 Transformer 中均表现良好，但在普通深层网络中差异更为显著。

---

## 归一化与分布式训练

### Q17 [基础] 比较 BatchNorm 与 LayerNorm 并确定各自的适用场景

**Q:** 两种归一化方法各自在哪个维度上操作，为何这使 BatchNorm 不适用于某些任务？

**A:** **批归一化**（Batch Normalization，BatchNorm，Ioffe & Szegedy, 2015）对批量中每个特征维度进行归一化：对于大小为 $B$ 的小批量，它计算每个神经元在 $B$ 个激活值上的均值和方差，然后归一化。每个特征的可学习仿射参数 $\gamma$ 和 $\beta$ 恢复表达能力。BatchNorm 的统计量依赖于批量大小：小批量产生嘈杂的估计，当 $B < 16$ 时性能急剧下降。更重要的是，BatchNorm 在同一批次的样本之间建立了依赖关系，使其与自回归推理（批量大小为 1）和 RNN（不同时间步会被混合）不兼容。

**层归一化**（Layer Normalization，LayerNorm，Ba et al., 2016）则对每个单独样本的特征维度进行独立归一化，计算单个样本所有 $d$ 个特征的均值和方差。这使统计量独立于批量大小和其他样本，适用于 RNN、Transformer 和变长序列。代价是 LayerNorm 对图像上的卷积网络效果较差，因为逐样本归一化会过度削弱空间结构。

在实践中：BatchNorm 适用于批量大小合理的卷积图像模型；LayerNorm 是跨 NLP 和视觉领域的 Transformer 模型的标准选择。

---

### Q18 [进阶] 分析 RMSNorm 在许多大型语言模型架构中取代 LayerNorm 的原因

**Q:** RMSNorm 从 LayerNorm 的计算中移除了什么，这种简化在理论和实验上有何依据？

**A:** **RMSNorm**（Root Mean Square Layer Normalization，均方根归一化，Zhang & Sennrich, 2019）是 LayerNorm 的简化版本，仅使用激活值的均方根进行归一化，省去了均值中心化：

$$\bar{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2}$$

这消除了标准 LayerNorm 中的均值减法（重中心化）和偏置参数 $\beta$。Zhang & Sennrich (2019) 证明 LayerNorm 中的重中心化操作对训练稳定性贡献甚微，而重缩放（按比例归一化）提供了大部分收益。由于操作更少且硬件效率更高，与 LayerNorm 相比，实验加速约为 $7$–$64\%$。

理论依据在于：激活值的尺度（而非均值）才是导致深层网络梯度流动不稳定的根本原因。RMSNorm 对激活向量施加球形约束——强制 $\|\mathbf{x}\|_2 = \sqrt{d}$——这对于稳定训练已经足够。LayerNorm 中的均值减法是对批统计量的类比，但在特征维度上这种类比较为牵强。

LLaMA、Gemma、Mistral 等现代大型语言模型均使用 RMSNorm 而非 LayerNorm，通常采用**前置归一化**（pre-norm）的方式（在注意力和前馈子层之前应用），而非后置归一化。前置归一化可防止残差流在传递至下一层前被缩小，从而避免深层残差网络中的梯度消失 (He et al., 2016)。

---

### Q19 [基础] 区分分布式深度学习中的三种主要并行策略

**Q:** 数据并行、模型并行和流水线并行各自如何在设备间分配训练工作？

**A:** **数据并行**（Data Parallelism）在每台设备上复制完整模型，并将小批量的不同子集分配给各个副本。每次反向传播后，梯度在设备间进行同步（全规约，all-reduce），所有副本的模型权重以相同方式更新。数据并行使批量大小与设备数量成正比，是最易实现的策略，但要求完整模型能放入单台设备的内存。

**模型并行**（Model Parallelism，即张量并行）将模型的权重矩阵拆分到不同设备上。在 Megatron-LM (Shoeybi et al., 2019) 中，注意力头和前馈层的权重行按列分割到不同设备，每个 Transformer 子层结束时进行全规约。这使得训练超过单台设备内存容量的模型成为可能，但会在每层边界引入频繁的设备间通信。

**流水线并行**（Pipeline Parallelism）将不同层分配给不同设备：设备 1 运行第 1 到第 $k$ 层，设备 2 运行第 $k{+}1$ 到第 $2k$ 层，以此类推。数据以微批次流的形式传输：设备 1 处理微批次 1 后立即开始处理微批次 2，同时设备 2 处理微批次 1。GPipe (Huang et al., 2019) 将这一方案形式化为同步流水线并行，配合微批次间的梯度累积。其局限性在于**流水线气泡**（pipeline bubble）：在每个批次的开始和结束时，流水线处于填充或排空状态，部分设备处于空闲——气泡比例为 $(D-1)/(D-1+M)$，其中 $D$ 为流水线深度，$M$ 为微批次数量。

在实践中，大规模训练（如 GPT-3 量级的模型）通常结合三种策略：节点间采用数据并行，节点内 GPU 间采用模型并行，节点间还采用流水线并行。

---

### Q20 [进阶] 解释 ZeRO 优化器各阶段如何减少数据并行训练中的内存冗余

**Q:** 数据并行工作节点之间存在哪些内存冗余，ZeRO 的三个阶段如何逐步消除这些冗余？

**A:** 在标准数据并行训练中，$N$ 个工作节点中的每个都保存以下内容的完整副本：(1) 优化器状态（如 Adam 的 $m_t$ 和 $v_t$，在 FP32 中通常为参数量的 $2\times$），(2) 梯度（与参数大小相同），以及 (3) 参数本身。对于具有 $\Psi$ 个参数的模型，每个工作节点占用 $16\Psi$ 字节（FP32 参数 $4\Psi$ + Adam 状态 $8\Psi$ + 梯度 $4\Psi$），$N$ 个工作节点的总内存为 $16N\Psi$。所有这些都是冗余的——每个工作节点保存了完全相同的副本 (Rajbhandari et al., 2020)。

**ZeRO**（Zero Redundancy Optimizer，零冗余优化器，Rajbhandari et al., 2020）分三个阶段消除这些冗余：

- **阶段 1** 将优化器状态分片到 $N$ 个工作节点。每个工作节点仅保存 $1/N$ 的 Adam $m_t$ 和 $v_t$ 状态。每节点内存降至 $4\Psi + 12\Psi/N$。
- **阶段 2** 在此基础上进一步分片梯度。每次反向传播后，梯度被规约-分散（reduce-scatter）到负责对应参数分片的工作节点。每节点内存降至 $4\Psi + 12\Psi/N$（梯度在规约-分散后被丢弃）。
- **阶段 3** 进一步分片参数本身。工作节点仅存储 $1/N$ 的参数；在每次前向或反向传播之前，参数通过全收集（all-gather）从各所有者处聚合。每节点内存降至 $16\Psi/N$，相比标准 DDP 实现 $N$ 倍的缩减。

阶段 3 在每次前向和反向传播时增加了全收集通信，但由于通信与计算可以重叠，吞吐量开销较小。Rajbhandari et al. (2020) 演示了在 400 块 V100 GPU 上使用 ZeRO 阶段 3 训练 1,700 亿参数模型的实验，这在标准 DDP 下是不可能实现的。ZeRO 现已成为 DeepSpeed 的核心，并被集成到 PyTorch FSDP（Fully Sharded Data Parallel，全分片数据并行）中。

---

## 快速参考

| # | 难度 | 主题 | 章节 |
|---|------|------|------|
| Q1 | 基础 | Adam 一阶/二阶矩估计 | 优化算法 |
| Q2 | 进阶 | AdamW 与 Adam 中 L2 正则化的对比 | 优化算法 |
| Q3 | 进阶 | RAdam 方差修正 | 优化算法 |
| Q4 | 进阶 | 二阶与一阶优化方法对比 | 优化算法 |
| Q5 | 基础 | 学习率预热 | 学习率调度 |
| Q6 | 进阶 | 带热重启的余弦退火（SGDR） | 学习率调度 |
| Q7 | 进阶 | 超收敛 | 学习率调度 |
| Q8 | 进阶 | 批量大小与学习率缩放 | 学习率调度 |
| Q9 | 基础 | Dropout 正则化 | 正则化与泛化 |
| Q10 | 进阶 | 自适应优化器中的权重衰减与 L2 正则化 | 正则化与泛化 |
| Q11 | 基础 | 梯度裁剪 | 正则化与泛化 |
| Q12 | 基础 | 标签平滑与模型校准 | 正则化与泛化 |
| Q13 | 基础 | 混合精度训练 | 训练效率 |
| Q14 | 进阶 | 梯度检查点的内存权衡 | 训练效率 |
| Q15 | 进阶 | 梯度累积与大批量训练的对比 | 训练效率 |
| Q16 | 进阶 | 激活函数与梯度流动 | 训练效率 |
| Q17 | 基础 | BatchNorm 与 LayerNorm 对比 | 归一化与分布式训练 |
| Q18 | 进阶 | 大型语言模型中的 RMSNorm 与 LayerNorm | 归一化与分布式训练 |
| Q19 | 基础 | 数据并行、模型并行与流水线并行 | 归一化与分布式训练 |
| Q20 | 进阶 | ZeRO 优化器内存缩减 | 归一化与分布式训练 |

## 参考文献

- Kingma & Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (2015)
- Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (2019)
- Liu et al., [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) (2020)
- Martens & Grosse, [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671) (2015)
- Dauphin et al., [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572) (2014)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Loshchilov & Hutter, [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) (2017)
- Huang et al., [Snapshot Ensembles: Train 1, Get M for Free](https://arxiv.org/abs/1704.00109) (2017)
- Smith & Topin, [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120) (2019)
- Goyal et al., [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) (2017)
- Srivastava et al., [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) (2014)
- Müller et al., [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) (2019)
- Pascanu et al., [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063) (2013)
- Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (2018)
- Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (2016)
- Hendrycks & Gimpel, [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) (2016)
- Ramachandran et al., [Swish: a Self-Gated Activation Function](https://arxiv.org/abs/1710.05941) (2017)
- Ioffe & Szegedy, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (2015)
- Ba et al., [Layer Normalization](https://arxiv.org/abs/1607.06450) (2016)
- Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (2019)
- He et al., [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (2016)
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019)
- Huang et al., [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) (2019)
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (2020)
