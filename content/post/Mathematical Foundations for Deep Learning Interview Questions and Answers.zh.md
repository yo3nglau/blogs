---
title: "深度学习数学基础：面试问题与推荐回答"
author: yo3nglau
date: '2026-05-14'
categories:
  - Interview
tags:
  - Deep Learning
  - Mathematics
  - Optimization
toc: true
---

## 线性代数

### Q1 [基础] 描述特征值与特征向量的几何含义

**Q：** 特征值与特征向量在几何上代表什么？它们为何有助于理解机器学习中的线性变换？

**A：** 对于方阵 $A$ 和非零向量 $v$，**特征向量**（eigenvector）方程 $Av = \lambda v$ 表明：将 $A$ 作用于 $v$ 后，所得向量与 $v$ 方向相同，只是被标量**特征值**（eigenvalue）$\lambda$ 进行了缩放。几何上，$v$ 是变换不会旋转的特殊方向——只会拉伸（若 $|\lambda| > 1$）、压缩（若 $|\lambda| < 1$）或反向（若 $\lambda < 0$）。其他所有向量既会被旋转，也会被缩放。

对于对称半正定矩阵——协方差矩阵和 Gram 矩阵天然属于这一类——所有特征值非负，所有特征向量两两正交。这种正交性意味着特征向量构成了一个自然坐标系，与变换的主方向对齐。实际上，数据集的协方差矩阵的特征向量指向方差最大的方向，特征值度量对应方向上的方差大小。注意力得分矩阵 $QK^\top$ 以及线性层中的权重矩阵，均可通过其谱来分析信号传播和条件数特性。

---

### Q2 [基础] 解释 SVD 及其如何将特征分解推广至非方阵

**Q：** 什么是奇异值分解？当矩阵不是方阵时，它与特征分解有何关系？

**A：** **奇异值分解**（SVD）将任意 $m \times n$ 矩阵 $A$ 分解为 $A = U\Sigma V^\top$，其中 $U \in \mathbb{R}^{m \times m}$ 和 $V \in \mathbb{R}^{n \times n}$ 是正交矩阵，$\Sigma \in \mathbb{R}^{m \times n}$ 是对角矩阵，对角元素为非负的**奇异值**（singular values）$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$。$U$ 的列向量是左奇异向量，$V$ 的列向量是右奇异向量。与特征分解不同，SVD 对任意形状的矩阵均有定义，无需矩阵可对角化。

SVD 与特征分解的联系是精确的：$A^\top A = V \Sigma^\top \Sigma V^\top$ 正是对称半正定矩阵 $A^\top A$ 的特征分解，因此 $A$ 的奇异值是 $A^\top A$ 特征值的平方根。同样地，$AA^\top = U \Sigma \Sigma^\top U^\top$。最大奇异值 $\sigma_1 = \|A\|_2$ 是算子范数，控制 $A$ 对输入向量的最大放大倍数。

在深度学习中，SVD 出现在权重矩阵分析、梯度分析（各层的雅可比矩阵）、注意力机制研究以及低秩适应方法（如 LoRA）中——LoRA 将权重更新表示为低秩因子 $\Delta W = BA$，其中 $\text{rank}(BA) \ll \min(m,n)$。

---

### Q3 [进阶] 通过 SVD 与最优低秩近似定理分析 PCA

**Q：** PCA 与 SVD 有何关系？保留前 $k$ 个奇异值有何理论保证？

**A：** **主成分分析**（PCA）在中心化数据矩阵 $X \in \mathbb{R}^{n \times d}$（每行为一个数据点，各列零均值）中寻找保留最大方差的 $k$ 维线性子空间。样本协方差为 $C = X^\top X / n$，其特征分解为 $C = V\Lambda V^\top$。主成分是 $V$ 的列向量，投影后的数据为 $Z = XV \in \mathbb{R}^{n \times k}$。

这恰好对应 $X$ 的 SVD：将 $X$ 写为 $X = U\Sigma V^\top$，$V$ 的列向量是右奇异向量（主方向），$U\Sigma$ 的列向量是在这些方向上的投影，而 $\sigma_i^2 / n$ 等于第 $i$ 个主成分所解释的方差。因此，PCA 和 SVD 是同一计算的两种视角。

只保留前 $k$ 个奇异值的理论依据是 **Eckart–Young–Mirsky 定理**：在所有秩为 $k$ 的矩阵 $B$ 中，截断 SVD $A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$ 使 Frobenius 范数近似误差 $\|A - B\|_F$ 最小。残差为 $\|A - A_k\|_F^2 = \sum_{i > k} \sigma_i^2$。这一最优性保证表明，舍弃小奇异值是在秩约束下最优的压缩方式，而非启发式操作。它也解释了为何在线性模型和最小二乘损失下，基于 SVD 的降维在信息意义上是最优的。

---

### Q4 [进阶] 解释矩阵条件数及其对深度网络训练的影响

**Q：** 矩阵的条件数是什么？病态条件在深度网络优化中如何体现？

**A：** 矩阵 $A$ 的**条件数**（condition number）为 $\kappa(A) = \sigma_{\max} / \sigma_{\min}$，即最大奇异值与最小奇异值之比。正交矩阵的 $\kappa = 1$（条件完美）；$\kappa \gg 1$ 表明矩阵病态。几何上，条件数衡量 $A$ 对单位球形状的扭曲程度：球体经变换后变为长轴比为 $\kappa$ 的椭球体。

在优化中，某点处 Hessian 矩阵 $H = \nabla^2 L$ 的条件数决定了梯度下降的收敛速度。对于二次损失，最优步长为 $2/(\lambda_{\max} + \lambda_{\min})$，收敛需要 $O(\kappa(H) \log(1/\epsilon))$ 步，而当 $\kappa = 1$ 时仅需 $O(\log(1/\epsilon))$ 步。在深度网络中，病态的权重矩阵使参数空间不同方向上的梯度量级差异悬殊，迫使使用全局保守的学习率，导致训练缓慢。

实际上，多个层面的手段共同对抗病态问题。**权重初始化**方案（Glorot & Bengio, 2010；He et al., 2015）通过选择方差使每一层大致保持信号幅度，将雅可比矩阵的谱范数保持在单位附近。**批归一化**（Ioffe & Szegedy, 2015）对每层的预激活值进行归一化，平滑损失景观、降低有效条件数，在实践中可支持更大的学习率和更快的收敛。梯度截断通过限制梯度范数（而非各分量）来处理梯度爆炸——这是深度循环网络中病态的另一种表现。

---

### Q5 [进阶] 描述矩阵微积分如何处理向量值与矩阵值的导数

**Q：** 梯度、雅可比矩阵与链式法则如何从标量推广到向量和矩阵输入？这对反向传播有何意义？

**A：** 对于标量函数 $f: \mathbb{R}^n \to \mathbb{R}$，**梯度**（gradient）$\nabla_x f \in \mathbb{R}^n$ 的分量为 $(\nabla_x f)_i = \partial f / \partial x_i$。对于向量函数 $f: \mathbb{R}^n \to \mathbb{R}^m$，**雅可比矩阵**（Jacobian）$J \in \mathbb{R}^{m \times n}$ 满足 $J_{ij} = \partial f_i / \partial x_j$。反向传播传播的是**向量-雅可比积**（VJP，vector-Jacobian product），而非完整的雅可比矩阵：给定上游梯度 $\bar{y} \in \mathbb{R}^m$（损失关于函数输出的梯度），对输入梯度的贡献为 $\bar{x} = J^\top \bar{y} \in \mathbb{R}^n$，计算代价为 $O(mn)$，无需显式构造 $J$。

深度学习中有几个反复出现的恒等式。对于线性层 $y = Wx$：
- $\partial L / \partial x = W^\top (\partial L / \partial y)$
- $\partial L / \partial W = (\partial L / \partial y)\, x^\top$

对于二次型 $f = x^\top A x$：$\nabla_x f = (A + A^\top)x$，当 $A$ 对称时简化为 $2Ax$。对于迹 $f = \mathrm{tr}(AB)$：$\partial f / \partial A = B^\top$。

其实际意义在于：网络中的每一层必须实现两个函数——前向传播（根据输入计算输出并缓存中间值）和反向传播（根据上游梯度计算 VJP）。逆模式自动微分（所有主流深度学习框架的底层机制）将这些 VJP 沿计算图链式相乘，使计算代价与参数数量无关，仅为两次前向传播的代价——当参数数量远超输出维度时，这一性质至关重要。

---

## 概率与统计

### Q6 [基础] 解释 MLE 及其如何推导常用损失函数

**Q：** 最大似然估计如何指导损失函数的选择？什么假设分别导向均方误差与交叉熵？

**A：** **最大似然估计**（MLE）选择使观测数据概率最大的模型参数：$\hat\theta = \arg\max_\theta \prod_i p_\theta(x_i)$。取对数后，乘积变为求和，变号后得到等价的最小化问题：$\hat\theta = \arg\min_\theta \sum_i -\log p_\theta(x_i)$。似然模型 $p_\theta$ 的选择直接决定了损失函数。

若给定输入 $x$ 的目标 $y$ 被建模为**高斯分布**：$p_\theta(y|x) = \mathcal{N}(f_\theta(x),\, \sigma^2)$，则 $-\log p_\theta(y|x) = (y - f_\theta(x))^2 / 2\sigma^2 + \text{const}$。对 $\theta$ 最小化此负对数似然等价于**均方误差**（MSE）回归。若 $y$ 被建模为**伯努利分布**：$p_\theta(y|x) = \sigma(f_\theta(x))^y (1-\sigma(f_\theta(x)))^{1-y}$，则负对数似然为**二元交叉熵**。对于具有 softmax 输出 $q_c$ 的 $C$ 类**分类**模型，则为多类**交叉熵** $-\log q_{y_\text{true}}$。

该框架还可推广到更特殊的损失函数。拉普拉斯似然 $p_\theta(y|x) \propto \exp(-|y - f_\theta(x)| / b)$ 给出 **L1（MAE）损失**，由于拉普拉斯分布尾部比高斯更重，其对异常值的鲁棒性更强。MLE 因此为几乎所有标准损失函数提供了有原则的推导：分布的选择编码了对数据噪声结构的假设。

---

### Q7 [基础] 描述高斯分布在深度学习中的作用

**Q：** 为何高斯分布在深度学习的理论与实践中如此普遍？

**A：** 密度函数为 $p(x) = (2\pi\sigma^2)^{-1/2}\exp(-(x-\mu)^2 / 2\sigma^2)$ 的**高斯分布**（Gaussian distribution）$\mathcal{N}(\mu, \sigma^2)$ 在深度学习中大量出现，根本原因有两点。第一，它是在固定均值和方差约束下的**最大熵分布**：在所有具有相同前两阶矩的分布中，高斯分布做出最少的额外假设，是与观测统计量相容的最无信息选择。第二，**中心极限定理**保证：无论各随机变量的个体分布如何，大量独立随机变量之和在分布上收敛到高斯，这解释了为何当结果由许多微小加性贡献构成时，高斯噪声模型是合适的。

实践中，高斯分布出现在权重初始化（小高斯噪声防止对称性破坏失败）、VAE 的潜变量先验（$p(z) = \mathcal{N}(0, I)$）、鲁棒性训练的高斯噪声增强，以及无限宽网络的理论分析（神经正切核；无限宽随机初始化网络的函数空间是一个高斯过程）中。多元高斯分布 $\mathcal{N}(\mu, \Sigma)$ 还具有**对线性变换封闭**的额外性质：若 $x \sim \mathcal{N}(\mu, \Sigma)$，$A$ 是线性映射，则 $Ax \sim \mathcal{N}(A\mu, A\Sigma A^\top)$。这种封闭性使高斯分布在概率机器学习中具有解析可处理性。

---

### Q8 [进阶] 分析偏差-方差分解与双下降现象

**Q：** 经典的偏差-方差权衡与现代过参数化模型中观察到的双下降现象有何关联？

**A：** **偏差-方差分解**（bias-variance decomposition）将模型 $\hat{f}$ 在点 $x$ 处的期望测试误差表示为：

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f^*(x))^2}_{\text{偏差}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{方差}} + \sigma^2_\text{noise}$$

偏差度量模型假设带来的系统误差；方差度量对特定训练集的敏感程度。经典图像预测测试误差呈 U 形曲线：小模型欠拟合（高偏差、低方差），大模型过拟合（低偏差、高方差），最优复杂度使二者之和最小。

**双下降**（double descent）现象（Belkin et al., 2019）挑战了这一图像。在**插值阈值**——模型容量恰好等于训练样本数的节点——测试误差出现峰值，因为模型被迫记住每个训练点，但这种记忆缺乏结构。当容量继续增长超过该阈值后，测试误差再度下降，甚至可以达到或超越经典最优点。这第二次下降的发生，是因为高度过参数化的模型有大量精确拟合训练数据的解；梯度下降的**隐式偏置**（implicit bias）倾向于选择其中范数最小或最平滑的解，这些解具有良好的泛化性。

双下降现象已在线性回归、核方法、随机森林和深度神经网络中被观测到（Belkin et al., 2019）。这意味着，在过参数化体制下，经典的偏差-方差框架是不完整的；当容量充足时，通过早停或显式权重惩罚进行正则化可能是不必要的，甚至有害的。该现象还取决于插值算法的选择：最小范数最小二乘表现出干净的双下降，而其他插值算法则未必。

---

### Q9 [进阶] 解释集中不等式及其在泛化理论中的作用

**Q：** 什么是集中不等式？它们如何用于推导泛化误差界？

**A：** **集中不等式**（concentration inequalities）量化随机变量在其均值附近的集中程度。**Markov 不等式**表明，对任意非负 $X$：$P(X \geq a) \leq \mathbb{E}[X] / a$——这是一个仅需有限期望的弱界。**Chebyshev 不等式**利用方差将其收紧：$P(|X - \mu| \geq k\sigma) \leq 1/k^2$，但尾部仍是多项式衰减。**Hoeffding 不等式**对有界独立随机变量 $X_1, \ldots, X_n \in [a_i, b_i]$ 给出强得多的结论：

$$P\!\left(\left|\bar{X} - \mathbb{E}[\bar{X}]\right| \geq t\right) \leq 2\exp\!\left(\frac{-2n^2 t^2}{\sum_i (b_i - a_i)^2}\right)$$

尾部以 $n$ 的指数速度衰减，使 Hoeffding 界对多样本均值远比 Chebyshev 更紧。

泛化理论将这些工具用于界定训练误差与测试误差之间的差距。**VC 维**界表明，VC 维为 $d$ 的假设类的泛化差距以 $O(\sqrt{d/n})$ 的速度缩放。**Rademacher 复杂度**通过衡量假设类对随机噪声的拟合能力给出更紧的实例相关界。**PAC-Bayes 界**用 $\mathrm{KL}(Q \| P)$（$Q$ 是已学模型的后验，$P$ 是先验）来表达泛化差距，为贝叶斯方法和权重先验的正则化解释提供了动机。这些界对实际神经网络而言很少足够紧，但它们定性地解释了为何更多数据、更简单模型或强先验能改善泛化。

---

### Q10 [进阶] 描述权重先验如何将贝叶斯推断与正则化联系起来

**Q：** MAP 估计框架如何将正则化与概率建模统一起来？贝叶斯视角如何超越惩罚优化？

**A：** **MAP 估计**（最大后验估计）最大化后验 $p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta)\, p(\theta)$。取负对数后：

$$\hat\theta_\text{MAP} = \arg\min_\theta \left[-\sum_i \log p(y_i | x_i, \theta) - \log p(\theta)\right]$$

第一项是负对数似然（标准损失），第二项 $-\log p(\theta)$ 充当由先验决定的正则化项。**各向同性高斯先验**（isotropic Gaussian prior）$p(\theta) = \mathcal{N}(0, \tau^2 I)$ 给出 $-\log p(\theta) = \|\theta\|_2^2 / 2\tau^2 + \text{const}$，这正是强度为 $\lambda = 1/(2\tau^2)$ 的 **L2 正则化**（权重衰减）。**拉普拉斯先验** $p(\theta) \propto \exp(-|\theta|/b)$ 给出 $-\log p(\theta) \propto \|\theta\|_1$，即 **L1 正则化**，通过将许多权重精确置零来诱导稀疏性。

超越 MAP，完整的**贝叶斯推断**维护参数上的后验分布而非点估计，对所有合理的 $\theta$ 加权平均预测：$p(y^* | x^*, \mathcal{D}) = \int p(y^* | x^*, \theta)\, p(\theta | \mathcal{D})\, d\theta$。这能产生校准更好的不确定性估计，但对神经网络而言计算上不可处理。近似推断方法应对这一挑战：测试时的 **Dropout** 对应在特定近似后验下进行蒙特卡洛采样（Gal & Ghahramani, 2016），将一种简单的正则化技术与有原则的概率解释联系起来。**拉普拉斯近似**利用观测到的 Hessian 在 MAP 估计附近拟合一个高斯后验，从而高效地计算后验预测分布。

---

## 微积分与优化

### Q11 [基础] 解释反向传播如何在计算图上实现链式法则

**Q：** 反向传播实际上计算什么？计算图的结构如何使其高效？

**A：** **反向传播**（backpropagation）是应用于神经网络计算图的逆模式自动微分。它通过应用链式法则计算标量损失 $L$ 关于所有参数的梯度：对任意中间变量 $z$ 及其上游后继 $y$，梯度满足 $\partial L / \partial z = (\partial y / \partial z)^\top (\partial L / \partial y)$，即**向量-雅可比积**（VJP）。

计算分两个阶段。**前向传播**（forward pass）按拓扑顺序求值每个操作，计算所有中间值并缓存反向传播所需的值。**反向传播**（backward pass）以逆序遍历计算图、累积梯度：每个节点接收上游梯度 $\partial L / \partial y$，计算 VJP $(\partial y / \partial z)^\top (\partial L / \partial y)$，并将结果向下游传递。$L$ 关于任意中间量的梯度是图中所有路径上 VJP 之和，无需显式构造任何雅可比矩阵。

效率源于复用：逆模式在单次反向传播中计算标量输出关于所有输入的梯度，代价与一次前向传播成正比（乘以一个小常数）。这与正向模式自动微分形成对比——正向模式每次传播一个输入方向的方向导数，在输入维度小而输出维度大时更为高效。由于深度学习的损失是标量而参数数量庞大，逆模式被普遍采用。

---

### Q12 [基础] 描述梯度下降的收敛性及动量的改进效果

**Q：** 什么决定了梯度下降的收敛速度？为什么动量能加速训练？

**A：** 对于步长 $\eta = 1/L$ 下的梯度下降 $\theta \leftarrow \theta - \eta \nabla L(\theta)$，应用于 $L$-光滑（Lipschitz 梯度）凸损失时，收敛速度保证为 $O(1/T)$。对于 $\mu$-强凸损失——具有唯一最小值和曲率下界——收敛是**线性**（几何）的：$L(\theta_T) - L(\theta^*) \leq (1 - \mu/L)^T [L(\theta_0) - L(\theta^*)]$。条件数 $\kappa = L/\mu$ 决定收敛速率：$\kappa$ 大意味着收敛慢——在距最小值较远处（某些方向损失平坦）梯度大，在接近最小值时（某些方向损失急剧弯曲）梯度小。

**动量**（momentum；Polyak, 1964）维护一个速度向量 $v_t$，更新方式为 $v_t = \beta v_{t-1} + \nabla L(\theta_{t-1})$，$\theta_t = \theta_{t-1} - \eta v_t$。通过累积历史梯度，动量在梯度符号频繁翻转的方向上（狭窄山谷）抑制震荡，在梯度符号一致的方向上放大步伐。**Nesterov 加速梯度**通过在超前点 $\theta - \beta v$ 处计算梯度，对光滑凸函数实现 $O(1/T^2)$ 的最优收敛速率。**Adam** 将动量（梯度的一阶矩估计）与**自适应学习率**（梯度平方的二阶矩估计）相结合：$\theta \leftarrow \theta - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$，对每个参数的学习率按梯度均方根的倒数进行缩放，使其对病态损失景观具有鲁棒性。

---

### Q13 [进阶] 分析深度网络的损失景观与鞍点的作用

**Q：** 深度网络的非凸损失景观与凸损失有何不同？这对一阶优化方法意味着什么？

**A：** 深度神经网络的损失景观高度**非凸**（non-convex）：存在指数级多的临界点 $\nabla L = 0$。临界点根据 Hessian 的特征值分类：**局部最小值**的所有特征值均为正；**局部最大值**的所有特征值均为负；**鞍点**（saddle point）的特征值有正有负。对于线性网络，所有局部最小值均可证明是全局最小值；但对于深度非线性网络，结构更为丰富。

Dauphin et al.（2014）借鉴随机矩阵理论的结论指出：在高维网络中，大多数临界点是鞍点而非局部最小值——关键在于，鞍点处的损失值通常接近全局最小损失。核心观察是：在高维空间中，Hessian 所有特征值同时为正的概率指数级小——大多数退化临界点是近平坦的（负曲率为零或接近零），而非深陷其中的局部陷阱。受小批量噪声扰动的梯度下降可在多项式时间内逃离严格鞍点（Jin et al., 2017），使一阶随机方法在实践中行之有效。

过参数化网络还呈现进一步的结构特征：在插值处（训练损失为零），全局最小值集合构成高维流形。SGD 的噪声引入了朝向该流形较平坦区域的**隐式偏置**，对应泛化能力更好的解。锐度感知最小化（SAM）通过在计算梯度更新前将权重在损失增加最大的方向上扰动，显式地寻求平坦最小值，代价是每步需要两倍的梯度评估。

---

### Q14 [进阶] 解释二阶优化及其在深度学习中鲜少使用的原因

**Q：** 牛顿法相比梯度下降有何优势？什么阻碍了它在深度网络中的实用化？

**A：** **牛顿法**（Newton's method）将参数更新为 $\theta \leftarrow \theta - H^{-1} \nabla L$，其中 $H = \nabla^2 L$ 是 Hessian 矩阵。对于强凸光滑函数，牛顿法在解附近实现**二次收敛**：每次迭代正确数字位数翻倍。它还具有**尺度不变性**——自动考虑不同参数维度的曲率差异，使最优点附近的有效条件数为 1。对于二次损失，一步即可收敛。

实际障碍十分严重。存储 $H \in \mathbb{R}^{p \times p}$ 需要 $O(p^2)$ 内存，求逆需要 $O(p^3)$ 计算量——对于 $p \gtrsim 10^6$ 的参数均不可处理。**拟牛顿方法**（BFGS、L-BFGS）利用历史梯度差的秩 2 更新近似 $H^{-1}$，将内存降至存储 $m$ 个梯度对的 $O(mp)$，广泛用于中小规模网络或最终微调阶段。

**自然梯度**（natural gradient）用**Fisher 信息矩阵** $F = \mathbb{E}[\nabla \log p_\theta \nabla \log p_\theta^\top]$ 取代欧氏度量，给出对模型重参数化不变的梯度。这在理论上很有吸引力——Fisher 度量捕捉了模型族的内禀曲率——但 $F$ 与 Hessian 有同样的 $O(p^2)$ 存储问题。**K-FAC**（Martens & Grosse, 2015）将 $F$ 近似为每层的 Kronecker 积 $F \approx A \otimes G$，其中 $A$ 和 $G$ 是激活值和预激活梯度的协方差矩阵，计算代价较低。这将每层的求逆代价降至 $O(n^3 + m^3)$，并在图像分类和强化学习中实现了实际加速，但实现较为复杂，且每步代价仍远高于 Adam。

---

### Q15 [进阶] 分析梯度消失与梯度爆炸及架构设计的解决方案

**Q：** 深度网络中梯度消失与梯度爆炸的原因是什么？哪些机制被证明最为有效？

**A：** 在深度为 $L$ 的网络中，损失关于第 1 层参数的梯度涉及沿整条计算路径的雅可比矩阵乘积：

$$\frac{\partial L}{\partial \theta^{(1)}} = \frac{\partial L}{\partial z^{(L)}} \prod_{\ell=2}^{L} \frac{\partial z^{(\ell)}}{\partial z^{(\ell-1)}} \cdot \frac{\partial z^{(1)}}{\partial \theta^{(1)}}$$

若每层的谱半径 $\rho\!\left(\partial z^{(\ell)} / \partial z^{(\ell-1)}\right) < 1$，$L-1$ 个雅可比矩阵的乘积将**指数级衰减**——即梯度消失（vanishing gradients）。若 $\rho > 1$，梯度则指数级增长——即梯度爆炸（exploding gradients）。两种情形都使早期层的训练实际上不可能，这是 2010 年代中期之前的主要障碍。

四种互补的解决方案被证明有效。**精心初始化**（Glorot & Bengio, 2010）设置 $\text{Var}(W) = 2/(\text{fan\_in} + \text{fan\_out})$，使信号方差通过线性层保持不变（Xavier/Glorot 初始化）。He et al.（2015）针对 ReLU 非线性对此进行了修正，设置 $\text{Var}(W) = 2/\text{fan\_in}$（Kaiming/He 初始化），确保初始化时激活值的期望幅度保持稳定。**批归一化**（Ioffe & Szegedy, 2015）在每个小批量内将预激活值归一化为零均值和单位方差，防止 sigmoid/tanh 激活饱和，使梯度幅度对网络深度相对不敏感。**残差连接**（He et al., 2016）将每个块改写为 $x^{(\ell+1)} = x^{(\ell)} + F(x^{(\ell)}, \theta^{(\ell)})$，使雅可比矩阵 $\partial x^{(\ell+1)} / \partial x^{(\ell)} = I + \partial F / \partial x^{(\ell)}$ 始终包含单位矩阵，无论网络多深都提供直接的梯度路径。**梯度截断**——当 $\|\nabla L\| > c$ 时对梯度重新缩放——处理循环网络中的梯度爆炸，因为循环网络的有效深度等于序列长度。

---

## 信息论

### Q16 [基础] 解释熵与交叉熵及其在分类训练中的应用

**Q：** 香农熵和交叉熵分别度量什么？为什么交叉熵是分类任务的自然损失？

**A：** **香农熵**（Shannon entropy）$H(p) = -\sum_x p(x) \log p(x) = \mathbb{E}_p[-\log p(X)]$ 度量分布 $p$ 的平均不确定性，或等价地，对 $p$ 的一次采样进行编码所需的最小期望比特数。$C$ 类上的均匀分布具有最大熵 $\log C$；将所有概率质量集中于一个结果的退化分布的熵为 $0$。

**交叉熵** $H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_p[-\log q(X)]$ 度量：用针对 $q$ 优化的编码来对 $p$ 的样本进行编码时的期望编码长度。它始终不小于 $H(p)$，当且仅当 $p = q$ 时取等。在多类分类中，真实标签分布 $p$ 是集中于类别 $y$ 的独热向量，模型输出 $q$ 是 softmax 概率向量。交叉熵损失简化为 $-\log q_y$，即模型赋予正确类别的对数概率的负值。

最小化交叉熵等价于 MLE：最大化 $\log q_y$ 使真实标签在模型下的似然最大。更深层的关系是 $H(p, q) = H(p) + \mathrm{KL}(p \| q)$：由于独热 $p$ 的 $H(p) = 0$，对 $q$ 最小化交叉熵等同于最小化 $\mathrm{KL}(p \| q)$，即使模型分布尽量接近经验标签分布（前向 KL 意义）。这一三重等价——MLE、交叉熵最小化、KL 最小化——是监督分类的数学基础。

---

### Q17 [基础] 描述 KL 散度、其非对称性及前向与反向 KL 的区别

**Q：** KL 散度度量什么？散度的方向为何在变分推断中至关重要？

**A：** 从分布 $q$ 到 $p$ 的 **KL 散度**（Kullback–Leibler divergence）为：

$$\mathrm{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\!\left[\log \frac{p}{q}\right]$$

它始终非负（Gibbs 不等式），当且仅当 $p = q$ 几乎处处成立时为零，且不对称：$\mathrm{KL}(p \| q) \neq \mathrm{KL}(q \| p)$。KL 散度不是度量，但它是一种 $f$-散度，在信息论和机器学习中起核心作用。

非对称性具有重要的行为后果。**前向 KL**（forward KL）$\mathrm{KL}(p \| q)$ 在 $p$ 下求期望：凡 $p(x) > 0$ 而 $q(x) = 0$ 之处，项 $p(x) \log(p(x)/q(x)) = +\infty$。因此，最小化前向 KL 迫使 $q$ 在 $p$ 有支撑的地方都有支撑——$q$ 是**覆盖型**（mass-covering，也称包含型）。当真实分布 $p$ 有多个模态时，这是合适的：$q$ 必须在所有模态上分散概率质量。

**反向 KL**（reverse KL）$\mathrm{KL}(q \| p)$ 在 $q$ 下求期望：凡 $q(x) > 0$ 而 $p(x) \approx 0$ 之处，项 $q(x) \log(q(x)/p(x))$ 很大。最小化反向 KL 迫使 $q$ 只在 $p$ 较大处集中——$q$ 是**寻模型**（mode-seeking，排他型）。在变分推断中，ELBO 目标对应最小化 $\mathrm{KL}(q_\phi(z|x) \| p_\theta(z|x))$（反向 KL），因为在变分分布 $q_\phi$ 下求期望是可处理的，而在真实后验 $p_\theta$ 下求期望则不然。由此产生的近似后验倾向于覆盖不足真实后验的多个模态，这是均场变分推断的一个已知局限。

---

### Q18 [进阶] 解释互信息及其在表示学习中的估计与最大化

**Q：** 互信息捕获了相关系数无法描述的什么信息？近期方法如何使其在自监督学习中切实可用？

**A：** **互信息**（mutual information）$I(X; Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y) = \mathrm{KL}(p(x,y) \| p(x)p(y))$ 度量已知 $Y$ 后关于 $X$ 不确定性的减少量，或等价地，两个随机变量之间共享的信息量。与 Pearson 相关系数不同，互信息能捕获任意统计依赖关系——包括非线性的——当且仅当 $X$ 与 $Y$ 独立时等于零。它是对称的且始终非负。

直接计算互信息需要联合密度 $p(x,y)$，在高维情形下不可处理。**MINE**（Belghazi et al., 2018）通过 **Donsker–Varadhan** 变分表示估计互信息：

$$I(X; Y) \geq \mathbb{E}_{p(x,y)}[T_\psi(x,y)] - \log\,\mathbb{E}_{p(x)p(y)}\!\left[e^{T_\psi(x,y)}\right]$$

训练神经网络 $T_\psi$ 最大化该下界，从而提供可微的互信息估计量。**InfoNCE / CPC**（Oord et al., 2018）利用噪声对比估计（NCE）提供另一种下界：给定一个正样本对 $(x, y)$ 和 $K$ 个负样本对 $(x, y_j^-)$，InfoNCE 目标为：

$$\mathcal{L}_\text{InfoNCE} = -\mathbb{E}\!\left[\log \frac{e^{f(x,y)}}{e^{f(x,y)} + \sum_{j=1}^K e^{f(x,y_j^-)}}\right]$$

它是 $-I(X; Y) + \log(K+1)$ 的上界。最大化 InfoNCE 即提升互信息的一个下界。这一原理是对比自监督方法（SimCLR、MoCo）的基础：通过最大化同一图像不同增强视角表示之间的互信息，模型学到对增强方式不变且能预测视角身份的特征——这些特征对下游任务有良好的迁移性。

---

### Q19 [进阶] 描述信息瓶颈原理及其与深度学习的联系

**Q：** 信息瓶颈最小化什么目标？它对深度网络应如何编码信息有何预测？

**A：** **信息瓶颈**（information bottleneck，IB；Tishby et al., 2000）形式化了寻找输入 $X$ 的紧凑表示 $Z$ 的目标——使 $Z$ 关于目标 $Y$ 的信息量最大。马尔可夫约束 $Y \to X \to Z$ 要求 $Z$ 由 $X$ 计算得到。IB 拉格朗日量为：

$$\max_{p(z|x)}\; I(Z; Y) - \beta\, I(Z; X)$$

当 $\beta = 0$ 时，解为 $Z = X$（无压缩）。当 $\beta \to \infty$ 时，解退化为常数（完全压缩，不保留任何相关信息）。**IB 曲线**——信息平面 $(I(Z;X),\, I(Z;Y))$ 上的 Pareto 前沿——代表表示复杂度与任务相关性之间的最优权衡。

Tishby & Schwartz-Ziv（2017）从这一视角解读深度网络的训练，声称网络先拟合训练标签（增大 $I(Z;Y)$），再经历**压缩阶段**（通过 SGD 中类似扩散的动力学减小 $I(Z;X)$）。这一说法引发了广泛关注，但随后受到质疑：表面上的压缩强烈依赖于激活函数和互信息估计量的分箱超参数。对于使用线性或 ReLU 激活的网络，压缩并未被一致观测到。IB 解释仍是一个实证支持有争议的活跃研究方向。

尽管如此，IB 框架提供了有用的概念词汇，并与已有模型相联系：$\beta$-VAE 优化了一个相关目标，其中 ELBO 重建项扮演 $I(Z;Y)$ 的角色，KL 惩罚项扮演 $I(Z;X)$ 的角色。IB 目标下的最优表示是由 $X$ 预测 $Y$ 的**充分统计量**（sufficient statistics），使 IB 成为表示学习系统应追求目标的形式化判据。

---

### Q20 [进阶] 推导 MLE、交叉熵最小化与 KL 散度最小化的等价性

**Q：** 为什么最小化交叉熵、最大化似然和最小化 KL 散度最终归结为同一个优化问题？

**A：** 设 $p_\text{data}$ 为真实数据分布，$p_\theta$ 为模型。基于 $n$ 个独立同分布样本 $\{x_i\}$ 的**经验 MLE** 目标为：

$$\hat\theta_\text{MLE} = \arg\max_\theta \frac{1}{n}\sum_{i=1}^n \log p_\theta(x_i)$$

由大数定律，当 $n \to \infty$ 时，此式收敛到 $\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$，因此渐近地：

$$\hat\theta_\text{MLE} \approx \arg\max_\theta\; \mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$$

从模型到数据的 **KL 散度**为：

$$\mathrm{KL}(p_\text{data} \| p_\theta) = \mathbb{E}_{p_\text{data}}\!\left[\log p_\text{data}(x)\right] - \mathbb{E}_{p_\text{data}}\!\left[\log p_\theta(x)\right]$$

由于 $\mathbb{E}_{p_\text{data}}[\log p_\text{data}(x)] = -H(p_\text{data})$ 关于 $\theta$ 是常数：

$$\arg\min_\theta\; \mathrm{KL}(p_\text{data} \| p_\theta) = \arg\max_\theta\; \mathbb{E}_{p_\text{data}}[\log p_\theta(x)] = \hat\theta_\text{MLE}$$

**交叉熵** $H(p_\text{data}, p_\theta) = -\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$ 恰好是 MLE 目标的负值，因此最小化交叉熵与 MLE 完全等价。综合以上：$H(p_\text{data}, p_\theta) = H(p_\text{data}) + \mathrm{KL}(p_\text{data} \| p_\theta)$，由于 $H(p_\text{data})$ 不依赖于 $\theta$，最小化交叉熵即最小化 KL 散度。

其实际含义深远。任何用交叉熵损失训练的神经网络都在隐式地拟合统计模型 $p_\theta$，以最小化与数据生成过程的散度——架构和参数化的选择决定了搜索的分布族 $\{p_\theta\}$。将交叉熵替换为不同的散度（例如反向 KL、$f$-散度）会产生不同的拟合行为：反向 KL 产生寻模型拟合，前向 KL（交叉熵）产生覆盖型拟合。这一统一也解释了为什么 softmax 分类器、语言模型的下一词元预测头和 VAE 解码器尽管解决表面上不同的任务，却共享相同的交叉熵训练目标。

---

## 快速参考

| # | 难度 | 主题 | 章节 |
|---|------|------|------|
| Q1 | 基础 | 特征值与特征向量的几何含义 | 线性代数 |
| Q2 | 基础 | SVD 及其与特征分解的关系 | 线性代数 |
| Q3 | 进阶 | 通过 SVD 实现 PCA 与最优低秩近似 | 线性代数 |
| Q4 | 进阶 | 矩阵条件数与训练稳定性 | 线性代数 |
| Q5 | 进阶 | 矩阵微积分、雅可比矩阵与 VJP | 线性代数 |
| Q6 | 基础 | 通过 MLE 推导损失函数 | 概率与统计 |
| Q7 | 基础 | 高斯分布在深度学习中的普遍性 | 概率与统计 |
| Q8 | 进阶 | 偏差-方差分解与双下降现象 | 概率与统计 |
| Q9 | 进阶 | 集中不等式与泛化界 | 概率与统计 |
| Q10 | 进阶 | 权重先验与贝叶斯正则化 | 概率与统计 |
| Q11 | 基础 | 链式法则与计算图上的反向传播 | 微积分与优化 |
| Q12 | 基础 | 梯度下降收敛性与动量 | 微积分与优化 |
| Q13 | 进阶 | 非凸损失景观与鞍点 | 微积分与优化 |
| Q14 | 进阶 | 二阶优化与 K-FAC | 微积分与优化 |
| Q15 | 进阶 | 梯度消失与梯度爆炸：成因与解决方案 | 微积分与优化 |
| Q16 | 基础 | 香农熵与分类中的交叉熵 | 信息论 |
| Q17 | 基础 | KL 散度及其非对称性 | 信息论 |
| Q18 | 进阶 | 互信息估计与对比学习 | 信息论 |
| Q19 | 进阶 | 信息瓶颈原理 | 信息论 |
| Q20 | 进阶 | MLE、交叉熵与 KL 最小化的等价性 | 信息论 |

## 参考文献

- Glorot & Bengio, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (2010)
- He et al., [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) (2015)
- Ioffe & Szegedy, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (2015)
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2016)
- Belkin et al., [Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Off](https://arxiv.org/abs/1812.11118) (2019)
- Gal & Ghahramani, [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) (2016)
- Dauphin et al., [Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization](https://arxiv.org/abs/1406.2572) (2014)
- Jin et al., [How to Escape Saddle Points Efficiently](https://arxiv.org/abs/1703.00887) (2017)
- Martens & Grosse, [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671) (2015)
- Tishby et al., [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057) (2000)
- Tishby & Schwartz-Ziv, [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810) (2017)
- Belghazi et al., [Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062) (2018)
- Oord et al., [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (2018)
