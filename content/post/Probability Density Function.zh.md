---
title: "概率密度函数：技术入门"
author: yo3nglau
date: '2026-05-15'
categories:
  - Deep Learning
tags:
  - Probability Theory
  - Generative Models
  - Mathematics
toc: true
---

## 引言

**概率密度函数**（probability density function，PDF）是概率在连续随机变量上分布的基本数学对象。概率质量函数为每个离散结果赋予确定的概率，而 PDF 则刻画了连续样本空间中每一点处的无穷小概率率：随机变量落在某区间内的概率，等于 PDF 在该区间上的积分。这一从求和到积分的转变并非仅是符号上的变化——它反映了离散概率与连续概率之间更深层的结构性差异，对模型的训练方式、数据分布的表示方式以及样本的生成方式都有深远影响。

PDF 的概念奠定了现代统计推断与机器学习的基础。每种参数化概率模型——高斯分布、指数分布、狄利克雷分布——都由其 PDF 定义，而通过最大似然估计来训练这类模型，本质上就是将 PDF 拟合到观测数据。超越经典统计学，PDF 在生成模型中以性质各异的角色出现：作为被参数化并加以变换的显式对象（归一化流），作为通过学习其梯度来建模的分布（基于分数的模型），或作为从未被写出的隐式目标（生成对抗网络）。

本文从研究生层次出发，系统阐述 PDF 的定义与关键性质，建立关于密度与概率之区别的直觉认识，并考察四个 AI 应用场景中 PDF 所扮演的结构性不同角色。

## 数学基础

设 $X$ 为取值于 $\mathbb{R}$ 的连续随机变量。若存在函数 $f_X: \mathbb{R} \to \mathbb{R}_{\geq 0}$，使得对任意满足 $a \leq b$ 的区间 $[a, b]$，$X$ 落在该区间内的概率为

$$P(a \leq X \leq b) = \int_a^b f_X(x)\, dx$$

则称 $f_X$ 为 $X$ 的**概率密度函数**。任意有效 PDF 须满足两个条件：非负性，即对所有 $x \in \mathbb{R}$ 有 $f_X(x) \geq 0$；以及归一性，即 $\int_{-\infty}^{\infty} f_X(x)\, dx = 1$。PDF 与**累积分布函数**（cumulative distribution function，CDF）$F_X(x) = P(X \leq x)$ 之间的关系由微积分基本定理给出：在 $F_X$ 可微处，$f_X(x) = F_X'(x)$。

高斯 PDF 是机器学习中最常用的参数族。对于均值为 $\mu \in \mathbb{R}$、方差为 $\sigma^2 > 0$ 的随机变量 $X$（记为 $X \sim \mathcal{N}(\mu, \sigma^2)$），其 PDF 为

$$f_X(x;\, \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

对于均值为 $\boldsymbol{\mu} \in \mathbb{R}^d$、正定协方差矩阵为 $\Sigma \in \mathbb{R}^{d \times d}$ 的 $d$ 维随机向量 $\mathbf{X}$（记为 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$），多元 PDF 为

$$f_\mathbf{X}(\mathbf{x};\, \boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

其中 $|\Sigma|$ 表示 $\Sigma$ 的行列式，$\Sigma^{-1}$ 为其逆矩阵。

生成模型的核心数学工具是**变量替换公式**（change-of-variables formula）。设 $\mathbf{X}$ 为具有 PDF $f_\mathbf{X}$ 的 $d$ 维随机向量，$g: \mathbb{R}^d \to \mathbb{R}^d$ 为可逆的可微映射。变换后的变量 $\mathbf{Y} = g(\mathbf{X})$ 的 PDF 为

$$f_\mathbf{Y}(\mathbf{y}) = f_\mathbf{X}(g^{-1}(\mathbf{y})) \cdot \left|\det J_{g^{-1}}(\mathbf{y})\right|$$

其中 $g^{-1}$ 是 $g$ 的逆映射，$J_{g^{-1}}(\mathbf{y})$ 是 $g^{-1}$ 在 $\mathbf{y}$ 处的 $d \times d$ 雅可比矩阵（其第 $(i,j)$ 项为 $\partial [g^{-1}]_i / \partial y_j$），$|\det \cdot|$ 表示行列式的绝对值。雅可比因子反映了 $g$ 对局部体积的拉伸或压缩程度：若映射膨胀了某区域，密度必须相应减小，以保持总概率质量为 1。

## 核心直觉

理解 PDF 的最佳方式是类比物理学中的质量密度。一根密度不均匀的细杆，其线密度为 $\rho(x)$（单位：kg/m），则线段 $[a, b]$ 内的总质量为 $\int_a^b \rho(x)\, dx$。密度 $\rho(x)$ 本身并非质量——它是一个率，表示在 $x$ 附近每单位长度所积累的质量。概率密度的运作方式完全相同：$f_X(x)$ 表示在 $x$ 附近每单位 $x$ 所积累的概率。要得到真正的概率，必须在具有正宽度的区间上积分。

密度解释最重要的推论是 PDF 的值可以超过 1。对于 $X \sim \mathcal{N}(0, \sigma^2)$，取 $\sigma = 0.1$，峰值为 $f_X(0) = \frac{1}{0.1 \cdot \sqrt{2\pi}} \approx 3.99$，但这不违反任何概率公理：该分布极度集中，因此密度相应很高，而对整个 $\mathbb{R}$ 的积分仍恰好为 1。值 $f_X(x_0) = 3.99$ 传达的信息是：在 $x_0$ 附近，概率以约每单位 $x$ 3.99 的速率积累——而非 $x_0$ 处的概率为 3.99 甚至 0.399。对于连续随机变量，单点概率始终为零：$P(X = x_0) = \int_{x_0}^{x_0} f_X(x)\, dx = 0$。

一个更微妙的含义是：PDF 的值并非尺度不变的——它依赖于测量单位。若 $X$ 以米为单位，通过 $Y = 100X$ 转换为厘米，则变量替换公式给出 $f_Y(y) = f_X(y/100) \cdot \frac{1}{100}$，峰值缩小为原来的 $\frac{1}{100}$。这种对参数化的敏感性正是变量替换公式中雅可比修正项不可或缺的原因：若省略该项，在重参数化下概率质量将无法守恒，从而使所有后续似然计算失效。

## 在人工智能中的应用

### 最大似然估计

**最大似然估计**（maximum likelihood estimation，MLE）是参数化概率模型的基础训练方法。给定具有 PDF $f(\mathbf{x};\, \theta)$ 的模型，其中 $\theta$ 表示可学习参数，$\mathbf{x} \in \mathbb{R}^d$ 为单个数据点；并给定从真实数据分布中独立抽取的 $N$ 个观测样本 $\mathbf{x}_1, \ldots, \mathbf{x}_N$，MLE 寻求使数据联合似然最大的参数：

$$\hat{\theta}_\mathrm{MLE} = \arg\max_\theta \sum_{i=1}^N \log f(\mathbf{x}_i;\, \theta)$$

其中，对数密度之和（对数似然）代替密度之积，以保证数值稳定性。最大化对数似然等价于最小化经验数据分布与模型分布之间的 KL 散度，而后者又等价于最小化两者之间的交叉熵——由此确立了 MLE 作为分类任务中交叉熵损失以及语言建模中下一词元预测损失的理论基础。在 GPT-3（Brown et al., 2020）中，训练目标是最大化每个下一词元 $x_t$ 在模型条件 PDF $f(x_t \mid x_1, \ldots, x_{t-1};\, \theta)$ 下的对数似然，其中 $x_1, \ldots, x_{t-1}$ 为前序上下文。

### 归一化流

**归一化流**（normalizing flows）通过一系列可逆神经网络层对易处理的基础分布进行变换，从而构造灵活的高维 PDF。起点是基础随机向量 $\mathbf{z} \in \mathbb{R}^d$，其 PDF $f_\mathbf{Z}$ 通常取为 $\mathcal{N}(\mathbf{0}, I)$，其中 $I$ 为 $d \times d$ 单位矩阵。由 $\theta$ 参数化的可逆可微映射 $g_\theta: \mathbb{R}^d \to \mathbb{R}^d$ 将 $\mathbf{z}$ 推入数据空间，得到 $\mathbf{x} = g_\theta(\mathbf{z})$。应用变量替换公式，$\mathbf{x}$ 上的诱导 PDF 为

$$f_\mathbf{X}(\mathbf{x};\, \theta) = f_\mathbf{Z}(g_\theta^{-1}(\mathbf{x})) \cdot \left|\det J_{g_\theta^{-1}}(\mathbf{x})\right|$$

其中 $g_\theta^{-1}$ 为 $g_\theta$ 的逆映射，$J_{g_\theta^{-1}}(\mathbf{x})$ 为其在 $\mathbf{x}$ 处的雅可比矩阵。该 PDF 精确且可微，因此参数可通过对数似然 $\log f_\mathbf{Z}(g_\theta^{-1}(\mathbf{x})) + \log|\det J_{g_\theta^{-1}}(\mathbf{x})|$ 直接用 MLE 训练。计算瓶颈在于雅可比行列式，对一般 $d \times d$ 矩阵的计算代价为 $O(d^3)$。Rezende & Mohamed (2015) 在变分推断场景中引入了归一化流；Dinh et al. (2016) 提出了实值非体积保持（RealNVP）变换——其雅可比矩阵为下三角形式的仿射耦合层——将行列式计算降至 $O(d)$ 的对角元素乘积。

### 基于分数的生成模型

PDF $p(\mathbf{x})$ 的**分数函数**（score function）是其关于数据点的梯度：$\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x} \log p(\mathbf{x})$，其中 $\nabla_\mathbf{x}$ 表示关于 $\mathbf{x}$ 的梯度。对于具有归一化常数 $Z$ 的非归一化密度 $\tilde{p}(\mathbf{x}) = p(\mathbf{x}) / Z$，分数满足 $\nabla_\mathbf{x} \log p(\mathbf{x}) = \nabla_\mathbf{x} \log \tilde{p}(\mathbf{x})$，因为 $\nabla_\mathbf{x} \log Z = 0$。这意味着分数与归一化常数无关——当 $Z$ 难以处理时（高维分布通常如此），这是一个关键优势。Song & Ermon (2019) 使用**去噪分数匹配**（denoising score matching）训练神经网络 $\mathbf{s}_\theta(\mathbf{x})$ 近似 $\nabla_\mathbf{x} \log p(\mathbf{x})$：对数据点添加高斯噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$ 得到 $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$，网络被训练以恢复噪声方向 $-\boldsymbol{\epsilon}/\sigma^2$，即噪声条件 PDF $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$ 的分数。采样则通过 Langevin 动力学实现：从 $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, I)$ 出发，迭代

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\alpha}{2}\, \mathbf{s}_\theta(\mathbf{x}_t) + \sqrt{\alpha}\, \mathbf{z}_t$$

其中 $\alpha > 0$ 为步长，$\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, I)$ 为每步独立的高斯噪声。Ho et al. (2020) 将这一框架重新表述为**去噪扩散概率模型**（denoising diffusion probabilistic models，DDPM），在固定高斯噪声调度下建立了 DDPM 去噪目标与分数匹配之间的形式等价性。

### 生成对抗网络

**生成对抗网络**（generative adversarial networks，GAN）（Goodfellow et al., 2014）在从不计算 PDF 的情况下实现分布匹配。由 $\phi$ 参数化的生成器网络 $G_\phi: \mathbb{R}^k \to \mathbb{R}^d$ 将潜在噪声 $\mathbf{z} \in \mathbb{R}^k$（采样自 $p_\mathbf{z}(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I_k)$，其中 $k$ 为潜在维度，$I_k$ 为 $k \times k$ 单位矩阵）映射为 $d$ 维数据空间 $\mathbb{R}^d$ 中的合成样本 $G_\phi(\mathbf{z})$。生成器在 $\mathbb{R}^d$ 上诱导出分布 $p_{G_\phi}$——即 $p_\mathbf{z}$ 通过 $G_\phi$ 的推前测度——其 PDF 从未被显式计算。取而代之的是，由 $\psi$ 参数化的判别器网络 $D_\psi: \mathbb{R}^d \to [0, 1]$ 估计样本 $\mathbf{x} \in \mathbb{R}^d$ 来自真实数据 PDF $p_\mathrm{data}$ 而非 $p_{G_\phi}$ 的概率。两个网络在极小极大博弈中进行训练，$D_\psi$ 最大化而 $G_\phi$ 最小化以下目标：

$$\mathbb{E}_{\mathbf{x} \sim p_\mathrm{data}}[\log D_\psi(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_\mathbf{z}}[\log(1 - D_\psi(G_\phi(\mathbf{z})))]$$

Goodfellow 等人证明，在最优判别器 $D_\psi^*$ 处，该目标等于 $p_\mathrm{data}$ 与 $p_{G_\phi}$ 之间的 **Jensen-Shannon 散度**的两倍减去 $\log 4$，因此训练 $G_\phi$ 等价于最小化真实数据 PDF 与生成数据 PDF 之间的散度——而无需以闭合形式写出任何一个 PDF。

## 核心要点

概率密度函数是一个率而非概率，其值只有在对正宽度区间积分时才具有意义；这一密度与概率的本质区别，渗透到所有依赖于它的学习算法之中。变量替换公式通过雅可比行列式将随机向量的 PDF 与其可逆变换的 PDF 联系起来，是归一化流的数学引擎，并在雅可比矩阵具有结构性时使精确似然计算变得可行。分数函数——对数 PDF 的梯度——将分布的几何结构与归一化常数分离，使基于分数的扩散模型无需显式估计 PDF，即可学习复杂数据分布的形状。生成对抗网络更进一步，完全舍弃了任何显式 PDF，通过判别器介导的散度最小化实现分布匹配。这四种范式共同勾勒出与 PDF 互动的方式谱系：直接计算、变换、微分，或隐式匹配。

## 参考文献

- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (2020)
- Rezende & Mohamed, [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) (2015)
- Dinh et al., [Density estimation using Real-valued Non-Volume Preserving (Real NVP) transformations](https://arxiv.org/abs/1605.08803) (2016)
- Song & Ermon, [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) (2019)
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Goodfellow et al., [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (2014)
