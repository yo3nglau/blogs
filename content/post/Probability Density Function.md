---
title: "Probability Density Function: A Technical Introduction"
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

## Introduction

A **probability density function** (PDF) is the fundamental mathematical object through which probability is distributed over a continuous random variable. Where a probability mass function assigns discrete probabilities to individual outcomes, a PDF characterizes the infinitesimal probability rate at every point in a continuous sample space: the probability of the variable falling in any interval is obtained by integrating the PDF over that interval. This shift from summation to integration is not merely notational — it reflects a deeper structural difference between discrete and continuous probability, one that has far-reaching consequences for how models are trained, how data distributions are represented, and how samples are generated.

The concept of a PDF sits at the foundation of modern statistical inference and machine learning. Every parametric probability model — Gaussian, exponential, Dirichlet — is defined by its PDF, and training such models by maximum likelihood estimation is equivalent to fitting a PDF to observed data. Beyond classical statistics, PDFs appear in generative modeling in qualitatively different roles: as explicit objects to be parameterized and transformed (normalizing flows), as distributions whose gradients are learned (score-based models), or as implicit targets to be matched without ever being written down (generative adversarial networks).

This post develops the definition and key properties of PDFs at a graduate level, builds intuition for what density means as opposed to probability, and examines four settings in AI where PDFs play structurally distinct roles.

## Mathematical Foundations

A continuous random variable $X$ taking values in $\mathbb{R}$ has a **probability density function** $f_X: \mathbb{R} \to \mathbb{R}_{\geq 0}$ if, for any interval $[a, b]$ with $a \leq b$, the probability of $X$ falling in that interval is given by

$$P(a \leq X \leq b) = \int_a^b f_X(x)\, dx$$

Two conditions characterize any valid PDF: non-negativity, $f_X(x) \geq 0$ for all $x \in \mathbb{R}$, and normalization, $\int_{-\infty}^{\infty} f_X(x)\, dx = 1$. The PDF is related to the **cumulative distribution function** (CDF) $F_X(x) = P(X \leq x)$ by the fundamental theorem of calculus: $f_X(x) = F_X'(x)$ wherever $F_X$ is differentiable.

The Gaussian PDF is the most prevalent parametric family in machine learning. For a random variable $X$ with mean $\mu \in \mathbb{R}$ and variance $\sigma^2 > 0$, written $X \sim \mathcal{N}(\mu, \sigma^2)$, the PDF is

$$f_X(x;\, \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

For a $d$-dimensional random vector $\mathbf{X}$ with mean $\boldsymbol{\mu} \in \mathbb{R}^d$ and positive definite covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$, written $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, the multivariate PDF is

$$f_\mathbf{X}(\mathbf{x};\, \boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where $|\Sigma|$ denotes the determinant of $\Sigma$ and $\Sigma^{-1}$ is its inverse.

A result central to generative modeling is the **change-of-variables formula**. Let $\mathbf{X}$ be a $d$-dimensional random vector with PDF $f_\mathbf{X}$, and let $g: \mathbb{R}^d \to \mathbb{R}^d$ be an invertible, differentiable map. The transformed variable $\mathbf{Y} = g(\mathbf{X})$ has PDF

$$f_\mathbf{Y}(\mathbf{y}) = f_\mathbf{X}(g^{-1}(\mathbf{y})) \cdot \left|\det J_{g^{-1}}(\mathbf{y})\right|$$

where $g^{-1}$ is the inverse of $g$, $J_{g^{-1}}(\mathbf{y})$ is the $d \times d$ Jacobian matrix of $g^{-1}$ evaluated at $\mathbf{y}$ (whose $(i,j)$-th entry is $\partial [g^{-1}]_i / \partial y_j$), and $|\det \cdot|$ denotes the absolute value of the determinant. The Jacobian factor accounts for how $g$ locally stretches or compresses volume: if the map expands a region, the density must shrink proportionally to preserve the total probability mass of 1.

## Core Intuition

The right way to think about a PDF is by analogy with mass density in physics. A thin rod with non-uniform linear density $\rho(x)$ measured in kg/m has total mass $\int_a^b \rho(x)\, dx$ in the segment $[a, b]$. The density $\rho(x)$ is not itself a mass — it is a rate, expressing how much mass accumulates per unit of length near $x$. Probability density works identically: $f_X(x)$ expresses how much probability accumulates per unit of $x$ near the point $x$. To obtain an actual probability, you must integrate over an interval of positive width.

The most consequential implication of this density interpretation is that PDF values can exceed 1. For $X \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.1$, the peak value is $f_X(0) = \frac{1}{0.1 \cdot \sqrt{2\pi}} \approx 3.99$, yet no axiom of probability is violated: the distribution is simply very concentrated, so the density is correspondingly high, while the integral over all of $\mathbb{R}$ remains exactly 1. A value $f_X(x_0) = 3.99$ conveys that probability is accumulating at a rate of roughly 3.99 per unit of $x$ near $x_0$ — not that the probability of $x_0$ is 3.99 or even 0.399. The point probability $P(X = x_0) = \int_{x_0}^{x_0} f_X(x)\, dx = 0$ for any single point of a continuous random variable.

A subtler implication is that PDF values are not scale-invariant: they depend on the units of measurement. If $X$ is measured in meters and rescaled to centimeters by $Y = 100X$, the change-of-variables formula gives $f_Y(y) = f_X(y/100) \cdot \frac{1}{100}$, shrinking the peak by a factor of 100. This sensitivity to parameterization is why the Jacobian correction in the change-of-variables formula is not optional: omitting it would fail to conserve total probability mass under reparameterization, invalidating any downstream likelihood computation.

## Applications in AI

### Maximum Likelihood Estimation

**Maximum likelihood estimation** (MLE) is the foundational training procedure for parametric probabilistic models. Given a model with PDF $f(\mathbf{x};\, \theta)$, where $\theta$ denotes the learnable parameters and $\mathbf{x} \in \mathbb{R}^d$ is a single data point, and given $N$ observations $\mathbf{x}_1, \ldots, \mathbf{x}_N$ drawn independently from the true data distribution, MLE seeks the parameters that maximize the joint likelihood of the data:

$$\hat{\theta}_\mathrm{MLE} = \arg\max_\theta \sum_{i=1}^N \log f(\mathbf{x}_i;\, \theta)$$

where the sum of log-densities replaces the product of densities for numerical stability. Maximizing the log-likelihood is equivalent to minimizing the KL divergence from the empirical data distribution to the model distribution, which in turn equals minimizing the cross-entropy between the two — establishing MLE as the theoretical basis for cross-entropy loss in classification and next-token prediction in language modeling. In GPT-3 (Brown et al., 2020), training maximizes the log-likelihood of each next token $x_t$ under the model's conditional PDF $f(x_t \mid x_1, \ldots, x_{t-1};\, \theta)$, where $x_1, \ldots, x_{t-1}$ is the preceding context.

### Normalizing Flows

**Normalizing flows** construct flexible, high-dimensional PDFs by transforming a tractable base distribution through a sequence of invertible neural network layers. The starting point is a base random vector $\mathbf{z} \in \mathbb{R}^d$ with a simple PDF $f_\mathbf{Z}$ — typically $\mathcal{N}(\mathbf{0}, I)$ where $I$ is the $d \times d$ identity matrix. An invertible differentiable map $g_\theta: \mathbb{R}^d \to \mathbb{R}^d$, parameterized by $\theta$, pushes $\mathbf{z}$ forward to the data space, yielding $\mathbf{x} = g_\theta(\mathbf{z})$. Applying the change-of-variables formula, the induced PDF over $\mathbf{x}$ is

$$f_\mathbf{X}(\mathbf{x};\, \theta) = f_\mathbf{Z}(g_\theta^{-1}(\mathbf{x})) \cdot \left|\det J_{g_\theta^{-1}}(\mathbf{x})\right|$$

where $g_\theta^{-1}$ is the inverse of $g_\theta$ and $J_{g_\theta^{-1}}(\mathbf{x})$ is the Jacobian of $g_\theta^{-1}$ evaluated at $\mathbf{x}$. This PDF is exact and differentiable, so parameters can be trained directly by MLE on the log-likelihood $\log f_\mathbf{Z}(g_\theta^{-1}(\mathbf{x})) + \log|\det J_{g_\theta^{-1}}(\mathbf{x})|$. The computational bottleneck is the Jacobian determinant, which costs $O(d^3)$ for a general $d \times d$ matrix. Rezende & Mohamed (2015) introduced normalizing flows in the variational inference setting; Dinh et al. (2016) proposed Real-valued Non-Volume Preserving (RealNVP) transformations — affine coupling layers whose Jacobians are lower-triangular — reducing the determinant computation to an $O(d)$ product of diagonal entries.

### Score-Based Generative Models

The **score function** of a PDF $p(\mathbf{x})$ is its gradient with respect to the data point: $\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x} \log p(\mathbf{x})$, where $\nabla_\mathbf{x}$ denotes the gradient with respect to $\mathbf{x}$. For an unnormalized density $\tilde{p}(\mathbf{x}) = p(\mathbf{x}) / Z$ with normalizing constant $Z$, the score satisfies $\nabla_\mathbf{x} \log p(\mathbf{x}) = \nabla_\mathbf{x} \log \tilde{p}(\mathbf{x})$, since $\nabla_\mathbf{x} \log Z = 0$. This means the score is independent of the normalizing constant — a key advantage when $Z$ is intractable, as it is for most high-dimensional distributions. Song & Ermon (2019) train a neural network $\mathbf{s}_\theta(\mathbf{x})$ to approximate $\nabla_\mathbf{x} \log p(\mathbf{x})$ using **denoising score matching**: data points are perturbed by adding Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$, yielding $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$, and the network is trained to recover the noise direction $-\boldsymbol{\epsilon}/\sigma^2$, which equals the score of the noise-conditional PDF $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})$. Samples are then drawn by Langevin dynamics: starting from $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, I)$ and iterating

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\alpha}{2}\, \mathbf{s}_\theta(\mathbf{x}_t) + \sqrt{\alpha}\, \mathbf{z}_t$$

where $\alpha > 0$ is the step size and $\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, I)$ is independent Gaussian noise at each step $t$. Ho et al. (2020) recast this framework as **denoising diffusion probabilistic models** (DDPM), establishing a formal equivalence between the DDPM denoising objective and score matching under a fixed Gaussian noise schedule.

### Generative Adversarial Networks

**Generative adversarial networks** (Goodfellow et al., 2014) achieve PDF matching without ever computing a PDF. A generator network $G_\phi: \mathbb{R}^k \to \mathbb{R}^d$, parameterized by $\phi$, maps latent noise $\mathbf{z} \in \mathbb{R}^k$ drawn from $p_\mathbf{z}(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I_k)$ — where $k$ is the latent dimension and $I_k$ is the $k \times k$ identity — to synthetic samples $G_\phi(\mathbf{z}) \in \mathbb{R}^d$ in the $d$-dimensional data space. The generator induces a distribution $p_{G_\phi}$ over $\mathbb{R}^d$ — the pushforward of $p_\mathbf{z}$ through $G_\phi$ — whose PDF is never explicitly computed. Instead, a discriminator network $D_\psi: \mathbb{R}^d \to [0, 1]$, parameterized by $\psi$, estimates the probability that a sample $\mathbf{x} \in \mathbb{R}^d$ came from the true data PDF $p_\mathrm{data}$ rather than $p_{G_\phi}$. The two networks are trained in a minimax game where $D_\psi$ maximizes and $G_\phi$ minimizes

$$\mathbb{E}_{\mathbf{x} \sim p_\mathrm{data}}[\log D_\psi(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_\mathbf{z}}[\log(1 - D_\psi(G_\phi(\mathbf{z})))]$$

Goodfellow et al. prove that at the optimal discriminator $D_\psi^*$, this objective equals twice the **Jensen-Shannon divergence** between $p_\mathrm{data}$ and $p_{G_\phi}$ minus $\log 4$, so training $G_\phi$ minimizes the divergence between the generated and true data PDFs — without either being available in closed form.

## Key Takeaways

A probability density function is a rate, not a probability, and its values carry meaning only when integrated over intervals of positive width; this density-versus-probability distinction propagates into every learning algorithm that depends on it. The change-of-variables formula, which relates the PDF of a random vector to the PDF of its invertible transformation through the Jacobian determinant, is the mathematical engine behind normalizing flows and makes exact likelihood computation tractable when the Jacobian is structured. The score function — the gradient of the log-PDF — separates the distributional geometry from the normalizing constant, enabling score-based diffusion models to learn the shape of complex data distributions without evaluating their densities directly. Generative adversarial networks go one step further, dispensing with any explicit PDF entirely and matching distributions through discriminator-mediated divergence minimization. These four settings reflect a spectrum of how the PDF can be engaged: computed directly, transformed, differentiated, or implicitly matched.

## Resources

- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (2020)
- Rezende & Mohamed, [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) (2015)
- Dinh et al., [Density estimation using Real-valued Non-Volume Preserving (Real NVP) transformations](https://arxiv.org/abs/1605.08803) (2016)
- Song & Ermon, [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) (2019)
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Goodfellow et al., [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (2014)
