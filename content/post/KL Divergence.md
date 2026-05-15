---
title: "KL Divergence: A Technical Introduction"
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

## Introduction

**Kullback-Leibler divergence**, introduced by Solomon Kullback and Richard Leibler in their 1951 paper on information sufficiency (Kullback & Leibler, 1951), is a measure of how one probability distribution $P$ differs from a reference distribution $Q$. Unlike a distance metric, it has a directional character: $D_{KL}(P \| Q)$ quantifies the information lost when $Q$ is used to approximate $P$, making it fundamentally asymmetric. This asymmetry is not a deficiency but a feature — it encodes a meaningful distinction between the "true" distribution and the "approximate" one, a distinction that maps directly onto the structure of statistical inference.

KL divergence sits at the intersection of information theory, statistics, and machine learning. Its connection to Shannon entropy makes it the natural tool for expressing how much a model's beliefs deviate from reality, and its variational properties underpin some of the most influential training objectives in modern AI. Variational autoencoders, trust region policy optimization, reinforcement learning from human feedback, and knowledge distillation all invoke KL divergence as a central element of their loss functions — not incidentally but structurally.

This post develops the mathematical foundations of KL divergence, builds intuition for its behavior, and examines its role in four major AI application areas at the graduate level.

## Mathematical Foundations

For discrete distributions $P$ and $Q$ defined over the same support $\mathcal{X}$, where $P(x)$ and $Q(x)$ denote the probability each distribution assigns to outcome $x$, the KL divergence from $Q$ to $P$ is defined as

$$D_{KL}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$

with the convention that $0 \log 0 = 0$ and that the expression equals $+\infty$ whenever $Q(x) = 0$ for some $x$ with $P(x) > 0$. For continuous distributions, the sum becomes an integral: $D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$. The logarithm is taken base $e$ throughout this post, giving units of nats; base-2 logarithms yield bits.

KL divergence is directly related to Shannon entropy and cross-entropy. Recall that the **entropy** of $P$ is $H(P) = -\sum_x P(x) \log P(x)$ and the **cross-entropy** of $Q$ relative to $P$ is $H(P, Q) = -\sum_x P(x) \log Q(x)$. Then

$$D_{KL}(P \| Q) = H(P, Q) - H(P)$$

which makes the information-theoretic interpretation precise: KL divergence is the excess expected code length when events drawn from $P$ are encoded using a code optimized for $Q$ instead of $P$.

Non-negativity, $D_{KL}(P \| Q) \geq 0$ with equality if and only if $P = Q$ almost everywhere, follows from **Gibbs' inequality**, which is itself a consequence of Jensen's inequality applied to the convex function $-\log$. Since $\log$ is strictly concave, $\mathbb{E}_P[\log(Q/P)] \leq \log(\mathbb{E}_P[Q/P]) = \log(1) = 0$, so $\mathbb{E}_P[\log(P/Q)] \geq 0$. Despite this non-negativity, KL divergence is not a metric: it is asymmetric ($D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ in general) and does not satisfy the triangle inequality.

For the specific case of two diagonal Gaussian distributions, which arises repeatedly in variational methods, the KL divergence admits a closed form. Let $q = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ be a $d$-dimensional diagonal Gaussian with mean $\mu \in \mathbb{R}^d$ and per-dimension variances $\sigma_j^2$, and let $p = \mathcal{N}(0, I)$ be the standard normal prior. Then

$$D_{KL}(q \| p) = \frac{1}{2} \sum_{j=1}^{d} \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right)$$

This expression is differentiable with respect to $\mu$ and $\sigma^2$, making it directly amenable to gradient-based optimization — a property that proves essential in variational autoencoders.

## Core Intuition

The cleanest way to internalize KL divergence is through coding theory. Suppose $P$ is the true distribution over messages and $Q$ is your assumed model. Shannon's source coding theorem guarantees that the optimal code for $Q$ assigns approximately $-\log Q(x)$ bits to message $x$. If you use this code but messages are actually drawn from $P$, the expected code length is $H(P, Q) = \mathbb{E}_P[-\log Q(x)]$. The optimal code for $P$ would achieve $H(P) = \mathbb{E}_P[-\log P(x)]$. The gap between the two — the wasted bits — is exactly $D_{KL}(P \| Q)$. Zero wasted bits means $P = Q$; large KL means your model is a poor match for reality.

The danger zone in this picture is when $Q(x) \approx 0$ for events $x$ that $P$ assigns meaningful probability. At such points, $\log(P(x)/Q(x))$ grows without bound, and even a small amount of probability mass under $P$ can make $D_{KL}(P \| Q)$ enormous. This is why minimizing $D_{KL}(P \| Q)$ — where $P$ is fixed and $Q$ is the approximation being optimized — forces $Q$ to cover all the modes of $P$: the penalty for missing any region that $P$ charges is effectively infinite. This behavior is called **zero-avoiding** or **mean-seeking**.

The reversed objective $D_{KL}(Q \| P)$ has the opposite character. Here $Q$ is the approximation and the log ratio is $\log(Q(x)/P(x))$; when $Q(x) > 0$ and $P(x) \approx 0$, the ratio blows up, so the optimizer is instead penalized for placing probability mass where $P$ does not. This drives $Q$ to concentrate on a single mode of $P$ rather than covering all of them, a behavior called **zero-forcing** or **mode-seeking**. The choice between minimizing forward KL ($D_{KL}(P \| Q)$) or reverse KL ($D_{KL}(Q \| P)$) thus encodes a fundamental modeling decision, with direct consequences for how generative models learn to represent multimodal distributions.

## Applications in AI

### Variational Inference and Variational Autoencoders

**Variational inference** reframes the intractable posterior computation problem $p_\theta(z|x)$ as an optimization problem: find the member of a tractable family of distributions $q_\phi(z|x)$ that is closest to the true posterior in KL divergence. The decomposition of the log-likelihood

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}\!\left[\log p_\theta(x|z)\right] - D_{KL}(q_\phi(z|x) \| p(z))}_{\mathcal{L}(\theta,\phi;\,x)} + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

shows that $\mathcal{L}(\theta, \phi; x)$, the **evidence lower bound** (ELBO), equals the log-likelihood minus a non-negative KL term. Since maximizing the ELBO tightens the bound, it simultaneously pushes $q_\phi$ toward the true posterior and $p_\theta$ toward generating data like $x$.

The **Variational Autoencoder** (Kingma & Welling, 2013) operationalizes this framework with a neural network encoder $q_\phi(z|x)$ parameterized as a diagonal Gaussian and a neural network decoder $p_\theta(x|z)$. The KL term $D_{KL}(q_\phi(z|x) \| p(z))$ in the ELBO acts as a regularizer, preventing the encoder from collapsing to a point mass and enforcing a smooth, structured latent space. Because $q_\phi$ and $p$ are both Gaussian, this term is computed in closed form using the expression derived above, enabling exact gradient computation for the entire training objective via the reparameterization trick.

### Trust Region Policy Optimization

In reinforcement learning, policy gradient methods that take large gradient steps can catastrophically degrade performance because a poor update changes the distribution of future states. **Trust Region Policy Optimization** (Schulman et al., 2015) addresses this by constraining each policy update to a trust region defined by KL divergence. Formally, the optimization problem is

$$\max_\theta \; \mathbb{E}_{s \sim \rho^{\pi_\mathrm{old}},\, a \sim \pi_\mathrm{old}}\!\left[\frac{\pi_\theta(a|s)}{\pi_\mathrm{old}(a|s)} A^{\pi_\mathrm{old}}(s, a)\right] \quad \text{s.t.} \quad \mathbb{E}_{s \sim \rho^{\pi_\mathrm{old}}}\!\left[D_{KL}(\pi_\mathrm{old}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta$$

where $\rho^{\pi_\mathrm{old}}$ is the state visitation distribution induced by $\pi_\mathrm{old}$, $s$ and $a$ are state and action, $A^{\pi_\mathrm{old}}$ is the advantage function, and $\delta$ is the trust region radius. The KL constraint ensures the new policy remains in a neighborhood of the old one where the importance-weighted objective is a reliable estimate of true performance improvement. Schulman et al. prove a monotonic improvement guarantee under this constraint, making TRPO the first policy gradient method with a provably safe update rule.

### Reinforcement Learning from Human Feedback

Training large language models to follow instructions requires optimizing a reward model derived from human preference judgments, but unconstrained maximization of such a reward signal causes **reward hacking** — the model finds degenerate outputs that achieve high reward without producing useful language. The standard remedy, introduced in the fine-tuning framework of Ziegler et al. (2019) and scaled in InstructGPT (Ouyang et al., 2022), is a KL penalty that keeps the fine-tuned policy $\pi$ close to a frozen reference model $\pi_\mathrm{ref}$:

$$\max_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)}\!\left[r(x, y)\right] - \beta \, D_{KL}(\pi(\cdot|x) \| \pi_\mathrm{ref}(\cdot|x))$$

Here $x$ is an input prompt drawn from the training distribution $\mathcal{D}$, $y$ is the model's generated response, $r(x, y)$ is the reward model's score, and $\beta$ is a coefficient that trades off reward maximization against distributional fidelity. When $\beta = 0$ the policy is unconstrained and degenerates; as $\beta$ increases, the policy stays closer to the reference but may underfit the reward signal. In practice $\beta$ is tuned to find a regime where the model improves on human preference metrics while retaining coherent language generation. The KL term in this objective is formally identical to the one used in TRPO but serves a different function: rather than ensuring safe policy improvement, it acts as a regularizer against distribution shift induced by imperfect reward modeling.

### Knowledge Distillation

**Knowledge distillation** (Hinton et al., 2015) transfers the learned representations of a large teacher network into a smaller student network by training the student to reproduce the teacher's output distribution rather than hard ground-truth labels. For a classification task with a teacher producing logits $z_t$ and a student producing logits $z_s$, the distillation loss is

$$\mathcal{L}_\mathrm{KD} = T^2 \cdot D_{KL}\!\left(\sigma(z_t / T) \,\|\, \sigma(z_s / T)\right)$$

where $\sigma$ denotes the softmax function and $T > 1$ is the **temperature** parameter. At $T = 1$ the softmax concentrates probability mass on the argmax class; at high $T$ it spreads probability more uniformly across classes, revealing the teacher's relative confidence structure — for example, that a particular "2" resembles a "7" more than a "1". The student trained on these softened distributions acquires richer supervision than one-hot labels provide, achieving better generalization than training from scratch at the same parameter count. The $T^2$ factor compensates for the reduced magnitude of the soft-target gradients at high temperature. Minimizing this forward KL with the teacher distribution fixed forces the student to cover all modes of the teacher's output, preventing it from ignoring low-probability classes that nonetheless encode semantic relationships.

## Key Takeaways

KL divergence measures the information cost of approximating one probability distribution with another, and its asymmetry — which direction is minimized — produces qualitatively different learned distributions: mean-seeking when minimizing the forward KL, mode-seeking when minimizing the reverse. This property is not merely a mathematical curiosity; it directly governs the behavior of variational autoencoders, trust region methods, RLHF reward shaping, and knowledge distillation, each of which embeds a KL term chosen to enforce a specific inductive bias. Across all these settings, the closed-form tractability of the Gaussian KL and the guarantee of non-negativity make it both computationally and theoretically convenient, cementing its role as one of the fundamental quantities in modern machine learning.

## Resources

- Kullback & Leibler, [On Information and Sufficiency](https://www.jstor.org/stable/2236703) (1951)
- Kingma & Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (2013)
- Schulman et al., [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (2015)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015)
- Ziegler et al., [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (2019)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022)
