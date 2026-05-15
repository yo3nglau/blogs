---
title: "Softmax: A Technical Introduction"
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

## Introduction

**Softmax** is a vector-valued function that maps a vector of real-valued logits to a probability distribution over a finite set of outcomes. Given an input $\mathbf{z} \in \mathbb{R}^K$, softmax produces a vector whose entries are non-negative and sum to one — placing the output on the interior of the $K$-dimensional probability simplex $\Delta^{K-1}$. The function's name reflects its role as a smooth, differentiable alternative to the argmax operation: where argmax returns a one-hot indicator for the largest logit, softmax distributes probability mass continuously across all classes, preserving gradient flow and enabling end-to-end training by backpropagation.

The function's origins trace to statistical physics, where the **Boltzmann distribution** describes the probability of a system occupying an energy state $E_i$ at temperature $T$ as proportional to $e^{-E_i/T}$. Identifying logit $z_i$ with negative energy $-E_i$ and setting $T = 1$ recovers the softmax formula. The same function appears in econometrics as the **multinomial logit model** (McFadden, 1974), and was introduced into the neural network literature by Bridle (1990) as a principled probabilistic output layer for multi-class classification.

Softmax is now one of the most ubiquitous operations in deep learning. It appears as the output layer of classifiers, as the normalizer inside the attention mechanism of Transformers, as a policy parameterization in reinforcement learning, and as the target of temperature calibration in post-hoc reliability correction. This post develops its mathematical properties at a graduate level and examines these four application areas in depth.

## Mathematical Foundations

For an input vector $\mathbf{z} = (z_1, \ldots, z_K)^\top \in \mathbb{R}^K$, where $K \geq 2$ is the number of classes, the softmax function $\sigma: \mathbb{R}^K \to \Delta^{K-1}$ is defined component-wise as

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K$$

Each output $\sigma(\mathbf{z})_i$ is strictly positive, and by construction $\sum_{i=1}^K \sigma(\mathbf{z})_i = 1$, so the output vector lies in the open probability simplex. A key algebraic property is **translation invariance**: for any scalar $c \in \mathbb{R}$, $\sigma(\mathbf{z} + c\mathbf{1}) = \sigma(\mathbf{z})$, where $\mathbf{1} \in \mathbb{R}^K$ is the all-ones vector. This follows because adding $c$ to every exponent multiplies numerator and denominator by the same factor $e^c$, which cancels. Translation invariance implies softmax is overparameterized: the $K$-dimensional input has only $K - 1$ effective degrees of freedom. Numerically, setting $c = -\max_j z_j$ before computing exponentials prevents overflow without changing the output — the standard implementation trick for numerical stability.

The Jacobian of softmax is needed for efficient backpropagation. Writing $S = \sum_{k=1}^K e^{z_k}$ for the normalizing sum, differentiating $\sigma_i = e^{z_i}/S$ with respect to $z_j$ yields $\sigma_i(1 - \sigma_i)$ when $i = j$ (by the quotient rule) and $-\sigma_i \sigma_j$ when $i \neq j$. These two cases unify into

$$\frac{\partial \sigma(\mathbf{z})_i}{\partial z_j} = \sigma(\mathbf{z})_i \bigl(\delta_{ij} - \sigma(\mathbf{z})_j\bigr)$$

where $\delta_{ij}$ is the Kronecker delta, equal to 1 if $i = j$ and 0 otherwise. The full Jacobian matrix $J \in \mathbb{R}^{K \times K}$ can be written as $J = \mathrm{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^\top$, where $\boldsymbol{\sigma} = \sigma(\mathbf{z})$ is the output vector. This structure makes Jacobian-vector products computable in $O(K)$ rather than $O(K^2)$ time.

Softmax is intimately connected to the **log-sum-exp** function $\mathrm{LSE}(\mathbf{z}) = \log \sum_{j=1}^K e^{z_j}$, which is a smooth convex upper bound on $\max_j z_j$. The softmax outputs are the partial derivatives of the log-sum-exp: $\sigma(\mathbf{z})_i = \partial \,\mathrm{LSE}(\mathbf{z}) / \partial z_i$. Equivalently, $\sigma(\mathbf{z})_i = \exp(z_i - \mathrm{LSE}(\mathbf{z}))$, which shows that each softmax output is the exponential of the corresponding logit minus the log-partition function. Introducing a **temperature** parameter $T > 0$, the tempered softmax is

$$\sigma(\mathbf{z};\, T)_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$

where dividing each logit by $T$ controls the sharpness of the output distribution.

## Core Intuition

Softmax is best understood as a differentiable relaxation of argmax. The argmax of $\mathbf{z}$ assigns probability 1 to the largest entry and 0 to all others — a deterministic, non-differentiable operation that blocks gradient flow. Softmax replaces this hard selection with a soft assignment: it assigns the highest probability to the largest logit but distributes the remaining probability mass across other entries in proportion to $e^{z_j}$. The larger the gap between logits, the more the output concentrates on the leading entry; in the limit of a single dominant logit, the softmax output approaches a one-hot vector.

The temperature parameter $T$ controls this sharpness continuously. As $T \to 0^+$, the ratios $z_i/T$ grow without bound and the softmax converges to the argmax: $\sigma(\mathbf{z};\, T) \to \mathbf{e}_c$ where $c = \arg\max_j z_j$ and $\mathbf{e}_c$ is the unit vector in the $c$-th direction. As $T \to \infty$, the ratios $z_i/T \to 0$ for all $i$, and $\sigma(\mathbf{z};\, T) \to \frac{1}{K}\mathbf{1}$, the uniform distribution over all $K$ classes. Temperature thus provides a single scalar dial that smoothly interpolates between certainty and maximum entropy, a property exploited in knowledge distillation, calibration, and sampling from language models.

The Boltzmann distribution analogy illuminates why softmax behaves this way. In statistical mechanics, a system in thermal equilibrium at temperature $T$ occupies energy state $E_i$ with probability $p_i \propto e^{-E_i/T}$. Identifying the negative logit $-z_i$ with energy $E_i$ recovers the tempered softmax. Low temperature corresponds to a system strongly biased toward its lowest-energy (highest-logit) state; high temperature corresponds to a system uniformly sampling all states regardless of energy. The logits play the role of negative energies, and the normalizing denominator $\sum_j e^{z_j}$ is the partition function — the object whose log is the free energy of the system.

## Applications in AI

### Multi-class Classification

In neural network classifiers, softmax converts the unnormalized output of the final linear layer — the **logit vector** $\mathbf{z} = W\mathbf{h} + \mathbf{b}$, where $\mathbf{h} \in \mathbb{R}^d$ is the penultimate hidden representation, $W \in \mathbb{R}^{K \times d}$ is the weight matrix, and $\mathbf{b} \in \mathbb{R}^K$ is the bias — into a categorical distribution $\hat{\mathbf{y}} = \sigma(\mathbf{z})$ over $K$ classes. Training minimizes the cross-entropy loss $\mathcal{L} = -\log \hat{y}_c = -\log \sigma(\mathbf{z})_c$, where $c \in \{1, \ldots, K\}$ is the index of the ground-truth class. Expanding using the log-sum-exp gives $\mathcal{L} = -z_c + \mathrm{LSE}(\mathbf{z})$, and differentiating yields the gradient

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sigma(\mathbf{z})_j - \mathbf{1}[j = c]$$

where $\mathbf{1}[j = c]$ equals 1 if $j = c$ and 0 otherwise. This gradient has a clean interpretation: it is the difference between the model's predicted probability for class $j$ and the true probability (0 or 1) for class $j$. The gradient is zero when the model assigns probability 1 to the correct class, and it points most strongly toward reducing the probability of the most overconfident incorrect class. BERT (Devlin et al., 2018) applies this exact formulation in its classification head, computing $\sigma(W\mathbf{h}_\mathrm{[CLS]})$ over the pooled representation $\mathbf{h}_\mathrm{[CLS]}$ of the special classification token.

### Scaled Dot-Product Attention

The **scaled dot-product attention** mechanism (Vaswani et al., 2017) uses softmax to convert raw similarity scores between query and key vectors into a probability distribution over value vectors. Given a query matrix $Q \in \mathbb{R}^{n \times d_k}$ containing $n$ queries each of dimension $d_k$, a key matrix $K \in \mathbb{R}^{m \times d_k}$ containing $m$ key vectors, and a value matrix $V \in \mathbb{R}^{m \times d_v}$ containing $m$ value vectors each of dimension $d_v$, the attention output is

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where the softmax is applied row-wise to the $n \times m$ score matrix $QK^\top / \sqrt{d_k}$, producing $n$ probability distributions each over the $m$ key positions, and the resulting $n \times m$ weight matrix is then multiplied by $V$ to yield an $n \times d_v$ output. Each row of the output is a convex combination of the value vectors, with weights determined by the softmax over dot-product similarities. The scaling factor $1/\sqrt{d_k}$ counteracts the tendency of dot products to grow in magnitude with $d_k$ — Vaswani et al. note that for large $d_k$, unscaled dot products push the softmax into regions with extremely small gradients, slowing learning. The softmax is essential here not just for normalization but for producing a differentiable weighted average that permits end-to-end training via backpropagation.

### Softmax Policy in Reinforcement Learning

In reinforcement learning over discrete action spaces, softmax provides a natural parameterization of a stochastic policy. Given a state $s$ and a set of possible actions $\mathcal{A} = \{a_1, \ldots, a_K\}$, a neural network with parameters $\theta$ produces a preference score $h_\theta(s, a) \in \mathbb{R}$ for each action $a$, and the policy is defined as

$$\pi_\theta(a \mid s) = \frac{\exp(h_\theta(s, a))}{\sum_{a' \in \mathcal{A}} \exp(h_\theta(s, a'))}$$

This parameterization ensures $\pi_\theta(\cdot \mid s)$ is a valid probability distribution for every $s$ and every value of $\theta$, and it is differentiable with respect to $\theta$, enabling policy gradient updates. The softmax policy avoids the hard commitment of greedy action selection while remaining trainable end-to-end; its entropy $H(\pi_\theta(\cdot \mid s)) = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$ is directly controlled by the scale of $h_\theta$, and adding an entropy bonus to the reward encourages exploration. Mnih et al. (2016) use this softmax policy parameterization in the actor network of A3C (Asynchronous Advantage Actor-Critic) for discrete-action Atari games, where the actor outputs a softmax distribution over the game's action set.

### Temperature Scaling and Calibration

A well-**calibrated** classifier is one whose predicted probability $p$ for an event matches the actual frequency with which that event occurs across instances where the model assigns probability $p$. Modern deep neural networks trained with cross-entropy are systematically miscalibrated: they tend to assign confidences close to 1 even on uncertain inputs, a phenomenon termed **overconfidence**. Guo et al. (2017) demonstrate this empirically and show that **temperature scaling** — dividing all logits by a single learned scalar $T > 1$ before applying softmax — is the most effective single-parameter post-hoc calibration method. The calibrated probability for class $i$ is $\hat{p}_i = \sigma(\mathbf{z}/T)_i$, where $\mathbf{z}$ is the vector of logits produced by the pre-trained model and $T > 0$ is fit by minimizing the negative log-likelihood on a held-out validation set. Because $T$ is a scalar applied uniformly across all logits, it does not change the argmax prediction — the ranking of classes is preserved — but it softens the probability distribution, reducing overconfidence. Temperature scaling is also central to knowledge distillation (Hinton et al., 2015), where $T > 1$ is used during training to expose the student network to the teacher's soft probability assignments, which carry more information about inter-class similarities than one-hot labels.

## Key Takeaways

Softmax is the canonical operation for converting real-valued logits into a probability distribution over a discrete set, and its differentiability is what makes multi-class neural networks trainable by gradient descent; the gradient of the cross-entropy loss with respect to the logits simplifies to the difference between predicted and true probabilities, a form that makes both the mathematics and the optimization transparent. Its connection to log-sum-exp establishes softmax as a smooth relaxation of argmax, with the temperature parameter providing continuous control over sharpness — from near-deterministic selection at low temperature to near-uniform distribution at high temperature. In attention mechanisms, this sharpness control governs how broadly or narrowly a query aggregates information from a context; in reinforcement learning, it balances exploration and exploitation through policy entropy; in calibration, it corrects the systematic overconfidence that cross-entropy training induces in large models. Across all these roles, the Boltzmann interpretation remains the unifying thread: softmax is a partition function in disguise, and tuning its temperature tunes the effective energy scale of the underlying probability model.

## Resources

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (2018)
- Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (2016)
- Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (2017)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015)
