---
title: "Exponential Moving Average: A Technical Introduction"
author: yo3nglau
date: '2026-05-15'
categories:
  - Deep Learning
tags:
  - Optimization
  - Signal Processing
  - Mathematics
toc: true
---

## Introduction

**Exponential Moving Average** (EMA), also known as the **exponentially weighted moving average**, is a recursive smoothing technique that assigns geometrically decaying weights to past observations. Unlike a simple moving average, which treats every observation within a fixed window equally, EMA gives the most recent data the greatest influence while older observations fade at an exponential rate. The concept originates in signal processing and control engineering, where it was used to track slowly varying physical quantities in the presence of measurement noise, and it now ranks among the most frequently employed numerical primitives in machine learning.

The reason EMA has migrated so naturally from engineering to AI is that machine learning constantly confronts the same fundamental tension: how to combine a noisy, rapidly changing signal with a stable, historically informed baseline. Gradient estimates during training fluctuate wildly from batch to batch; value targets in reinforcement learning shift as the policy improves; representations produced by an encoder change continuously as its weights are updated. In each case, a slowly evolving EMA of the signal of interest provides a kind of temporal inertia — a smooth, conservative estimate that resists the influence of individual outliers while remaining sensitive to persistent trends. This post develops the mathematics of EMA from first principles and examines its role in four areas of contemporary AI research: adaptive gradient optimization, target networks in reinforcement learning, momentum encoders in self-supervised learning, and parameter averaging in generative model training.

## Mathematical Foundations

The EMA of a sequence of observations $x_1, x_2, \ldots$ is defined by the recursion

$$v_t = \beta \, v_{t-1} + (1 - \beta) \, x_t$$

where $v_t$ is the smoothed estimate at time step $t$, $x_t$ is the raw observation at step $t$, and $\beta \in (0, 1)$ is the **decay factor** (also called the smoothing coefficient or momentum parameter). Initializing $v_0 = 0$ and unrolling the recursion shows that the smoothed estimate is a weighted superposition of all past observations,

$$v_t = (1 - \beta) \sum_{k=0}^{t-1} \beta^k \, x_{t-k} + \beta^t v_0$$

where $k$ indexes how many steps in the past an observation lies, and the weight assigned to observation $x_{t-k}$ decays as $\beta^k$. The total weight on the observations is $1 - \beta^t$, which converges to $1$ as $t \to \infty$, so for large $t$ the estimate approaches a proper convex combination of past inputs. A decay factor close to $1$, such as $\beta = 0.999$, produces a heavily smoothed, slowly varying estimate that averages over a large effective window, while a smaller value such as $\beta = 0.5$ makes the estimate far more responsive to recent inputs at the cost of greater sensitivity to noise.

When $v_0 = 0$, the estimate in the early steps is systematically biased toward zero because the weights on the observations sum to only $1 - \beta^t < 1$ rather than $1$. This **initialization bias** is corrected by rescaling:

$$\hat{v}_t = \frac{v_t}{1 - \beta^t}$$

where $\hat{v}_t$ is the bias-corrected estimate. The correction is significant only in the first $O(1/(1-\beta))$ steps, which corresponds to the effective memory length of the EMA; for large $t$, the denominator $1 - \beta^t$ is indistinguishable from $1$ and $\hat{v}_t \approx v_t$. Two important analytical properties follow from the geometric weighting structure. First, the EMA is a linear filter, and in the frequency domain it acts as a low-pass filter, attenuating high-frequency fluctuations while preserving slow trends. Second, the effective number of observations being averaged is approximately $1/(1-\beta)$, so a practitioner choosing $\beta = 0.9$ is implicitly averaging over roughly ten steps, while $\beta = 0.99$ extends the effective window to about one hundred steps.

## Core Intuition

The most useful way to think about EMA is as a fading memory. At each step, the running estimate is formed by blending the incoming observation with the existing memory in proportions $(1-\beta) : \beta$. Any single observation therefore contributes a weight that decays geometrically as more observations arrive: after $k$ additional steps, an observation's residual influence on the current estimate is proportional to $\beta^k$. The number of steps until this influence falls below half of its initial value — the **half-life** of an observation — is $\log 2 / \log(1/\beta) \approx 0.693/(1-\beta)$ for $\beta$ near $1$. This relationship gives practitioners an intuitive handle on the $\beta$ parameter: targeting a half-life of one hundred steps implies $\beta \approx 1 - 0.693/100 \approx 0.993$.

The most common misunderstanding is that a larger $\beta$ is unconditionally better because it produces a smoother signal. Smoothness comes at the direct cost of **lag**: a high-$\beta$ EMA tracks a sudden, persistent shift in the underlying signal very slowly, requiring many steps before the estimate catches up to the new regime. In optimization problems, this trade-off manifests as a choice between using a stable but sluggish estimate (high $\beta$) and a noisy but reactive one (low $\beta$). A second subtlety concerns the initialization bias: even when the true signal is constant from the very first step, the estimate $v_t$ rises only gradually toward its true value because early steps have $\beta^t \gg 0$ and the correction factor $1 - \beta^t$ is far from $1$. The bias-corrected form $\hat{v}_t$ eliminates this artifact and is essential whenever accurate estimates in the very first steps of a procedure are operationally important.

## Applications in AI

### Adaptive Gradient Optimization

The most pervasive use of EMA in machine learning is within adaptive gradient methods, where it provides stable, low-noise estimates of first- and second-order gradient statistics. The **Adam** optimizer (Kingma & Ba, 2015) maintains two EMA quantities updated at each training step: the first moment estimate $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$, which tracks the running mean of the gradient $g_t$, and the second moment estimate $s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2$, which tracks the uncentered variance of the gradient. After bias correction, the parameter update for each coordinate is proportional to $\hat{m}_t / (\sqrt{\hat{s}_t} + \epsilon)$, where $\epsilon$ is a small stabilizing constant. This ratio normalizes the gradient by an estimate of its recent magnitude, enabling a single global learning rate to serve all parameters simultaneously: coordinates whose gradients have been consistently large receive proportionally smaller updates, while rarely activated parameters receive larger ones.

The bias correction is not an optional refinement in Adam but an operational necessity. With the recommended default $\beta_2 = 0.999$, the uncorrected second moment $s_t$ at the very first step is $(1 - 0.999) g_1^2 = 0.001 \, g_1^2$, which is one thousand times smaller than the true squared gradient. Without correction, the update $g_t / \sqrt{s_t}$ would be inflated by a factor of roughly $\sqrt{1000} \approx 32$, causing catastrophically large parameter changes in the early iterations. The bias-corrected $\hat{s}_1 = s_1 / (1 - 0.999) = g_1^2$ restores the proper scale. RMSProp, an earlier adaptive method, employs EMA of squared gradients without the bias correction or the first moment, illustrating how the same EMA mechanism can be adapted with varying levels of statistical rigor.

### Target Networks in Deep Reinforcement Learning

In value-based deep reinforcement learning, the EMA-based soft update addresses a fundamental stability problem: when the same network simultaneously defines the current policy and provides the regression targets for its own training, the targets shift with every parameter update, creating a moving-target problem that can destabilize or diverge the learning process. The original **Deep Q-Network** (Mnih et al., 2013) introduced a target network — a lagged copy of the online network — whose parameters are held fixed for a prescribed number of steps and then replaced with a hard copy of the current weights. While effective, hard copying creates abrupt discontinuities in the target signal each time the copy is made, which can induce oscillations in the subsequent training phase.

The **Deep Deterministic Policy Gradient** algorithm (Lillicrap et al., 2016) replaced the periodic hard copy with a soft update: at every training step, the target network parameters $\theta^-$ are blended toward the online parameters $\theta$ by

$$\theta^- \leftarrow \tau \, \theta + (1 - \tau) \, \theta^-$$

where $\tau \in (0, 1)$ is the soft-update coefficient, typically set to a small value such as $0.005$. Because $\tau \ll 1$, the target network changes by only a tiny amount each step, and the target values it produces shift smoothly and slowly. The online network therefore chases a target that moves continuously but at a much slower rate than it does itself — a separation of timescales that stabilizes the regression problem considerably. The EMA formulation is strictly preferable to hard copies in continuous control settings because it never resets the target network to an abruptly new position; instead, the target tracks the online network's trajectory with a tunable lag controlled entirely by $\tau$.

### Momentum Encoders in Self-Supervised Learning

A third major role for EMA arises in contrastive self-supervised learning, where maintaining a large dictionary of consistent negative representations is essential but computationally prohibitive if done through gradient descent on a large batch. **Momentum Contrast** (He et al., 2020) addresses this with a momentum encoder: a key encoder whose parameters $\theta_k$ are not optimized directly by backpropagation but are instead updated at each step by EMA of the query encoder parameters $\theta_q$,

$$\theta_k \leftarrow m \, \theta_k + (1 - m) \, \theta_q$$

where $m$ is the momentum coefficient, set to $0.999$ in the original paper. Because the key encoder evolves so slowly, representations it produces over consecutive training steps remain approximately consistent with one another: a key encoded at step $t$ and a key encoded at step $t - 1000$ were produced by nearly identical encoders, so they can be meaningfully compared in the contrastive loss without introducing large systematic errors. This temporal consistency allows MoCo to maintain a queue of up to $65{,}536$ negative keys drawn from recent mini-batches — a count far larger than any feasible single batch — while keeping those keys approximately on-distribution relative to the current encoder. The slow-moving momentum encoder is not merely a computational shortcut; it imposes a specific learning dynamic that ensures the contrastive loss signal is stable enough for the online encoder to learn meaningful representations rather than collapsing to trivial solutions.

### Generative Model Parameter Averaging

In the training of diffusion-based generative models, EMA over a network's own parameter trajectory serves as a post-hoc denoising step on the optimization path. **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) maintain a shadow copy $\bar{\theta}$ of the denoising network weights that is never updated by gradient descent but instead tracks the gradient-updated weights $\theta$ through an EMA applied at every training iteration. At inference time, generation is performed using $\bar{\theta}$ exclusively, while $\theta$ continues to evolve through gradient steps during training. Ho et al. (2020) observed that the EMA weights yield substantially better Fréchet Inception Distance scores and visual sample quality than the online checkpoint at any given step, and this practice has since become a standard component of diffusion model training pipelines.

The theoretical motivation traces back to **Polyak-Ruppert averaging** (Polyak & Juditsky, 1992), which established that for strongly convex stochastic optimization problems, the time-averaged iterate of stochastic gradient descent converges at the minimax-optimal rate and typically outperforms the final SGD iterate even in finite time. In the non-convex landscape of diffusion model training, the formal guarantees do not directly apply, but the empirical mechanism is the same: SGD iterates $\theta_1, \theta_2, \ldots$ oscillate around a basin of a local minimum, and their running average $\bar{\theta}$ lies closer to the basin center than any individual iterate. By maintaining an exponentially weighted running average rather than a uniform one, EMA gives more weight to recent iterates and adjusts more smoothly to gradual shifts in the effective loss landscape caused by the evolving learning rate schedule.

## Key Takeaways

Exponential Moving Average is a mechanism for converting a noisy or rapidly changing signal into a stable, informative estimate using nothing more than a single tunable decay factor $\beta$ and a one-line recursion. Its geometric weighting structure connects it directly to first-order low-pass filtering and to the classical Polyak-Ruppert averaging theory of stochastic optimization, giving it a mathematical pedigree that justifies its widespread adoption. In modern AI, EMA functions as a universal stabilization primitive whose specific role varies by context — smoothing gradient moments in Adam, maintaining lagged targets in reinforcement learning, enforcing temporal consistency in contrastive encoders, and averaging out stochastic noise in generative model training — but whose underlying mechanism is always the same: older information is discounted exponentially so that the estimate remains dominated by recent, relevant observations. A practitioner who understands the half-life interpretation of $\beta$, the operational significance of the bias correction, and the fundamental lag-smoothness trade-off will find that EMA can be precisely tuned rather than merely guessed at, and that its apparent simplicity conceals a depth of behavior that continues to reward careful analysis.

## Resources

- Kingma & Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (2015)
- Mnih et al., [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (2013)
- Lillicrap et al., [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) (2016)
- He et al., [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) (2020)
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Polyak & Juditsky, [Acceleration of Stochastic Approximation by Averaging](https://epubs.siam.org/doi/10.1137/0330046) (1992)
