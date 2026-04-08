---
title: "Training Techniques & Optimization: Interview Questions and Answers"
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

## Optimization Algorithms

### Q1 [Basic] Explain how Adam optimizes parameters using first and second moment estimates

**Q:** How does the Adam optimizer use gradient history to adapt learning rates for each parameter?

**A:** **Adam** (Adaptive Moment Estimation) maintains two running statistics per parameter: a first moment $m_t$ (exponential moving average of gradients) and a second moment $v_t$ (exponential moving average of squared gradients) (Kingma & Ba, 2015). At each step $t$, these are updated as:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

where $g_t$ is the gradient and $\beta_1 = 0.9$, $\beta_2 = 0.999$ are the decay rates. Because $m_t$ and $v_t$ are initialized to zero, Adam applies bias correction to counteract the initialization bias:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The parameter update is then $\theta_t = \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$, where $\epsilon$ is a small constant for numerical stability. The effect is that parameters with historically large gradients receive smaller effective learning rates, while parameters with small or infrequent gradients are updated more aggressively. This adaptive scaling often provides faster convergence than SGD with a fixed learning rate, particularly for sparse or noisy gradients.

---

### Q2 [Advanced] Analyze why Adam's L2 regularization fails and how AdamW corrects it

**Q:** What is the mathematical difference between adding L2 regularization to the loss versus directly decoupling weight decay in Adam?

**A:** In standard SGD, adding an L2 penalty $\frac{\lambda}{2}\|\theta\|^2$ to the loss is exactly equivalent to **weight decay** because the gradient becomes $g_t + \lambda\theta$, and the update subtracts $\eta\lambda\theta$ from the parameter — shrinking it by a fixed fraction. This equivalence breaks in Adam (Loshchilov & Hutter, 2019).

When L2 regularization is added to the loss in Adam, the regularization gradient $\lambda\theta$ gets absorbed into $g_t$ and then divided by $\sqrt{\hat{v}_t} + \epsilon$. For parameters with large historical gradients, $\sqrt{\hat{v}_t}$ is large, so the effective weight decay shrinks disproportionately — parameters that are updated frequently are regularized less than parameters updated infrequently. This breaks the intended uniform shrinkage.

**AdamW** fixes this by applying weight decay directly to the parameters, outside the adaptive update:

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_{t-1}\right)$$

The decay term $\lambda\theta_{t-1}$ is no longer scaled by the second moment, so it acts as proper weight decay regardless of gradient history. Loshchilov & Hutter (2019) showed that AdamW consistently outperforms Adam with L2 on image classification and language modelling benchmarks, and it has become the default optimizer for training large language models.

---

### Q3 [Advanced] Identify the variance problem Adam encounters early in training and how RAdam resolves it

**Q:** Why does Adam sometimes diverge or produce unstable updates in the early training steps, and what mechanism does RAdam use to address this?

**A:** At the start of training, the second moment $v_t$ is close to zero because $\beta_2^t \approx 1$ for small $t$ and the exponential moving average has had few updates. After bias correction, $\hat{v}_t$ can still have very high variance — the denominator $\sqrt{\hat{v}_t}$ is unreliable, producing erratically large or small effective learning rates. This initialization instability can cause early divergence or lock the optimizer into a poor region of the loss landscape before the estimate stabilizes (Liu et al., 2020).

**RAdam** (Rectified Adam) addresses this by computing the approximate maximum length of the simple variance of $\hat{v}_t$, denoted $\rho_t$:

$$\rho_t = \rho_\infty - \frac{2t\beta_2^t}{1 - \beta_2^t}, \quad \rho_\infty = \frac{2}{1-\beta_2} - 1$$

When $\rho_t$ exceeds a threshold (approximately 4), the second moment estimate is considered sufficiently stable and the adaptive update is applied with a variance rectification term $r_t$. When $\rho_t$ is too small, RAdam falls back to SGD with momentum, avoiding the unreliable adaptive scaling entirely.

In practice, RAdam eliminates the need for a warmup schedule in many settings. Liu et al. (2020) demonstrated that RAdam matches or improves upon Adam with warmup across machine translation and language model fine-tuning tasks without requiring warmup hyperparameter tuning.

---

### Q4 [Advanced] Compare first-order and second-order optimization methods for deep learning

**Q:** What information does a second-order optimizer use that first-order methods ignore, and why have second-order methods not replaced Adam in large-scale training?

**A:** First-order methods like SGD and Adam use only the gradient $\nabla_\theta \mathcal{L}$, which points in the steepest ascent direction. Second-order methods additionally use the **Hessian** $H = \nabla^2_\theta \mathcal{L}$ (or an approximation), which encodes the curvature of the loss surface. The Newton update $\theta \leftarrow \theta - H^{-1}\nabla_\theta\mathcal{L}$ rescales the gradient by the inverse curvature, enabling larger steps in flat directions and smaller steps in sharp directions. This can achieve quadratic convergence near a minimum, compared to the linear convergence of gradient descent.

The fundamental obstacle is that for a model with $d$ parameters, storing and inverting the full Hessian costs $O(d^2)$ memory and $O(d^3)$ compute — entirely intractable for models with billions of parameters. Practical approximations include diagonal Hessian estimates and **K-FAC** (Kronecker-Factored Approximate Curvature), which approximates the Fisher information matrix using Kronecker products of smaller matrices (Martens & Grosse, 2015). K-FAC has been successfully applied to convolutional and recurrent models, achieving faster convergence per epoch but at a high per-step overhead.

A deeper issue is that modern deep networks are not locally convex: the loss surface contains saddle points, flat regions, and sharp minima (Dauphin et al., 2014). Second-order methods, designed for convex optimization, can be attracted to saddle points via curvature and may converge to sharp minima that generalize poorly. Empirically, the flat minima that SGD with noise finds tend to generalize better than the precise minima that Newton-type methods converge to. These combined factors — memory cost, computational overhead, and generalization concerns — explain why first-order methods remain dominant in large-scale deep learning.

---

## Learning Rate Scheduling

### Q5 [Basic] Describe why learning rate warmup is used at the start of training

**Q:** What happens to the optimization dynamics without a warmup phase, and why does starting from a small learning rate help?

**A:** At the very beginning of training, the model parameters are randomly initialized and the gradient estimates are unreliable — both because the model's outputs are nearly random and because the second moment accumulator in Adam is zero (see Q3). Starting with a large learning rate in this regime can push parameters into regions of the loss surface that are difficult to escape, destabilize batch normalization statistics, or cause NaN losses in mixed-precision training.

**Warmup** linearly or exponentially increases the learning rate from near zero to the target learning rate over a fixed number of steps (commonly 1,000–10,000 steps for large models). During this phase, the optimizer takes cautious small steps while accumulating reliable gradient statistics. Once the second moment estimate has stabilized, the full learning rate can be applied safely.

Warmup became standard practice with the Transformer architecture (Vaswani et al., 2017), which uses the schedule $\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5},\, t \cdot T_w^{-1.5})$ where $T_w$ is the warmup period. In practice, linear warmup followed by cosine decay is now the most common schedule for large language model pre-training.

---

### Q6 [Advanced] Explain how cosine annealing with warm restarts encourages exploration of the loss landscape

**Q:** How does periodically resetting the learning rate in SGDR improve generalization compared to monotonically decaying schedules?

**A:** Standard step decay or polynomial decay reduce the learning rate monotonically, causing the optimizer to converge progressively toward a local minimum. **SGDR** (Stochastic Gradient Descent with Warm Restarts) periodically resets the learning rate to its maximum value following a cosine decay within each cycle (Loshchilov & Hutter, 2017):

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)$$

where $T_{\text{cur}}$ is the number of steps since the last restart and $T_i$ is the total steps in the current cycle.

The restart mechanism serves two purposes. First, after each cooldown to $\eta_{\min}$, the model is in a local region of the loss surface; the restart launches a large step that can escape the current basin and explore a different region. Second, the ensemble effect: by snapshotting the model weights at each cycle minimum, one can ensemble multiple models at effectively zero extra training cost — **Snapshot Ensembles** (Huang et al., 2017) exploited this to achieve competitive accuracy without multiple independent training runs.

Warm restarts also naturally produce a cosine annealing schedule when used with a single cycle (no restarts), which is now standard in practice. The key advantage over step decay is the smooth continuous schedule, which avoids the sudden loss spikes that can accompany discrete learning rate drops.

---

### Q7 [Advanced] Characterize super-convergence and the conditions under which it emerges

**Q:** Under what training regime can a network converge in far fewer iterations than standard schedules permit, and what explains this phenomenon?

**A:** **Super-convergence** refers to the phenomenon where a model trains in an order of magnitude fewer iterations than standard training when using a large maximum learning rate with a cyclic schedule (Smith & Topin, 2019). Rather than requiring dozens of epochs with carefully decayed learning rates, super-convergence trains to comparable accuracy in a single cycle with a learning rate that is 10–100× larger than conventional practice.

Smith & Topin (2019) attributed super-convergence to a regularization effect of large learning rates: large steps prevent the optimizer from converging to sharp minima, effectively acting as regularization. This reduces the need for other regularizers (dropout, weight decay), and they found that reducing these during the large-learning-rate phase further enabled super-convergence.

The phenomenon is more readily observed in specific settings: relatively small datasets, networks with batch normalization, and residual connections. On large-scale datasets like ImageNet, super-convergence is harder to reproduce reliably. Smith & Topin (2019) demonstrated it on CIFAR-10 with ResNets, achieving test accuracy above $93\%$ in 10,000 iterations compared to the standard ~100,000. The **learning rate range test** — ramping the LR from a small to a large value across a short run and observing the loss — is their proposed diagnostic for identifying the appropriate maximum learning rate.

---

### Q8 [Advanced] Analyze the relationship between batch size and learning rate at scale

**Q:** Why does increasing batch size require adjusting the learning rate, and what are the practical limits of this scaling?

**A:** When training with **large minibatches**, the gradient estimate has lower variance than with small batches: the standard deviation of the minibatch gradient scales as $\sigma / \sqrt{B}$ where $B$ is the batch size. This means a gradient step with a large batch is more accurate but covers less diversity per unit of wall time. Goyal et al. (2017) empirically established the **linear scaling rule**: when multiplying the batch size by $k$, multiply the learning rate by $k$ as well. Intuitively, $k$ small-batch steps cover a similar region of parameter space as one large-batch step with $k\times$ learning rate. Using this rule and a 5-epoch linear warmup, they trained ResNet-50 on ImageNet in one hour with a batch size of 8,192, matching the accuracy of the standard 256-batch 90-epoch training.

The linear scaling rule holds approximately in the **noise-dominated regime** where the gradient signal is much smaller than the noise. It breaks in the **curvature-dominated regime** (very large batches), where the step size is limited by the sharpest directions of the loss surface rather than gradient noise. In this regime, further increasing $B$ and $\eta$ causes divergence. The critical batch size at which the transition occurs depends on the dataset and architecture; for ImageNet with ResNet-50, degradation becomes visible above roughly $B = 16{,}384$.

Practical large-batch training thus requires: linear warmup (Goyal et al., 2017), batch normalization without reducing the effective per-device batch size too much, and sometimes ghost batch normalization or layer normalization as alternatives when distributed per-device batches are tiny.

---

## Regularization & Generalization

### Q9 [Basic] Describe how dropout prevents overfitting in neural networks

**Q:** What does dropout do to the network during training, and why does this reduce overfitting?

**A:** **Dropout** randomly sets each neuron's activation to zero with probability $p$ during each forward pass of training (Srivastava et al., 2014). The neuron's output is scaled by $\frac{1}{1-p}$ to preserve the expected activation magnitude (inverted dropout). During inference, all neurons are active with no scaling.

The key intuition is that dropout prevents neurons from co-adapting: a neuron cannot rely on specific other neurons always being present, so it must learn features that are individually useful. Srivastava et al. (2014) showed that this is equivalent to training an exponential number of different network architectures and averaging their predictions — a form of implicit model ensembling. They demonstrated reductions in test error on vision, speech, and text tasks, with $p = 0.5$ being a robust default for hidden layers and $p = 0.8$ for input layers.

In modern practice, dropout is used less in convolutional networks (where spatial dropout or weight decay is preferred) but remains widely used in transformer attention layers and feed-forward layers. The optimal dropout rate depends on model size: larger models with more capacity to overfit benefit from higher $p$.

---

### Q10 [Advanced] Explain why proper weight decay is more principled than L2 regularization for adaptive optimizers

**Q:** Beyond the AdamW correction, what theoretical justification makes weight decay the preferred regularization for neural networks trained with adaptive methods?

**A:** The theoretical grounding connects to the **maximum a posteriori (MAP)** interpretation of regularization. Adding an L2 penalty $\frac{\lambda}{2}\|\theta\|^2$ to the loss corresponds to placing an isotropic Gaussian prior $\mathcal{N}(0, 1/\lambda)$ on the weights and finding the MAP estimate under this prior. With SGD, the resulting update is equivalent to weight decay, and the regularization is correctly applied relative to the gradient signal.

With adaptive optimizers, this Bayesian interpretation breaks because the effective learning rate is scaled per-parameter by $1/(\sqrt{\hat{v}_t} + \epsilon)$. The implicitly defined prior under Adam with L2 is no longer isotropic — parameters with larger historical gradients have their prior effectively weakened. This means the regularization is strongest on rarely-updated parameters (which may not need it) and weakest on frequently-updated parameters (which are most likely to overfit).

Decoupled weight decay as in AdamW ensures the prior remains isotropic: every parameter is shrunk toward zero by the same fractional amount $\lambda$ per step, regardless of gradient history (Loshchilov & Hutter, 2019). From an optimization geometry perspective, this corresponds to performing gradient descent within a ball constraint rather than adding a penalty to the gradient, which is a more principled regularization. The difference is most pronounced for large-scale models where some parameters are updated very frequently (attention projections) and others infrequently (embedding rows for rare tokens).

---

### Q11 [Basic] Identify when gradient clipping is necessary and how to set the clipping threshold

**Q:** What training pathologies motivate gradient clipping, and how should the clipping norm be calibrated?

**A:** **Gradient clipping** caps the norm of the gradient vector before applying an optimizer update. The most common form is global norm clipping: if $\|\nabla_\theta\mathcal{L}\|_2 > \tau$, the gradient is rescaled to $\tau \cdot \nabla_\theta\mathcal{L} / \|\nabla_\theta\mathcal{L}\|_2$. Pascanu et al. (2013) introduced this in the context of recurrent neural networks, where gradients can grow exponentially through time due to **exploding gradients** — repeated matrix multiplications through many unrolled timesteps drive the gradient norm toward infinity, causing catastrophically large parameter updates.

Beyond RNNs, gradient clipping is beneficial in any deep architecture trained with large learning rates, during fine-tuning when the loss surface has sharp curvature, and in mixed-precision training where FP16 can produce incorrect gradient scales. The clipping threshold $\tau$ is typically set by monitoring gradient norms during early training: a value of 1.0 is the common default, but setting it at the 95th percentile of early-training gradient norms is a principled approach. If clipping is triggered on more than a small fraction of steps, it suggests the learning rate is too large or the gradient accumulation length is too long.

---

### Q12 [Basic] Explain how label smoothing affects training dynamics and model calibration

**Q:** What does label smoothing change in the cross-entropy loss, and how does it affect the confidence of model predictions?

**A:** Standard cross-entropy training uses **hard labels** — one-hot vectors that assign probability 1 to the correct class and 0 to all others. This encourages the model to output logits that drive the correct-class softmax probability toward 1 and all others toward 0, producing an infinitely deep well in the loss surface. **Label smoothing** (Müller et al., 2019) replaces the hard target with a soft distribution:

$$q_i = (1 - \epsilon)\cdot\mathbf{1}[i = y] + \frac{\epsilon}{K}$$

where $\epsilon$ is the smoothing factor (typically 0.1) and $K$ is the number of classes. The correct class target becomes $1 - \epsilon + \epsilon/K$ and incorrect classes receive $\epsilon/K$ probability mass.

The practical effect is that the model is discouraged from becoming overconfident: it cannot achieve zero loss by making the logit gap arbitrarily large. Müller et al. (2019) showed that label smoothing produces better-calibrated models — the softmax probabilities are more aligned with empirical accuracy — and improves generalization on image classification. However, they also found that label smoothing hurts knowledge distillation: a teacher trained with label smoothing produces softer, less informative probability distributions, reducing the information transferred to the student. This highlights that label smoothing is a regularizer that should be removed or reduced when the model is used as a teacher.

---

## Training Efficiency

### Q13 [Basic] Describe mixed precision training and why it preserves accuracy despite using lower precision

**Q:** What numerical formats does mixed precision training use, and how does it avoid the accuracy loss that lower precision typically causes?

**A:** **Mixed precision training** (Micikevicius et al., 2018) uses 16-bit floating point (FP16) for forward and backward passes and 32-bit (FP32) for weight updates. Modern hardware — NVIDIA Tensor Cores, Google TPUs — achieves 2–8× higher throughput for FP16 matrix operations and consumes half the memory, enabling larger batches or models.

Three techniques preserve accuracy despite reduced precision. First, a **master copy** of the weights is maintained in FP32; after each gradient update, the FP16 weights are derived from it. Small gradient updates that would underflow to zero in FP16 accumulate correctly in FP32. Second, **loss scaling** multiplies the loss by a large constant (e.g., $2^{15}$) before the backward pass, shifting gradient values out of the FP16 subnormal range where they would underflow. The gradients are then unscaled before the weight update. Third, a subset of operations (batch normalization statistics, loss computation) are kept in FP32 to preserve numerical stability.

Micikevicius et al. (2018) demonstrated that mixed precision matches FP32 accuracy on image classification (ResNet-50, ImageNet), speech recognition, and language modelling with no architectural changes. Modern training frameworks (PyTorch AMP, JAX) automate loss scaling and the FP16/FP32 cast boundaries, making mixed precision training the default in practice.

---

### Q14 [Advanced] Analyze the computational trade-off in gradient checkpointing

**Q:** How does gradient checkpointing reduce peak memory usage during backpropagation, and when is the added compute cost justified?

**A:** During standard backpropagation, all intermediate activations from the forward pass must be stored in memory until their corresponding backward pass computation — memory cost is $O(L)$ for a network with $L$ layers. **Gradient checkpointing** (Chen et al., 2016) reduces this by storing only a subset of activations (the **checkpoints**) and recomputing intermediate activations from the nearest checkpoint during the backward pass.

The optimal checkpointing strategy stores activations at intervals of $\sqrt{L}$ steps, requiring $O(\sqrt{L})$ memory at the cost of one additional forward pass through each segment. Chen et al. (2016) showed that this reduces activation memory from $O(L)$ to $O(\sqrt{L})$ while increasing compute by roughly $33\%$ (since each segment is computed twice). For very deep networks or large sequence lengths in transformers, the memory saving far outweighs the compute overhead.

The cost-benefit is context-dependent. For a transformer processing long sequences, activation memory scales as $O(B \cdot T \cdot d)$ where $B$ is batch size, $T$ is sequence length, and $d$ is model dimension — gradient checkpointing can be the difference between fitting a model in GPU memory and running out of memory entirely. It is less justified when memory is not the bottleneck (e.g., small models on high-memory hardware) because the $33\%$ compute overhead directly reduces training throughput. In practice, frameworks like PyTorch offer per-layer checkpointing control, allowing practitioners to selectively checkpoint the most memory-intensive layers (attention blocks) while leaving lighter layers uncheckpointed.

---

### Q15 [Advanced] Evaluate gradient accumulation as a substitute for large-batch training

**Q:** How does accumulating gradients across micro-batches simulate training with a larger batch, and what effects are not reproduced?

**A:** **Gradient accumulation** performs $k$ forward-backward passes on consecutive micro-batches of size $B_\mu$ before applying a single optimizer update, effectively simulating a batch of size $k \cdot B_\mu$ without storing all samples in memory simultaneously. The gradients from each micro-batch are summed (or averaged) before the update step. This enables large-batch training on hardware with limited memory, or when the desired batch size exceeds what a single GPU can hold.

The simulation is exact for the gradient update itself: $\nabla\mathcal{L}(B_\mu \cup \ldots \cup B_\mu^{(k)}) = \frac{1}{k}\sum_{i=1}^k \nabla\mathcal{L}(B_\mu^{(i)})$. However, **batch normalization** is not faithfully reproduced: BatchNorm computes mean and variance statistics over each micro-batch independently, not over the full effective batch. This means the normalization statistics differ from true large-batch training and the running statistics used at inference are computed on micro-batches. For models using LayerNorm or RMSNorm (most transformers), this issue does not arise.

A subtler discrepancy is that gradient accumulation does not reduce wall time proportionally to batch size — it performs $k$ sequential forward-backward passes, so throughput scales linearly with the number of accumulation steps. True large-batch training with data parallelism reduces wall time by parallelizing across devices. Gradient accumulation is therefore a memory solution, not a throughput solution, and should not be conflated with distributed large-batch training for wall-time efficiency.

---

### Q16 [Advanced] Explain how activation function choice affects gradient flow and training stability

**Q:** What properties of an activation function determine whether gradients propagate well through deep networks, and how have modern choices addressed ReLU's limitations?

**A:** The ideal activation function for gradient flow should be non-saturating (gradient does not vanish for large inputs), smooth (providing useful gradient signal in transition regions), and cheap to compute. **ReLU** ($f(x) = \max(0, x)$) solved the saturation problem of sigmoid and tanh: its gradient is either 0 or 1, so it does not saturate for positive activations. However, ReLU has the **dying ReLU** problem: if a neuron's pre-activation is consistently negative (e.g., due to a large negative bias from a bad update), its gradient is permanently zero and the neuron never recovers.

**GELU** (Gaussian Error Linear Unit) $f(x) = x\Phi(x)$, where $\Phi$ is the standard normal CDF, avoids the hard zero-gate of ReLU by softly gating the input based on its magnitude (Hendrycks & Gimpel, 2016). This allows small negative activations to pass through, preventing the dying neuron problem, and produces smoother gradients. GELU has become the dominant activation in transformer architectures.

**Swish** $f(x) = x \cdot \sigma(\beta x)$ (Ramachandran et al., 2017) is a self-gated variant that generalizes smoothly between linear ($\beta \to 0$) and ReLU-like ($\beta \to \infty$) behavior. Both GELU and Swish share the property of being unbounded above (preventing vanishing gradients for large positive inputs), smooth (providing useful gradient signal in transition regions), and having a non-monotone character — the slight dip below zero for small negative inputs acts as a soft gate. For very deep networks, the activation function interacts with initialization and normalization placement; residual connections and LayerNorm largely mitigate the sensitivity to activation choice, which is why both GELU and ReLU work well in transformers but the difference matters more in plain deep networks.

---

## Normalization & Distributed Training

### Q17 [Basic] Compare BatchNorm and LayerNorm and identify when each is appropriate

**Q:** What dimension does each normalization method operate over, and why does this make BatchNorm unsuitable for certain tasks?

**A:** **Batch Normalization** (Ioffe & Szegedy, 2015) normalizes each feature dimension across the batch: for a mini-batch of $B$ samples, it computes the mean and variance of each neuron across the $B$ activations, then normalizes. The learned affine parameters $\gamma$ and $\beta$ per feature restore representational capacity. BatchNorm's statistics depend on batch size: small batches produce noisy estimates, and performance degrades sharply when $B < 16$. More importantly, BatchNorm creates dependencies across examples in the same batch, making it incompatible with auto-regressive inference (where the batch size is 1) and with RNNs (where different timesteps would be mixed).

**Layer Normalization** (Ba et al., 2016) instead normalizes across the feature dimension for each individual example independently, computing mean and variance over all $d$ features of a single sample. This makes the statistics independent of batch size and of other examples, making it suitable for RNNs, transformers, and variable-length sequences. The tradeoff is that LayerNorm is less effective for CNNs on images, where spatial structure makes per-example normalization overly aggressive.

In practice: BatchNorm is preferred for convolutional image models with reasonable batch sizes; LayerNorm is standard for transformer-based models across NLP and vision.

---

### Q18 [Advanced] Analyze why RMSNorm has replaced LayerNorm in many large language model architectures

**Q:** What does RMSNorm remove from the LayerNorm computation, and what justifies this simplification both theoretically and empirically?

**A:** **RMSNorm** (Zhang & Sennrich, 2019) is a simplification of LayerNorm that normalizes using only the root mean square of activations, without centering by the mean:

$$\bar{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2}$$

This eliminates the mean subtraction (re-centering) and the bias parameter $\beta$ present in standard LayerNorm. Zhang & Sennrich (2019) showed that the re-centering operation in LayerNorm contributes little to training stability, while the re-scaling (normalization by scale) provides most of the benefit. The empirical speedup is approximately $7$–$64\%$ compared to LayerNorm due to fewer operations and better hardware efficiency.

The theoretical justification is that the scale of activations, not their mean, is what destabilizes gradient flow in deep networks. RMSNorm imposes a spherical constraint on the activation vector — enforcing $\|\mathbf{x}\|_2 = \sqrt{d}$ — which is sufficient for stable training. The mean subtraction in LayerNorm is motivated by an analogy to batch statistics, but in the feature dimension this analogy is weaker.

Modern LLMs such as LLaMA, Gemma, and Mistral use RMSNorm over LayerNorm, typically applied as **pre-norm** (before the attention and FFN sublayers) rather than post-norm. Pre-norm placement prevents gradient vanishing through deep residual stacks, as the residual stream is never scaled down before being passed to the next layer (He et al., 2016).

---

### Q19 [Basic] Distinguish the three primary parallelism strategies in distributed deep learning

**Q:** How do data parallelism, model parallelism, and pipeline parallelism divide the work of training across devices?

**A:** **Data parallelism** replicates the full model on every device and assigns a different subset of the mini-batch to each replica. Gradients are synchronized (all-reduced) across devices after each backward pass, and the model weights are updated identically on every replica. Data parallelism scales batch size proportionally to the number of devices and is the simplest strategy to implement, but requires that the full model fits in a single device's memory.

**Model parallelism** (tensor parallelism) partitions the model's weight matrices across devices. In Megatron-LM (Shoeybi et al., 2019), attention heads and feed-forward weight rows are split column-wise across devices, with an all-reduce at the end of each transformer sublayer. This enables training models larger than a single device's memory but introduces frequent inter-device communication at each layer boundary.

**Pipeline parallelism** assigns different layers to different devices, so device 1 runs layers 1–$k$, device 2 runs layers $k{+}1$–$2k$, and so on. Data flows as a stream of micro-batches: device 1 processes micro-batch 1, then starts micro-batch 2 while device 2 processes micro-batch 1. GPipe (Huang et al., 2019) formalized this as synchronous pipeline parallelism with gradient accumulation across micro-batches. The limitation is **pipeline bubble**: devices are idle at the start and end of each batch when the pipeline is filling or draining — bubble fraction scales as $(D-1)/(D-1+M)$ where $D$ is pipeline depth and $M$ is number of micro-batches.

In practice, large-scale training (e.g., GPT-3-scale models) combines all three strategies: data parallelism across nodes, model parallelism within a node's GPUs, and pipeline parallelism across nodes.

---

### Q20 [Advanced] Explain how the ZeRO optimizer stages reduce memory redundancy in data-parallel training

**Q:** What memory is redundant across data-parallel workers, and how do ZeRO's three stages progressively eliminate this redundancy?

**A:** In standard data-parallel training, each of $N$ workers holds a complete copy of: (1) the optimizer states (e.g., Adam's $m_t$ and $v_t$, typically $2\times$ the parameter count in FP32), (2) the gradients (same size as parameters), and (3) the parameters themselves. For a model with $\Psi$ parameters, each worker uses $16\Psi$ bytes ($4\Psi$ parameters in FP32 + $8\Psi$ Adam states + $4\Psi$ gradients), with total memory across $N$ workers being $16N\Psi$. All of this is redundant — every worker holds identical copies (Rajbhandari et al., 2020).

**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al., 2020) eliminates this redundancy in three stages:

- **Stage 1** partitions the optimizer states across $N$ workers. Each worker holds only $1/N$ of the Adam $m_t$ and $v_t$ states. Memory per worker reduces to $4\Psi + 12\Psi/N$.
- **Stage 2** additionally partitions the gradients. After each backward pass, gradients are reduced to the worker responsible for the corresponding parameter shard. Memory per worker reduces to $4\Psi + 12\Psi/N$ (gradients are discarded after the reduce-scatter).
- **Stage 3** partitions the parameters themselves. Workers only store $1/N$ of the parameters; before each forward or backward pass, parameters are gathered via all-gather from the respective owners. Memory per worker becomes $16\Psi/N$, a factor of $N$ reduction over DDP.

Stage 3 adds an all-gather communication per forward and backward pass, but since communication and computation can overlap, the throughput overhead is modest. Rajbhandari et al. (2020) demonstrated training a 170-billion-parameter model on 400 V100 GPUs with ZeRO Stage 3, which would be impossible with standard DDP. ZeRO is now the foundation of DeepSpeed and is integrated into PyTorch FSDP (Fully Sharded Data Parallel).

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Adam first/second moment estimates | Optimization Algorithms |
| Q2 | Advanced | AdamW vs L2 regularization in Adam | Optimization Algorithms |
| Q3 | Advanced | RAdam variance rectification | Optimization Algorithms |
| Q4 | Advanced | Second-order vs first-order optimization | Optimization Algorithms |
| Q5 | Basic | Learning rate warmup | Learning Rate Scheduling |
| Q6 | Advanced | Cosine annealing with warm restarts (SGDR) | Learning Rate Scheduling |
| Q7 | Advanced | Super-convergence | Learning Rate Scheduling |
| Q8 | Advanced | Batch size and learning rate scaling | Learning Rate Scheduling |
| Q9 | Basic | Dropout regularization | Regularization & Generalization |
| Q10 | Advanced | Weight decay vs L2 for adaptive optimizers | Regularization & Generalization |
| Q11 | Basic | Gradient clipping | Regularization & Generalization |
| Q12 | Basic | Label smoothing and model calibration | Regularization & Generalization |
| Q13 | Basic | Mixed precision training | Training Efficiency |
| Q14 | Advanced | Gradient checkpointing memory trade-off | Training Efficiency |
| Q15 | Advanced | Gradient accumulation vs large-batch training | Training Efficiency |
| Q16 | Advanced | Activation functions and gradient flow | Training Efficiency |
| Q17 | Basic | BatchNorm vs LayerNorm | Normalization & Distributed Training |
| Q18 | Advanced | RMSNorm vs LayerNorm in LLMs | Normalization & Distributed Training |
| Q19 | Basic | Data, model, and pipeline parallelism | Normalization & Distributed Training |
| Q20 | Advanced | ZeRO optimizer memory reduction | Normalization & Distributed Training |

## Resources

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
