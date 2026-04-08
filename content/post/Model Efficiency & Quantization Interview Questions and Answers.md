---
title: "Model Efficiency & Quantization: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-08'
categories:
  - Interview
tags:
  - Deep Learning
  - Quantization
  - Model Compression
toc: true
---

## Quantization Fundamentals

### Q1 [Basic] Explain the key design choices in uniform quantization

**Q:** What numerical decisions determine how a floating-point weight tensor is mapped to integers, and how do symmetric and asymmetric schemes differ?

**A:** **Uniform quantization** maps a floating-point range $[x_{\min}, x_{\max}]$ to $2^b$ evenly spaced integer values using two parameters: a **scale factor** $s$ and a **zero-point** $z$. The bit-width $b$ determines the number of representable levels — INT8 provides 256 levels, INT4 provides 16. The scale is set by the tensor's range, and the clamp operation introduces **clipping error** for values outside the chosen range; calibration-time decisions about where to set $x_{\min}$ and $x_{\max}$ trade clipping against rounding error (Nagel et al., 2021).

**Symmetric** quantization sets $z = 0$ and clips symmetrically ($x_{\min} = -x_{\max}$), which simplifies computation: zero maps exactly to the integer 0 and the zero-point can be dropped from the multiply-accumulate path. **Asymmetric** quantization allows $z \neq 0$ to shift the quantization grid to cover the actual data range more efficiently for skewed distributions — post-ReLU activations, for example, are always non-negative and benefit from an asymmetric grid that concentrates levels in $[0, x_{\max}]$. The downside is a zero-point correction term in every matrix multiply.

Granularity is a third dimension: **per-tensor** quantization uses a single $s$ and $z$ for the entire weight matrix, while **per-channel** quantization uses separate values per output channel. Different output channels can have very different dynamic ranges, so per-channel calibration reduces quantization error substantially at negligible inference cost. Nagel et al. (2021) provide a comprehensive analysis of the interactions between these choices across CNN and transformer architectures.

---

### Q2 [Basic] Distinguish post-training quantization from quantization-aware training

**Q:** How do PTQ and QAT differ in when quantization is applied relative to training, and what determines which approach is appropriate?

**A:** **Post-training quantization** (PTQ) applies quantization to a fully trained floating-point model without any gradient-based optimization. A small calibration dataset (typically a few hundred to a few thousand samples) is passed through the model to collect activation statistics for setting scale and zero-point per layer. PTQ requires no access to the original training pipeline and produces a quantized model in minutes. The limitation is that quantization error cannot be compensated by weight updates, so accuracy degrades more than QAT — especially below 8 bits — and sensitive layers may require more careful calibration (Nagel et al., 2021).

**Quantization-aware training** (QAT) inserts **fake quantization** nodes into the forward pass during training: each tensor is quantized and immediately dequantized, injecting quantization error into the loss while keeping the computation in floating point for gradient propagation. Because the rounding operation has zero gradient, the **straight-through estimator** (STE) passes gradients through the clamp as if it were the identity (Jacob et al., 2018). This allows the optimizer to learn weights that are more tolerant of quantization, typically recovering most of PTQ's accuracy loss at INT8 and enabling usable accuracy at INT4.

In practice, PTQ is preferred when training data is unavailable, when targeting INT8 (where PTQ accuracy loss is modest), or when rapid deployment is needed. QAT is worth the infrastructure cost when targeting INT4 or lower, or when task-specific accuracy requirements are tight.

---

### Q3 [Advanced] Explain why activation quantization is harder than weight quantization

**Q:** What properties of neural network activations create quantization challenges that do not arise for weights, and how does this problem intensify in large language models?

**A:** Weights are **static** after training: their distribution is fully known at quantization time, and optimal per-channel scale factors can be computed exactly from the saved checkpoints. Activations are **dynamic**: their distribution changes with every input. A scale factor calibrated on a representative dataset may be mismatched for out-of-distribution inputs, causing either excessive clipping (large values saturate) or excessive rounding error (scale set too conservatively, wasting bit-width on the normal range). This input-dependence forces either conservative static calibration or expensive per-token dynamic quantization at inference time, each carrying accuracy or latency costs.

More fundamentally, activation distributions in transformer models can be highly non-Gaussian. Dettmers et al. (2022) showed that in OPT and BLOOM models above approximately 6B parameters, a small fraction (roughly $0.1\%$) of activation dimensions consistently exhibit values 100× larger than the rest — **systematic outlier channels** that appear reliably across all inputs once the model reaches a critical scale. Quantizing the entire tensor to the scale of these outliers wastes nearly all representable integers on the normal range; quantizing to the normal range causes outliers to saturate. Neither is acceptable at INT8.

A third challenge is that per-channel quantization — highly effective for weights — is harder to apply to activations indexed by (batch, token, channel): different tokens in the same batch may have very different per-channel statistics, making a static per-channel calibration unreliable. The combination of dynamic range, input-dependence, and systematic outliers explains why activations require specialized solutions (SmoothQuant, LLM.int8()) rather than the straightforward per-channel PTQ that works well for weights.

---

### Q4 [Advanced] Describe how GPTQ achieves accurate weight-only quantization of billion-parameter models

**Q:** What optimization principle does GPTQ apply to minimize layer-wise quantization error, and what algorithmic changes make it tractable at 175B-parameter scale?

**A:** GPTQ is a one-shot PTQ method grounded in the **Optimal Brain Quantization** (OBQ) framework (Frantar et al., 2022). OBQ extends the Optimal Brain Surgeon (OBS) approach from pruning to quantization: when weight $w_q$ is rounded to $\hat{w}_q$, the remaining unquantized weights in the same layer are updated to compensate for the introduced error. The optimal update for the remaining weights $F$ when quantizing index $q$ is:

$$\delta_F = -\frac{w_q - \hat{w}_q}{[\mathbf{H}_F^{-1}]_{qq}} \cdot (\mathbf{H}_F^{-1})_{:,q}$$

where $\mathbf{H}_F = 2\mathbf{X}\mathbf{X}^T$ is the Hessian of the layer's output reconstruction error with respect to weights (for a linear layer with calibration input $\mathbf{X}$). This greedy per-weight compensation exploits second-order information to choose better quantization targets.

Naïve OBQ scales as $O(d^3)$ per layer (from Hessian inversion) and processes one weight at a time — intractable for layers with millions of parameters. GPTQ introduces two modifications. First, all weights within a column block of width 128 are quantized simultaneously, accumulating the inverse-Hessian updates in a **lazy batch**: Hessian inverse rows needed for future updates are precomputed once and reused, reducing redundant computation. Second, rather than dynamically reordering weights by inverse-Hessian diagonal (as OBQ does), GPTQ processes weights in column order, which is hardware-friendly and cache-efficient. Frantar et al. (2022) quantized OPT-175B to INT4 in approximately 4 GPU-hours on a single A100, with less than 1 perplexity point of degradation on WikiText-2.

---

## Quantization for Large Language Models

### Q5 [Advanced] Explain how AWQ identifies and protects salient weights during quantization

**Q:** What observation motivates AWQ's selective treatment of weight channels, and how are the optimal protection scales determined without retraining?

**A:** **AWQ** (Activation-aware Weight Quantization) is built on the finding that weight channels corresponding to large-magnitude activations contribute disproportionately to quantization error: approximately $1\%$ of weight channels dominate the accuracy loss when quantized to INT4 (Lin et al., 2023). The naive fix — keeping these salient channels in FP16 while quantizing the rest — is hardware-inefficient because mixing precision at fine channel granularity introduces irregular memory access.

AWQ instead applies a per-channel **activation-aware scale** before quantization. Multiplying a weight channel by $s > 1$ reduces its relative quantization error ($\Delta w / w$ decreases as $w$ grows), effectively allocating more precision to it without changing the integer format. The corresponding activation dimension is divided by $s$ to preserve the matrix product. The scale is applied offline to the weight matrix and absorbed into the preceding normalization layer's affine parameters, so inference cost is unchanged.

The optimal per-channel scales are found by minimizing output reconstruction error on a calibration set:

$$s^* = \arg\min_{s} \|\mathbf{W}_q(\mathbf{s})\mathbf{x} - \mathbf{W}\mathbf{x}\|^2$$

where $\mathbf{W}_q(\mathbf{s})$ denotes $\mathbf{W}$ quantized after scaling by $s$. Lin et al. (2023) parameterized $s_j = \bar{a}_j^\alpha$ where $\bar{a}_j$ is the mean activation magnitude of channel $j$ and $\alpha \in [0, 1]$ is grid-searched over roughly 20 values — making the search efficient and requiring only a small calibration set. On LLaMA-7B with INT4, AWQ achieved lower perplexity than GPTQ at comparable hardware throughput, showing that activation-guided scaling is a more effective proxy for quantization sensitivity than pure second-order weight information.

---

### Q6 [Basic] Contrast weight-only and weight-activation quantization in terms of deployment trade-offs

**Q:** When would a practitioner prefer quantizing only weights versus both weights and activations, and what hardware characteristics drive this choice?

**A:** **Weight-only quantization** (W4A16 or W8A16) stores model weights in low precision but performs matrix multiplications in FP16, dequantizing weights on the fly before the operation. The primary gain is **memory bandwidth reduction**: in LLM auto-regressive decoding with batch size 1, the bottleneck is loading weight matrices from GPU HBM to on-chip compute units — weight-only quantization at INT4 reduces this transfer by $4\times$, directly improving token throughput. Methods such as GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2023) follow this strategy, enabling larger models on the same hardware with minimal accuracy loss.

**Weight-activation quantization** (W8A8) quantizes both tensors, enabling the use of integer arithmetic units — INT8 tensor cores on NVIDIA GPUs offer higher peak throughput than FP16 units, beneficial when the operation is **compute-bound**. This regime occurs at large batch sizes where the arithmetic intensity (FLOPs per byte) is high enough to saturate the ALUs rather than the memory bus. Methods such as SmoothQuant (Xiao et al., 2022) and LLM.int8() (Dettmers et al., 2022) target this regime, enabling high-throughput serving of many concurrent requests. The cost is the added difficulty of quantizing activations accurately (Q3).

In practice, single-user LLM inference is strongly memory-bandwidth-bound (W4A16 preferred), while serving many concurrent requests with large batches is often compute-bound (W8A8 preferred). The transition point depends on model size, hardware generation, and serving batch size.

---

### Q7 [Advanced] Describe how SmoothQuant migrates quantization difficulty from activations to weights

**Q:** What is the mathematical identity at the core of SmoothQuant, and why does shifting the scale from activations to weights solve the outlier problem?

**A:** **SmoothQuant** exploits a key structural observation: the outlier channels in transformer activations are **consistent across tokens** — the same feature dimensions are large for all inputs (Xiao et al., 2022). This consistency means the outlier pattern can be characterized offline and countered with a fixed per-channel transformation. For any per-channel scale vector $\mathbf{s} \in \mathbb{R}^{C_{\text{in}}}$, the linear layer output is preserved exactly by:

$$\mathbf{Y} = \mathbf{X}\mathbf{W} = (\mathbf{X}\operatorname{diag}(\mathbf{s})^{-1}) \cdot (\operatorname{diag}(\mathbf{s})\mathbf{W})$$

Setting $s_j$ proportional to the observed magnitude of channel $j$ in activations, the transformed activation $\hat{\mathbf{X}} = \mathbf{X}\operatorname{diag}(\mathbf{s})^{-1}$ has outlier channels scaled down to the same range as other channels, while the transformed weight $\hat{\mathbf{W}} = \operatorname{diag}(\mathbf{s})\mathbf{W}$ has those columns scaled up. Weights are already easy to quantize, and the per-channel scale for weights can simply be absorbed into the quantization scale — the difficulty migrates to a medium that handles it well.

The migration strength is controlled by a per-channel parameter $\alpha$:

$$s_j = \frac{\max(|\mathbf{X}_{:,j}|)^\alpha}{\max(|\mathbf{W}_{j,:}|)^{1-\alpha}}$$

with $\alpha = 0.5$ as a robust default. The scales are folded into the preceding LayerNorm's affine parameters offline, adding zero inference overhead. Xiao et al. (2022) demonstrated near-lossless INT8 inference for OPT-175B and BLOOM-176B on both perplexity and downstream zero-shot benchmarks, achieving the first practical W8A8 quantization of 176B-scale models.

---

### Q8 [Advanced] Explain how LLM.int8() handles activation outliers in transformer models

**Q:** What computational decomposition does LLM.int8() use to enable accurate INT8 inference, and why does this approach become necessary specifically for large models?

**A:** **LLM.int8()** (Dettmers et al., 2022) addresses the outlier problem through **mixed-precision decomposition**: rather than quantizing the entire activation tensor, it separates the matrix multiplication based on whether each input feature dimension contains an outlier. Feature dimensions whose activation values exceed a threshold (typically $|x| > 6$) are extracted into an **outlier matrix** $\mathbf{X}_o \in \mathbb{R}^{B \times C_o}$; the remainder forms $\mathbf{X}_r \in \mathbb{R}^{B \times C_r}$. The weight matrix is partitioned correspondingly into $\mathbf{W}_o$ and $\mathbf{W}_r$:

$$\mathbf{Y} = \mathbf{X}_o \mathbf{W}_o^T + \operatorname{Int8}(\mathbf{X}_r) \cdot \operatorname{Int8}(\mathbf{W}_r)^T$$

The outlier portion (roughly $0.1\%$ of feature dimensions) is computed in FP16; the remaining $\sim 99.9\%$ uses INT8 tensor cores. The INT8 path dominates cost; the FP16 path adds modest overhead. Because outlier channels are persistent — Dettmers et al. (2022) showed they emerge as a phase transition at model scale ($\gtrsim 6$B parameters) and are consistent across all inputs — the outlier set can be determined once from calibration data and reused at inference.

The practical motivation is memory: a FP16 175B-parameter model requires approximately 350 GB of GPU memory; LLM.int8() reduces weight storage to roughly 175 GB (INT8), enabling inference of 176B models on 4× 48 GB GPUs versus 8× in FP16. Dettmers et al. (2022) reported less than $1\%$ performance degradation across OPT and BLOOM models on standard zero-shot benchmarks.

---

## Pruning & Sparsity

### Q9 [Basic] Contrast structured and unstructured pruning in their impact on hardware utilization

**Q:** Why does achieving 90% sparsity through unstructured pruning often fail to deliver real inference speedup, while structured pruning at 50% sparsity does?

**A:** **Unstructured pruning** removes individual weights regardless of their positions, producing irregular zero patterns scattered throughout the weight matrices. Despite high sparsity ratios, modern GPUs and TPUs execute matrix multiplications on dense rectangular blocks — they do not skip computation over individual zeros unless the zeros happen to conform to a specific hardware-supported pattern (such as NVIDIA A100's 2:4 structured sparsity). Without special sparse kernels, a $90\%$-sparse dense matrix still occupies the same memory layout as the original and executes in approximately the same time (Han et al., 2016). Even with sparse format (CSR, CSC), the irregular indexing overhead and poor vectorization often reduce the speedup below what the sparsity ratio would suggest.

**Structured pruning** removes entire computational units — output channels, attention heads, FFN neurons, or whole transformer layers — producing a smaller **dense** model. A model with $50\%$ of its output channels removed runs the corresponding matrix multiplications at $50\%$ of the original cost on any hardware that supports dense operations, with no sparse format overhead. The pruned dimension simply disappears from the weight tensor shape.

The practical implication is that unstructured pruning is primarily a **storage compression** technique — reducing model size on disk and in memory — while structured pruning directly reduces latency and memory at inference time. For deployment on commodity hardware (CPUs, standard GPUs), structured pruning is the path to measurable speedup.

---

### Q10 [Advanced] Explain the lottery ticket hypothesis and what it reveals about neural network overparameterization

**Q:** What did Frankle and Carlin find about sparse subnetworks within dense networks, and why does the initialization of the subnetwork matter?

**A:** The **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019) states that a randomly initialized dense network contains a sparse subnetwork — the "winning ticket" — that, when trained in isolation starting from its **original initialization** (not re-initialized randomly), converges in a comparable number of steps to similar or better accuracy than the full network. Frankle & Carlin (2019) found these tickets via **iterative magnitude pruning**: train the network, prune the $p\%$ of weights with smallest final magnitude, reset remaining weights to their initialization values, and repeat. After several rounds, sparse subnetworks at $10$–$20\%$ of original size matched the full network's accuracy on MNIST and CIFAR-10.

The critical and surprising finding is that the **initial weight values matter**: taking the winning subnetwork but re-initializing it randomly instead of using the original initialization destroys the performance advantage. The winning ticket's initial values appear to encode a favorable optimization trajectory — perhaps better gradient signal early in training. This implies that the dense network's role is partially as a search procedure: it implicitly identifies which sparse substructure to preserve and which initialization is effective for it.

For larger networks (ResNet-50, language models), Frankle & Carlin (2019) found that the pure initialization reset no longer works: weights must be reset to their values at a small number of early training steps (**rewinding**) rather than initialization. This "late resetting" requirement suggests that the useful inductive bias for large networks is established in early training dynamics, not at initialization. The hypothesis has implications for understanding why overparameterization aids optimization — it increases the probability of containing a well-initialized winning ticket — and motivates research into finding sparse subnetworks before training rather than after.

---

### Q11 [Advanced] Describe how SparseGPT enables one-shot pruning of large language models

**Q:** What makes iterative pruning infeasible for LLMs, and what algorithmic principle does SparseGPT use to prune without any gradient updates?

**A:** Iterative pruning requires gradient-based fine-tuning after each pruning round to recover accuracy. For ResNets, this is standard practice, but for 100B+ parameter LLMs even a single epoch of fine-tuning requires processing the full pre-training corpus with backpropagation — computationally infeasible outside large research clusters, and often impractical even within them. **SparseGPT** (Frantar & Alistarh, 2023) removes the need for any gradient updates by solving, for each linear layer, a weight reconstruction problem: find sparse weights $\hat{\mathbf{W}}$ that minimize output reconstruction error on a small calibration set:

$$\min_{\hat{\mathbf{W}}} \|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_F^2 \quad \text{s.t.} \quad \|\hat{\mathbf{W}}\|_0 \leq k$$

This is solved using the same OBS/OBQ second-order framework as GPTQ (Q4): when a weight is pruned (set to zero), the remaining row weights are updated by the inverse-Hessian correction $\delta_F$ to compensate for the introduced error. Processing is done row-by-row, which is parallelizable, and uses the lazy batch-update trick to amortize Hessian operations across weight blocks.

Frantar & Alistarh (2023) demonstrated that OPT-175B can be pruned to $50\%$ unstructured sparsity with less than 1 perplexity point of degradation on WikiText-2, completing in approximately 4 GPU-hours on a single A100. At $2{:}4$ structured sparsity (the NVIDIA-native format that enables hardware speedup), SparseGPT achieved roughly 2 perplexity points of degradation — a usable accuracy level. They also showed that combining SparseGPT with quantization (sparse-quantized models) maintained competitive accuracy at higher compression ratios, suggesting that sparsity and quantization are complementary along different compression axes.

---

### Q12 [Advanced] Compare static and dynamic sparse training and how RigL advances the state of the art

**Q:** Why does training a sparse network from a random fixed mask underperform the dense-then-prune pipeline, and how does RigL overcome this without a dense training phase?

**A:** **Static sparse training** — fixing a random sparsity mask at initialization and training only the selected weights — typically underperforms the dense-then-prune pipeline at the same final sparsity. The lottery ticket hypothesis explains why: random sparse initialization lacks the favorable structure that magnitude pruning discovers after dense training. The winning ticket's value comes from both its topology and its initial weights; a random mask with reinitialized weights captures neither. Static sparse training thus starts from a structurally poor subgraph that is difficult to optimize even at high learning rates.

**Dynamic sparse training** alternates between sparse training steps and mask update steps that grow new connections while pruning existing ones. **RigL** (Rigging the Lottery, Evci et al., 2020) grows connections by computing the **instantaneous gradient** of the loss with respect to currently-zero weights during the sparse forward pass — weights with large gradients would reduce the loss significantly if activated. Connections are pruned by **weight magnitude**, since small active weights contribute little to the output. The mask is updated periodically every $\Delta T$ steps, with a cosine-decaying update fraction that starts large (aggressive topology exploration) and decreases as training converges.

Evci et al. (2020) showed that RigL at the same training FLOPs as standard dense training matches or outperforms iterative magnitude pruning with fine-tuning at high sparsity ($80$–$99\%$) on ImageNet with ResNets. At a $2\times$ FLOP budget, RigL surpasses the dense baseline at $90\%$ sparsity — the first sparse training method to exceed dense training quality at matching FLOP cost. The key advantage is discovering task-adaptive sparsity patterns during training rather than relying on the post-hoc magnitude ordering found by the pruning pipeline.

---

## Knowledge Distillation & Low-Rank Compression

### Q13 [Basic] Explain how soft-target knowledge distillation works and what the teacher provides beyond hard labels

**Q:** What does a teacher model's output distribution encode that one-hot labels discard, and how does temperature scaling affect the quality of the distillation signal?

**A:** **Knowledge distillation** (Hinton et al., 2015) trains a smaller student model to match the full output distribution of a larger teacher, rather than only the ground-truth hard label. A well-trained teacher assigns non-trivial probabilities to incorrect classes — e.g., for an image of a car, it might output $p(\text{car}) = 0.7$, $p(\text{truck}) = 0.2$, $p(\text{bus}) = 0.08$. These **soft targets** encode the teacher's knowledge about **inter-class similarity**: the large car/truck probability reflects that these categories are visually similar and often confused. Hard labels (car=1, all others=0) discard this relational information entirely. The student trained on soft targets learns not just "this is a car" but also "cars resemble trucks more than they resemble birds," which provides richer supervision.

The student minimizes a weighted combination of the cross-entropy with hard labels and the KL divergence from the teacher's softened distribution:

$$\mathcal{L} = (1-\lambda)\,\mathcal{L}_{\text{CE}}(y,\, \sigma(\mathbf{z}_s)) + \lambda T^2\,\mathcal{L}_{\text{KL}}(\sigma(\mathbf{z}_t/T),\, \sigma(\mathbf{z}_s/T))$$

where $T$ is a **temperature** parameter. At $T = 1$, the teacher's distribution is sharply peaked and provides little signal about secondary classes. Raising $T$ flattens the distribution, amplifying the relative probabilities of non-dominant classes. The $T^2$ factor compensates for the reduced gradient magnitude at high temperature (Hinton et al., 2015). Typical values are $T \in [2, 20]$, chosen by validation.

---

### Q14 [Advanced] Describe feature-level distillation and the challenges of aligning intermediate representations

**Q:** What does feature-level distillation transfer that output distillation cannot, and what makes matching intermediate representations across different-capacity networks difficult?

**A:** Output distillation compresses the teacher's knowledge into a $K$-class probability vector. Feature-level distillation additionally transfers the **internal representations** learned in intermediate layers — spatial feature maps, attention patterns, or hidden states — which contain richer structural information. **FitNets** (Romero et al., 2015) proposed training the student to minimize the MSE between its intermediate activations and a designated "hint" layer of the teacher:

$$\mathcal{L}_{\text{hint}} = \frac{1}{2}\|f_t(\mathbf{x}) - W_r f_s(\mathbf{x})\|_2^2$$

where $W_r$ is a learned **regressor** that projects the student's feature dimension to match the teacher's. This projection is necessary because teacher and student layers typically have different widths; without it the student would need to exactly replicate the teacher's dimensionality at every intermediate layer, which is architecturally constraining.

A deeper difficulty is that different-capacity networks learn qualitatively different internal representations even when solving the same task. A large teacher may distribute information across many channels; a small student may concentrate the same information into fewer, denser features. Forcing the student to precisely mimic the teacher's high-dimensional feature maps can be an overly rigid constraint that prevents it from finding its own efficient representation. MSE over activations is also dominated by channels with high variance, which may not be the most informative ones. **Relational Knowledge Distillation** (RKD, Park et al., 2019) addresses this by transferring relational structure — pairwise distances and angles between sample embeddings — rather than absolute activation values, allowing the student freedom to find its own representation geometry while preserving the relative structure the teacher learned.

---

### Q15 [Basic] Explain how LoRA reduces trainable parameters during fine-tuning without modifying the original weights

**Q:** What low-rank structure does LoRA introduce, and how does it preserve the pre-trained model while enabling task-specific adaptation?

**A:** **LoRA** (Low-Rank Adaptation, Hu et al., 2022) is motivated by the observation that the weight update $\Delta W$ needed to adapt a large pre-trained model to a downstream task has low intrinsic rank — the task-specific adaptation lies in a much lower-dimensional subspace than the full weight matrix. Rather than updating the full $d \times k$ weight matrix, LoRA parameterizes the update as a product of two small matrices:

$$W' = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$. The original weights $W_0$ are frozen; only $A$ and $B$ are trained. For a GPT-3-scale attention projection with $d = k = 12{,}288$ and $r = 8$, trainable parameters drop from $\sim 151\text{M}$ to $\sim 196\text{K}$ — a $770\times$ reduction per matrix.

$A$ is initialized from a small Gaussian and $B$ to zero, so $\Delta W = BA = 0$ at initialization — training starts from the exact pre-trained output. At inference, the LoRA weights are merged by computing $W' = W_0 + BA$ once, incurring no inference overhead. Hu et al. (2022) showed that LoRA applied to the attention weight matrices of GPT-3 matches or exceeds full fine-tuning on downstream NLP benchmarks while reducing trainable parameters by $10{,}000\times$. Multiple task-specific LoRA adapters can be maintained simultaneously for a single base model, with negligible per-adapter storage cost.

---

### Q16 [Advanced] Describe how truncated SVD approximates and compresses linear and convolutional layers

**Q:** How does low-rank matrix decomposition reduce parameter count and FLOPs in a layer, and what properties of trained weight matrices limit its effectiveness?

**A:** Given a weight matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$, its singular value decomposition (SVD) is $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ with singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$. The **truncated rank-$r$ approximation** retains only the top $r$ singular triplets:

$$\hat{\mathbf{W}}_r = \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^T = (\mathbf{U}_r \boldsymbol{\Sigma}_r^{1/2})(\boldsymbol{\Sigma}_r^{1/2}\mathbf{V}_r^T)$$

This replaces one matrix multiply of cost $O(mn)$ with two sequential multiplies of costs $O(mr)$ and $O(rn)$, totaling $O(r(m+n))$ — a FLOPs reduction factor of $mn / (r(m+n))$. Storage similarly reduces from $mn$ to $r(m+n)$.

For convolutional networks, Denton et al. (2014) applied SVD to reshaped filter banks, achieving approximately $2\times$ speedup on VGG-style networks with roughly $1\%$ top-5 accuracy degradation on ImageNet. The fundamental limitation is that the singular value spectrum of weight matrices in trained networks tends to **decay slowly**: many singular values have similar magnitudes, so a rank-$r$ truncation must use large $r$ to avoid significant approximation error. Layers near the input and square layers ($m \approx n$) tend to have flatter spectra and resist compression. The compression ratio is thus highly layer-dependent — most of the spectral mass may be concentrated in the top few singular values for some layers and spread broadly for others — making non-uniform rank allocation (larger $r$ for critical layers) important for achieving a good accuracy-compression trade-off in practice.

---

## Efficient Architectures & Inference

### Q17 [Basic] Explain how depthwise separable convolutions reduce computation compared to standard convolutions

**Q:** How does factorizing a convolution into depthwise and pointwise stages change the total operation count, and where does the computational saving come from?

**A:** A **standard convolution** with $M$ input channels, $N$ output channels, kernel size $D_K \times D_K$, and output spatial size $D_F \times D_F$ performs $D_K^2 \cdot M \cdot N \cdot D_F^2$ multiply-accumulate (MAC) operations: each of the $N$ output filters slides over all $M$ input channels.

A **depthwise separable convolution** (Howard et al., 2017) factorizes this into two stages. First, a **depthwise convolution** applies one $D_K \times D_K$ filter independently per input channel, costing $D_K^2 \cdot M \cdot D_F^2$ MACs (captures spatial patterns, no channel mixing). Second, a **pointwise convolution** ($1 \times 1$) mixes channel information to produce $N$ output features, costing $M \cdot N \cdot D_F^2$ MACs. Total cost: $(D_K^2 + N) \cdot M \cdot D_F^2$. The reduction ratio relative to standard convolution is:

$$\frac{D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2}{D_K^2 \cdot M \cdot N \cdot D_F^2} = \frac{1}{N} + \frac{1}{D_K^2}$$

For $D_K = 3$ and large $N$, this approaches $1/9 \approx 8$–$9\times$ reduction. Howard et al. (2017) demonstrated that MobileNets built entirely from depthwise separable convolutions achieve roughly $1\%$ lower top-1 accuracy on ImageNet than VGG-16 while using approximately $27\times$ fewer MACs and $32\times$ fewer parameters — establishing depthwise separable convolutions as the key primitive for mobile-scale vision models.

---

### Q18 [Advanced] Describe how FlashAttention reduces memory and increases throughput in transformer computation

**Q:** What is the memory and bandwidth bottleneck in standard scaled dot-product attention, and how does FlashAttention's tiling approach eliminate it without changing the mathematical output?

**A:** Standard scaled dot-product attention $\text{softmax}(\mathbf{QK}^T/\sqrt{d})\mathbf{V}$ requires materializing the $N \times N$ attention score matrix in GPU HBM (high-bandwidth memory), where $N$ is the sequence length. This is an $O(N^2)$ memory footprint that becomes prohibitive for long sequences. More critically, attention is **IO-bound** rather than compute-bound: modern GPUs have far more arithmetic throughput than memory bandwidth, so the repeated HBM reads and writes of large activation tensors dominate wall time. Dao et al. (2022) estimated that naïve PyTorch attention spends the majority of its time on memory reads/writes rather than FLOPs.

**FlashAttention** eliminates HBM materialization of the $N \times N$ matrix by computing attention in **tiles** sized to fit in on-chip SRAM. For each tile of queries $\mathbf{Q}_i$, it iterates over all tiles of keys $\mathbf{K}_j$ and values $\mathbf{V}_j$, maintaining running statistics using the **online softmax** algorithm: a running maximum $m_i$ and normalization constant $\ell_i$ are updated incrementally, allowing the partial attention output to be corrected without storing all scores:

$$m_i^{(j)} = \max\!\left(m_i^{(j-1)},\, \operatorname{rowmax}\!\left(\mathbf{Q}_i \mathbf{K}_j^T / \sqrt{d}\right)\right)$$

The final output is assembled tile by tile with a single pass, never writing the $N \times N$ matrix to HBM. The mathematical result is identical to standard attention (Dao et al., 2022).

FlashAttention reduces HBM memory from $O(N^2)$ to $O(N)$ and achieves 2–4× wall-clock speedup over PyTorch attention on A100 GPUs for sequence lengths 1K–16K. FlashAttention-2 (Dao, 2023) further improved throughput by roughly $2\times$ through better parallelization across the sequence dimension and reduction of non-matmul FLOPs in the attention kernel.

---

### Q19 [Advanced] Explain how speculative decoding accelerates auto-regressive token generation

**Q:** What inefficiency in standard token-by-token generation does speculative decoding exploit, and how does it guarantee that the output distribution is unchanged?

**A:** In standard auto-regressive generation, the target LLM generates one token per forward pass. For large models, each forward pass is **memory-bandwidth-bound**: weight matrices must be loaded from HBM for every token, and at batch size 1 the GPU arithmetic units are largely idle while waiting for data. Generating a sequence of $T$ tokens requires $T$ sequential forward passes, each underutilizing the available compute (Leviathan et al., 2022).

**Speculative decoding** uses a small, fast **draft model** to propose $k$ candidate tokens in $k$ cheap sequential steps. These $k$ tokens are then verified in a **single parallel forward pass** of the target model. Because the target pass over $k$ tokens loads weights once and processes all positions simultaneously, it costs approximately the same as one standard token generation step — yet it potentially accepts multiple tokens. When all $k$ draft tokens are accepted, the algorithm achieves $k\times$ throughput; if some are rejected, at least one new token is always produced.

Correctness is guaranteed by a rejection sampling scheme: draft token $x_i$ at position $i$ is accepted with probability $\min(1,\, p(x_i|x_{<i}) / q(x_i|x_{<i}))$, where $p$ is the target distribution and $q$ is the draft distribution. If rejected, a new token is sampled from a corrected distribution $\text{norm}(\max(0,\, p(\cdot) - q(\cdot)))$. The resulting token sequence has the **exact same distribution as the target model** sampling alone — no approximation is introduced (Leviathan et al., 2022).

The expected speedup scales as approximately $\frac{1-\alpha^{k+1}}{(1-\alpha)(1 + k \cdot c)}$, where $\alpha$ is the per-token acceptance probability and $c$ is the cost ratio of one draft step to one target step. Leviathan et al. (2022) reported 2–3× speedup on T5-XXL using T5-Small as the draft model, with identical output quality, demonstrating that the performance of the expensive target model is fully preserved.

---

### Q20 [Basic] Describe how differentiable architecture search finds efficient neural network architectures

**Q:** What made early NAS methods prohibitively expensive, and how does DARTS make architecture search tractable through continuous relaxation?

**A:** Classical Neural Architecture Search (Zoph & Le, 2017) uses a reinforcement learning controller to sample candidate architectures, trains each from scratch on the target dataset to evaluate validation accuracy, and updates the controller based on the reward. Zoph & Le (2017) required 800 GPUs running for 28 days to discover competitive architectures on CIFAR-10 — the cost of evaluating each architecture independently makes exhaustive or RL-guided search over large spaces impractical for most research settings.

**DARTS** (Differentiable Architecture Search, Liu et al., 2019) makes search tractable through **continuous relaxation**: each edge $(i, j)$ in the architecture cell (a DAG where nodes are feature maps) holds a mixture over $|\mathcal{O}|$ candidate operations, weighted by softmax-normalized architecture parameters $\boldsymbol{\alpha}$:

$$\bar{o}^{(i,j)}(\mathbf{x}) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i,j)})} \cdot o(\mathbf{x})$$

The architecture parameters $\boldsymbol{\alpha}$ and network weights $\mathbf{w}$ are jointly optimized by **bilevel optimization**: $\mathbf{w}$ is updated on training data, $\boldsymbol{\alpha}$ on validation data, alternating within a single training run on the full dataset. After search, the discrete architecture is derived by selecting the highest-weighted operation on each edge. DARTS reduces search cost from thousands of GPU-days to approximately 4 GPU-days on CIFAR-10, discovering architectures that transfer competitively to ImageNet without retraining (Liu et al., 2019).

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Uniform quantization: scale, zero-point, granularity | Quantization Fundamentals |
| Q2 | Basic | PTQ vs QAT | Quantization Fundamentals |
| Q3 | Advanced | Why activations are harder to quantize than weights | Quantization Fundamentals |
| Q4 | Advanced | GPTQ one-shot weight quantization | Quantization Fundamentals |
| Q5 | Advanced | AWQ salient-weight-aware quantization | Quantization for Large Language Models |
| Q6 | Basic | Weight-only vs weight-activation quantization | Quantization for Large Language Models |
| Q7 | Advanced | SmoothQuant activation-to-weight difficulty migration | Quantization for Large Language Models |
| Q8 | Advanced | LLM.int8() mixed-precision decomposition | Quantization for Large Language Models |
| Q9 | Basic | Structured vs unstructured pruning | Pruning & Sparsity |
| Q10 | Advanced | Lottery ticket hypothesis | Pruning & Sparsity |
| Q11 | Advanced | SparseGPT one-shot LLM pruning | Pruning & Sparsity |
| Q12 | Advanced | Static vs dynamic sparse training and RigL | Pruning & Sparsity |
| Q13 | Basic | Soft-target knowledge distillation | Knowledge Distillation & Low-Rank Compression |
| Q14 | Advanced | Feature-level distillation and relational KD | Knowledge Distillation & Low-Rank Compression |
| Q15 | Basic | LoRA low-rank fine-tuning | Knowledge Distillation & Low-Rank Compression |
| Q16 | Advanced | Truncated SVD layer compression | Knowledge Distillation & Low-Rank Compression |
| Q17 | Basic | Depthwise separable convolutions | Efficient Architectures & Inference |
| Q18 | Advanced | FlashAttention IO-aware tiling | Efficient Architectures & Inference |
| Q19 | Advanced | Speculative decoding | Efficient Architectures & Inference |
| Q20 | Basic | DARTS differentiable architecture search | Efficient Architectures & Inference |

## Resources

- Nagel et al., [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) (2021)
- Jacob et al., [Quantization and Training of Neural Networks for Inference at Integer Arithmetic](https://arxiv.org/abs/1712.05877) (2018)
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2022)
- Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (2023)
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) (2022)
- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (2022)
- Han et al., [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) (2016)
- Frankle & Carlin, [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) (2019)
- Frantar & Alistarh, [SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot](https://arxiv.org/abs/2301.00774) (2023)
- Evci et al., [Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/1911.11134) (2020)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015)
- Romero et al., [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) (2015)
- Park et al., [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) (2019)
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2022)
- Denton et al., [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736) (2014)
- Howard et al., [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (2017)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022)
- Zoph & Le, [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) (2017)
- Liu et al., [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (2019)
