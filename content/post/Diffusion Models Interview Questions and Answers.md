---
title: "Diffusion Models: Interview Questions and Answers"
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

## Foundations of Diffusion Models

### Q1 [Basic] Describe the forward and reverse processes in DDPM

**Q:** How do the forward and reverse processes in DDPM define a generative model, and what mathematical structure makes the reverse process tractable?

**A:** **DDPM** (Ho et al., 2020) defines a generative model through two Markov chains. The **forward process** $q(x_{1:T}|x_0)$ gradually corrupts a data sample $x_0$ by adding small amounts of Gaussian noise over $T$ steps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\right)$$

where $\{\beta_t\}_{t=1}^T$ is a fixed variance schedule. A key property is the closed-form marginal: defining $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$, we have

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t) I\right)$$

which means noisy samples at any timestep can be drawn directly without iterating through all preceding steps. At $T = 1000$ with a linear schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$, $x_T$ is nearly isotropic Gaussian (Ho et al., 2020).

The **reverse process** $p_\theta(x_{0:T-1}|x_T)$ starts from Gaussian noise and denoises step-by-step. The true reverse posterior $q(x_{t-1}|x_t, x_0)$ is analytically tractable and itself Gaussian. The model learns $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$ to approximate this posterior. Generation proceeds by sampling $x_T \sim \mathcal{N}(0, I)$ and repeatedly applying the learned denoising step — a process rooted conceptually in the nonequilibrium thermodynamics framework of Sohl-Dickstein et al. (2015).

---

### Q2 [Basic] Explain the connection between score matching and denoising

**Q:** What is the score function, and how does denoising score matching provide a practical training objective for score-based generative models?

**A:** The **score function** of a distribution $p(x)$ is its log-density gradient $\nabla_x \log p(x)$. Score-based generative models learn this gradient field, then use it to draw samples via **Langevin dynamics**: iteratively moving along the score with added noise. Song & Ermon (2019) showed that a neural network trained to estimate $\nabla_x \log p(x)$ can generate high-quality samples — but the standard score matching objective requires computing a trace of the Hessian, which is intractable for high-dimensional data.

**Denoising score matching** bypasses this by training the network to denoise corrupted data instead. Given a clean sample $x_0$ and noise $\epsilon \sim \mathcal{N}(0, I)$, we corrupt it to $x_t = x_0 + \sigma_t \epsilon$ and train $s_\theta(x_t, \sigma_t)$ to predict $\nabla_{x_t} \log p_{\sigma_t}(x_t)$. Vincent (2011) showed that this denoising objective has the same optimum as the intractable score matching objective. The connection is that the optimal denoiser predicts the clean data, and the residual points in the direction of the score.

Song & Ermon (2019) extended this to a multi-scale framework (**NCSN**): training a single score network conditioned on a sequence of noise levels $\sigma_1 < \sigma_2 < \cdots < \sigma_L$, and using annealed Langevin dynamics at inference. This directly motivated DDPM, where the noise predictor $\epsilon_\theta(x_t, t)$ is equivalent to a rescaled score estimate: $s_\theta(x_t, t) \approx -\epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}$.

---

### Q3 [Advanced] Explain the SDE framework and how it unifies score-based and DDPM models

**Q:** How does Song et al.'s SDE perspective unify DDPM and NCSN, and what new capabilities does the continuous-time formulation unlock?

**A:** Song et al. (2021) showed that both DDPM and NCSN are discretizations of two continuous-time SDEs with the same mathematical structure:

$$dx = f(x, t)\,dt + g(t)\,dW$$

where $W$ is a standard Wiener process. **VP-SDE** (Variance Preserving) is the continuous limit of DDPM's forward process with $f(x,t) = -\frac{1}{2}\beta(t) x$ and $g(t) = \sqrt{\beta(t)}$. **VE-SDE** (Variance Exploding) is the continuous limit of NCSN with $f = 0$ and $g(t) = \sqrt{d[\sigma^2(t)]/dt}$.

Both SDEs have an exact reverse SDE:

$$dx = \left[f(x,t) - g^2(t)\nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{W}$$

and a corresponding **probability flow ODE** whose marginals match those of the forward SDE but which evolves deterministically:

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2} g^2(t) \nabla_x \log p_t(x)$$

The continuous formulation unlocks three capabilities not available in discrete DDPM. First, the probability flow ODE can be integrated with any off-the-shelf numerical ODE solver, enabling exact likelihood computation via the instantaneous change-of-variables formula. Second, the ODE trajectory defines a deterministic mapping between noise and data, enabling latent space interpolation and editing. Third, flexible noise schedules become first-class — any smooth schedule from $p_T \approx \mathcal{N}(0, I)$ to $p_0 = p_\text{data}$ defines a valid model, and the trained score network generalizes across the full continuum of noise levels.

---

### Q4 [Advanced] Analyze the ELBO decomposition and the simplified training objective

**Q:** How does the DDPM training objective relate to the ELBO, and why does the simplified noise-prediction loss empirically outperform the full variational bound?

**A:** The DDPM generative model defines a variational lower bound (ELBO) on log-likelihood:

$$\log p_\theta(x_0) \geq -\mathcal{L}_\text{VLB} = \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

Expanding the ELBO yields a sum of KL terms comparing the learned reverse step $p_\theta(x_{t-1}|x_t)$ to the analytical posterior $q(x_{t-1}|x_t, x_0)$ at each timestep. Because $q(x_{t-1}|x_t, x_0)$ is Gaussian with known mean $\tilde{\mu}_t(x_t, x_0)$, the per-step KL reduces to a squared-error between the learned and true posterior means — and the mean can be reparameterized as a noise prediction:

$$\mathcal{L}_t \propto \mathbb{E}_{x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, t)\|^2\right]$$

Ho et al. (2020) proposed the **simplified objective** that discards the timestep-dependent weighting:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\, t)\|^2\right]$$

Empirically, the simplified loss outperforms the weighted ELBO for sample quality, even though it does not optimize an explicit likelihood bound. The intuition is that the VLB weights small-$t$ terms (nearly clean images) more heavily because the posterior variance is small; the simplified loss treats all timesteps uniformly, giving more gradient signal from the high-noise regime where the model has more to learn. Nichol & Dhariwal (2021) subsequently showed that learning the variance schedule allows optimizing a hybrid of the simplified loss and VLB, recovering better likelihood estimates.

---

## Accelerated Sampling

### Q5 [Basic] Explain how DDIM enables deterministic and accelerated sampling

**Q:** What is DDIM's key insight that allows deterministic generation and step-skipping without retraining the noise predictor?

**A:** **DDIM** (Song et al., 2020) starts from the observation that DDPM's forward process $q(x_{1:T}|x_0)$ does not need to be Markovian. Song et al. (2020) derived a non-Markovian forward process with the same marginals $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$ as DDPM — meaning the same noise predictor $\epsilon_\theta$ is applicable — but a different reverse update:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0(x_t) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t, t) + \sigma_t\,\epsilon$$

where $\hat{x}_0(x_t) = (x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t))/\sqrt{\bar{\alpha}_t}$ is the predicted clean image and $\sigma_t$ controls stochasticity. Setting $\sigma_t = 0$ makes the update deterministic: the same noise input $x_T$ always produces the same output $x_0$.

The deterministic ODE formulation enables **step-skipping**: instead of iterating through $t = T, T-1, \ldots, 1$, DDIM can evaluate the noise predictor at a subsequence of $S \ll T$ timesteps. With $S = 50$ steps, DDIM achieves generation quality comparable to DDPM with $T = 1000$ steps — a $20\times$ speedup — because the ODE trajectory is smooth and can be integrated coarsely. The same noise predictor trained with the DDPM objective works directly, requiring no retraining.

---

### Q6 [Advanced] Explain how DPM-Solver accelerates sampling using ODE theory

**Q:** What mathematical structure of the diffusion ODE does DPM-Solver exploit, and how does it achieve high-quality generation in very few steps?

**A:** The probability flow ODE from the SDE framework (Song et al., 2021) can be written in a form that reveals a **semi-linear structure**. Lu et al. (2022) rewrote the ODE in terms of the log-SNR $\lambda_t = \log(\sqrt{\bar{\alpha}_t} / \sqrt{1-\bar{\alpha}_t})$:

$$\frac{dx_\lambda}{d\lambda} = x_\lambda - \frac{\sqrt{1-e^{-2\lambda}}}{e^{-\lambda}} \epsilon_\theta(x_\lambda, \lambda)$$

The linear part of this ODE (the $x_\lambda$ term) has an exact solution. The nonlinear part (the score term) is approximated using Taylor expansions of $\epsilon_\theta$ with respect to $\lambda$. DPM-Solver-2 uses a second-order expansion and DPM-Solver-3 uses a third-order expansion — both computed using only function evaluations of $\epsilon_\theta$ at a small number of points.

The key advantage over DDIM (which is effectively a first-order ODE solver in this framework) is that higher-order methods can take larger steps with the same approximation error. DPM-Solver-2 achieves near-perfect sample quality in 20 steps (compared to DDIM's 50–100), and DPM-Solver-3 in as few as 10 steps on CIFAR-10 and ImageNet (Lu et al., 2022). The method works with any DDPM-trained model without retraining and is compatible with both noise-prediction and data-prediction parameterizations, making it a drop-in replacement for DDIM in deployed systems.

---

### Q7 [Advanced] Describe consistency models and their relationship to diffusion

**Q:** How do consistency models achieve single-step generation, and what distinguishes consistency distillation from consistency training?

**A:** **Consistency models** (Song et al., 2023) define a consistency function $f_\theta(x_t, t)$ that maps any point on a PF-ODE trajectory at any noise level $t$ back to the same origin $x_0$. The self-consistency property requires:

$$f_\theta(x_t, t) = f_\theta(x_{t'}, t') \quad \text{for all } t, t' \text{ on the same ODE trajectory}$$

with the boundary condition $f_\theta(x_0, 0) = x_0$ (the function is the identity at zero noise). Generation is then single-step: sample $x_T \sim \mathcal{N}(0, \sigma_T^2 I)$ and compute $f_\theta(x_T, T)$.

**Consistency distillation** (CD) trains $f_\theta$ by distilling from a pre-trained diffusion model. Given adjacent points $(x_{t_{n+1}}, x_{t_n})$ on an ODE trajectory — where $x_{t_n}$ is obtained by one step of a numerical ODE solver from $x_{t_{n+1}}$ — the loss minimizes:

$$\mathcal{L}_\text{CD} = \mathbb{E}\!\left[d\!\left(f_\theta(x_{t_{n+1}}, t_{n+1}),\, f_{\theta^-}(x_{t_n}, t_n)\right)\right]$$

where $d(\cdot, \cdot)$ is a metric (LPIPS in practice) and $\theta^-$ is an exponential moving average of $\theta$.

**Consistency training** (CT) bootstraps from scratch without a pre-trained diffusion model by estimating ODE adjacencies from the data distribution alone, at the cost of somewhat lower sample quality than CD. Song et al. (2023) reported single-step FID of 3.55 on CIFAR-10 with CD, approaching the quality of DDIM with 10 steps, and multi-step consistency sampling (2–3 function evaluations) further closing the gap to full diffusion quality.

---

### Q8 [Advanced] Compare flow matching to diffusion models

**Q:** How does flow matching differ from diffusion in its training objective and sampling paths, and what practical advantages does this bring?

**A:** Diffusion models learn to reverse a noising process by predicting the score $\nabla_x \log p_t(x)$ or equivalently the noise $\epsilon_t$. The forward process defines curved trajectories in data space (ellipsoidal paths governed by the SNR schedule), and the reverse ODE must trace these same curved paths accurately. This curvature requires many NFEs (network function evaluations) for precise integration.

**Flow matching** (Lipman et al., 2022) sidesteps this by directly learning a vector field $v_\theta(x, t)$ that transports noise $x_0 \sim \mathcal{N}(0, I)$ to data $x_1 \sim p_\text{data}$ via an ODE $dx/dt = v_\theta(x, t)$. The conditional flow matching (CFM) objective conditions on individual data points:

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t,\, q(x_1),\, p_t(x | x_1)}\!\left[\|v_\theta(x, t) - u_t(x|x_1)\|^2\right]$$

Choosing **straight-line paths** $x_t = (1-t)x_0 + tx_1$ gives a constant conditional vector field $u_t(x_t|x_1) = x_1 - x_0$, making training trivially simple. Straight paths require fewer integration steps because the ODE trajectory has zero curvature — the same step size that would fail for curved diffusion paths integrates straight-line flows exactly.

**Rectified Flow** (Liu et al., 2022) is a parallel development using the same linear interpolation idea. Both flow matching and rectified flow have been adopted in production models: Stable Diffusion 3 (Esser et al., 2024) uses a flow matching objective with a Multimodal DiT architecture, reporting improved training efficiency and sample quality at comparable compute. The absence of a defined noise schedule also simplifies ablating model design choices.

---

## Conditional Generation

### Q9 [Basic] Explain classifier-free guidance and how it controls generation

**Q:** How does classifier-free guidance steer a diffusion model toward a target condition without using a separate classifier?

**A:** **Classifier-free guidance** (CFG; Ho & Salimans, 2022) trains a single conditional diffusion model that can also run unconditionally by randomly dropping the condition $c$ during training (replacing it with a null token $\varnothing$ with some probability, typically 10–20%). At inference, the model runs two forward passes per denoising step — once with the condition and once without — and linearly extrapolates in score space:

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w\,\bigl(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\bigr)$$

where $w \geq 1$ is the guidance scale. This is equivalent to amplifying the component of the score that points toward condition-consistent samples. The guidance scale trades off fidelity against diversity: $w = 1$ recovers standard conditional sampling; $w \gg 1$ concentrates the distribution toward high-probability modes, producing sharper and more condition-faithful images at the cost of lower variety.

CFG requires no separate classifier and produces cleaner guidance gradients than classifier-based alternatives. It has become the standard conditioning mechanism in systems like Stable Diffusion (typical $w = 7.5$), DALL-E 2, and Imagen, where it substantially improves text-image alignment.

---

### Q10 [Basic] Describe ControlNet and how it adds spatial conditioning

**Q:** How does ControlNet extend a pre-trained diffusion model to accept new spatial conditioning signals without degrading existing generation quality?

**A:** **ControlNet** (Zhang & Agrawala, 2023) addresses the challenge of adapting a large pre-trained diffusion U-Net to new spatial inputs — edge maps, depth maps, human pose keypoints, segmentation masks — without catastrophically forgetting its pre-trained capabilities.

The architecture creates a **trainable copy** of the U-Net's encoding blocks while keeping the original model frozen. The spatial conditioning input (e.g., a Canny edge map) is fed into the trainable copy alongside the noisy image. The outputs of the trainable blocks are added back to the frozen decoder via **zero convolutions** — $1\times 1$ convolutional layers initialized with zero weights and zero biases. Because these layers start at exactly zero output, the ControlNet initially has no effect on the model's output, preserving all pre-trained capabilities. As training progresses, the zero convolutions learn to route the conditioning information into the frozen decoder.

The key insight is that zero initialization provides a safe starting point: fine-tuning can proceed at larger learning rates than would be tolerable without it, because the initial gradient of the output with respect to the conditioning signal is exactly zero. Zhang & Agrawala (2023) demonstrated that a single ControlNet trained on one spatial conditioning type generalizes well, and that multiple ControlNets can be composed at inference time by summing their contributions to the decoder.

---

### Q11 [Advanced] Analyze the trade-offs between classifier guidance and classifier-free guidance

**Q:** What are the practical and theoretical differences between classifier guidance and CFG, and when does each fail?

**A:** **Classifier guidance** (Dhariwal & Nichol, 2021) adds the gradient of a separately trained noisy classifier $p_\phi(y|x_t)$ to the score:

$$\tilde{\nabla}_{x_t} \log p(x_t) = \nabla_{x_t} \log p_t(x_t) + w\,\nabla_{x_t} \log p_\phi(y|x_t)$$

This requires training a classifier specifically on noisy data at all timesteps, separate from the diffusion model. On ImageNet 256$\times$256, Dhariwal & Nichol (2021) used ADM (their improved U-Net backbone) with classifier guidance to achieve FID 4.59 at $w = 1.0$, demonstrating that diffusion models could outperform state-of-the-art GANs for the first time on class-conditional ImageNet.

**CFG** avoids the separate classifier but produces a qualitatively different type of guidance. The key theoretical distinction: classifier guidance computes gradients of a discriminative classifier, which can contain adversarial artifacts — the classifier learns to exploit low-level texture statistics not present in real data, causing the guided samples to become oversharpened in ways that look realistic to the classifier but unnatural to humans. CFG's extrapolation in score space cannot produce such artifacts because both score estimates come from the same generative model.

Both methods share the fundamental **diversity-fidelity trade-off**: higher $w$ increases precision (samples are closer to the mode of the conditional distribution) at the cost of recall (fewer modes are sampled, diversity decreases). This tradeoff is well-captured by the Precision/Recall framework: guidance pushes Precision up and Recall down monotonically with $w$. At very high guidance scales, FID begins to increase because the distribution collapses to a small set of highly typical samples, diverging from the full diversity of the training distribution.

---

### Q12 [Advanced] Analyze text encoder choice for text-to-image diffusion models

**Q:** How does the choice of text encoder affect text-image alignment in diffusion models, and what does Imagen's finding reveal about the importance of language model scale?

**A:** Early text-to-image diffusion models conditioned on CLIP text embeddings (Rombach et al., 2022) because CLIP representations were already aligned with visual concepts through contrastive training on image-text pairs. CLIP's 77-token context window and its training objective (maximizing image-text similarity) produce embeddings well-suited to coarse semantic alignment but limited for compositional instructions or rare visual concepts.

**Imagen** (Saharia et al., 2022) made a striking discovery: replacing CLIP with a large frozen language model (**T5-XXL**, 4.6B parameters) trained solely on text substantially improved text-image alignment, particularly for compositional prompts, unusual word orderings, and domain-specific terminology not well-represented in image-text corpora. The finding was counter-intuitive because T5 was never trained with visual supervision; yet its richer linguistic representations — learned from the full distribution of text — encode syntactic and semantic structure that CLIP's contrastive objective does not capture.

The mechanism is that T5 produces separate per-token embeddings that the diffusion U-Net's cross-attention layers can spatially route to corresponding image regions. CLIP produces a single global embedding or a relatively shallow per-token representation, limiting its ability to express compositional descriptions like "a red cube on top of a blue sphere next to a green cylinder." Saharia et al. (2022) showed that scaling the text encoder (from T5-Small to T5-XXL) improves DrawBench human preference scores even when the image decoder scale is fixed — indicating that text understanding, not image generation capacity, was the primary bottleneck for complex prompts.

---

### Q13 [Basic] Describe how noise schedule design affects diffusion model training and sampling

**Q:** What are the key choices in designing a noise schedule, and why does the cosine schedule improve upon the linear schedule for image generation?

**A:** The **noise schedule** $\{\beta_t\}$ or equivalently $\{\bar{\alpha}_t\}$ controls the signal-to-noise ratio (SNR) at each timestep and determines how the training loss is distributed across the denoising difficulty spectrum. Ho et al. (2020) used a linear schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. While effective for 256$\times$256 images, this schedule front-loads noise too aggressively: at low resolutions or in later timesteps, the image is already nearly pure noise, wasting the model's capacity on nearly featureless inputs.

Nichol & Dhariwal (2021) proposed the **cosine schedule**:

$$\bar{\alpha}_t = \cos^2\!\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

with a small offset $s = 0.008$ to prevent $\bar{\alpha}_t$ from reaching exactly zero. This keeps the SNR trajectory smoother and more uniform across timesteps, spending more steps at intermediate noise levels where the model must learn to distinguish coherent structure. The cosine schedule meaningfully improved log-likelihood and sample quality on CIFAR-10 and ImageNet (Nichol & Dhariwal, 2021).

A deeper principle is that the optimal schedule depends on data resolution and content: lower-resolution images have simpler structure and require less noise to be destroyed, so a given $\bar{\alpha}_t$ corresponds to a higher effective SNR at low resolution. This **resolution-SNR mismatch** is one reason cascaded models (like Imagen) use separate schedules for each resolution stage.

---

## Large-Scale Architectures

### Q14 [Basic] Explain how latent diffusion models reduce computational cost

**Q:** How does running diffusion in a learned latent space rather than pixel space reduce computation, and what does the encoder-decoder architecture contribute?

**A:** Running a diffusion model directly in high-resolution pixel space scales poorly: a 512$\times$512$\times$3 image requires U-Net attention layers operating over $512^2 = 262,144$ spatial positions, making both training and sampling prohibitively expensive. **Latent Diffusion Models** (LDM; Rombach et al., 2022) address this by first training a perceptual compression model — a VQ-VAE or KL-regularized VAE — that encodes images to a compact latent space, then running the diffusion process entirely in that latent space.

The encoder reduces an image to a latent $z \in \mathbb{R}^{h \times w \times c}$ with a spatial downsampling factor of $f \in \{4, 8, 16\}$ (so a 512$\times$512 image maps to a $64 \times 64$ or $32 \times 32$ latent). The diffusion U-Net operates at this latent resolution, reducing attention complexity by $f^2$- to $f^4$-fold. The decoder then maps the generated latent back to pixel space.

The encoder-decoder contributes more than just compression: the VAE is trained to produce a latent space where structurally similar images are nearby, and the KL or VQ regularization prevents the latent space from collapsing or exploding. The diffusion model therefore operates in a semantically meaningful feature space where global structure (object layout, scene composition) and local texture are naturally separated across different latent dimensions, making the denoising task more tractable than pixel-space denoising where the model must simultaneously reason about both scales. Rombach et al. (2022) demonstrated that with an $f = 4$ downsampling factor, LDM achieves competitive FID scores at a fraction of the compute of pixel-space models.

---

### Q15 [Advanced] Analyze the DiT architecture and its scaling properties

**Q:** What does DiT replace in the standard diffusion U-Net, and what evidence shows that Transformer-based diffusion scales more favorably than U-Net-based diffusion?

**A:** **DiT** (Diffusion Transformer; Peebles & Xie, 2023) replaces the U-Net backbone used in most latent diffusion models with a pure Vision Transformer. Input latent patches are flattened into a sequence of tokens, and $N$ standard Transformer blocks with modified conditioning process the sequence. The key architectural choice is **adaLN-Zero** conditioning: timestep $t$ and class label $y$ are used to predict scale and shift parameters for adaptive layer normalization — similar to FiLM conditioning — with the output projection initialized to zero so the initial output of each Transformer block is zero-residual.

DiT eliminates the multi-scale U-Net structure (encoder, bottleneck, skip-connected decoder) in favor of a flat sequence of identical Transformer blocks operating at a single spatial resolution. This makes scaling straightforward: increasing depth $N$, width $d$, or patch size $p$ all translate to predictable increases in GFLOPs and parameter count.

Peebles & Xie (2023) conducted systematic scaling experiments on class-conditional ImageNet 256$\times$256 generation, finding a clean power-law relationship between compute and FID. **DiT-XL/2** (675M parameters, patch size 2) achieves FID 2.27, outperforming all prior diffusion models including ADM on the same benchmark. Crucially, unlike U-Net-based models whose FID improvement saturates at certain scales, DiT's performance improves monotonically with compute, suggesting that ViT-style architectures — which have demonstrated similar favorable scaling in discriminative tasks — transfer their scaling efficiency to generative modeling. This architecture has since been adopted for large-scale video generation.

---

### Q16 [Advanced] Describe the two-stage design of DALL-E 2

**Q:** How does DALL-E 2's hierarchical architecture separate semantic and visual generation, and what are its implications for generation diversity and editability?

**A:** **DALL-E 2** (Ramesh et al., 2022) uses a two-stage architecture that explicitly separates semantic understanding from visual synthesis. The first stage is a **prior**: given a CLIP text embedding $z_t$, a diffusion model (or autoregressive model) generates a CLIP image embedding $z_i$. The second stage is a **decoder** (also called UNCLIP): given $z_i$, a diffusion model conditioned on the CLIP image embedding generates the full-resolution image.

The motivation is the CLIP embedding space's structure: CLIP image embeddings capture the semantic content of an image (objects, scene type, style) while being largely invariant to exact pixel-level details (lighting, texture, camera angle). Conditioning the image decoder on $z_i$ rather than on text directly allows the model to generate many visually distinct images that share the same semantic content — different valid "renderings" of the concept encoded in $z_i$.

This separation has two practical consequences. First, **image variation**: by sampling different decodings from the same $z_i$, DALL-E 2 can produce multiple semantically consistent but visually diverse images from a single text prompt. Second, **image editing via embedding interpolation**: interpolating between the CLIP embeddings of two images $z_i^{(1)}$ and $z_i^{(2)}$ produces a smooth semantic trajectory between them, enabling editing operations like style transfer that are difficult to express as text prompts. Ramesh et al. (2022) demonstrated these capabilities, though the two-stage design also introduces error accumulation: the prior must produce an embedding that is both semantically faithful to the text and within the distribution the decoder can handle well.

---

### Q17 [Advanced] Analyze cascaded diffusion in Imagen

**Q:** How does Imagen's cascaded design achieve 1024×1024 generation, and what role do its dynamic thresholding and text encoder choices play?

**A:** **Imagen** (Saharia et al., 2022) generates 1024$\times$1024 images through a three-stage cascade: a base model generates 64$\times$64 images from text, a super-resolution model upsamples to 256$\times$256, and a second super-resolution model upsamples to 1024$\times$1024. Each stage is conditioned on its lower-resolution input (for the SR stages) and on T5-XXL text embeddings. All stages use pixel-space U-Nets rather than latent diffusion.

A key technical contribution is **dynamic thresholding**. Standard diffusion clamps pixel values to $[-1, 1]$ during sampling (static thresholding). At high guidance scales ($w \gg 1$), the predicted $\hat{x}_0$ often exceeds this range, causing color saturation and artifacts. Dynamic thresholding instead clips to the $s$-th percentile of the absolute value of the prediction, then rescales: if $|\hat{x}_{0,j}| > s$ for any coordinate $j$, rescale the entire tensor by $s / \max_j |\hat{x}_{0,j}|$. This preserves the relative structure of the prediction while preventing outlier coordinates from dominating. Dynamic thresholding enables training with large guidance scales that would otherwise produce saturated images, and it is what allows Imagen to operate effectively at $w \geq 7$ across all cascade stages.

Saharia et al. (2022) found that text encoder scale was the single most impactful factor for text-image alignment — more so than diffusion model size — and that the cascaded design distributes the generation problem sensibly: the base model handles global composition while the SR models add fine-grained texture and detail. Imagen reports FID 7.27 on COCO zero-shot evaluation (Saharia et al., 2022).

---

## Video, Inverse Problems, and Evaluation

### Q18 [Basic] Describe how video diffusion models extend image diffusion

**Q:** What architectural extensions does video diffusion require, and what are the core challenges that distinguish video from image generation?

**A:** **Video Diffusion Models** (Ho et al., 2022) extend the U-Net backbone to the temporal dimension by replacing 2D spatial convolutions with **3D space-time convolutions** and adding **temporal attention** layers that attend over all frames at each spatial position. The forward process adds noise independently to each frame, and the denoising U-Net learns to jointly denoise the full video clip.

The core computational challenge is scale: a video of $F$ frames at resolution $H \times W$ requires attention over $F \cdot H \cdot W$ positions. For a 16-frame 256$\times$256 video, this is $1,048,576$ positions — intractable for full 3D attention. Ho et al. (2022) addressed this with **factored space-time attention**: separate spatial attention (attending within each frame) and temporal attention (attending across frames at fixed spatial positions), reducing complexity from $O(F^2 H^2 W^2)$ to $O(F^2 + H^2 W^2)$ per token.

The deeper challenge is **temporal consistency**: the model must generate semantically coherent content across frames — objects should maintain their appearance, physics should be plausible, camera motion should be smooth. This requires the model to internalize spatiotemporal dynamics, not just per-frame appearance. Video-specific training data at scale is the principal bottleneck, as temporally consistent video is far more expensive to collect and annotate than images. Long-form video generation introduces an additional difficulty: maintaining consistency over many seconds requires reasoning about narrative or event structure that exceeds what a fixed-length context window can capture.

---

### Q19 [Advanced] Explain how diffusion models solve linear inverse problems

**Q:** How can a pre-trained unconditional diffusion model be applied to solve measurement-conditioned reconstruction problems without task-specific training?

**A:** A linear inverse problem asks: given a measurement $y = Ax + \eta$ where $A$ is a known degradation operator (downsampling, masking, blurring) and $\eta$ is noise, reconstruct the clean signal $x$. The goal is to sample from the posterior $p(x|y) \propto p(y|x)\,p(x)$, where $p(x)$ is the data prior. A pre-trained diffusion model implicitly encodes this prior through its score function $\nabla_x \log p_t(x)$.

**DDRM** (Kawar et al., 2022) exploits the SVD of $A = U\Sigma V^\top$ to decouple the problem. At each denoising step, the solution is projected onto the subspace consistent with the measurement in the spectral domain, blending the diffusion prior for unmeasured directions with measurement fidelity for measured directions. This gives exact measurement consistency and requires only one forward pass of the noise predictor per step.

**DPS** (Diffusion Posterior Sampling; Chung et al., 2022) handles non-linear and non-uniform degradations by approximating the likelihood gradient. At each step, the noise predictor provides an estimate $\hat{x}_0 = \mu_\theta(x_t, t)$ of the clean image. The likelihood term is approximated as:

$$\nabla_{x_t} \log p(y|x_t) \approx \nabla_{x_t} \log p(y|\hat{x}_0(x_t))$$

and added as an additional guidance term to the standard score update. The approximation is justified by the self-consistency of the Tweedie estimator: $\hat{x}_0$ is the minimum mean squared error estimate of $x_0$ given $x_t$. DPS handles degradations for which no SVD decomposition exists (nonlinear forward models, phase retrieval), generalizing well beyond DDRM's assumptions at the cost of additional computational overhead per step.

---

### Q20 [Advanced] Analyze FID and its limitations as an evaluation metric for generative models

**Q:** What does FID measure, and what systematic limitations make it an incomplete characterization of generative model quality?

**A:** **FID** (Fréchet Inception Distance; Heusel et al., 2017) measures the Fréchet distance between the Inception-v3 feature distributions of real and generated images:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of real and generated Inception features respectively. Lower FID indicates that the generated distribution is closer to the real distribution in this feature space.

FID has four systematic limitations. First, **Inception feature inadequacy**: the feature space was optimized for ImageNet classification, not perceptual similarity. Generative artifacts that fool the classifier (wrong textures, hallucinated objects) may go unpenalized, while valid but unusual compositions may inflate FID. Second, **sample size sensitivity**: stable FID estimates require $\geq 50,000$ generated and reference samples; evaluating with fewer samples introduces high variance that obscures real model differences. Third, **conflation of precision and recall**: FID is a single scalar that improves both when the generated distribution covers the real distribution well and when it produces high-fidelity samples. A model that generates only a small set of perfect images can achieve low FID if those images closely match the training distribution modes.

The Precision/Recall framework directly addresses this by separately measuring whether generated samples fall within the real distribution manifold (Precision) and whether the real distribution manifold is covered by generated samples (Recall). For text-to-image models, **CLIP Score** measures semantic alignment between generated images and text prompts — a dimension FID cannot capture. Human preference scores (from pairwise comparisons via systems like ELO rating) remain the gold standard for evaluating perceptual quality and prompt faithfulness in practice, but are expensive to collect at scale.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | DDPM forward and reverse processes | Foundations of Diffusion Models |
| Q2 | Basic | Score matching and denoising | Foundations of Diffusion Models |
| Q3 | Advanced | SDE framework unifying diffusion models | Foundations of Diffusion Models |
| Q4 | Advanced | ELBO and simplified training objective | Foundations of Diffusion Models |
| Q5 | Basic | DDIM deterministic sampling | Accelerated Sampling |
| Q6 | Advanced | DPM-Solver ODE-based acceleration | Accelerated Sampling |
| Q7 | Advanced | Consistency models | Accelerated Sampling |
| Q8 | Advanced | Flow matching vs. diffusion | Accelerated Sampling |
| Q9 | Basic | Classifier-free guidance | Conditional Generation |
| Q10 | Basic | ControlNet spatial conditioning | Conditional Generation |
| Q11 | Advanced | Classifier guidance vs. CFG trade-offs | Conditional Generation |
| Q12 | Advanced | Text encoder choice: T5 vs. CLIP | Conditional Generation |
| Q13 | Basic | Noise schedule design | Conditional Generation |
| Q14 | Basic | Latent diffusion models | Large-Scale Architectures |
| Q15 | Advanced | DiT architecture and scaling | Large-Scale Architectures |
| Q16 | Advanced | DALL-E 2 two-stage design | Large-Scale Architectures |
| Q17 | Advanced | Cascaded diffusion in Imagen | Large-Scale Architectures |
| Q18 | Basic | Video diffusion models | Video, Inverse Problems, and Evaluation |
| Q19 | Advanced | Diffusion for linear inverse problems | Video, Inverse Problems, and Evaluation |
| Q20 | Advanced | FID limitations and evaluation metrics | Video, Inverse Problems, and Evaluation |

## Resources

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
