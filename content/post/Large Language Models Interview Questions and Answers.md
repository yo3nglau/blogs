---
title: "Large Language Models: Interview Questions and Answers"
author: yo3nglau
date: '2026-03-31'
categories:
  - Interview
tags:
  - Deep Learning
  - Large Language Model
  - NLP
toc: true
---

## Pretraining & Architecture

### Q1 [Basic] What is the pretraining objective of large language models?

**Q:** What training objective do LLMs like GPT use during pretraining, and why is it effective?

**A:** The dominant pretraining objective for autoregressive LLMs is next token prediction, also called causal language modeling. Given a sequence of tokens x_1, x_2, ..., x_{t-1}, the model learns to predict the next token x_t by minimizing cross-entropy loss over the vocabulary. This is applied at every position in the sequence simultaneously during training, making it highly data-efficient: a single document of n tokens provides n-1 training examples.

Despite its simplicity, this objective is remarkably powerful. The model must implicitly learn grammar, facts, reasoning patterns, and world knowledge to predict the next token well across diverse text. This is why scaling next-token prediction to large datasets and model sizes leads to emergent capabilities — the model acquires general-purpose language understanding as a byproduct of predicting text.

It is worth distinguishing this from masked language modeling (MLM), used by BERT-style models, which masks random tokens and predicts them from both left and right context. Autoregressive (causal) LMs can only attend to past tokens, making them natural generators — they can be prompted to continue any prefix. MLM models produce better representations for classification tasks but cannot generate text directly. GPT, LLaMA, and most modern LLMs use autoregressive next-token prediction.

---

### Q2 [Basic] What are Scaling Laws and what do they tell us about LLM training?

**Q:** What are neural scaling laws, and how do they inform decisions about model size and training data?

**A:** Scaling laws (Kaplan et al., 2020) are empirical power-law relationships between a language model's performance (measured by loss) and three resources: the number of model parameters N, the training dataset size D, and the total compute budget C. The key finding is that loss decreases smoothly and predictably as any of these resources increases, following L ∝ N^{-α}, L ∝ D^{-β}, and L ∝ C^{-γ} for empirically determined exponents.

The Chinchilla paper (Hoffmann et al., 2022) refined these findings with a critical practical insight: for a fixed compute budget, the optimal allocation is to train a model of roughly N parameters on approximately 20N tokens of data. This overturned the prevailing practice of training very large models on relatively small datasets — GPT-3 (175B parameters) was trained on only ~300B tokens, which the Chinchilla analysis showed was severely undertrained. A smaller model trained on more data (as Chinchilla-70B was) can match or exceed a larger undertrained model at the same compute cost.

Scaling laws have direct engineering implications: they allow teams to predict the performance of a larger run from smaller experiments before committing the full compute budget, and they prescribe how to allocate a fixed budget between model size and data. Modern models like LLaMA and Mistral follow Chinchilla-optimal or data-rich training recipes, using 1–2 trillion tokens for models in the 7B–70B range — far exceeding the original Chinchilla recommendation.

---

### Q3 [Basic] How do RoPE, ALiBi, and absolute positional encodings differ?

**Q:** What are the main approaches to positional encoding in LLMs, and what are their trade-offs?

**A:** Absolute positional encodings (sinusoidal or learned) were used in the original Transformer: a fixed or learned vector is added to each token embedding before the first layer, encoding its absolute position. The limitation is poor generalization to sequence lengths longer than those seen during training — the model has simply never seen positions beyond the training cutoff, so performance degrades sharply at longer sequences.

Rotary Position Embedding (RoPE, Su et al., 2021) encodes position by rotating the query and key vectors in the attention computation by an angle proportional to their absolute positions. Crucially, the dot product of a rotated query at position m with a rotated key at position n depends only on their relative difference m-n, giving RoPE both absolute and relative position awareness. RoPE has become the dominant choice for modern LLMs (LLaMA, Mistral, Qwen, GPT-NeoX) due to its strong performance and compatibility with extensions for longer contexts.

ALiBi (Attention with Linear Biases, Press et al., 2022) takes a different approach: instead of adding position information to token embeddings, it subtracts a fixed linear penalty from attention scores based on the distance between query and key positions. No positional embeddings are added to tokens at all. The key advantage is length generalization: models trained with ALiBi on short sequences extrapolate more smoothly to longer ones, since the linear bias pattern extends naturally beyond the training length. The trade-off is that ALiBi models tend to slightly underperform RoPE models at the training length, making ALiBi more attractive when out-of-distribution length generalization is the priority.

---

### Q4 [Advanced] How can LLMs be extended to handle very long contexts (e.g., from 4K to 128K tokens)?

**Q:** What are the main techniques for extending the context window of a pretrained LLM?

**A:** A model pretrained at 4K tokens cannot simply be deployed at 128K tokens because its positional encodings have never seen positions beyond 4K. For RoPE-based models, the rotary angles are computed using a base frequency (typically 10,000), and positions beyond the training length produce rotation angles outside the range the model learned to handle. The result is severe degradation in attention patterns and output quality.

The most widely used extension technique is RoPE frequency scaling. Position Interpolation (PI) scales down all position indices so that the training range maps to the new longer context — if the model saw positions 0 to 4095, a 32K context uses positions 0 to 4095 scaled by 32K/4K = 8. This avoids out-of-distribution positions but compresses the positional resolution, making nearby tokens harder to distinguish. NTK-Aware scaling addresses this by scaling only high-frequency components of RoPE (which encode short-range structure) less aggressively than low-frequency components (which encode long-range structure). YaRN (Peng et al., 2023) further refines this with attention scaling and requires less fine-tuning data to adapt successfully.

Even with successful positional extension, long-context performance faces a second challenge known as "lost in the middle": empirical studies show that LLMs are better at using information at the beginning and end of a long context than in the middle, even when they nominally support the full context length. Architecturally, sliding window attention (used in Mistral) and sparse attention patterns address this by ensuring every token has strong local attention while limiting global attention to reduce compute. Effective long-context deployment therefore requires both position encoding adaptation and, ideally, fine-tuning on long-context data to teach the model to use the full window.

---

### Q5 [Advanced] What is the Mixture of Experts (MoE) architecture and what advantages does it offer?

**Q:** How does a Mixture of Experts architecture work, and why is it used in large language models?

**A:** In a standard dense Transformer, every token is processed by every parameter in the feed-forward network (FFN) at each layer. A Mixture of Experts (MoE) model replaces each FFN with a set of E parallel expert networks and a learned router. For each token, the router selects the top-k experts (typically k=1 or k=2) and only those experts process that token. The outputs of the selected experts are weighted by the router's softmax scores and summed to produce the layer output.

The primary advantage is parameter efficiency: an MoE model can have significantly more total parameters than a dense model while using the same amount of compute per token. For example, Mixtral 8x7B has 47B total parameters but each token activates roughly 13B parameters (2 of 8 experts per layer), giving it the inference cost of a 13B dense model with the capacity of a much larger one. This allows MoE models to achieve better quality than a comparably-priced dense model.

The main challenges are load balancing and communication overhead. Without explicit regularization, the router tends to collapse — always selecting the same one or two experts, wasting the others. This is addressed with an auxiliary load-balancing loss that encourages roughly equal token distribution across experts. In distributed training and inference, different experts may live on different GPUs, requiring all-to-all communication for each MoE layer — this adds latency and memory overhead compared to dense models. MoE is therefore most advantageous in training scenarios where total parameter count matters more than per-device memory, and in inference at large batch sizes where communication overhead amortizes over many tokens.

---

## Alignment & Fine-tuning

### Q6 [Basic] What is Supervised Fine-Tuning (SFT) and how is it used?

**Q:** What is supervised fine-tuning in the context of LLMs, and what does it accomplish?

**A:** Supervised Fine-Tuning (SFT) is the process of continuing to train a pretrained LLM on a curated dataset of (instruction, response) pairs using standard cross-entropy loss. The pretrained model has learned rich language understanding but produces text that continues its training distribution — raw web text, books, and code — rather than responses to instructions. SFT shifts the model's output distribution toward helpful, instruction-following responses.

A key finding from the LIMA paper (Zhou et al., 2023) is that data quality matters far more than quantity for SFT: a model fine-tuned on just 1,000 carefully curated instruction-response pairs can match or exceed models trained on hundreds of thousands of lower-quality examples. This suggests that pretraining encodes the knowledge, while SFT primarily teaches the model the format and style of responding — a surface-level alignment that requires surprisingly little data.

Parameter-efficient fine-tuning methods, most notably LoRA (Low-Rank Adaptation), are now standard for SFT. LoRA freezes the pretrained weights and adds small trainable low-rank matrices to the attention layers, reducing trainable parameters by 100–1000x. This makes SFT feasible on consumer GPUs and reduces the risk of catastrophic forgetting of pretrained knowledge. QLoRA further reduces memory by quantizing the frozen base model to 4-bit while keeping the LoRA adapters in higher precision.

---

### Q7 [Basic] What is RLHF and how does it work?

**Q:** Explain the Reinforcement Learning from Human Feedback (RLHF) pipeline and how it improves LLM alignment.

**A:** RLHF (Ouyang et al., 2022 — InstructGPT) is a three-stage pipeline for training LLMs to follow human preferences. The first stage is SFT: fine-tune the pretrained model on high-quality demonstration data to produce a baseline instruction-following model. The second stage trains a reward model (RM): human annotators compare pairs of model responses to the same prompt and indicate which they prefer, and a separate model is trained to predict these human preferences — effectively learning a scalar score representing response quality.

The third stage uses reinforcement learning, specifically Proximal Policy Optimization (PPO), to update the SFT model (the "policy") to maximize the reward assigned by the reward model. To prevent the policy from drifting too far from the SFT model — which would lead to reward hacking (producing outputs that score high on the RM but are not actually helpful) — a KL divergence penalty is added to the loss: the total reward is r_RM(x, y) - β · KL(π_θ || π_SFT). The β hyperparameter controls the tradeoff between maximizing reward and staying close to the SFT distribution.

RLHF was the technique behind the jump from GPT-3 to InstructGPT and later ChatGPT, demonstrating that alignment on human preferences dramatically improved helpfulness and reduced harmful outputs even compared to much larger unaligned models. The RM acts as a scalable proxy for human judgment, enabling the policy to be optimized across far more prompts than human annotators could evaluate directly.

---

### Q8 [Basic] What is Direct Preference Optimization (DPO)?

**Q:** What is DPO and how does it differ from RLHF in aligning language models?

**A:** Direct Preference Optimization (DPO, Rafailov et al., 2023) is an alternative to RLHF that achieves the same alignment objective — training the model to prefer human-preferred responses — without requiring an explicit reward model or reinforcement learning. The key mathematical insight is that the RLHF objective (maximizing reward while minimizing KL divergence from a reference policy) has a closed-form optimal solution, and this solution can be used to directly reparametrize the problem as a supervised classification loss on preference pairs.

The DPO loss takes the form: -log σ(β · log(π_θ(y_w|x) / π_ref(y_w|x)) - β · log(π_θ(y_l|x) / π_ref(y_l|x))), where y_w is the preferred response and y_l is the dispreferred response. Intuitively, this loss increases the relative log-probability of preferred responses and decreases that of dispreferred responses, scaled by how much the current policy differs from the reference (the SFT model). No reward model training and no PPO training loop are required — the entire alignment is accomplished in a single supervised training phase on preference pairs.

DPO's main advantages are simplicity and stability: it eliminates the reward model, removes the RL training loop (which is notoriously sensitive to hyperparameters), and trains end-to-end with a straightforward classification loss. Its main limitation compared to RLHF is that it is an offline method — it can only learn from a fixed dataset of preference pairs and cannot explore new responses or collect online feedback. RLHF with PPO can in principle generate new responses and collect additional human feedback during training, giving it more flexibility for complex alignment objectives.

---

### Q9 [Advanced] What are the main challenges and limitations of RLHF?

**Q:** What problems arise in practice when applying RLHF to large language models?

**A:** Reward hacking is the most fundamental challenge in RLHF. Because the reward model is an imperfect proxy for human preferences, the policy can learn to exploit its blind spots — producing responses that score high on the RM but are not actually helpful or safe. Common manifestations include excessive verbosity (longer responses often score higher even when brevity is better), sycophancy (the model agrees with the user rather than giving accurate answers), and repetition of certain phrases the RM has learned to reward. The KL penalty mitigates but does not fully prevent this.

The reward model itself introduces several failure modes. Human annotators disagree substantially on which response is better — studies show inter-annotator agreement often below 75% — introducing noise into the RM's training signal. Annotators also bring systematic biases: they may prefer confident-sounding responses over accurate ones, longer responses over concise ones, or responses that match their own views. The RM is also trained on a limited distribution of prompts and response pairs; it may generalize poorly to prompts that differ significantly from its training distribution.

Scalable oversight is a deeper, longer-horizon challenge. As LLMs become more capable than humans in specialized domains, human feedback becomes less reliable — annotators cannot evaluate the correctness of a highly technical response they do not understand. This motivates research into AI-assisted oversight: using a more capable model to evaluate responses (RLAIF — RL from AI Feedback) or using structured debate and verification protocols where the AI's claims can be checked more easily than fully trusted.

---

### Q10 [Advanced] What is the alignment tax, and how do RLHF and DPO compare on this dimension?

**Q:** What is the alignment tax and how does it affect the choice between RLHF and DPO?

**A:** The alignment tax refers to the degradation in performance on standard capability benchmarks (such as math, coding, and factual knowledge tests) that often occurs after applying alignment fine-tuning. A model that scores 80% on a math benchmark before RLHF may drop to 75% afterward, even as it becomes more helpful and safer in conversational settings. This occurs because RLHF optimizes for human preference ratings, which do not perfectly correlate with benchmark accuracy — the model partially shifts away from its pretrained capabilities toward the stylistic and behavioral patterns rewarded by human annotators.

The alignment tax is not inevitable, and its severity depends heavily on implementation details. Using a sufficiently strong KL penalty prevents excessive drift from the SFT model. Training the reward model on high-quality annotations across diverse domains (not just conversational prompts) reduces the misalignment between RM scores and actual capability. More recent approaches like Constitutional AI (Bai et al., 2022) and RLAIF attempt to preserve capabilities by using AI-generated feedback that is less susceptible to the stylistic biases of human annotators.

DPO tends to exhibit a smaller alignment tax than PPO-based RLHF in practice, partly because its offline nature and direct optimization of the preference objective are less prone to the reward-hacking dynamics that cause capability degradation. However, DPO has its own failure modes: because it is an offline method operating on a fixed dataset, it can fail to generalize when the distribution of preference pairs does not cover the capabilities the model needs to preserve. In high-stakes deployments where both helpfulness and capability preservation are critical, the best current practice is iterative: alternate rounds of SFT, preference data collection, and DPO or PPO fine-tuning with careful evaluation on both alignment and capability benchmarks between rounds.

---

## Inference & Efficiency

### Q11 [Basic] What is the KV Cache and how does it speed up LLM inference?

**Q:** What is the KV Cache in LLM inference and why is it important?

**A:** During autoregressive text generation, the model generates one token at a time. At each step, self-attention must compute queries, keys, and values for every token in the sequence. Without caching, generating the t-th token requires computing attention over all t-1 previous tokens from scratch — the key and value matrices for every past token are recomputed at every generation step, making inference O(n^2) in total time for a sequence of length n.

The KV Cache solves this by storing the key (K) and value (V) matrices for all previously generated tokens and reusing them at subsequent steps. Only the new token's K and V need to be computed at each step; all past K and V are loaded from cache. This reduces the per-step attention computation from O(n) to O(1) in terms of new computation, making generation time scale linearly rather than quadratically with sequence length. In practice, this optimization is essential — without it, generating a 1,000-token response would require ~500,000x more attention computation than generating the first token.

The trade-off is memory: the KV Cache grows as O(n · d_model · num_layers · 2) with sequence length, and for large models with long contexts, it can consume more memory than the model weights themselves. This motivated architectural variants that reduce KV cache size. Multi-Query Attention (MQA) uses a single set of K/V heads shared across all query heads. Grouped Query Attention (GQA, used in LLaMA-2 and Mistral) is a middle ground: K/V heads are grouped, with multiple query heads sharing each K/V head. GQA reduces cache size by a factor equal to the grouping ratio while recovering most of MQA's accuracy.

---

### Q12 [Basic] How does model quantization work and what are its trade-offs?

**Q:** What is model quantization, and what are the key trade-offs between different quantization levels?

**A:** Quantization reduces the numerical precision used to represent model weights and/or activations, from the training precision (typically BFloat16 or Float16) to lower-bit integer formats such as INT8 or INT4. A weight quantized from FP16 to INT8 uses 8 bits instead of 16, halving memory consumption for that parameter. Quantization also speeds up inference on hardware that has optimized INT8 matrix multiplication kernels (most modern GPUs and all mobile processors).

Post-training quantization (PTQ) applies quantization after training without modifying the training process. The simplest approach (round-to-nearest) often works well for INT8 weights with minimal accuracy loss, especially for large models where individual weight errors average out. INT4 quantization is more challenging: naive rounding causes significant accuracy degradation. Methods like GPTQ (Frantar et al., 2022) address this by using second-order information (the Hessian of the loss) to minimize the reconstruction error of each layer's outputs after quantization, rather than minimizing per-weight rounding error. AWQ (Lin et al., 2023) identifies the 1% of weights that are most sensitive to quantization and protects them by scaling, enabling high-quality INT4 quantization without second-order computation.

A key subtlety is that weight quantization (reducing precision of stored model weights) is much easier than activation quantization (reducing precision of intermediate tensors during the forward pass). Activations have dynamic ranges that change per input, making fixed quantization ranges less accurate. For this reason, most production deployments use weight-only INT4 quantization (weights stored in 4-bit, dequantized to FP16 for computation) rather than fully quantized INT4 inference. The effective speedup comes from reduced memory bandwidth for loading weights, not from INT4 arithmetic itself.

---

### Q13 [Basic] How does speculative decoding accelerate LLM inference?

**Q:** What is speculative decoding and how does it reduce LLM inference latency?

**A:** Autoregressive generation is inherently sequential: each token depends on all previous ones, so tokens must be generated one at a time. Modern GPUs are highly parallel processors that are significantly underutilized when processing a single token per forward pass — the arithmetic intensity is too low to saturate GPU compute. Speculative decoding (Leviathan et al., 2022; Chen et al., 2023) addresses this by generating multiple candidate tokens in parallel and verifying them in a single pass of the large model.

The procedure uses a small, fast draft model to generate k candidate tokens autoregressively. These k tokens are then verified in a single forward pass of the large target model, which processes all k positions in parallel. For each position, the target model's probability distribution is compared to the draft model's. Tokens are accepted or rejected using a rejection sampling scheme designed to ensure the output distribution is identical to what the large model alone would produce: token i is accepted with probability min(1, p_target(x_i) / p_draft(x_i)), and if rejected, a corrected token is sampled from the adjusted distribution. On average, β·k tokens are accepted per draft-verify cycle, where β is the acceptance rate, giving a theoretical speedup of β·k / (cost_draft·k + cost_target) over running the large model alone.

In practice, speculative decoding achieves 2–3x speedups for typical text generation tasks where the draft and target models are sufficiently aligned. The speedup is largest when the draft model is small (cheap to run) and the acceptance rate is high (draft tokens are usually correct). Self-speculative decoding variants use early exit from the large model's own layers as the draft, eliminating the need for a separate draft model. The technique is now integrated into major inference frameworks including vLLM and Hugging Face TGI.

---

### Q14 [Advanced] How does continuous batching improve LLM inference throughput?

**Q:** What is continuous batching and why is it important for serving LLMs efficiently?

**A:** Naive LLM serving processes requests in static batches: a group of sequences is collected, processed together until all sequences in the batch have finished generating, and only then is the next batch started. This is highly inefficient because sequences in the same batch have different lengths — shorter sequences finish early but their GPU slots sit idle waiting for the longest sequence to complete. The GPU is substantially underutilized, and new requests queue up even though compute is available.

Continuous batching (also called iteration-level scheduling or in-flight batching) addresses this by operating at the granularity of individual generation steps rather than full sequences. After each forward pass (which generates one token per active sequence), the scheduler checks which sequences have finished and immediately replaces them with new requests from the queue. This means the batch composition changes dynamically at every step: at any given forward pass, the batch contains sequences at different stages of generation, some newly started and some nearly complete. The GPU remains fully utilized because new sequences fill slots the moment they become available.

PagedAttention (Kwon et al., 2023), implemented in vLLM, is an essential complement to continuous batching. KV cache memory must be pre-allocated for each sequence, and traditional static allocation wastes memory because the final sequence length is unknown in advance. PagedAttention manages KV cache memory in fixed-size pages (like virtual memory in an operating system), allocating new pages on demand as a sequence grows and immediately freeing them when the sequence completes. This eliminates memory fragmentation, allows the system to serve more concurrent sequences, and enables efficient copy-on-write for beam search. Together, continuous batching and PagedAttention are the core techniques behind high-throughput LLM serving systems, enabling throughput improvements of 10–20x over naive batching.

---

### Q15 [Advanced] What role does FlashAttention play in LLM training and inference?

**Q:** How does FlashAttention improve LLM training and inference, and what are its underlying principles?

**A:** Standard self-attention materializes the full n × n attention matrix in GPU HBM (global memory) before applying softmax and multiplying by the value matrix. For a sequence of length n, this requires O(n^2) HBM memory reads and writes. On modern GPUs, HBM bandwidth is the primary bottleneck for attention — the arithmetic is fast, but loading and storing the large attention matrix is slow. For a sequence of 32K tokens, the attention matrix alone requires ~4GB of HBM per layer per head in FP16.

FlashAttention (Dao et al., 2022) restructures the attention computation to avoid materializing the full n × n matrix. It processes the query, key, and value matrices in tiles that fit in the GPU's fast on-chip SRAM. Within each tile, it computes partial attention scores, applies a numerically stable online softmax, and accumulates the weighted value sum — all without writing intermediate results to HBM. The full output is assembled from these tiles using the online softmax normalization trick. This reduces HBM memory usage from O(n^2) to O(n) while keeping the computation mathematically exact (no approximation). FlashAttention-2 (2023) further improves GPU utilization through better work partitioning across thread blocks, achieving near-peak arithmetic throughput on A100 GPUs.

In LLM training, FlashAttention is critical for long-context training: it makes 32K–128K context lengths feasible on standard GPU clusters that could not otherwise fit the attention matrices in memory. In inference, FlashAttention reduces memory pressure during the prefill phase (processing the input prompt), where all prompt tokens are attended over simultaneously. For the decode phase (generating one token at a time), the attention pattern is a single row of the attention matrix (new token attending to all cached keys), which FlashDecoding parallelizes more efficiently than the original FlashAttention. These optimizations collectively make long-context LLM inference practical at production scale.
