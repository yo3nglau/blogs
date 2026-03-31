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

---

## Applications & Engineering

### Q16 [Basic] What is Retrieval-Augmented Generation (RAG) and what problems does it solve?

**Q:** What is RAG, how does it work, and what advantages does it offer over a standard LLM?

**A:** Retrieval-Augmented Generation (RAG, Lewis et al., 2020) is an architecture that augments an LLM's generation with information retrieved from an external knowledge base at inference time. The standard pipeline has three components: an offline indexing stage that encodes a document corpus into dense vector embeddings (using a separate encoder model) and stores them in a vector database; an online retrieval stage that encodes the user's query into the same embedding space and retrieves the top-k most similar document chunks using approximate nearest-neighbor search; and a generation stage where the retrieved chunks are prepended to the prompt, giving the LLM relevant context before it generates its answer.

RAG solves several fundamental limitations of standalone LLMs. First, it addresses the knowledge cutoff: LLM weights encode knowledge only up to their training date, while a RAG system's retrieval index can be updated in real time with new documents. Second, it reduces hallucination on factual queries by grounding the model's response in retrieved evidence — the model generates based on documents it can "see" in context rather than relying on imprecisely memorized facts. Third, it enables specialization to proprietary or domain-specific knowledge that was not present in public training data, without requiring expensive fine-tuning.

The retrieval quality is the critical bottleneck: RAG is only as good as the documents it retrieves and the chunking strategy used to split them. Sparse retrieval (BM25, TF-IDF) works well for keyword-heavy queries; dense retrieval (DPR, E5, BGE embeddings) handles semantic similarity better. Hybrid retrieval combines both. Re-ranking retrieved chunks with a cross-encoder before passing them to the LLM further improves precision. Chunking strategy — how documents are split into retrievable pieces — significantly affects whether the relevant information is captured in a single chunk or split across multiple.

---

### Q17 [Basic] What are the core techniques in prompt engineering?

**Q:** What are the most important prompt engineering techniques and when should each be used?

**A:** Prompt engineering refers to the design of inputs to an LLM to elicit better outputs without modifying the model's weights. The most fundamental distinction is between zero-shot prompting (providing only a task description with no examples) and few-shot prompting (including 2–8 input-output examples in the prompt). Few-shot prompting reliably improves performance on tasks where the desired format or reasoning style is not obvious from a task description alone, and the specific examples chosen matter significantly — examples that are diverse and representative of the target distribution consistently outperform random selections.

Chain-of-Thought (CoT) prompting (Wei et al., 2022) dramatically improves performance on reasoning tasks by including intermediate reasoning steps in either the few-shot examples or by adding "Let's think step by step" to a zero-shot prompt. CoT is effective because it forces the model to allocate tokens to intermediate reasoning rather than jumping directly to an answer — the model "thinks" by generating reasoning text that conditions its final output. Self-consistency (Wang et al., 2022) extends CoT by sampling multiple independent reasoning chains and taking a majority vote over their final answers, further improving accuracy on tasks with deterministic answers.

Structural techniques include role prompting ("You are an expert data scientist"), output format specification (instructing the model to respond in JSON, bullet points, or a specific template), and explicit constraint statements ("Answer in 3 sentences or fewer"). For complex multi-step tasks, it is often more effective to decompose the task into a sequence of simpler prompts with verification steps between them (prompt chaining) rather than attempting everything in one prompt. A practical guideline is to start with a clear, direct task description — most prompt engineering failures stem from ambiguous instructions rather than missing tricks.

---

### Q18 [Advanced] How do LLM-based agents use tools and planning?

**Q:** What are LLM agents, how do they use external tools, and what are the main architectural patterns?

**A:** An LLM agent is a system where an LLM serves as the central reasoning engine, iteratively deciding what actions to take, executing those actions via external tools, and updating its plan based on the results. Unlike a single-turn LLM call, an agent operates in a loop: observe current state → reason about what to do → call a tool → observe the tool's result → reason again. This loop continues until the agent determines the task is complete or a stopping condition is reached.

Tool use is enabled through function calling (native to models like GPT-4 and Claude) or structured output prompting. The LLM is given a description of available tools (e.g., web search, code execution, database queries, calculator, API calls) in its system prompt. When it decides a tool is needed, it outputs a structured call specifying the tool name and arguments; an orchestration layer executes the call and returns the result as a new observation in the context. This enables agents to access real-time information, execute code, read and write files, and interact with external services — capabilities far beyond what the LLM's weights alone can provide.

ReAct (Yao et al., 2022) is the dominant planning pattern: the model alternates between Thought (reasoning about the current state) and Action (selecting a tool call), with Observation (the tool result) appended after each action. This interleaving of reasoning and action allows the model to adapt its plan based on what it observes. For complex, multi-step tasks, the main failure modes are error accumulation (a mistake in an early step compounds in later steps), hallucinated tool calls (the model generates plausible-looking but incorrect tool arguments), and getting stuck in reasoning loops. Mitigations include explicit verification steps, structured output schemas that constrain tool call formats, and human-in-the-loop confirmation for high-stakes actions.

---

### Q19 [Advanced] What are the limitations of RAG and how does it compare to long-context LLMs?

**Q:** What are RAG's main failure modes, and when is a long-context LLM a better choice than RAG?

**A:** RAG's most fundamental limitation is retrieval quality: if the correct document chunk is not retrieved, the LLM cannot generate the right answer regardless of its capabilities. Retrieval fails when the query's semantics are poorly matched by the embedding space (especially for complex, multi-hop questions), when the relevant information is spread across multiple chunks that are not retrieved together, or when the chunking strategy cuts across the boundaries of the relevant information. For tasks requiring synthesis across many documents (e.g., "What are all the risks mentioned across these 50 reports?"), RAG's top-k retrieval fundamentally cannot surface all relevant content simultaneously.

Long-context LLMs (with 128K–1M token context windows) offer an alternative: load the entire knowledge base or all relevant documents directly into the context, eliminating retrieval entirely. This avoids retrieval errors, naturally handles multi-document reasoning, and simplifies the pipeline significantly. However, it scales poorly: at 128K tokens, a single inference call processes 128,000 tokens regardless of whether most of them are relevant, which is expensive both in compute and in the "lost in the middle" problem — models attend poorly to information in the middle of very long contexts even when they nominally support the context length.

The practical choice depends on the knowledge base size and query type. For knowledge bases of tens to hundreds of documents, long-context LLMs are increasingly practical and often outperform RAG on multi-document reasoning tasks. For knowledge bases of thousands to millions of documents, RAG remains necessary for scalability. Hybrid approaches — use RAG to retrieve a candidate set, then use a long-context LLM to reason over all retrieved chunks simultaneously — combine the scalability of retrieval with the reasoning quality of full-context attention and represent the current state of the art for complex knowledge-intensive tasks.

---

### Q20 [Advanced] What causes LLM hallucinations and how can they be mitigated?

**Q:** Why do LLMs hallucinate, and what are the most effective strategies for reducing hallucination?

**A:** Hallucination — the generation of plausible-sounding but factually incorrect or unsupported content — arises from several distinct mechanisms. The most fundamental is that LLMs are trained to produce fluent, contextually appropriate text, not to be accurate: the training objective (next-token prediction) rewards producing text that looks like human writing, not text that is verifiably true. When a model's pretraining data contains incorrect or conflicting information, or when the model has not memorized a fact reliably (because it appeared rarely in training data), the model may confabulate a plausible-sounding answer rather than expressing uncertainty.

Hallucination also arises from the decoding process: higher temperature sampling introduces randomness that can lead the model away from its highest-probability (and typically most reliable) completions. Beam search or greedy decoding reduces this randomness but does not eliminate hallucination caused by the model's knowledge gaps. Additionally, models tend to be overconfident — they often express incorrect statements with the same fluency and confidence as correct ones, because training data does not systematically pair uncertain claims with uncertainty markers.

Effective mitigation strategies span architecture, training, and inference. Grounding via RAG is the most reliable approach for factual queries: if the answer comes from a retrieved document, the model can be instructed to only state what is explicitly supported by the context ("answer only based on the provided documents"), dramatically reducing factual hallucination. RLHF with factuality rewards can train models to prefer accurate, hedged responses over confident-sounding incorrect ones. At inference time, self-consistency sampling (generate multiple responses and select the majority answer) reduces hallucination on factual questions with deterministic answers. Uncertainty quantification approaches — asking the model to assess its own confidence or generating multiple samples and measuring agreement — can identify when the model is likely hallucinating, even if they cannot always correct it. For production systems, the most reliable strategy combines RAG for grounding, careful instruction design that asks models to cite sources and express uncertainty, and output validation where answers are verified against retrieved evidence.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Pretraining objective (next token prediction) | Pretraining & Architecture |
| Q2 | Basic | Scaling Laws | Pretraining & Architecture |
| Q3 | Basic | Positional encoding: RoPE vs ALiBi vs absolute | Pretraining & Architecture |
| Q4 | Advanced | Long-context extension | Pretraining & Architecture |
| Q5 | Advanced | Mixture of Experts (MoE) | Pretraining & Architecture |
| Q6 | Basic | Supervised Fine-Tuning (SFT) | Alignment & Fine-tuning |
| Q7 | Basic | RLHF pipeline | Alignment & Fine-tuning |
| Q8 | Basic | Direct Preference Optimization (DPO) | Alignment & Fine-tuning |
| Q9 | Advanced | RLHF challenges and limitations | Alignment & Fine-tuning |
| Q10 | Advanced | Alignment tax and RLHF vs DPO | Alignment & Fine-tuning |
| Q11 | Basic | KV Cache | Inference & Efficiency |
| Q12 | Basic | Model quantization (INT8/INT4) | Inference & Efficiency |
| Q13 | Basic | Speculative decoding | Inference & Efficiency |
| Q14 | Advanced | Continuous batching | Inference & Efficiency |
| Q15 | Advanced | FlashAttention in LLM inference | Inference & Efficiency |
| Q16 | Basic | Retrieval-Augmented Generation (RAG) | Applications & Engineering |
| Q17 | Basic | Prompt engineering techniques | Applications & Engineering |
| Q18 | Advanced | LLM agents: tool use and planning | Applications & Engineering |
| Q19 | Advanced | RAG limitations and long context vs RAG | Applications & Engineering |
| Q20 | Advanced | Hallucination: causes and mitigation | Applications & Engineering |

## Resources

- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020)
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla, 2022)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, 2022)
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (DPO, 2023)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE, 2021)
- Press et al., [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) (ALiBi, 2022)
- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Dao et al., [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023)
- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (RAG, 2020)
