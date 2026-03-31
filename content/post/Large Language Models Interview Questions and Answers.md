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
