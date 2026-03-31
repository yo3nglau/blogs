---
title: "Transformer and Vision Transformer: Interview Questions and Answers"
author: yo3nglau
date: '2026-03-31'
categories:
  - Interview
tags:
  - Deep Learning
  - Transformer
  - Vision Transformer
toc: true
---

## Transformer Fundamentals

### Q1 [Basic] What is the Self-Attention mechanism?

**Q:** Can you explain what the Self-Attention mechanism is and how it works?

**A:** Self-Attention allows each token in a sequence to attend to every other token, enabling the model to capture dependencies regardless of distance. For each position, the mechanism computes a weighted sum of all values in the sequence, where the weights reflect how relevant each position is to the current one.

The computation proceeds in three steps. First, each input embedding is projected into three vectors: a Query (Q), a Key (K), and a Value (V). Second, attention scores are computed as the dot product of Q with all Keys, scaled by the square root of the key dimension to prevent the softmax from saturating (without scaling, large dot products push softmax into regions with near-zero gradients): score = QK^T / sqrt(d_k). Third, these scores are passed through a softmax to obtain attention weights, which are then used to compute a weighted sum of the Values.

The key advantage over recurrence is that all positions are processed simultaneously, and the model can directly access any position in the sequence with equal computational cost, making it highly effective for capturing long-range dependencies.

---

### Q2 [Basic] What are Query, Key, and Value in Self-Attention?

**Q:** What do the Query, Key, and Value vectors represent in the Attention mechanism?

**A:** Query, Key, and Value are three learned linear projections of the input embeddings, each with its own weight matrix (W_Q, W_K, W_V). The intuition comes from information retrieval: the Query represents what the current position is looking for, the Key represents what each position can offer, and the Value represents the actual content to be retrieved.

Attention weights are computed by comparing the Query of one position against the Keys of all positions. A high dot product between a Query and a Key means that position is highly relevant and will receive a larger weight in the final weighted sum of Values. This allows the model to selectively focus on the most relevant parts of the sequence for each position.

In practice, the weight matrices are learned end-to-end during training, so the model learns to encode useful queries, keys, and values for the task at hand without any explicit supervision on the attention patterns themselves.

---

### Q3 [Basic] What is Multi-Head Attention and why is it used?

**Q:** What is Multi-Head Attention and what advantage does it offer over single-head attention?

**A:** Multi-Head Attention runs h independent attention operations (heads) in parallel, each with its own Q, K, V projections into a lower-dimensional subspace. The outputs of all heads are concatenated and projected back to the original dimension via a final linear layer.

The motivation is that a single attention head is constrained to represent one type of relationship between positions. With multiple heads, different heads can attend to different aspects simultaneously — for example, one head might capture syntactic dependencies while another captures semantic similarity, or in vision, one head might attend to local texture while another captures global structure.

The total computational cost is kept roughly constant by dividing the model dimension equally across heads: if d_model = 512 and h = 8, each head operates in a 64-dimensional subspace, so the total parameter count is similar to a single full-dimensional attention.

---

### Q4 [Basic] What is Positional Encoding and why does Transformer need it?

**Q:** Why does the Transformer need positional encoding, and how does the original paper implement it?

**A:** Self-Attention is permutation invariant: if you shuffle the input tokens, the attention weights change but the model has no built-in way to know the original order. This is fundamentally different from RNNs, which process tokens sequentially and inherently encode position through recurrence. Without positional information, the Transformer would treat "The cat sat on the mat" identically to "mat the on sat cat The."

The original Transformer paper (Vaswani et al., 2017) addresses this with fixed sinusoidal positional encodings added to the input embeddings before the first layer. For each position pos and each dimension i:

PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

The alternating sine/cosine functions at different frequencies allow the model to learn to attend by relative positions, since PE(pos+k) can be expressed as a linear function of PE(pos). Later work replaced fixed encodings with learned positional embeddings (BERT, GPT) or relative positional encodings (Rotary Position Embedding, ALiBi), which can generalize better to sequence lengths not seen during training.

---

### Q5 [Basic] What is the difference between the Encoder and Decoder in the original Transformer?

**Q:** How does the Encoder differ from the Decoder in the Transformer architecture?

**A:** The Encoder processes the entire input sequence in parallel using bidirectional self-attention, meaning each position can attend to every other position in both directions. Each encoder layer consists of a Multi-Head Self-Attention sublayer followed by a Feed-Forward Network (FFN) sublayer, each wrapped with residual connections and Layer Normalization. The encoder produces a sequence of contextualized representations that capture the full input context.

The Decoder generates the output sequence autoregressively, one token at a time. It has three sublayers: a Masked Self-Attention layer (which prevents each position from attending to future positions, preserving the autoregressive property), a Cross-Attention layer (where the Queries come from the decoder and the Keys and Values come from the encoder output), and an FFN layer. The cross-attention allows the decoder to selectively focus on relevant parts of the input sequence at each generation step.

In modern practice, many architectures use only the encoder (e.g., BERT for representation learning) or only the decoder (e.g., GPT for generation), since the original encoder-decoder design is primarily suited for sequence-to-sequence tasks like machine translation.

---

### Q6 [Advanced] What is the computational complexity of Self-Attention and how can it be optimized?

**Q:** What is the time and space complexity of Self-Attention, and what approaches exist to reduce it?

**A:** Standard Self-Attention has O(n^2 · d) time complexity and O(n^2) memory complexity, where n is the sequence length and d is the model dimension. The bottleneck is the attention matrix QK^T, which is n × n. For a sequence of 1,000 tokens, this is manageable; for 10,000 tokens (common in document processing or high-resolution images), the quadratic cost becomes prohibitive.

Several approaches have been proposed to reduce this cost. Sparse attention methods (Longformer, BigBird) restrict each token to attend only to a subset of positions — for example, a local window plus a few global tokens — reducing complexity to O(n · w) where w is the window size. Linear attention approximations (Performer, Linear Transformer) reformulate the attention computation to avoid materializing the full n × n matrix, achieving O(n) complexity at the cost of some approximation.

FlashAttention (Dao et al., 2022) is a hardware-aware exact attention implementation that computes attention in tiles that fit in fast on-chip SRAM, avoiding repeated reads from slow GPU HBM (global memory). It achieves O(n^2) time but O(n) memory and is 2-4× faster in practice due to reduced memory I/O — it has become the standard implementation in most modern frameworks. For vision, Swin Transformer addresses the quadratic cost by computing attention within fixed local windows rather than globally.

---

### Q7 [Advanced] Why does the Transformer use Layer Normalization instead of Batch Normalization?

**Q:** What is the reason Transformers use Layer Normalization rather than Batch Normalization?

**A:** Batch Normalization normalizes across the batch dimension, computing mean and variance statistics over a mini-batch for each feature. This works well for fixed-size inputs in computer vision (e.g., image classification) but has two problems in the Transformer setting: first, sequences in NLP have variable lengths, making it difficult to define a consistent batch-level statistic; second, with small batch sizes — common in large-model training — BN statistics become noisy and unstable.

Layer Normalization instead normalizes across the feature dimension for each individual sample, independent of the batch. This means the statistics are computed per token, making it robust to variable sequence lengths and batch size. The normalization is: LN(x) = (x - μ) / σ · γ + β, where μ and σ are computed over the d_model features of that single token.

An important design choice is whether to apply LN before or after each sublayer (Pre-LN vs Post-LN). The original Transformer paper uses Post-LN (apply after residual addition), which can lead to unstable gradients in deep networks. Pre-LN (apply before the sublayer, inside the residual branch) has been shown empirically and theoretically to produce more stable gradient flow, and is now standard in most modern Transformer implementations including GPT-2 and onward.

---

### Q8 [Advanced] Why do Transformers parallelize better than RNNs during training?

**Q:** What architectural property makes Transformers more parallelizable than RNNs during training?

**A:** RNNs have a fundamental sequential dependency: the hidden state h_t is computed from h_{t-1} and the current input x_t. This means the computation at position t cannot begin until position t-1 is complete, making it impossible to parallelize across time steps. For a sequence of length n, this creates a critical path of n sequential operations regardless of hardware.

The Transformer eliminates this dependency. In Self-Attention, the output at every position is a function only of the input embeddings and the learned weight matrices — not of any previously computed hidden state. This means all n output representations can be computed simultaneously via matrix multiplication: Q = XW_Q, K = XW_K, V = XW_V, output = softmax(QK^T / sqrt(d_k))V. On modern GPUs and TPUs, these are highly optimized batched matrix operations.

It is important to note that this parallelism applies only during training. At inference time, autoregressive Transformer decoders (GPT-style) must still generate tokens one at a time, since each new token depends on all previously generated tokens. Techniques like speculative decoding and parallel decoding attempt to recover some inference-time parallelism, and models like BERT that use the encoder only remain fully parallel at inference.
