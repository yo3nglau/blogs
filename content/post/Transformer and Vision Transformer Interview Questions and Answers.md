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

The computation proceeds in three steps. First, each input embedding is projected into three vectors: a Query (Q), a Key (K), and a Value (V). Second, attention scores are computed as the dot product of Q with all Keys, scaled by the square root of the key dimension to prevent extremely small gradients: score = QK^T / sqrt(d_k). Third, these scores are passed through a softmax to obtain attention weights, which are then used to compute a weighted sum of the Values.

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
