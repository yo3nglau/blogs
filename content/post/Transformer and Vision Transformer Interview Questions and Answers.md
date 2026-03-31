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

---

## Vision Transformer (ViT)

### Q9 [Basic] How does Patch Embedding work in ViT?

**Q:** How does ViT convert an image into a sequence of tokens?

**A:** ViT (Dosovitskiy et al., 2020) divides the input image into a grid of non-overlapping fixed-size patches. For a 224×224 image with a patch size of 16×16, this produces (224/16)^2 = 196 patches. Each patch is flattened into a 1D vector of size 16×16×3 = 768 (for RGB images) and then linearly projected to the model dimension d_model using a learned weight matrix. This linear projection layer is the patch embedding.

Learnable 1D position embeddings are added to each patch embedding to provide spatial information, since the Transformer itself is permutation invariant. The resulting sequence of 196 embedded patches, plus a prepended [CLS] token, is fed into the standard Transformer encoder.

Conceptually, this design treats image patches as the equivalent of words in NLP: each patch is a "token," and the Transformer learns to model relationships between patches via Self-Attention. The patch size is a key hyperparameter — smaller patches give more tokens and finer granularity but increase sequence length quadratically.

---

### Q10 [Basic] What is the role of the [CLS] token in ViT?

**Q:** What does the [CLS] token do in Vision Transformer?

**A:** The [CLS] (classification) token is a learnable embedding prepended to the sequence of patch embeddings before the Transformer encoder. It does not correspond to any image patch. After the sequence passes through all Transformer layers, the [CLS] token's output representation is used as the global image representation and fed into the classification head (a linear layer or MLP) to produce the final prediction.

The rationale is that through Self-Attention, the [CLS] token can attend to all patch tokens and aggregate information from the entire image. Because the [CLS] token has no fixed spatial meaning, it is free to learn to aggregate the most task-relevant information across all patches during training.

This design was borrowed directly from BERT (Devlin et al., 2019), where the [CLS] token serves the same aggregation role for sentence-level classification. An alternative is global average pooling over all patch token outputs, which has been shown to perform comparably or better in some settings, but the [CLS] token design has remained standard in ViT.

---

### Q11 [Basic] Why does ViT require large amounts of training data?

**Q:** Why does ViT typically need large-scale datasets to match CNN performance?

**A:** CNNs incorporate strong inductive biases by design: convolutional filters enforce locality (each filter sees only a local patch), and weight sharing across spatial positions encodes translation equivariance. These biases are well-matched to natural images and allow CNNs to learn effective visual features from relatively small datasets.

ViT, in contrast, has minimal inductive bias. Patch embeddings are flat and unstructured; the Self-Attention mechanism treats all patches equally and must learn from data alone that nearby patches tend to be more related than distant ones. Without enough data, ViT overfits and fails to learn the spatial structure that CNNs get for free from their architecture.

The original ViT paper demonstrated this clearly: ViT trained on ImageNet-1k (1.2M images) underperforms ResNets, but when pretrained on ImageNet-21k (14M images) or JFT-300M (300M images), ViT matches or exceeds comparable CNNs. This data dependency is the primary practical limitation of pure ViT models, and has motivated data-efficient variants (DeiT) and self-supervised pretraining approaches (MAE, DINO).

---

### Q12 [Basic] What are the core differences between ViT and CNNs?

**Q:** What are the fundamental architectural and behavioral differences between ViT and CNNs?

**A:** The most fundamental difference is the receptive field. In a CNN, each neuron in early layers sees only a small local patch; the receptive field grows layer by layer. In ViT, every patch token attends to every other patch token from the very first layer — the receptive field is global from the start. This means ViT can capture long-range spatial dependencies that CNNs can only model in later, deeper layers.

In terms of inductive biases, CNNs have locality (filters are spatially local), weight sharing (same filter applied across the image), and translation equivariance built in. ViT has none of these — it learns them from data. This makes ViT more flexible but more data-hungry.

From a practical standpoint, CNNs tend to outperform ViTs when labeled data is limited, while ViTs become competitive or superior at scale (large datasets, large models). ViTs also scale more predictably with model size and data, following power-law scaling behavior similar to language models. Additionally, the Transformer architecture is more amenable to multi-modal extensions (e.g., combining image and text tokens in a unified sequence), which has made ViT the dominant backbone in modern vision-language models.

---

### Q13 [Advanced] What is the inductive bias problem in ViT?

**Q:** What does it mean that ViT has less inductive bias than CNNs, and why does this matter?

**A:** Inductive bias refers to the set of assumptions an architecture makes about the structure of data, independent of what it learns from training examples. CNNs encode two strong inductive biases: locality (convolutions operate on local spatial neighborhoods, assuming nearby pixels are more related) and translation equivariance (the same feature detector is applied everywhere, assuming that a feature is useful regardless of where it appears in the image).

ViT intentionally discards these biases. Patch embeddings treat each patch as an unordered token, and Self-Attention computes relationships between all pairs of patches with no preference for spatially close pairs. The model must learn from data that, for example, adjacent patches tend to be more correlated than distant ones — something a CNN gets for free.

This has two consequences. On the negative side, ViT requires substantially more data to reach the same performance as CNNs, because it cannot rely on architectural shortcuts. On the positive side, the lack of hard-coded biases means ViT is more flexible: it can learn non-local patterns that CNNs might miss, and the same architecture can be applied with minimal modification to other modalities (text, audio, point clouds) by simply changing the tokenization strategy. Hybrid architectures — using a CNN for early local feature extraction and a Transformer for global reasoning — attempt to get the best of both worlds.

---

### Q14 [Advanced] How does DeiT address ViT's data dependency problem?

**Q:** What techniques does DeiT introduce to train ViT-like models efficiently without large proprietary datasets?

**A:** DeiT (Data-efficient Image Transformers, Touvron et al., 2021) shows that a ViT-sized model can be trained competitively on ImageNet-1k alone (without JFT or ImageNet-21k pretraining) through two main contributions: aggressive data augmentation and knowledge distillation from a CNN teacher.

On the augmentation side, DeiT applies a combination of RandAugment, Mixup, CutMix, random erasing, and repeated augmentation. These techniques significantly expand the effective training distribution, compensating for the lack of additional data and helping ViT learn robust features without overfitting.

The key architectural innovation is the distillation token. Alongside the [CLS] token, DeiT prepends a learnable distillation token to the patch sequence. During training, this token is supervised to match the hard labels produced by a pretrained CNN teacher (typically a RegNet or EfficientNet), while the [CLS] token is supervised with the true labels via cross-entropy. At inference, predictions from both tokens are ensembled. The distillation token allows the student ViT to inherit the inductive biases of the CNN teacher implicitly through the soft supervision signal, without requiring explicit architectural changes.

---

### Q15 [Advanced] How does Swin Transformer's shifted window attention work?

**Q:** What problem does Swin Transformer solve, and how does its shifted window mechanism work?

**A:** Swin Transformer (Liu et al., 2021) addresses two limitations of ViT: the O(n^2) complexity of global attention, which becomes prohibitive for high-resolution images, and the lack of a hierarchical feature representation (ViT produces a single-scale feature map, while CNNs produce multi-scale pyramids useful for dense prediction tasks like detection and segmentation).

Swin partitions the image into non-overlapping local windows of fixed size (e.g., 7×7 patches) and computes Self-Attention only within each window. If the image has N patches and each window has M patches, the complexity drops from O(N^2) to O(N · M), which is linear in image size for fixed M. This makes high-resolution processing feasible.

The "shifted window" mechanism addresses the limitation that attention within fixed windows creates no cross-window communication. In alternating Transformer layers, Swin shifts the window partition by half a window size in both height and width directions. This causes windows from the previous layer to straddle window boundaries in the current layer, allowing indirect information flow between adjacent windows. To maintain efficiency, shifted windows that cross image boundaries are handled with a cyclic shifting and masking trick rather than padding. Swin also uses patch merging layers (similar to CNN pooling) to progressively reduce spatial resolution and double channel dimensions, producing the hierarchical feature maps needed for dense prediction.
