---
title: "Multi-label Learning: Interview Questions and Answers"
author: yo3nglau
date: '2026-03-31'
categories:
  - Interview
tags:
  - Deep Learning
  - Multi-label Learning
  - Classification
toc: true
---

## Fundamentals

### Q1 [Basic] What is multi-label image classification, and how does it differ from multi-class and multi-task classification?

**Q:** Can you explain the problem formulation of multi-label image classification and how it differs from related settings?

**A:** In multi-class image classification, each image belongs to exactly one of $L$ mutually exclusive categories. A softmax output enforces a probability distribution that sums to 1, and a single cross-entropy loss drives the model toward one dominant class. Examples include ImageNet classification and digit recognition.

Multi-label image classification relaxes the mutual-exclusivity constraint: each image can simultaneously belong to any subset of $L$ classes. The canonical example is MS-COCO, where a single image may contain "person", "bicycle", and "traffic light" all at once. Formally, the goal is to learn $f: \mathcal{X} \to \{0,1\}^L$, mapping each image to a binary label vector. Architecturally this means replacing the softmax with $L$ independent sigmoid activations — each output $\hat{p}_j = \sigma(z_j) \in (0,1)$ estimates the probability that label $j$ is present, independently of the others.

Multi-task classification is a different setting: it involves multiple heterogeneous tasks (e.g., predicting object category, bounding box, depth, and pose simultaneously), each with its own loss and output head. The tasks share a backbone but have semantically distinct outputs. Multi-label classification is a special case of multi-task where all tasks are binary predictions over a homogeneous label set, sharing not just the backbone but also the loss structure.

---

### Q2 [Basic] How is the standard CNN + sigmoid + BCE baseline constructed for multi-label image classification?

**Q:** What does the standard multi-label image classification baseline look like in practice?

**A:** The standard baseline follows a simple three-part structure: backbone → pooling → classification head. A convolutional backbone (e.g., ResNet-50, ResNet-101) or Vision Transformer (e.g., ViT-B/16) processes the input image and produces a feature map or token sequence. Global average pooling collapses spatial dimensions into a single vector $f(x) \in \mathbb{R}^d$. A fully-connected layer then projects this to $L$ logits $z \in \mathbb{R}^L$, and sigmoid activations $\hat{p}_j = \sigma(z_j)$ produce per-label probabilities.

Training minimizes Binary Cross-Entropy (BCE) averaged over all label-instance pairs:

$$\mathcal{L}_\text{BCE} = -\frac{1}{NL} \sum_{i=1}^N \sum_{j=1}^L \left[ y_{ij} \log \hat{p}_{ij} + (1 - y_{ij}) \log(1 - \hat{p}_{ij}) \right]$$

At inference, a threshold $\tau$ (default 0.5) converts probabilities to binary predictions: $\hat{y}_j = \mathbf{1}[\hat{p}_j \geq \tau]$.

This baseline is essentially the neural-network instantiation of Binary Relevance: one independent binary classifier per label, sharing a common feature extractor. Its simplicity and scalability make it a strong and widely-used starting point. The critical limitation is the independence assumption built into both the sigmoid activation and the BCE loss — labels are treated as conditionally independent given the image, ignoring co-occurrence structure. This limitation motivates graph-based and attention-based models that explicitly capture label correlations.

---

### Q3 [Basic] What is Binary Relevance, and why does modeling label correlations matter?

**Q:** What is the Binary Relevance approach, and what problem arises from ignoring label correlations?

**A:** Binary Relevance (BR) is the simplest problem transformation strategy for multi-label learning: it decomposes the $L$-label problem into $L$ independent binary classification problems, one per label. Each binary classifier $f_j: \mathcal{X} \to \{0,1\}$ is trained and applied separately, with no information shared between labels at prediction time. In the deep learning era, a shared backbone with $L$ independent sigmoid outputs and BCE loss is the neural equivalent of BR.

BR is computationally attractive: classifiers can be trained in parallel, the approach scales to any number of labels, and it makes no assumptions about the label distribution. If the labels were truly conditionally independent given the image, BR would be statistically optimal.

In practice, labels are rarely independent. On MS-COCO, "bicycle" and "person" co-occur far more than chance; "umbrella" and "rain" are semantically linked; "fork" and "dining table" almost always appear together. Ignoring these correlations means the model misses opportunities to improve prediction consistency — it might confidently predict "dog" while missing "leash", even though the latter strongly implies the former. Empirically, models that explicitly leverage label co-occurrence (e.g., ML-GCN) consistently outperform the BR baseline on standard benchmarks, especially for tail labels whose individual training signal is sparse but whose co-occurrence with common labels is informative.

---

### Q4 [Advanced] How do Classifier Chains model label dependencies, and what are their failure modes?

**Q:** Describe the Classifier Chains method and explain where and why it breaks down.

**A:** Classifier Chains (CC), introduced by Read et al. (2009), extends Binary Relevance by arranging the $L$ labels in a fixed order $y_1, y_2, \ldots, y_L$ and training a chain of binary classifiers where each classifier $f_j$ receives both the original image features $x$ and the predictions of all preceding classifiers:

$$f_j\!\left(x,\, \hat{y}_1, \ldots, \hat{y}_{j-1}\right) \to \hat{y}_j$$

This corresponds to factoring the joint label distribution as:

$$P(y_1, \ldots, y_L \mid x) = \prod_{j=1}^{L} P\!\left(y_j \mid x,\, y_1, \ldots, y_{j-1}\right)$$

The key advantage over BR is that downstream classifiers can exploit label correlations: if $\hat{y}_1 = 1$ (dog detected), classifier $f_2$ can use this signal to boost the probability of "leash". In theory, with a perfect ordering and perfect preceding predictions, CC recovers the exact joint distribution.

The first failure mode is **order sensitivity**. The chain encodes a directed dependency structure determined by the label ordering, but the true dependency graph is undirected and unknown. Labels placed early cannot benefit from labels placed late, regardless of their actual correlation. Random orderings lose important dependencies, and optimal ordering requires prior knowledge of the dependency structure.

The second is **error propagation**. During inference, $\hat{y}_1, \ldots, \hat{y}_{j-1}$ are model predictions, not ground truth. Incorrect early predictions cascade through the chain, compounding errors. A single early misclassification can invalidate downstream decisions. Ensemble of Classifier Chains (ECC) mitigates both failure modes by training multiple chains with different random orderings and aggregating by majority vote, but at the cost of $k \times L$ total classifiers for $k$ chains.

---

### Q5 [Advanced] How is the label co-occurrence graph constructed, and what quality issues arise?

**Q:** How do you build a label co-occurrence graph from training data, and what can go wrong?

**A:** A label co-occurrence graph $G = (V, E)$ has one node per label and edge weights encoding statistical dependencies between labels. The most common construction uses conditional probability: for labels $i$ and $j$, compute $A_{ij} = P(y_j = 1 \mid y_i = 1)$ from training set counts:

$$A_{ij} = \frac{\text{count}(y_i = 1 \wedge y_j = 1)}{\text{count}(y_i = 1)}$$

This produces an asymmetric matrix (co-occurrence is not symmetric in general). To obtain a symmetric graph, the matrix is often symmetrized as $A_\text{sym} = (A + A^\top) / 2$, then symmetrically normalized as $\hat{A} = D^{-1/2} A_\text{sym} D^{-1/2}$ for use in GCNs.

Several quality issues arise in practice. First, **spurious correlations from dataset bias**: if a dataset was collected in a particular context (e.g., most outdoor images in COCO happen to also contain people), the graph encodes these collection biases rather than true semantic relationships. A GCN trained on this graph will learn to predict "person" whenever an outdoor scene is detected, regardless of whether a person is actually present.

Second, **small-sample noise for rare labels**: labels with few positive examples produce unreliable conditional probability estimates. A label appearing in only 50 images yields high-variance statistics.

Third, **density vs. sparsity trade-off**: a dense graph with low-threshold edges introduces noisy long-range connections; a too-sparse graph misses useful correlations. ML-GCN addresses this with a sparsification threshold $\tau$ — entries below $\tau$ are zeroed out — and a re-weighting scheme that down-weights very high conditional probabilities (near 1.0) to prevent trivial shortcuts where one label always predicts another.

---
