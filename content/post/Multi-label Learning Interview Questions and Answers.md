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

## Architecture & Modeling

### Q6 [Basic] How does ML-GCN use a Graph Convolutional Network to learn label-aware classifiers?

**Q:** What is the core idea of ML-GCN, and how does it improve upon the standard sigmoid-BCE baseline?

**A:** ML-GCN (Multi-Label Graph Convolutional Network, Chen et al., CVPR 2019) reframes the classification head as a graph-conditioned output, rather than a fixed linear projection. The key insight is that each label's classifier (a weight vector $w_j \in \mathbb{R}^d$) should be informed by the classifiers of labels that frequently co-occur with it. ML-GCN computes all $L$ classifier vectors jointly by propagating label embeddings through a GCN over the co-occurrence graph.

Concretely, the pipeline has two branches. The image branch runs a CNN backbone (ResNet-101) to produce a global feature vector $f(x) \in \mathbb{R}^d$. The label branch initializes each label as a $d_0$-dimensional word vector (from GloVe embeddings of the label name), stacks these into $X^{(0)} \in \mathbb{R}^{L \times d_0}$, and applies two GCN layers:

$$X^{(l+1)} = \text{ReLU}\!\left(\hat{A}\, X^{(l)}\, W^{(l)}\right)$$

The GCN output $W = X^{(2)} \in \mathbb{R}^{L \times d}$ serves as the classification head. Final multi-label scores are $z = W \cdot f(x) \in \mathbb{R}^L$, trained with BCE. Because $W$ is produced by GCN propagation over the co-occurrence graph, each $w_j$ encodes not only label $j$'s own semantics but also the semantics of its graph neighbors — labels that frequently co-occur share discriminative information. ML-GCN achieved state-of-the-art mAP of 83.0 on MS-COCO (with ResNet-101) at publication, establishing GCN-based label correlation modeling as a standard technique.

---

### Q7 [Basic] How does Q2L (Query2Label) apply Transformer decoding to multi-label image prediction?

**Q:** What is the design of Q2L and why is it effective for multi-label image classification?

**A:** Query2Label (Q2L, Liu et al., NeurIPS 2021) treats multi-label classification as a set prediction problem and solves it with a Transformer decoder architecture inspired by DETR. The core idea is simple: use $L$ learnable label embeddings as queries in a cross-attention decoder that attends over the image's spatial feature tokens. Each query is responsible for predicting one label, and its output representation is passed through a binary classifier (a linear layer + sigmoid).

The decoder's cross-attention mechanism allows each label query to selectively attend to the image regions most relevant to its label. For example, the query corresponding to "bicycle" will attend strongly to wheel and frame regions, while "person" attends to body regions. Label co-occurrence is modeled implicitly: all queries attend over the same image features, so queries for co-occurring labels tend to activate overlapping image regions, creating inter-label consistency without an explicit co-occurrence graph.

In practice, Q2L uses a ViT or Swin Transformer as the image encoder, producing a sequence of patch tokens. The label queries are learned parameters initialized randomly and trained end-to-end. Self-attention layers among the label queries (in a full encoder-decoder stack) allow explicit label-to-label communication. Q2L achieved 91.3 mAP on MS-COCO with a TResNet-L backbone + CutMix + ASL training, outperforming ML-GCN by a substantial margin and requiring no hand-crafted co-occurrence graph.

---

### Q8 [Advanced] How does ML-Decoder scale multi-label classification to large label spaces?

**Q:** What problem does ML-Decoder solve compared to Q2L, and how does its grouped decoder design work?

**A:** Q2L's cross-attention decoder has complexity $O(L \cdot N)$ per layer, where $L$ is the number of labels and $N$ is the number of image tokens. For $L = 80$ (MS-COCO), this is manageable. But for $L = 1{,}000$ or $L = 10{,}000$ (common in product classification or web-scale tagging), the attention becomes prohibitively expensive and the $L$ query parameters themselves consume significant memory.

ML-Decoder (Ridnik et al., 2021) addresses this with a grouped decoding strategy. Instead of one query per label, ML-Decoder groups the $L$ labels into $G$ groups ($G \ll L$, e.g., $G = 111$ for $L = 80{,}000$). Within each group, a single shared query vector attends to the image tokens via cross-attention, producing a group representation. This group representation is then split into $L/G$ per-label representations through a lightweight per-label MLP head. The total cross-attention complexity drops from $O(L \cdot N)$ to $O(G \cdot N)$, making large-label-space classification feasible.

A key design choice in ML-Decoder is the use of "queries as embeddings": rather than random initialization, the group queries are initialized from the averaged word vectors of the labels assigned to each group, providing semantic priors that accelerate convergence. ML-Decoder achieves performance competitive with Q2L on MS-COCO while being approximately $10\times$ faster at inference for large label spaces. It has become the preferred classification head for production-scale multi-label systems.

---

### Q9 [Advanced] How does CLIP enable zero-shot and few-shot multi-label image classification?

**Q:** How can CLIP be applied to multi-label classification without multi-label fine-tuning?

**A:** CLIP (Radford et al., 2021) learns a joint image-text embedding space by training on 400 million image-text pairs with a contrastive objective: cosine similarity between aligned image-text pairs is maximized, and between misaligned pairs minimized. The result is an embedding space where visual concepts and their textual descriptions are geometrically close.

For zero-shot multi-label classification, each label $j$ is converted to a text prompt (e.g., "a photo of a {label}") and embedded by the text encoder to obtain $t_j \in \mathbb{R}^{d}$. The query image is embedded by the image encoder to $v \in \mathbb{R}^{d}$. Per-label relevance scores are computed as cosine similarities $s_j = v^\top t_j / (\|v\| \|t_j\|)$, and a threshold is applied to produce binary predictions. No multi-label training data is required.

The main limitation is threshold calibration: CLIP scores are not calibrated probabilities, and the optimal threshold varies by label, dataset, and prompt template. In practice, a small held-out labeled set is used to tune per-label thresholds even in the zero-shot regime. For few-shot adaptation, prompt tuning (CoCoOp, ProDA) learns soft prompt tokens while keeping CLIP weights frozen, achieving strong performance with as few as 1–16 labeled examples per label. CLIP-based methods have demonstrated competitive multi-label mAP on MS-COCO and NUS-WIDE without any standard multi-label training, making them particularly attractive when labeled data is scarce or when the label set changes frequently.

---

### Q10 [Advanced] How does self-supervised pretraining (MAE, DINO) benefit multi-label image classification?

**Q:** Why do MAE- and DINO-pretrained ViTs transfer better to multi-label tasks than supervised ImageNet pretraining?

**A:** Supervised ImageNet pretraining optimizes the backbone to predict a single dominant class label per image. This drives the model to focus on the most discriminative region — the region that most reliably distinguishes one class from all others. For a "Labrador" image, the feature map activates strongly on the dog's face and body, largely ignoring background objects. This single-class bias is harmless for classification but problematic for multi-label tasks where all co-occurring objects must be localized and recognized.

MAE (He et al., 2022) masks $75\%$ of image patches and trains a ViT encoder to reconstruct the full image from only $25\%$ visible patches via a lightweight decoder. To reconstruct masked regions, the encoder must build rich, contextual representations of the entire visible scene — it cannot afford to focus on a single object. This produces holistic features that encode multiple co-occurring objects, scene context, and fine-grained texture. Empirically, MAE-pretrained ViT-L/H features show significantly higher multi-label recall on MS-COCO compared to supervised ViT of the same size.

DINO (Caron et al., 2021) trains a ViT via self-distillation: a student network is trained to match the output of a momentum-updated teacher on different augmented views of the same image, without any labels. A remarkable emergent property of DINO features is that the [CLS] token's self-attention maps cleanly segment foreground objects from background, separating multiple co-occurring objects without any segmentation supervision. These attention maps transfer directly to multi-label localization, enabling DINO-pretrained models to achieve strong multi-label performance with a simple linear probe. Fine-tuning strategy: unfreeze the last 2–4 Transformer blocks and the classification head; freezing earlier blocks prevents overfitting the holistic representations to the multi-label training set.

---
## Loss & Training

### Q11 [Basic] Why is Binary Cross-Entropy used as the loss function for multi-label classification, and what is its core limitation?

**Q:** What justifies the use of BCE for multi-label training, and where does it fall short?

**A:** Multi-label classification is structurally a set of $L$ independent binary classification problems sharing a feature extractor. For each label $j$ and instance $i$, the model outputs $\hat{p}_{ij} = \sigma(z_{ij}) \in (0,1)$ — the estimated probability that label $j$ is present. If these $L$ binary variables are treated as conditionally independent given the image, the joint negative log-likelihood factorizes into a sum of per-label binary cross-entropies:

$$\mathcal{L}_\text{BCE} = -\frac{1}{NL} \sum_{i=1}^N \sum_{j=1}^L \left[ y_{ij} \log \hat{p}_{ij} + (1 - y_{ij}) \log(1 - \hat{p}_{ij}) \right]$$

This contrasts with softmax cross-entropy, which enforces a normalized probability distribution across mutually exclusive classes. Sigmoid + BCE imposes no normalization constraint, allowing any subset of outputs to be simultaneously high — precisely what multi-label prediction requires. BCE is also simple to implement, differentiable everywhere, and scales linearly with $L$ and $N$.

The core limitation is the **conditional independence assumption**. In reality, labels are correlated: "surfboard" and "ocean" frequently co-occur; "tie" and "person" almost always appear together. By treating labels independently, BCE wastes potential information in label co-occurrence patterns and can produce inconsistent predictions (e.g., "surfboard" predicted positive while "person" is negative, despite "person" being nearly always present in surfing images). This limitation motivates graph-regularized losses, pairwise ranking losses, and the structured label modeling approaches covered in Q6–Q7.

---

### Q12 [Basic] What is Focal Loss, and how does it address class imbalance in multi-label settings?

**Q:** How does Focal Loss work and why is it beneficial for multi-label image classification?

**A:** Focal Loss was introduced by Lin et al. (2017) for single-stage object detection to address the extreme foreground-background imbalance problem. The same challenge manifests in multi-label classification: for each training instance, most of the $L$ labels are negative (e.g., in an MS-COCO image with 3 positive labels out of 80, 77 labels are negative). Under standard BCE, easy negatives — true negatives the model correctly predicts with high confidence — contribute the majority of the total loss and gradient, swamping the signal from hard positives and hard negatives.

Focal Loss modifies BCE by adding a modulating factor $(1 - p_t)^\gamma$, where $p_t$ is the model's estimated probability for the correct label:

$$\mathcal{L}_\text{FL}(p_t) = -(1 - p_t)^\gamma \log p_t$$

When $\gamma = 0$, this reduces to standard BCE. For easy examples (high $p_t$), the factor $(1 - p_t)^\gamma$ approaches 0, down-weighting their contribution. For hard examples (low $p_t$), the factor remains close to 1, preserving their gradient. The focusing parameter $\gamma$ (typically $\gamma \in \{0.5, 1, 2\}$) controls how aggressively easy examples are down-weighted.

In multi-label settings, Focal Loss is applied per-label independently, replacing the BCE term for each label. It consistently improves recall on tail labels compared to plain BCE, because tail-label positives are typically hard examples that Focal Loss preserves, while the easy negatives that dominate the BCE gradient are suppressed. However, Focal Loss applies the same $\gamma$ to both positives and negatives symmetrically — a limitation addressed by Asymmetric Loss (Q14).

---

### Q13 [Basic] How are RandAugment, Mixup, and CutMix applied in multi-label image classification?

**Q:** What data augmentation strategies are most effective for multi-label image classification, and how do they interact with multi-label targets?

**A:** Data augmentation is critical for training multi-label image classifiers, especially when using ViT-based backbones that lack the inductive biases of CNNs. The three most impactful strategies are RandAugment, Mixup, and CutMix.

**RandAugment** (Cubuk et al., 2020) stochastically applies $N$ augmentation operations (chosen from a set including rotation, color jitter, sharpness, shear, translate, etc.) each with the same magnitude $M$. For multi-label classification, RandAugment is applied at the image level and the label vector remains unchanged — the augmented image is paired with the same set of ground-truth labels as the original. RandAugment expands the effective training distribution without requiring manual augmentation policy design.

**Mixup** creates synthetic training examples by linearly interpolating two images and their label vectors: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$ and $\tilde{y} = \lambda y_i + (1-\lambda)y_j$, where $\lambda \sim \text{Beta}(\alpha, \alpha)$. In multi-label settings, the mixed label vector is naturally a soft multi-label target — the model is trained to predict a blend of both label sets, which regularizes the output probabilities and improves calibration.

**CutMix** replaces a rectangular region of image $x_i$ with the corresponding region from image $x_j$. Labels are mixed proportionally to the area of the pasted region: $\tilde{y} = \lambda y_i + (1-\lambda) y_j$ where $\lambda$ is the proportion of pixels from $x_i$. CutMix is particularly well-suited to multi-label classification because different objects can genuinely appear in different spatial regions — the mixing is semantically plausible. DeiT showed that combining RandAugment + Mixup + CutMix + Repeated Augmentation can train ViT-sized models competitively on ImageNet-1k alone; the same combination is standard practice for multi-label ViT training.

---

### Q14 [Advanced] What is Asymmetric Loss, and what specific problem does it solve beyond Focal Loss?

**Q:** How does Asymmetric Loss (ASL) improve on Focal Loss for multi-label classification?

**A:** Asymmetric Loss (ASL, Ben-Baruch et al., ICCV 2021) identifies a structural asymmetry in multi-label training that symmetric losses like BCE and Focal Loss fail to exploit. In a typical multi-label image (e.g., MS-COCO with $L=80$), each instance has approximately 3 positive labels and 77 negative labels. Negative examples vastly outnumber positives. More importantly, the two populations have fundamentally different characteristics: most negatives are easy (the model correctly predicts low probability with high confidence), while many positives are hard (tail labels, occluded objects).

Focal Loss's single $\gamma$ parameter applies the same down-weighting schedule to both positives and negatives. Setting a high $\gamma$ to suppress easy negatives also down-weights easy positives — but easy positives are not a problem and should be preserved. Setting a low $\gamma$ to preserve positives fails to suppress the deluge of easy negatives.

ASL solves this with two modifications. First, **asymmetric focusing**: apply $\gamma^+ \approx 0$ to positives (no down-weighting, preserve all positive gradient) and $\gamma^- > \gamma^+$ to negatives (aggressively suppress easy negatives):

$$\mathcal{L}_\text{ASL} = \begin{cases} (1-p)^{\gamma^+} \log p & \text{if } y=1 \\ p_m^{\gamma^-} \log(1-p_m) & \text{if } y=0 \end{cases}$$

Second, **probability shifting**: replace the negative probability with $p_m = \max(p - m, 0)$, where $m \geq 0$ is a shift margin. Any negative prediction with $p < m$ contributes zero loss — these are discarded as trivially easy. This acts as automatic online hard negative mining. Typical hyperparameters: $\gamma^+ = 0$, $\gamma^- = 4$, $m = 0.05$. ASL consistently outperforms BCE and Focal Loss by $1$–$3$ mAP points on MS-COCO and OpenImages and has become the de facto loss function for state-of-the-art multi-label image classification.

---

### Q15 [Advanced] How do you train a multi-label classifier when annotations are incomplete?

**Q:** What strategies exist for handling missing or partial labels during multi-label training?

**A:** In practice, multi-label annotations are frequently incomplete. Large-scale datasets are often annotated with exhaustive labels only for a subset of images, or via crowdsourcing where annotators miss some relevant labels. If all unobserved labels are naively treated as negatives in BCE, the model receives incorrect supervision: true positives are penalized as negatives, systematically suppressing recall.

The simplest effective strategy is the **observed-only loss**: compute BCE only over the labels that were explicitly annotated for each instance, and ignore the remaining (unobserved) labels. This requires storing a per-instance annotation mask $m_{ij} \in \{0,1\}$ indicating whether label $j$ was annotated for instance $i$:

$$\mathcal{L}_\text{obs} = -\frac{1}{\sum m_{ij}} \sum_{i,j} m_{ij} \left[ y_{ij} \log \hat{p}_{ij} + (1-y_{ij}) \log(1-\hat{p}_{ij}) \right]$$

**Pseudo-labeling with iterative refinement** uses the model's own confident predictions to fill in missing annotations. After an initial training phase, predictions with $\hat{p}_{ij} > \tau_\text{high}$ or $\hat{p}_{ij} < \tau_\text{low}$ for unobserved labels are treated as soft pseudo-labels in subsequent training epochs. This iterates until convergence.

**SARB (Semantic-Aware Representation Blending, Pu et al., 2022)** constructs pseudo-complete label vectors by blending features of similar training images: for an image $x_i$ with incomplete labels, find semantically similar images $x_j$ and transfer their observed labels to $x_i$ where they are missing. This exploits the assumption that similar images share similar label sets.

**Label smoothing** applies a small negative-label weight reduction $\epsilon$: replace $y_{ij} = 0$ with $y_{ij} = \epsilon$ for unobserved labels. This reduces overconfident penalization without requiring an annotation mask, at the cost of slightly noisier gradients. In practice, the observed-only loss combined with pseudo-labeling is the most principled approach when annotation masks are available.

---

## Evaluation & Practical Considerations

### Q16 [Basic] What are the key evaluation metrics for multi-label image classification, and when is each appropriate?

**Q:** What metrics are used to evaluate multi-label classifiers, and what does each one measure?

**A:** Multi-label evaluation metrics fall into two families: **classification metrics** (requiring a threshold to produce binary predictions) and **ranking metrics** (operating on the raw score ordering).

The most common classification metrics are: **Hamming Loss** ($\text{HL} = \frac{1}{NL} \sum_{i,j} \mathbf{1}[\hat{y}_{ij} \neq y_{ij}]$), measuring the fraction of incorrect label predictions — but misleading when the label space is sparse, since always predicting "negative" achieves near-zero HL; **Subset Accuracy**, the strictest metric requiring exact match of the full label set; and **Instance F1 / Macro F1 / Micro F1**, which balance precision and recall at the label or instance level (covered in Q17).

The dominant ranking metric for multi-label image classification is **mAP (mean Average Precision)**: for each label, compute the area under the precision-recall curve across all confidence thresholds, then average over all $L$ labels. mAP is the standard benchmark metric on MS-COCO and PASCAL VOC. It rewards both correct label prediction and confident scoring of positives above negatives, without requiring threshold selection.

**Precision@$k$** measures the fraction of the top-$k$ predicted labels that are truly relevant — appropriate when the system must return exactly $k$ tags (e.g., image tagging APIs). **NDCG@$k$** additionally rewards correct labels appearing earlier in the top-$k$ list. For most CV research on MS-COCO, mAP is the primary metric; for recommendation and retrieval systems, P@$k$ and NDCG@$k$ are more deployment-relevant.

---

### Q17 [Basic] What is the difference between micro-averaged and macro-averaged F1 in multi-label evaluation?

**Q:** When should you report micro F1 versus macro F1 for a multi-label image classifier?

**A:** Both metrics aggregate per-label statistics into a single F1 score, but they weight labels differently.

**Micro F1** aggregates all true positives, false positives, and false negatives across every label before computing F1:

$$F1_\text{micro} = \frac{2 \sum_j \text{TP}_j}{2 \sum_j \text{TP}_j + \sum_j \text{FP}_j + \sum_j \text{FN}_j}$$

Because frequent labels contribute more counts, micro F1 is dominated by head classes. A model that achieves high recall on "person" (the most common label in MS-COCO) while completely failing on "toothbrush" (rare) will still report high micro F1.

**Macro F1** computes $F1_j$ independently for each label $j$, then averages equally:

$$F1_\text{macro} = \frac{1}{L} \sum_{j=1}^L F1_j$$

Every label contributes equally regardless of frequency. A model must perform well on rare tail labels to achieve high macro F1 — making it the right metric when all labels are operationally important.

The practical guideline: for systems where label importance is proportional to frequency (e.g., social media image tagging where common tags drive most queries), micro F1 is representative. For systems with equal-importance labels (e.g., medical image annotation where rare pathologies matter as much as common ones), macro F1 is essential. Standard practice in multi-label CV research is to report **both** alongside mAP, as the micro-macro gap reveals how well the model handles the long tail of rare labels.

---

### Q18 [Advanced] How do you select and tune classification thresholds in a multi-label system?

**Q:** What are the main strategies for converting multi-label scores to binary predictions, and which is most effective?

**A:** Multi-label scores $\hat{p}_j \in (0,1)$ must be thresholded to produce binary predictions $\hat{y}_j = \mathbf{1}[\hat{p}_j \geq \tau_j]$. The choice of $\tau_j$ significantly affects the precision-recall trade-off and downstream F1 / Hamming Loss.

**Fixed global threshold ($\tau = 0.5$)** assumes all labels are well-calibrated and share the same optimal operating point. In practice, different labels have different base rates and score distributions — "person" (common, high base rate) will tend to have higher sigmoid scores than "toothbrush" (rare, low base rate), so a single threshold is simultaneously too high for rare labels and too low for common ones. This approach is only defensible as a quick baseline.

**Per-label threshold tuning** tunes a separate $\tau_j$ for each label on a held-out validation set to maximize label-level F1:

$$\tau_j^* = \arg\max_\tau F1_j(\tau) \text{ on validation set}$$

This is the most reliably effective strategy and is standard in competitive multi-label systems. The cost is $L$ additional hyperparameters — manageable in practice via a grid search over $[0.05, 0.95]$ with step 0.05.

**Post-hoc temperature scaling** fits a single scalar $T$ (temperature) on the validation set to minimize NLL: logits become $z_j / T$, making all label scores simultaneously more or less extreme. This improves calibration globally but cannot correct per-label miscalibration.

**Instance-wise top-$k$** prediction outputs the $k$ labels with highest scores per image, where $k$ is set equal to the dataset's average label cardinality. This is threshold-free and useful when a fixed number of predictions is required.

In benchmark settings, per-label threshold tuning consistently outperforms a fixed $\tau = 0.5$ by 2–5 F1 points and is the standard evaluation protocol for MS-COCO multi-label benchmarks.

---

### Q19 [Advanced] What is Extreme Multi-Label Classification, and what methods address its challenges?

**Q:** How does Extreme Multi-Label Classification differ from standard multi-label learning, and what are the state-of-the-art approaches?

**A:** Extreme Multi-Label Classification (XML) refers to settings where the label space size $L$ ranges from thousands to millions ($L \in [10^3, 10^6]$), while each instance has only a small number of relevant labels (typically 5–10). Applications include Amazon product categorization ($L \approx 670{,}000$), Wikipedia article tagging ($L \approx 500{,}000$), and ad keyword prediction ($L > 10^6$).

XML presents challenges absent from standard multi-label learning. **Scalability**: a dense $L \times d$ classification head with $L = 10^6$ and $d = 256$ requires 256 GB of memory — infeasible. **Tail label learning**: most labels have fewer than 10 positive training examples, making reliable learning extremely difficult. **Evaluation**: exact set prediction is intractable; P@$k$ and NDCG@$k$ (for $k \in \{1, 3, 5\}$) are the universal metrics.

Representative approaches: **Tree-based methods** (FastXML, Parabel, PECOS) partition the label space into a hierarchical binary tree of clusters. At inference, the model traverses only $O(\log L)$ branches, enabling sub-millisecond prediction for millions of labels. **Two-stage neural methods** (AttentionXML, X-Transformer, LightXML) use a fast first-stage retrieval model (often a shallow TF-IDF or lightweight neural model) to produce a shortlist of $\sim 200$–$500$ candidate labels, then re-rank within the shortlist using a more expensive Transformer. **Negative sampling**: training with all $L$ negatives per example is infeasible; hard negatives are sampled from the shortlist produced by the retrieval stage, enabling efficient training. XML benchmarks (EUR-Lex-4K, Amazon-670K, Wiki-500K) are the standard evaluation datasets.

---

### Q20 [Advanced] How would you build a strong multi-label image classifier in practice with limited labeled data?

**Q:** What is a principled approach to training a multi-label image classifier when only limited annotated data is available?

**A:** With limited labeled data, the most impactful decisions are backbone selection, loss function, and leveraging label semantics — in that order.

**Step 1 — Start with a self-supervised pretrained backbone.** A ViT-B/16 pretrained with MAE or DINO significantly outperforms supervised ImageNet pretraining in the low-data regime, because self-supervised features encode multiple co-occurring objects rather than a single discriminative region (Q10). Use CLIP if zero-shot or cross-modal transfer is relevant. Freeze the first 8–10 Transformer blocks; fine-tune only the last 2 blocks and the classification head to avoid destroying pretrained representations with limited gradient signal.

**Step 2 — Use Asymmetric Loss.** With limited data, the positive-negative imbalance is more severe (fewer examples per tail label). ASL's probability shifting and asymmetric focusing concentrate gradient on hard informative examples, improving tail-label recall without requiring explicit class reweighting (Q14).

**Step 3 — Apply aggressive data augmentation.** Use RandAugment + CutMix together. CutMix is especially valuable in the limited-data regime: it synthesizes new training images by combining two existing ones, doubling effective diversity while preserving semantic plausibility through mixed multi-label targets (Q13).

**Step 4 — Leverage label text semantics.** Initialize the classification head from CLIP text embeddings of the label names ($t_j \in \mathbb{R}^d$ for each label $j$) rather than random initialization. This provides a semantically meaningful prior even for labels with very few training examples, accelerating convergence and improving generalization to rare labels.

**Step 5 — Handle incomplete annotations.** In limited-data regimes, annotations are often partial. Apply the observed-only loss to avoid penalizing potentially positive but unannotated instances, then use pseudo-labeling to fill in missing labels once an initial model is trained (Q15).

**Step 6 — Tune per-label thresholds on a small validation set.** Even with 100–200 validation images, per-label threshold tuning reliably improves F1 by 2–5 points over the default $\tau = 0.5$.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Multi-label vs. multi-class and multi-task | Fundamentals |
| Q2 | Basic | CNN + sigmoid + BCE baseline | Fundamentals |
| Q3 | Basic | Binary Relevance and label correlation | Fundamentals |
| Q4 | Advanced | Classifier Chains | Fundamentals |
| Q5 | Advanced | Label co-occurrence graph | Fundamentals |
| Q6 | Basic | ML-GCN | Architecture & Modeling |
| Q7 | Basic | Q2L / Query2Label | Architecture & Modeling |
| Q8 | Advanced | ML-Decoder | Architecture & Modeling |
| Q9 | Advanced | CLIP for multi-label | Architecture & Modeling |
| Q10 | Advanced | MAE / DINO pretraining | Architecture & Modeling |
| Q11 | Basic | Binary Cross-Entropy | Loss & Training |
| Q12 | Basic | Focal Loss | Loss & Training |
| Q13 | Basic | RandAugment, Mixup, CutMix | Loss & Training |
| Q14 | Advanced | Asymmetric Loss (ASL) | Loss & Training |
| Q15 | Advanced | Missing / partial labels | Loss & Training |
| Q16 | Basic | Evaluation metrics overview | Evaluation & Practical Considerations |
| Q17 | Basic | Micro vs. Macro F1 | Evaluation & Practical Considerations |
| Q18 | Advanced | Threshold selection | Evaluation & Practical Considerations |
| Q19 | Advanced | Extreme Multi-Label Classification | Evaluation & Practical Considerations |
| Q20 | Advanced | Building a classifier with limited data | Evaluation & Practical Considerations |

## Resources

- Chen et al., [Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582) (ML-GCN, 2019)
- Liu et al., [Query2Label: A Simple Transformer Way to Multi-Label Classification](https://arxiv.org/abs/2107.10834) (Q2L, 2021)
- Ridnik et al., [ML-Decoder: Scalable and Versatile Classification Head](https://arxiv.org/abs/2111.12933) (2021)
- Ben-Baruch et al., [Asymmetric Loss For Multi-Label Classification](https://arxiv.org/abs/2009.14119) (ASL, 2021)
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP, 2021)
- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (MAE, 2022)
- Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (DINO, 2021)
- Zhang & Zhou, [A Review on Multi-Label Learning Algorithms](https://doi.org/10.1109/TKDE.2013.39) (2014)
