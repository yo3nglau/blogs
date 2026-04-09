---
title: "Self-Supervised & Contrastive Learning: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-09'
categories:
  - Interview
tags:
  - Deep Learning
  - Self-Supervised Learning
  - Contrastive Learning
toc: true
---

## Contrastive Learning Foundations

### Q1 [Basic] Explain the InfoNCE loss and what makes it a useful learning objective for representations

**Q:** What does the InfoNCE objective optimize, and how do the temperature and negative count affect the quality of the learned representations?

**A:** The **InfoNCE** (Information Noise-Contrastive Estimation) loss, introduced in Contrastive Predictive Coding (Oord et al., 2018), trains an encoder to identify a positive sample among $K$ negatives. For two augmented views $z_i$ and $z_j$ of the same image, with $K$ negatives $\{z_k\}$:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(z_i \cdot z_j / \tau)}{\exp(z_i \cdot z_j / \tau) + \sum_{k=1}^{K} \exp(z_i \cdot z_k / \tau)}\right]$$

This is a cross-entropy over a $(K{+}1)$-way classification problem — the encoder learns to push positive pairs together and negative pairs apart in a normalized embedding space. Oord et al. (2018) showed that minimizing InfoNCE maximizes a lower bound on the mutual information $I(z_i; z_j)$ between the two views, tightening as $K$ increases.

The **temperature** $\tau$ controls the sharpness of the distribution. A low $\tau$ concentrates gradients on hard negatives and can cause training instability if too small; a high $\tau$ provides a smoother but weaker gradient signal. SimCLR found $\tau = 0.07$ effective for normalized embeddings (Chen et al., 2020). The **negative count** $K$ determines how well the lower bound on mutual information approximates the true value — empirically, performance degrades substantially when fewer than a few thousand negatives are available, motivating large-batch training or memory queue approaches.

---

### Q2 [Basic] Explain how data augmentation defines semantic similarity in contrastive learning

**Q:** What role do augmentation choices play in shaping the invariances encoded in the learned representation?

**A:** In contrastive learning, two augmented views of the same image form a positive pair — the model learns that these are semantically equivalent. The augmentation policy therefore directly specifies which transformations the representation should be **invariant** to. A model trained with color jitter will discard color information; a model trained with heavy cropping will discard absolute spatial position.

SimCLR (Chen et al., 2020) performed a systematic ablation showing that the combination of random crop, color distortion (jitter + random grayscale), and Gaussian blur was critical. Random cropping contributed most to performance because it forces the encoder to recognize objects across different scales and spatial positions while implicitly creating patches with different color statistics — a harder task than any single augmentation alone. Chen et al. (2020) found that using crop alone achieved substantially lower accuracy than crop combined with color distortion, suggesting the two augmentations are complementary: cropping alone permits a shortcut via low-level color statistics shared between crops.

The coupling between augmentation choice and representation content is a fundamental limitation of the paradigm. Tasks requiring fine-grained color discrimination — such as estimating material properties or species identification from plumage color — are harmed by color-jitter invariance baked in during pretraining. Designing augmentation policies that preserve task-relevant signals while still providing a challenging pretext task remains an active research problem.

---

### Q3 [Advanced] Describe how MoCo scales contrastive learning to large negative sets without proportionally scaling batch size

**Q:** What two mechanisms does MoCo introduce to decouple the number of negatives from GPU memory constraints, and what problem does the momentum encoder specifically solve?

**A:** Standard end-to-end contrastive methods require all negatives to be encoded in the same forward pass, so the number of negatives is bounded by the batch size — typically a few hundred per GPU. The InfoNCE loss degrades significantly with fewer than several thousand negatives, making large-batch training necessary and memory-expensive.

**MoCo** (Momentum Contrast, He et al., 2020) decouples negative count from batch size through a **memory queue** that stores the $K$ most recently encoded key representations ($K = 65536$ in the original paper). Each mini-batch encodes a new set of keys, enqueues them, and dequeues the oldest entries — at any time the queue contains representations from many past mini-batches, providing a large pool of negatives at no extra memory cost beyond the queue itself.

The challenge with a queue is **representation inconsistency**: keys encoded in earlier mini-batches were produced by an older version of the encoder. If the encoder changes rapidly via gradient descent, the queue contains representations from many different "models," making the contrastive signal noisy. MoCo solves this with a **momentum encoder** $f_k$ that is not trained by gradient descent but updated as an exponential moving average of the query encoder $f_q$:

$$\theta_k \leftarrow m\theta_k + (1-m)\theta_q$$

with $m = 0.999$. The slow update ($99.9\%$ of the previous value retained each step) ensures that key representations in the queue were produced by a nearly identical model — consistency is maintained without requiring gradient-based training of the key encoder. He et al. (2020) achieved 60.6% top-1 linear evaluation with ResNet-50 on ImageNet, establishing the momentum encoder as a broadly useful primitive later adopted by BYOL, DINO, and other methods.

---

### Q4 [Advanced] Explain how SimCLR achieves strong representations without a memory queue

**Q:** What design choices allow SimCLR to produce competitive features using only in-batch negatives, and what does the projection head contribute to the quality of the frozen backbone features?

**A:** **SimCLR** (Chen et al., 2020) eliminates the memory queue by encoding both views of every sample in the same forward pass and treating all other samples in the batch as negatives. For batch size $N$, each sample has $2(N-1)$ negatives — the two views of all other $N-1$ images. This requires large batches (4096–8192) to provide sufficient negative contrast, trading memory efficiency for architectural simplicity.

Two design choices are central to SimCLR's performance. First, a **nonlinear projection head** — a two-layer MLP applied on top of the backbone before the contrastive loss — improves downstream linear evaluation accuracy by approximately 10 percentage points over using backbone features directly. The projection head allows the contrastive loss to specialize a subspace of the representation for the contrastive objective, warping it in ways that benefit InfoNCE optimization but would damage the general-purpose backbone features if applied directly. Discarding the projection head at inference time preserves the backbone's broader representational structure.

Second, **stronger augmentations** than prior work: random crop, color distortion (random jitter plus random grayscale), and Gaussian blur combine into a challenging pretext task where different crops of the same image can differ dramatically in color and spatial content. These hard positive pairs require the encoder to identify high-level semantic content — object identity rather than texture — to succeed at the task.

Chen et al. (2020) showed that ResNet-50 trained with SimCLR for 1000 epochs achieves 76.5% top-1 linear evaluation on ImageNet, narrowing the gap with supervised ResNet-50 pretraining and substantially outperforming all prior self-supervised methods at the same architecture scale.

---

## Self-Supervised Learning without Negatives

### Q5 [Basic] Explain how BYOL avoids representational collapse without using any negative samples

**Q:** What asymmetries in BYOL's architecture prevent the model from converging to a degenerate constant representation?

**A:** **BYOL** (Bootstrap Your Own Latent, Grill et al., 2020) trains on positive pairs only — two augmented views of the same image — which raises the immediate concern of trivial collapse: an encoder that maps all inputs to the same vector satisfies any similarity objective at zero cost.

BYOL prevents collapse through two structural asymmetries. First, an **online network** (updated by gradient descent) and a **target network** (updated by an exponential moving average of the online weights, with momentum coefficient $m = 0.996$) use the same architecture but are never identical. The online network includes an additional **predictor** MLP applied after its projector; the target network has no predictor. The objective is to minimize the cosine distance between the online predictor output and the target projector output — the predictor must track a moving target, preventing both networks from trivially agreeing on a constant.

Second, the loss is applied **asymmetrically**: gradients from the online loss only update the online network; the target network is updated by the slow EMA, not by the loss gradient. Grill et al. (2020) showed empirically that using a trainable target network (gradient descent on both branches) causes collapse, while using the EMA update maintains stability. The EMA update acts as a form of implicit self-distillation — the online network learns to match a smoothed, consistent version of itself.

BYOL with ResNet-50 (1000 epochs) achieved 74.3% top-1 linear evaluation on ImageNet, matching or exceeding SimCLR without any negatives, directly challenging the then-prevailing assumption that negatives were strictly necessary to prevent collapse.

---

### Q6 [Basic] Describe the role of stop-gradient in SimSiam and how it prevents representational collapse

**Q:** What does the stop-gradient operation do mechanistically, and why does removing it lead to collapse?

**A:** **SimSiam** (Chen & He, 2021) further simplifies BYOL by removing the momentum encoder entirely. Both augmented views are encoded by the same shared network; a predictor maps one branch's output, and the loss is the negative cosine similarity between the predictor output and the other branch's representation. A **stop-gradient** ($\text{sg}$) is applied to the branch that does not go through the predictor:

$$\mathcal{L} = -\frac{p_1}{\|p_1\|_2} \cdot \frac{\text{sg}(z_2)}{\|z_2\|_2}$$

Without stop-gradient, gradients flow symmetrically through both branches, and the optimizer finds the degenerate constant-vector solution that sets the cosine similarity to 1 trivially. With stop-gradient, one branch is held fixed during each parameter update — the predictor $p_1$ must match a fixed target $z_2$, making collapse difficult because the target is not co-optimized with the predictor in the same step.

Chen & He (2021) interpreted stop-gradient as implementing an implicit expectation-maximization: one branch provides a "pseudo-label" (the fixed target) while the other branch updates to match it, alternating in a way that mirrors EM updates. They showed both analytically and empirically that this alternation prevents collapse at a fixed point corresponding to a non-degenerate representation. SimSiam with ResNet-50 (800 epochs) achieves 71.3% top-1 linear evaluation on ImageNet — competitive with methods that require much more engineering, demonstrating that neither negatives nor momentum encoders are strictly necessary.

---

### Q7 [Advanced] Explain how BarlowTwins uses cross-correlation to enforce redundancy reduction

**Q:** What does BarlowTwins' objective optimize, and how does the redundancy reduction principle connect information theory to preventing dimensional collapse?

**A:** **BarlowTwins** (Zbontar et al., 2021) takes a fundamentally different approach to self-supervised learning: rather than contrasting sample pairs, it regularizes the **cross-correlation matrix** between the embedding batches of two augmented views. Given $d$-dimensional embeddings $Z^A$ and $Z^B$ with each dimension batch-normalized to zero mean and unit variance, the cross-correlation matrix is:

$$\mathcal{C}_{ij} = \frac{\sum_b z^A_{b,i}\, z^B_{b,j}}{\sqrt{\sum_b (z^A_{b,i})^2}\cdot \sqrt{\sum_b (z^B_{b,j})^2}}$$

The loss pushes $\mathcal{C}$ toward the identity matrix:

$$\mathcal{L}_{BT} = \underbrace{\sum_i (1 - \mathcal{C}_{ii})^2}_{\text{invariance}} + \lambda \underbrace{\sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2}_{\text{redundancy reduction}}$$

The diagonal terms enforce **invariance**: the same feature dimension should respond identically to both augmented views. The off-diagonal terms enforce **redundancy reduction**: different feature dimensions should be as decorrelated as possible across the two views, preventing multiple dimensions from encoding the same information. Zbontar et al. (2021) drew inspiration from Barlow's (1961) redundancy reduction principle from computational neuroscience, which hypothesizes that the goal of sensory coding is to represent stimuli with statistically independent components.

A key practical advantage is that BarlowTwins is naturally stable across batch sizes — the cross-correlation is computed over the batch dimension independently of whether pairs are in-batch negatives. Zbontar et al. (2021) reported 73.2% top-1 linear evaluation with ResNet-50 (1000 epochs), comparable to BYOL, without requiring momentum networks, memory queues, or negative pairs.

---

### Q8 [Advanced] Describe VICReg's three regularization terms and how each prevents a different failure mode

**Q:** What specific failure mode does each of VICReg's three loss components address, and how does making these terms explicit clarify the connection to other self-supervised methods?

**A:** **VICReg** (Variance-Invariance-Covariance Regularization, Bardes et al., 2022) decomposes the self-supervised objective into three terms with explicit roles. Given embeddings $Z$ and $Z'$ from two augmented views (each of shape $N \times d$):

$$\mathcal{L}_{\text{VIC}} = \lambda\, s(Z, Z') + \mu\, [v(Z) + v(Z')] + \nu\, [c(Z) + c(Z')]$$

The **invariance** term $s(Z, Z') = \frac{1}{N}\sum_i \|z_i - z'_i\|^2$ is the mean squared Euclidean distance between paired embeddings — it directly minimizes the difference between representations of the same image under different augmentations. Without the other terms, invariance alone drives collapse: the trivial solution is $Z = Z' = \mathbf{0}$.

The **variance** term $v(Z) = \frac{1}{d}\sum_j \max\!\bigl(0,\, 1 - \sqrt{\text{Var}(z_{:,j}) + \epsilon}\bigr)$ penalizes any feature dimension whose per-batch standard deviation falls below 1. This directly targets **complete collapse** (all embeddings at one point) and **partial collapse** (only some dimensions active): any dimension with low variance contributes a positive loss, forcing the model to actively use all $d$ dimensions. This term replaces the role of negative pairs in contrastive methods — negatives prevent collapse by repulsion, while the variance term prevents it by direct regularization.

The **covariance** term $c(Z) = \frac{1}{d}\sum_{i \neq j} [C(Z)]_{ij}^2$, where $C(Z)$ is the normalized covariance matrix of $Z$, penalizes off-diagonal covariance — equivalent to BarlowTwins' redundancy reduction. Together, variance and covariance terms ensure full-rank, decorrelated representations. Bardes et al. (2022) achieved 73.2% top-1 linear evaluation with ResNet-50 (1000 epochs), matching BarlowTwins, while providing a transparent decomposition that exposes the role of each term independently.

---

## Multimodal Contrastive Learning

### Q9 [Basic] Explain how CLIP constructs a joint vision-language embedding space

**Q:** What training objective does CLIP use to align image and text representations, and what properties of the resulting embedding space enable downstream use?

**A:** **CLIP** (Contrastive Language-Image Pre-Training, Radford et al., 2021) trains an image encoder and a text encoder jointly on 400 million image-text pairs collected from the internet. The training objective is a symmetric contrastive loss: for a batch of $N$ (image, text) pairs, the model maximizes the cosine similarity of the $N$ matched pairs and minimizes it for the $N^2 - N$ unmatched pairs. Each direction contributes an independent cross-entropy loss:

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left[\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\right]$$

where $\mathcal{L}_{I \to T}$ is the cross-entropy treating each image's matched text as the positive among all $N$ texts in the batch, and vice versa. The encoders are a ViT (or ResNet) for images and a Transformer for text; both project to the same $d$-dimensional space before computing similarities.

The resulting space aligns visual and semantic content: the image of a dog and the phrase "a photograph of a dog" map to nearby embeddings, regardless of breed, pose, or image style. This alignment is learned purely from the co-occurrence of images and their captions — no manual category labels are used. At inference, the embedding space supports zero-shot classification (comparing an image embedding to class-name text embeddings), cross-modal retrieval (finding the most relevant image for a text query), and linear classification (training a linear classifier on image embeddings with a small labeled set).

---

### Q10 [Basic] Describe zero-shot image classification with CLIP and what enables its generalization

**Q:** How does CLIP perform zero-shot classification without any task-specific training, and what factors determine where it succeeds or fails?

**A:** CLIP performs zero-shot classification by converting each class name into a text prompt of the form "a photo of a {class}" and computing the cosine similarity between the image embedding and each prompt embedding. The predicted class is the one whose prompt has the highest similarity — no fine-tuning or labeled examples are needed.

Radford et al. (2021) evaluated zero-shot CLIP ViT-L/14 on 27 downstream classification benchmarks and found 76.2% top-1 accuracy on ImageNet, matching supervised ResNet-50 performance without seeing any ImageNet labels during pretraining. The generalization comes from two sources: (1) the breadth of internet text covers visual concepts across far more categories and domains than manually curated label taxonomies; (2) natural language descriptions encode contextual information about visual appearance that is absent from integer class labels.

However, zero-shot performance degrades for fine-grained or specialized domains underrepresented in internet text — medical imaging, satellite imagery, and specialized technical diagrams. The model also struggles with tasks that require counting, spatial reasoning, or distinguishing visually similar objects with subtly different labels. **Prompt engineering** substantially affects accuracy: Radford et al. (2021) found that ensembling over 80 prompt templates (varying sentence structure and context, e.g., "a photo of a {c}, a type of food") improved zero-shot ImageNet accuracy by approximately 3.5% relative to a single default prompt.

---

### Q11 [Advanced] Explain how ALIGN demonstrates that scale can compensate for label noise in vision-language pretraining

**Q:** What data pipeline choices distinguish ALIGN from CLIP, and what does ALIGN's performance reveal about the relative importance of data quality versus data quantity?

**A:** CLIP used a carefully constructed dataset with deduplication, quality filtering, and removal of images whose text descriptions matched ImageNet labels (to prevent test-set contamination). **ALIGN** (Jia et al., 2021) made the opposite choice: train on 1.8 billion image-text pairs collected with minimal filtering — only basic frequency-based thresholding to remove extremely rare terms and alt-text shorter than 3 tokens. This preserves the massive scale of internet image-text co-occurrences at the cost of substantially noisier supervision.

The core architecture follows the same symmetric contrastive objective as CLIP, using EfficientNet-L2 as the image encoder and BERT-Large as the text encoder, projecting to a shared 640-dimensional space. Despite approximately $4.5\times$ more data and noisier labels, the training signal remains sufficient because the contrastive objective is robust to individual noisy pairs — a single mismatched image-text pair contributes a diffuse gradient signal distributed across the batch, and with billions of pairs the aggregate signal remains informative.

ALIGN achieves comparable zero-shot ImageNet classification accuracy to CLIP's best models (~76% top-1) and outperforms supervised ImageNet pretraining on cross-modal retrieval tasks (Flickr30K and MSCOCO). Jia et al. (2021) also demonstrated 85.5% fine-tuning accuracy on ImageNet by further training the image encoder with labeled data. The key implication for the field: if noisy large-scale data is sufficient for competitive performance, the bottleneck shifts from data curation to compute and model capacity — a conclusion that motivated subsequent scaling-focused work in multimodal learning.

---

### Q12 [Advanced] Describe how SigLIP's sigmoid loss improves over the softmax InfoNCE for large-scale vision-language training

**Q:** What is the computational bottleneck in CLIP's softmax contrastive loss at scale, and how does the sigmoid reformulation eliminate it?

**A:** CLIP's softmax contrastive loss requires a global normalization constant — the denominator sums over all $N$ text embeddings for each image query (and vice versa). In distributed training across $M$ devices each holding $N/M$ samples, computing this normalization requires an all-gather communication operation to collect all $N$ embeddings before computing the softmax. At large batch sizes and many devices, this all-gather dominates communication overhead and creates a synchronization bottleneck.

**SigLIP** (Sigmoid Loss for Language Image Pre-Training, Zhai et al., 2023) replaces the softmax with independent per-pair sigmoid cross-entropy:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i,j} \log \sigma\!\left(y_{ij} \cdot (z_i \cdot z_j / \tau) - b\right)$$

where $y_{ij} = +1$ for matched pairs and $y_{ij} = -1$ for unmatched pairs, and $b$ is a learned bias. Each pair is evaluated independently with a binary sigmoid — there is no normalization constant and no dependence on other pairs in the batch. This eliminates the all-gather entirely: each device can compute its local loss over its own pairs and the global loss is the average.

Beyond efficiency, sigmoid loss enables effective training with **smaller batch sizes**: the softmax loss requires large $N$ so that the denominator provides a strong contrastive signal; sigmoid loss treats each pair independently, so smaller batches yield equally informative gradients per pair. Zhai et al. (2023) showed that SigLIP models outperform comparable CLIP models on zero-shot benchmarks, particularly in multilingual and low-resource language settings where image-text pair density is lower per language and smaller effective batch sizes are unavoidable.

---

## Masked Image Modeling

### Q13 [Basic] Explain the masked autoencoding pretext task and how MAE implements it for vision

**Q:** What does the masked autoencoding pretext task require the encoder to learn, and how does MAE's architecture differ from a symmetric autoencoder?

**A:** **Masked Autoencoders** (MAE, He et al., 2022) adapt BERT's masked language modeling pretext task to vision. An image is divided into non-overlapping $16 \times 16$ patches; a large fraction of patches (75% by default) are randomly masked; the unmasked patches are encoded by a ViT encoder; a lightweight decoder receives the encoder outputs interleaved with learned mask tokens and reconstructs the raw pixel values of the masked patches. The loss is mean-squared error over masked patches only — unmasked patches do not contribute.

The pretext task forces the encoder to learn to infer content from partial observations: to reconstruct a large masked region, the model must understand semantic structure — object shapes, part relationships, scene context — because adjacent visible patches alone are insufficient when most of the image is hidden. He et al. (2022) trained ViT-L/16 for 1600 epochs on ImageNet and achieved 85.9% top-1 fine-tuning accuracy, surpassing all prior self-supervised methods and matching the performance of supervised ViT pretraining at the same scale.

The architecture is **asymmetric**: the encoder processes only the visible (unmasked) tokens and never sees mask tokens, while the decoder operates on the full set including mask tokens. This asymmetry is an efficiency design — the encoder sees $25\%$ of patches, reducing encoder FLOPs by approximately $4\times$ relative to processing all patches. The decoder, which is much lighter (typically 8 Transformer blocks at 512 dimensions versus the encoder's 24 blocks at 1024 dimensions), handles the reconstruction task and is discarded at inference time.

---

### Q14 [Advanced] Describe why MAE uses a high masking ratio and what the asymmetric design achieves

**Q:** Why does a 75% masking ratio outperform lower ratios, and what are the downstream performance trade-offs of MAE's asymmetric architecture relative to contrastive methods?

**A:** A low masking ratio makes the reconstruction task trivially solvable without learning semantic content: adjacent visible patches share low-level texture statistics, and a model can interpolate masked regions by local pattern completion without forming a global understanding of the image. He et al. (2022) ablated masking ratios from 10% to 90% and found that performance on downstream fine-tuning peaks broadly between 60–80%, with 75% as the default. At very high ratios ($\geq 90\%$), reconstruction becomes ill-posed — insufficient context exists to identify even the category of the masked object, and the task provides ambiguous supervision.

The 75% ratio creates a task where the model must simultaneously track multiple widely-separated visible patches and reason about their relationship to infer the large masked regions. Block masking (masking contiguous rectangular regions rather than random patches) is slightly more effective at the same ratio because contiguous masks are harder to fill by interpolation from nearby visible patches, requiring longer-range reasoning. The difficulty-driven regime is similar to what self-supervised NLP methods learned: BERT's 15% mask rate, which was designed for token-level linguistic prediction, is insufficient for pixels because image information is far more spatially redundant than text.

The key downstream trade-off compared to contrastive methods is the **linear probing gap**: MAE ViT-L achieves approximately 76% top-1 under linear evaluation — lower than contrastive methods like MoCo v3 ViT-L (81.1%) — but achieves higher fine-tuning accuracy (85.9%). Pixel-level reconstruction does not force the encoder to organize representations into linearly separable class clusters; instead it produces features that require nonlinear adaptation. He et al. (2022) argued this is acceptable since fine-tuning is the practical deployment protocol, but the gap motivated subsequent work combining masked modeling with objectives that also preserve linearly separable structure (e.g., DINO with masked prediction).

---

### Q15 [Advanced] Explain how DINO uses self-distillation to produce semantically structured ViT features

**Q:** What training procedure does DINO use, and why do ViT features trained with DINO exhibit emergent semantic segmentation properties?

**A:** **DINO** (Caron et al., 2021) implements **self-distillation without labels**: a student ViT is trained to match the output probability distribution of a teacher ViT on different crops of the same image. The teacher is updated by an exponential moving average of the student (as in MoCo and BYOL); the student is trained by minimizing the cross-entropy between the student's softmax output and the teacher's softmax output over a shared set of prototypes (the final linear layer acts as a soft clustering head). A **centering** operation — subtracting a running mean from teacher outputs — prevents collapse by ensuring no single output dimension dominates.

DINO uses a **multi-crop** strategy: the student sees both two global views (crops covering $\geq 50\%$ of the image) and six local crops (smaller patches covering $\sim 20\%$); the teacher sees only global views. The student's loss is computed over all crops against both global teacher views. This local-to-global correspondence task — predicting global context from a local patch — forces the student to recognize objects at many scales and understand their global structure from limited local evidence.

The most striking emergent property is that the [CLS] token's attention weights over patch tokens — a byproduct of ViT's global pooling — spontaneously partition the image into semantically coherent foreground-background regions consistent with human segmentation boundaries (Caron et al., 2021). This property does not appear in supervised ViT training or in CNN features, suggesting that the self-distillation objective on multiple crops induces a qualitatively different internal organization. ViT-S/16 trained with DINO achieves 77.0% top-1 linear evaluation on ImageNet and enables competitive k-NN retrieval ($74.5\%$ top-1) and video object segmentation without any fine-tuning.

---

### Q16 [Advanced] Describe how DINOv2 scales self-supervised learning with curated data and combined objectives

**Q:** What does DINOv2 change relative to DINO in terms of data, objectives, and resulting representation quality?

**A:** **DINOv2** (Oquab et al., 2023) identifies data quality rather than model architecture as the primary limitation of previous self-supervised methods at scale. Rather than training on ImageNet (1.28M images) or unfiltered web crawls, DINOv2 assembles **LVD-142M** — a 142 million image curated dataset constructed by retrieving images from a large uncurated web pool using nearest-neighbor search seeded from a set of diverse high-quality source datasets. This retrieval-based curation removes exact duplicates and biases toward visual diversity without requiring manual annotation.

The training objective combines two components: the standard DINO self-distillation cross-entropy loss between student and teacher outputs, and a **masked image modeling** objective in which the student must predict the teacher's patch-level representations for masked image patches (rather than pixel values). This masked patch prediction, operating in the teacher's feature space rather than pixel space, provides dense local supervision absent from the global DINO objective — the model must encode fine-grained local structure in individual patch embeddings, not just the [CLS] global summary.

The combination resolves the linear probing gap: DINOv2 ViT-g/14 achieves 86.5% top-1 accuracy under linear evaluation on ImageNet (Oquab et al., 2023) — substantially higher than contrastive methods and unlike MAE, which requires fine-tuning to reach peak performance. More significantly, DINOv2 features transfer to dense prediction tasks (semantic segmentation on ADE20K, monocular depth estimation on NYUd) with a frozen backbone and a single linear head, demonstrating that the combined objective produces spatial feature quality that generalizes to pixel-level understanding.

---

## Theory, Evaluation, and Transfer

### Q17 [Basic] Explain what linear probing measures about a representation and how it differs from fine-tuning

**Q:** Why do some self-supervised methods perform well under fine-tuning but poorly under linear probing, and what does this gap reveal about the representation?

**A:** **Linear probing** trains a single linear classifier on top of frozen backbone features; no encoder weights are updated. The accuracy of the linear classifier directly measures the **linear separability** of the representation — whether class information is encoded in a way accessible via a hyperplane in the embedding space.

**Fine-tuning** updates all encoder weights along with the classifier head using labeled data. This allows the encoder to reconfigure its representations, so class separability at initialization matters less — nonlinearly organized features can be linearly separated after a few layers of adaptation.

The gap between these protocols reveals the structure imposed by the pretraining objective. MAE pretraining produces representations with high fine-tuning accuracy (~85.9% ViT-L) but lower linear probing accuracy (~76%) because pixel-level reconstruction does not require the encoder to organize representations into semantically clustered regions. The model learns to encode spatial texture and structural relationships that are useful for reconstruction but not necessarily linearly class-separable. Contrastive objectives, by contrast, directly push different-class negatives apart and same-class positives together in the representation space, producing linearly separable features at the cost of potentially over-constraining the representation's structure. Methods achieving high performance under both protocols — such as DINOv2 (Oquab et al., 2023) with 86.5% linear probing — are generally considered to have learned the most broadly useful representations.

---

### Q18 [Basic] Describe how self-supervised pretraining improves sample efficiency in low-data and semi-supervised settings

**Q:** How does the benefit of self-supervised pretraining change as the amount of labeled data varies, and what explains the diminishing returns at large labeled set sizes?

**A:** Self-supervised pretraining provides a general-purpose initialization that substantially reduces the number of labeled examples needed to achieve a target accuracy. In the semi-supervised setting, the encoder is fine-tuned with a small labeled fraction while the pretraining weights provide a strong starting point. SimCLR (Chen et al., 2020) reported 48.3% top-1 accuracy on ImageNet using only 1% of ImageNet labels (approximately 12,800 labeled images) with ResNet-50 — more than doubling the accuracy of a supervised-from-scratch baseline at the same labeled budget.

The benefit is largest when labeled data is scarce because pretraining substitutes for labels in learning low-level and mid-level visual features: edge detectors, texture representations, and part detectors learned from unlabeled data do not need to be re-learned from the small labeled set. As the labeled set grows, a supervised model can increasingly learn these features on its own, and the gap between pretrained and randomly initialized models shrinks. At large labeled set sizes (full ImageNet), the advantage of self-supervised pretraining over supervised pretraining diminishes substantially.

The transfer advantage is also domain-dependent: CLIP's pretraining on broad web image-text pairs (Radford et al., 2021) transfers effectively to specialized domains (satellite imagery, medical imaging) where few-shot fine-tuning is practical, precisely because the web corpus exposes the model to diverse visual contexts that single-domain pretraining cannot provide. Domain-specific self-supervised pretraining on unlabeled in-domain data can further improve over general web pretraining for specialized applications.

---

### Q19 [Advanced] Explain the uniformity-alignment decomposition and what it implies about representational collapse

**Q:** How do Wang and Isola's two metrics characterize the quality of a representation, and what does each failure mode look like geometrically?

**A:** Wang & Isola (2020) proposed decomposing contrastive representation quality into two independently measurable geometric properties on the unit hypersphere. **Alignment** measures how close positive pairs map:

$$\mathcal{L}_{\text{align}}(\alpha) = \mathbb{E}_{(x,x^+)\sim p_{\text{pos}}}\!\left[\|f(x) - f(x^+)\|_2^\alpha\right]$$

**Uniformity** measures how evenly the embedding distribution covers the hypersphere, using a Gaussian potential kernel:

$$\mathcal{L}_{\text{uniform}}(t) = \log\, \mathbb{E}_{x,y \sim p_{\text{data}}}\!\left[e^{-t\|f(x) - f(y)\|_2^2}\right]$$

Wang & Isola (2020) proved that the InfoNCE loss asymptotically decomposes into the sum of alignment and uniformity: the numerator encourages positive pairs to align, and the denominator's normalization requires negatives to be spread out — a uniform distribution maximizes the partition function, providing the strongest contrastive signal.

The two failure modes correspond geometrically. **Complete collapse** (all representations at one point) achieves perfect alignment ($\mathcal{L}_{\text{align}} = 0$) but the worst possible uniformity (all mass concentrated at a single pole). **Dimensional collapse** (representations on a low-dimensional subspace) achieves good alignment and partial uniformity within the subspace but wastes the remaining dimensions — the effective distribution is a delta function in the collapsed directions rather than the full hypersphere. Methods like VICReg's variance term and BarlowTwins' redundancy reduction explicitly target uniformity: they force the marginal embedding distribution to use all dimensions, which corresponds to uniformity on the full hypersphere. The framework provides a principled vocabulary for comparing different self-supervised objectives — a method that improves alignment at the cost of uniformity (or vice versa) is trading one type of representation quality for another.

---

### Q20 [Advanced] Describe why dimensional collapse occurs and how non-contrastive methods prevent it through explicit regularization

**Q:** What mechanism drives representations to collapse onto a low-dimensional subspace, and how do the regularization terms in BarlowTwins and VICReg specifically counteract this?

**A:** **Dimensional collapse** refers to learned representations that span a low-dimensional subspace of the $d$-dimensional embedding space, despite being embedded at full dimensionality. The rank of the empirical embedding matrix is far below $d$ — in extreme cases, thousands of training examples map to a single line through the origin. This is qualitatively different from complete collapse (a single point) because dimensionally collapsed representations may still contain class information along the active dimensions, but they discard all information that could be encoded in the unused orthogonal directions.

The mechanism driving dimensional collapse is **spectral imbalance in the contrastive loss**. The InfoNCE gradient signal is dominated by the high-variance directions of the embedding covariance: the model can reduce loss most efficiently by concentrating representations along the few directions that explain most of the variation in the data. Once a small set of directions captures the primary class signal, the loss provides little gradient pressure to populate the remaining orthogonal directions. This is analogous to why PCA retains a small number of components — the first few explain most variance — but here the model actively shrinks the inactive dimensions rather than just failing to populate them.

BarlowTwins (Zbontar et al., 2021) addresses dimensional collapse by penalizing all off-diagonal elements of the cross-correlation matrix equally, regardless of which dimension pairs they involve. Any pair of dimensions that are correlated across the two views incurs a loss, which forces the encoder to discover $d$ decorrelated representations rather than concentrating the signal in a few correlated components. VICReg (Bardes et al., 2022) uses the variance term to impose a direct floor on each dimension's activity level: any feature dimension with per-batch standard deviation below 1 is penalized, which explicitly prevents any dimension from collapsing to a near-constant. The two approaches target the same failure mode from complementary angles — BarlowTwins prevents inter-dimension correlation (removing redundancy), VICReg prevents intra-dimension variance collapse (ensuring activity) — and both can be understood as maximizing the effective rank of the representation matrix or equivalently maximizing the entropy of the marginal embedding distribution on the hypersphere, the uniformity term in the Wang & Isola (2020) framework.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | InfoNCE loss and mutual information lower bound | Contrastive Learning Foundations |
| Q2 | Basic | Data augmentation as semantic similarity in contrastive learning | Contrastive Learning Foundations |
| Q3 | Advanced | MoCo momentum encoder and memory queue | Contrastive Learning Foundations |
| Q4 | Advanced | SimCLR large-batch in-batch negatives and projection head | Contrastive Learning Foundations |
| Q5 | Basic | BYOL asymmetric online-target architecture | Self-Supervised Learning without Negatives |
| Q6 | Basic | SimSiam stop-gradient and collapse prevention | Self-Supervised Learning without Negatives |
| Q7 | Advanced | BarlowTwins cross-correlation and redundancy reduction | Self-Supervised Learning without Negatives |
| Q8 | Advanced | VICReg variance-invariance-covariance regularization | Self-Supervised Learning without Negatives |
| Q9 | Basic | CLIP symmetric contrastive loss for vision-language alignment | Multimodal Contrastive Learning |
| Q10 | Basic | CLIP zero-shot classification and prompt engineering | Multimodal Contrastive Learning |
| Q11 | Advanced | ALIGN scale vs. noise trade-off in vision-language pretraining | Multimodal Contrastive Learning |
| Q12 | Advanced | SigLIP sigmoid loss eliminating softmax all-gather overhead | Multimodal Contrastive Learning |
| Q13 | Basic | MAE masked autoencoding pretext task and asymmetric design | Masked Image Modeling |
| Q14 | Advanced | MAE 75% masking ratio and downstream linear probing trade-off | Masked Image Modeling |
| Q15 | Advanced | DINO self-distillation and emergent semantic attention maps | Masked Image Modeling |
| Q16 | Advanced | DINOv2 curated data and combined self-distillation + masked prediction | Masked Image Modeling |
| Q17 | Basic | Linear probing vs. fine-tuning as evaluation protocols | Theory, Evaluation, and Transfer |
| Q18 | Basic | Sample efficiency of self-supervised pretraining in semi-supervised settings | Theory, Evaluation, and Transfer |
| Q19 | Advanced | Uniformity-alignment decomposition of contrastive loss | Theory, Evaluation, and Transfer |
| Q20 | Advanced | Dimensional collapse mechanism and regularization-based prevention | Theory, Evaluation, and Transfer |

## Resources

- Oord et al., [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (2018)
- He et al., [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) (2020)
- Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) (2020)
- Grill et al., [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733) (2020)
- Wang & Isola, [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) (2020)
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021)
- Jia et al., [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) (2021)
- Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) (2021)
- Zbontar et al., [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) (2021)
- Chen & He, [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) (2021)
- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (2022)
- Bardes et al., [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906) (2022)
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (2023)
- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (2023)
