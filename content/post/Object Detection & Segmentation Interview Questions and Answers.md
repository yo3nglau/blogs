---
title: "Object Detection & Segmentation: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-10'
categories:
  - Interview
tags:
  - Deep Learning
  - Computer Vision
  - Object Detection
toc: true
---

## Detection Foundations

### Q1 [Basic] Explain anchor-based detection: IoU matching, regression targets, and NMS

**Q:** How does an anchor-based detector assign training labels to anchor boxes, and why is non-maximum suppression necessary at inference?

**A:** An anchor-based detector tiles a fixed set of anchor boxes at every spatial location on the feature map, varying across predefined scales and aspect ratios. In Faster R-CNN (Ren et al., 2015), each location uses 3 scales × 3 aspect ratios = 9 anchors. An anchor is labeled **positive** if its maximum IoU with any ground-truth box exceeds 0.7 (or if it is the highest-IoU anchor for a particular GT box to ensure coverage of rare objects), and **negative** if its maximum IoU with all GT boxes falls below 0.3. Anchors with IoU in $[0.3, 0.7)$ are ignored during training to avoid ambiguous gradients.

Regression targets are encoded as offsets relative to the anchor: $t_x = (x - x_a)/w_a$, $t_y = (y - y_a)/h_a$, $t_w = \log(w/w_a)$, $t_h = \log(h/h_a)$. The log-space parameterization ensures scale invariance and keeps regression targets in a compact numerical range.

At inference the detector produces hundreds of thousands of candidate boxes, most heavily overlapping. **Non-Maximum Suppression (NMS)** greedily selects the highest-scoring box, suppresses all boxes with IoU above a threshold (typically 0.5), and repeats. This reduces redundant predictions to a small final set. The limitation of greedy NMS is that closely spaced objects of the same class may be mutually suppressed; **Soft-NMS** addresses this by decaying rather than eliminating overlapping box scores, improving recall for dense scenes.

---

### Q2 [Basic] Describe Feature Pyramid Networks and their role in multi-scale detection

**Q:** What problem does FPN solve in object detection, and how does its architecture achieve multi-scale feature representations?

**A:** Convolutional backbone features exhibit a fundamental trade-off: deep feature maps (e.g., the final stride-32 layer) have strong semantic content but coarse spatial resolution, making small-object detection unreliable; shallow feature maps have high spatial fidelity but weak semantics. Running an independent detector at each backbone scale is computationally wasteful and does not share information across scales.

FPN (Lin et al., 2017a) constructs a feature pyramid by combining the semantic richness of deep layers with the spatial precision of shallow layers through a **top-down pathway with lateral connections**. Starting from the deepest backbone stage $C_5$, FPN upsamples (nearest-neighbor 2$\times$) and adds a lateral $1 \times 1$ convolution of the corresponding bottom-up feature map at each resolution, producing $\{P_2, P_3, P_4, P_5\}$ with uniform channel depth (256). A separate RPN and detection head is applied at each pyramid level, with objects assigned to levels by size: larger objects use coarser levels and smaller objects use finer levels.

This design adds negligible compute over the backbone and improves COCO AP by approximately 8 points over a single-scale C4-feature baseline (Lin et al., 2017a). Extensions such as PANet add a bottom-up path augmentation to strengthen low-level features, and BiFPN introduces learnable weighted fusion, both adopted in subsequent YOLO and EfficientDet families.

---

### Q3 [Advanced] Compare GIoU, DIoU, and CIoU as bounding box regression losses

**Q:** Why do L1 and L2 losses on box coordinates fall short for bounding box regression, and how do the IoU-family losses address these failures?

**A:** L1 and L2 losses on the four coordinate or $(c_x, c_y, w, h)$ parameterization suffer from two problems: (1) they are not scale-invariant—a 10-pixel error on a 20-pixel box is far worse than on a 200-pixel box, yet receives the same gradient; (2) they do not directly optimize the evaluation metric (IoU), so optimizing them does not necessarily improve detection AP. The plain IoU loss $\mathcal{L}_{\text{IoU}} = 1 - \text{IoU}$ corrects scale invariance but produces zero gradient when predicted and ground-truth boxes do not overlap, a common situation early in training.

**GIoU** (Rezatofighi et al., 2019) adds a penalty based on the smallest enclosing box $C$ containing both boxes:

$$\mathcal{L}_{\text{GIoU}} = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|}$$

The extra term is non-zero even for non-overlapping boxes, providing a learning signal by penalizing the wasted area in the enclosing box.

**DIoU** (Zheng et al., 2020) directly penalizes the normalized center distance, converging faster than GIoU when the predicted box is far from the target:

$$\mathcal{L}_{\text{DIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b_{gt})}{c^2}$$

where $\rho$ is the Euclidean center distance and $c$ is the diagonal of $C$.

**CIoU** (Zheng et al., 2020) adds an aspect-ratio consistency term $\alpha v$ to DIoU, where $v = \frac{4}{\pi^2}\left(\arctan\frac{w_{gt}}{h_{gt}} - \arctan\frac{w}{h}\right)^2$ and $\alpha = v / (1 - \text{IoU} + v)$ weights its contribution by IoU. CIoU is the default regression loss in modern YOLO variants and FCOS, consistently outperforming GIoU and DIoU by 0.5–1.5 AP on COCO.

---

### Q4 [Advanced] Explain Focal Loss and how it solves class imbalance in one-stage detectors

**Q:** Why does extreme foreground-background imbalance degrade one-stage detector training, and how does Focal Loss address it?

**A:** A one-stage detector applied to an image evaluates on the order of $10^4$–$10^5$ candidate locations, of which fewer than 0.01% correspond to objects. During training, background (easy negative) examples dominate the cross-entropy loss in both count and magnitude: even though each individual easy-negative contribution is small (e.g., $-\log(0.999) \approx 0.001$), their aggregate swamps the gradients from the rare foreground examples. Prior systems used **hard negative mining** (e.g., SSD keeps negatives at 3:1 with positives by score), which discards many informative gradients and requires careful sampling.

**Focal Loss** (Lin et al., 2017b) reshapes the cross-entropy loss to down-weight easy examples continuously, without discrete sampling:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

The modulating factor $(1-p_t)^\gamma$ reduces the contribution of well-classified examples: at $p_t = 0.95$ (easy background), the factor is $(0.05)^2 = 0.0025$ for $\gamma = 2$, suppressing their gradient to near zero. Hard, misclassified examples (low $p_t$) retain loss close to standard cross-entropy. The class-balancing weight $\alpha_t$ handles dataset-level imbalance separately.

**RetinaNet** (Lin et al., 2017b) demonstrates that combining Focal Loss with a simple ResNet-FPN backbone and shared class/box subnets at each pyramid level achieves 40.8 AP on COCO test-dev with a ResNet-101 backbone—matching or exceeding two-stage detectors of the same era at similar inference speed, showing that one-stage detectors underperformed not due to architecture but due to the training loss.

---

## Two-Stage Detection

### Q5 [Basic] Trace the R-CNN to Faster R-CNN evolution

**Q:** What computational inefficiencies did each step in the R-CNN → Fast R-CNN → Faster R-CNN progression address?

**A:** **R-CNN** (Girshick et al., 2014) established the region proposal + CNN pipeline: Selective Search generates ~2,000 region proposals per image, each is warped to a fixed size and independently forwarded through an AlexNet-style CNN to extract features, then classified by per-class SVMs with a separate box regressor. This approach achieved strong detection accuracy but was extremely slow (~47 seconds per image at test time) because the CNN ran independently for each of ~2,000 proposals.

**Fast R-CNN** (Girshick, 2015) inverts the ordering: run the CNN once on the full image to produce a shared feature map, then project each region proposal onto that feature map and extract a fixed-size representation via **RoI Pooling** (dividing the projected region into a $7 \times 7$ grid and max-pooling each cell). Classification and regression are performed jointly by a multi-task loss from a shared fully-connected head. End-to-end training is now possible (except for proposal generation). Test speed improves to ~2 seconds per image, with most time spent on Selective Search.

**Faster R-CNN** (Ren et al., 2015) eliminates the external proposal algorithm by introducing the **Region Proposal Network (RPN)**, a small fully-convolutional network that shares the same backbone features with the detection head. RPN predicts objectness scores and box deltas at each spatial location for $k$ anchors, and produces ordered proposal sets via NMS. The result is an end-to-end trainable pipeline running at ~5 fps with a VGG-16 backbone—orders of magnitude faster than R-CNN—while achieving better accuracy due to learned proposals.

---

### Q6 [Advanced] Analyze the design of the Region Proposal Network

**Q:** How does the RPN generate proposals, and what are the key design choices that make it effective?

**A:** The RPN in Faster R-CNN (Ren et al., 2015) is a fully-convolutional network applied to the shared backbone feature map (typically stride 16). At each spatial location it uses a $3 \times 3$ sliding window and $k = 9$ anchors (3 scales: 128, 256, 512 px; 3 aspect ratios: 1:1, 1:2, 2:1). Two sibling $1 \times 1$ convolutions produce **objectness scores** ($2k$ outputs) and **box regression deltas** ($4k$ outputs).

**Label assignment**: a positive anchor is any anchor with IoU $> 0.7$ with a GT box, or the highest-IoU anchor for a GT box (to ensure every GT box has at least one positive). A negative anchor has IoU $< 0.3$ with all GT boxes. The mini-batch samples 256 anchors with a 1:1 positive-to-negative ratio when possible.

**Proposal selection**: the RPN sorts all anchors by objectness score, takes the top $N$ (12,000 train / 6,000 test), applies NMS at IoU $= 0.7$, and passes the top $M$ (2,000 train / 300 test) proposals to the detector head. This two-stage filtering efficiently discards low-quality and redundant regions.

The shared backbone between RPN and detector is critical: feature computation is not repeated, and the two networks can benefit from joint training. Fine-tuning only the task-specific layers while keeping shared layers fixed (alternating four-step training) or joint end-to-end training both work well in practice.

---

### Q7 [Advanced] Explain why RoI Align outperforms RoI Pooling for instance segmentation

**Q:** What is the misalignment introduced by RoI Pooling, and how does RoI Align eliminate it?

**A:** **RoI Pooling** projects a floating-point proposal $(x_1, y_1, x_2, y_2)$ onto the feature map and immediately **quantizes** the coordinates to the nearest integer grid position. It then divides the resulting integer region into a $H \times W$ grid (e.g., $7 \times 7$), again quantizing each bin boundary. Each bin is max-pooled. The two rounds of quantization introduce misalignment: the pooled region can be off by up to half a stride (e.g., 8 pixels at feature stride 16), causing a systematic spatial offset between the input proposal and the pooled features.

For **bounding box classification and regression**, this misalignment is tolerable—detection is inherently translation-tolerant at the scale of detection head predictions. But for **instance segmentation**, where a 28×28 binary mask must align precisely with the ground-truth object boundary, even a few pixels of offset produces blurry or incorrect masks.

**RoI Align** (He et al., 2017) removes all quantization. Given a floating-point RoI, it divides it into $H \times W$ bins with floating-point boundaries and samples $2 \times 2$ points uniformly within each bin, computing feature values at each sample point by **bilinear interpolation** from the four surrounding grid points. The sampled values are then aggregated by average or max pooling. This maintains sub-pixel accuracy throughout. Ablating from RoI Pooling to RoI Align on the Mask R-CNN baseline improves mask AP by 3.1 points (14.0 → 17.1) on COCO while leaving box AP essentially unchanged—confirming that the benefit is specific to pixel-level prediction.

---

### Q8 [Advanced] Describe Cascade R-CNN and the distribution mismatch problem

**Q:** What is the distribution mismatch problem in two-stage detection, and how does Cascade R-CNN address it with progressive IoU thresholds?

**A:** A standard two-stage detector trains its detection head at a fixed IoU threshold $u$ (typically 0.5). At this threshold, a predicted box is positive if IoU $\geq 0.5$ with a GT box, accepting proposals with noisy localization. **Raising $u$ to 0.7** forces the head to produce more precise regressions, but introduces a **distribution mismatch**: at training time, few proposals have IoU $\geq 0.7$ (since the RPN is trained at $u = 0.5$), leading to very sparse positive samples, overfitting to those few examples, and degraded performance when applied to the distribution of IoU $\approx 0.5$–$0.7$ proposals actually produced at inference.

**Cascade R-CNN** (Cai & Vasconcelos, 2018) addresses this by training a sequence of detectors with monotonically increasing IoU thresholds:
- **Stage 1**: $u_1 = 0.5$. Takes RPN proposals (IoU $\approx 0.5$–$0.7$), applies a detection head, and outputs refined boxes.
- **Stage 2**: $u_2 = 0.6$. Takes stage-1 outputs as proposals (now IoU $\approx 0.6$–$0.75$), trains another head matched to this higher-quality proposal distribution.
- **Stage 3**: $u_3 = 0.7$. Takes stage-2 outputs and trains a head for high-quality localization.

At each stage, the input proposals match the training distribution of that head, resolving the mismatch. The key insight is that **quality improves stage by stage**: a detector trained at $u_3 = 0.7$ only performs well when its inputs are already near-correct, which only happens after sequential refinement. Cascade R-CNN achieves 42.8 AP on COCO test-dev with a ResNet-101 backbone, compared to 37.8 AP for the equivalent single-stage Faster R-CNN, a gain of 5.0 AP with only modest additional inference cost (three small heads vs one).

---

## One-Stage and Transformer-Based Detection

### Q9 [Basic] Describe FCOS and the role of centerness in anchor-free detection

**Q:** How does FCOS reformulate object detection as per-pixel regression, and what problem does the centerness branch solve?

**A:** FCOS (Tian et al., 2019) treats detection as a **fully convolutional, anchor-free, per-pixel prediction problem**. For each spatial location $(x, y)$ on a feature map, FCOS predicts a 4D distance vector $(l, r, t, b)$ representing the distances from the location to the left, right, top, and bottom edges of the enclosing ground-truth box (if the location falls inside any GT box), along with a classification score. No anchor boxes are needed; hyperparameters for scales and aspect ratios disappear.

Multi-level prediction with FPN resolves ambiguity when multiple objects of different sizes overlap at the same location: each FPN level $P_l$ is responsible for objects whose regression targets satisfy a size constraint ($P_3$: $[0, 64]$ px, $P_4$: $[64, 128]$ px, $P_5$: $[128, 256]$ px, $P_6$: $[256, 512]$ px, $P_7$: $[512, \infty)$).

The **centerness branch** addresses a quality problem: pixels near object boundaries can predict valid but inaccurate bounding boxes. Centerness is a scalar target:

$$\text{centerness} = \sqrt{\frac{\min(l, r)}{\max(l, r)} \times \frac{\min(t, b)}{\max(t, b)}} \in [0, 1]$$

It is high at the geometric center of a box and falls off toward edges. During NMS, predicted box scores are multiplied by centerness, so low-quality off-center predictions are suppressed without additional post-processing. FCOS reaches 44.7 AP on COCO with a ResNet-101-FPN backbone, competitive with anchor-based methods while being conceptually simpler (Tian et al., 2019).

---

### Q10 [Basic] Summarize the YOLO design philosophy and its architectural evolution

**Q:** What is the core design principle behind YOLO, and how have key architectural improvements from YOLOv3 onward advanced detection performance?

**A:** YOLO's core principle is **unified, single-pass detection**: the entire image is processed once through a CNN that simultaneously predicts bounding boxes and class probabilities across a spatial grid, enabling real-time inference without the sequential proposal-then-classify pipeline of two-stage methods.

**YOLOv3** (Redmon & Farhadi, 2018) introduced three key advances over earlier versions: (1) a 53-layer residual backbone (Darknet-53) with skip connections for better gradient flow; (2) multi-scale prediction at three resolutions (similar in spirit to FPN), using 3 anchors per scale (9 total, k-means clustered from the training set); (3) independent logistic classifiers per class rather than softmax, enabling multi-label predictions. YOLOv3 achieves 55.3 AP$_{50}$ on COCO in 51ms—an excellent speed-accuracy trade-off for deployment.

Subsequent versions refined several subsystems. **CSP connections** (YOLOv5/7) split feature maps across a partial residual branch, reducing computation while maintaining accuracy. **Mosaic augmentation** (YOLOv5) tiles four images per sample, forcing the model to detect small objects in varied contexts and effectively quadrupling dataset diversity. **Decoupled heads** (YOLOv6/8) separate classification and regression into independent branches, improving both tasks since their gradient dynamics differ. **Anchor-free reformulations** in YOLOv8 adopt FCOS-style per-pixel regression, eliminating anchor hyperparameter tuning. These incremental improvements have pushed YOLO-family models to 53–55 AP on COCO while maintaining sub-10ms inference on modern GPUs.

---

### Q11 [Advanced] Analyze DETR's bipartite matching loss and its slow convergence

**Q:** How does DETR eliminate anchors and NMS through bipartite matching, and why does it converge so much more slowly than CNN-based detectors?

**A:** DETR (Carion et al., 2020) frames detection as a **set prediction problem**: $N$ learned object queries (typically 100) are processed by a Transformer decoder that cross-attends to encoder features of the image. Each query produces one prediction: a class label and a box $(c_x, c_y, w, h)$ in normalized image coordinates. The model predicts exactly $N$ objects per image; unused slots predict the $\varnothing$ (no-object) class.

Training uses **Hungarian matching** to find the minimum-cost bipartite assignment between $N$ predictions and $M$ ground-truth objects (padded with $N - M$ $\varnothing$ slots):

$$\hat{\sigma} = \underset{\sigma \in \mathfrak{S}_N}{\arg\min} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

$$\mathcal{L}_{\text{match}} = -\hat{p}_{\sigma(i)}(c_i) + \lambda_{\text{L1}} \|b_i - \hat{b}_{\sigma(i)}\|_1 + \lambda_{\text{GIoU}} \mathcal{L}_{\text{GIoU}}(b_i, \hat{b}_{\sigma(i)})$$

The matched assignment is then used for the training loss. One-to-one matching eliminates duplicate predictions, making NMS unnecessary. DETR achieves 42.0 AP on COCO val with ResNet-50 after 500 epochs.

The **slow convergence** (500 epochs vs ~37 for Faster R-CNN) stems from the cross-attention mechanism: at initialization, each object query attends nearly uniformly across all image positions. The model must gradually learn to specialize each query to a distinct region. For DETR to discover that a particular query should attend to the top-left region, the optimizer must propagate gradient information through the full encoder-decoder stack many times. Additionally, early-training predictions are widely scattered, making Hungarian matching unstable (small prediction perturbations can cause entirely different matching assignments, producing incoherent gradients).

---

### Q12 [Advanced] Explain how DINO-DETR achieves fast convergence

**Q:** What do DAB-DETR and DN-DETR contribute, and how does DINO-DETR integrate these to dramatically close the convergence gap with CNN detectors?

**A:** Three innovations compound to solve DETR's convergence problem.

**DAB-DETR** (Liu et al., 2022) reformulates object queries as explicit anchor boxes $(c_x, c_y, w, h)$ rather than opaque learned embeddings. The $x, y$ components are updated by each decoder layer via attention offsets, providing direct spatial feedback. The $w, h$ components modulate the positional encoding of keys and queries—wider queries attend to wider regions, providing a strong spatial prior from the start. This alone cuts convergence from 500 to ~50 epochs.

**DN-DETR** (included in the DINO paper) addresses training instability from matching. In addition to $N$ learnable queries, it adds noisy copies of GT boxes as **denoising queries** with known targets. These bypass Hungarian matching (targets are pre-assigned) and directly train the decoder to reconstruct GT boxes from noisy inputs. A denoising attention mask prevents the two query groups from attending to each other. The denoising auxiliary loss stabilizes early training by providing consistent gradient signals regardless of the instability of Hungarian assignments.

**DINO** (Zhang et al., 2022) combines DAB-DETR anchor queries, contrastive denoising (separate positive/negative noise groups for a richer training signal), and mixed query selection (initialize some anchor queries from encoder top-scoring regions rather than purely learned embeddings). DINO with ResNet-50 achieves 49.0 AP on COCO val with 12 training epochs, compared to 42.0 AP for vanilla DETR at 500 epochs. With a Swin-L backbone and multi-scale features, DINO reaches 63.3 AP—the state of the art among detection transformers—while requiring only 24 epochs.

---

## Instance Segmentation

### Q13 [Basic] Describe the Mask R-CNN architecture and its mask branch design

**Q:** How does Mask R-CNN extend Faster R-CNN with instance segmentation, and what design choices make the mask branch effective?

**A:** Mask R-CNN (He et al., 2017) adds a third parallel branch to Faster R-CNN alongside the existing classification and regression heads. The mask branch takes a $14 \times 14$ RoI feature (from RoI Align at stride 4 or 8) and applies four $3 \times 3$ convolutions, each followed by ReLU, then a $2 \times 2$ deconvolution to upsample to $28 \times 28$, and finally a $1 \times 1$ convolution producing $K$ binary masks—one per class. The mask loss is **binary cross-entropy applied only to the GT class channel** for each proposal, avoiding competition between classes in mask space: during training, class $c$'s mask head does not receive gradient from other classes.

This decoupled design—predict masks independently for each class, select at inference using the detected class—has two benefits: (1) the mask predictor does not need to distinguish between classes, only to segment "this instance" vs. background; (2) it avoids the mask-classification coupling that led to quality degradation in earlier FCN-based segmenters.

Mask R-CNN with ResNet-50-FPN achieves 37.1 mask AP on COCO test-dev at ~5 fps. The use of **RoI Align** (over RoI Pooling) is critical: the pixel-accurate feature alignment enables sharp, well-positioned masks, contributing approximately 3 AP over the same architecture with RoI Pooling (He et al., 2017).

---

### Q14 [Advanced] Explain SOLOv2's kernel prediction framework for instance segmentation

**Q:** How does SOLOv2 predict instance masks without detection or RoI operations, and what is the decoupled SOLO strategy?

**A:** SOLOv2 (Wang et al., 2020) frames instance segmentation as **dynamic kernel prediction**: the network predicts a unique convolutional kernel for each instance, then applies it to a shared feature map to produce the binary mask.

The model has two branches operating on FPN features:
1. **Mask kernel branch**: divides the feature space into an $S \times S$ grid (e.g., $40 \times 40$ at the finest level). Each grid cell $(i, j)$ is responsible for an instance whose center falls in that cell, and predicts a $D$-dimensional kernel vector $\mathbf{E}_{i,j} \in \mathbb{R}^D$ (typically $D = 256$).
2. **Mask feature branch**: applies a series of convolutions to produce a dense feature map $\mathbf{F} \in \mathbb{R}^{H \times W \times D}$ enriched with coordinate channels (appended $x, y$ grids) for positional awareness.

The mask for instance $(i, j)$ is:

$$\mathbf{M}_{i,j} = \text{Conv}(\mathbf{F},\, \mathbf{E}_{i,j})$$

a single $1 \times 1$ convolution of $\mathbf{F}$ using $\mathbf{E}_{i,j}$ as the kernel weights.

**Decoupled SOLO** reduces computational cost by factorizing the kernel prediction: instead of one $S \times S$ grid producing a full $D$-dimensional kernel, separate horizontal and vertical branches predict $D/2$-dimensional vectors, forming the final kernel by concatenation. This replaces $S^2$ predictions with $2S$ predictions, dramatically reducing the number of active kernel slots at large grid sizes.

SOLOv2 achieves 37.1 mask AP on COCO val with ResNet-50-FPN at 36.1ms—matching Mask R-CNN accuracy at roughly half the latency—by eliminating RoI Align, region proposals, and sequential instance processing (Wang et al., 2020).

---

### Q15 [Advanced] Describe CondInst's conditional convolution approach to instance segmentation

**Q:** How does CondInst use instance-specific convolutional controllers, and why does this formulation avoid the limitations of both detect-then-segment and direct prediction methods?

**A:** CondInst (Tian et al., 2020) produces instance masks through a **conditional convolution**: each detected instance generates a small set of dynamic filter weights (a "controller") that are applied to a shared feature map to produce its mask. This bridges the efficiency of fully-convolutional methods with the precision of instance-specific representations.

From the detection head (an FCOS-style head producing per-pixel predictions), a **controller branch** outputs $c$ filter parameters at each positive location, forming three dynamic convolutional layers with weight shapes $8 \times 8 \times 5$, $8 \times 8 \times 8$, and $8 \times 1$ (totalling $c = 169$ parameters). These lightweight dynamic weights are applied sequentially to a shared **mask feature branch**, which outputs a $H/8 \times W/8 \times 8$ feature map.

The key to position-awareness without bounding boxes is the **relative coordinate map**: two feature channels encoding each pixel's $x$ and $y$ distance (normalized) to the predicted instance center are concatenated to the shared features. This allows the mask head to distinguish which instance it is segmenting at inference without any explicit RoI operation.

The design avoids RoI Align entirely (eliminating misalignment artifacts) and processes all instances in a single forward pass of the shared mask branch, with only the lightweight dynamic convolutions being instance-specific. CondInst achieves 34.7 mask AP on COCO val with ResNet-50-FPN at 31.8ms, versus Mask R-CNN's 35.0 AP at 69.9ms—similar accuracy at roughly half the latency (Tian et al., 2020).

---

### Q16 [Advanced] Compare the paradigms of instance segmentation and their design trade-offs

**Q:** What are the key architectural paradigms for instance segmentation, and when does each approach have an advantage?

**A:** Four major paradigms organize the instance segmentation literature.

**Detect-then-segment** (Mask R-CNN, He et al., 2017) applies ROI Align to each proposal's feature region, then runs a per-instance mask head. The strong coupling between localization and segmentation quality means accurate boxes yield accurate masks. This paradigm is the most deployment-proven and benefits directly from improvements to two-stage detection (e.g., Cascade R-CNN). Its limitation is sequential RoI processing: inference time scales with the number of instances, and RoI Align introduces a fixed spatial resolution bottleneck for large objects.

**Direct kernel prediction** (SOLOv2, Wang et al., 2020; CondInst, Tian et al., 2020) avoids detection and RoI operations by predicting instance-specific convolution kernels applied to a shared feature map. Inference time is largely independent of the instance count (parallel mask generation), and the fully-convolutional design integrates naturally into single-stage pipelines. The trade-off is that associating predictions to instances is implicit (grid cells for SOLO, controllers for CondInst), which can fail when many small instances overlap in the same grid cell.

**Transformer query-based** (Mask2Former, Cheng et al., 2022) uses learned object queries that cross-attend to multi-scale image features, producing per-query mask embeddings matched to ground truth via Hungarian assignment—the same bipartite matching principle as DETR but extended to masks. This paradigm achieves the highest mask quality and unifies instance, semantic, and panoptic segmentation in one architecture, at the cost of higher memory and compute from multi-scale attention.

**Bottom-up** methods predict pixel-level embeddings and group them into instances via clustering or learned affinities. These have no explicit object count limit but struggle to separate touching instances with similar appearance and require non-trivial post-processing.

For real-time deployment, kernel-based methods (CondInst, SOLOv2) offer the best latency-accuracy trade-off. For highest accuracy in research settings, query-based transformers dominate. For well-supported industrial deployment, the detect-then-segment paradigm remains the most extensively validated.

---

## Semantic, Panoptic Segmentation, and Foundation Models

### Q17 [Basic] Describe FCN and DeepLab's contributions to semantic segmentation

**Q:** How did FCN enable end-to-end semantic segmentation, and what is ASPP in DeepLabv3+?

**A:** **FCN** (Long et al., 2015) established the dominant paradigm for semantic segmentation by replacing the fully connected layers of classification CNNs (AlexNet, VGG) with $1 \times 1$ convolutional layers, making the network fully convolutional and capable of processing inputs of arbitrary spatial dimension with dense output. The final feature map is bilinearly upsampled to the input resolution. Skip connections from earlier layers (FCN-8s merges stride-8, stride-16, and stride-32 predictions) recover spatial detail lost during downsampling. FCN demonstrated that pixel-wise cross-entropy loss on dense label maps can be optimized end-to-end, replacing hand-crafted pipelines.

The **DeepLab** series addresses FCN's limited receptive field and loss of resolution. Inserting **dilated (atrous) convolutions** (replacing stride with dilation rate $r$) expands the receptive field by a factor of $r^2$ without reducing resolution and without additional parameters, allowing the backbone to maintain stride-8 or stride-4 output instead of stride-32.

**DeepLabv3+** (Chen et al., 2018) builds on this with two improvements. The **Atrous Spatial Pyramid Pooling (ASPP)** module applies parallel dilated convolutions with rates $\{1, 6, 12, 18\}$ plus global average pooling, concatenates the outputs, and applies a $1 \times 1$ fusion convolution—capturing multi-scale context from fine textures to large regions in a single module. An **encoder-decoder structure** then refines these pooled features: the ASPP output is combined with low-level features (stride-4) via a lightweight decoder ($1 \times 1$ conv + $3 \times 3$ conv + $4\times$ bilinear upsample), recovering sharp boundaries at low cost. DeepLabv3+ with Xception-65 achieves 89.0 mIoU on PASCAL VOC 2012 and 82.1 mIoU on Cityscapes (Chen et al., 2018).

---

### Q18 [Advanced] Explain Mask2Former as a universal segmentation architecture

**Q:** How does Mask2Former unify instance, semantic, and panoptic segmentation, and what is masked cross-attention?

**A:** Mask2Former (Cheng et al., 2022) achieves state-of-the-art performance on all three segmentation tasks—instance, semantic, and panoptic—using a single architecture and loss function, differing only in how prediction-to-target assignment is performed (Hungarian matching for instance/panoptic, global assignment for semantic).

The architecture has three components. The **pixel decoder** applies multi-scale deformable attention to backbone feature maps, progressively upsampling to produce high-resolution per-pixel features at $1/4$ input resolution—capturing fine spatial detail that a simple FPN misses for segmentation.

The **Transformer decoder** processes $N$ learned queries through $L$ layers, each containing: (1) self-attention among queries, (2) masked cross-attention from queries to image features, and (3) a feed-forward network. The key innovation is **masked cross-attention**: each query $q$ is only permitted to attend to image positions where its predicted binary mask $\mathbf{M}_q$ is positive (foreground):

$$\text{Attention}(Q, K, V) \text{ restricted to positions where } \mathbf{M}_q = 1$$

This focuses attention on the relevant object region, preventing queries from being distracted by background and dramatically improving efficiency and quality for dense segmentation—especially small objects—compared to the full cross-attention used in DETR.

Each decoder layer predicts both mask and class label, and auxiliary losses are applied at every layer to provide dense supervision. The final per-query binary mask (from dot product of query embedding with pixel decoder features) and class label are used for prediction.

Mask2Former with ResNet-50 achieves 57.8 PQ on COCO panoptic, 57.7 AP on COCO instance, and 57.6 mIoU on ADE20K semantic segmentation (Cheng et al., 2022)—the first architecture to reach state of the art on all three tasks simultaneously with a single model.

---

### Q19 [Basic] Describe the Segment Anything Model (SAM) and its promptable design

**Q:** What are the three components of SAM, and how does prompt-driven segmentation work?

**A:** SAM (Kirillov et al., 2023) is a foundation model for image segmentation trained on SA-1B, a dataset of 1.1 billion masks across 11 million images collected through a data engine combining manual annotation, semi-automatic labeling, and fully automatic mask generation.

SAM has three components designed for interactive, real-time mask prediction.

The **image encoder** is a MAE-pretrained ViT-H that processes a $1024 \times 1024$ image into a $64 \times 64 \times 256$ image embedding. It runs once per image; its output is cached to serve multiple prompt queries at negligible additional cost.

The **prompt encoder** converts diverse prompt types to dense and sparse embedding representations. Points (foreground/background clicks) and box corners are encoded with positional encoding and learned type embeddings. Coarse mask inputs are processed by a lightweight convolutional encoder. Text prompts are embedded using CLIP's text encoder. All prompt types are projected to the same embedding space.

The **mask decoder** is a lightweight two-layer Transformer that alternates query-to-image attention (prompt embeddings attending to image embedding) and image-to-query attention (image embedding attending to prompt embeddings), followed by an upsampling MLP that projects to $256 \times 256$ resolution. To handle prompt ambiguity (a single point may indicate a part, an object, or a group), SAM predicts three candidate masks with associated confidence scores, letting the user select the appropriate granularity.

Full mask prediction once the image embedding is cached takes under 50ms, enabling real-time interactive annotation workflows.

---

### Q20 [Advanced] Analyze SAM's capabilities, failure modes, and evolution toward open-vocabulary segmentation

**Q:** Where does SAM excel and where does it fail, and how does composing SAM with other models extend its capabilities?

**A:** SAM (Kirillov et al., 2023) excels in several respects. Its **zero-shot transfer** to unseen domains is strong for natural images—segmenting objects from a single click or bounding box with no fine-tuning. The interactive workflow dramatically reduces annotation time: SAM-assisted annotation on diverse benchmarks requires a fraction of the clicks needed for pixel-by-pixel labeling. SAM also provides useful masks for **promptable compositing** applications where region boundaries are needed without semantic labels.

**Failure modes** fall into several categories. First, SAM produces class-agnostic masks: it segments object regions but assigns no semantic labels, making it unsuitable for classification-dependent tasks without an external head. Second, the encoding bottleneck ($1024 \times 1024 \rightarrow 64 \times 64$) causes detail loss for very fine structures—thin wires, hair, text characters, fence posts—where the $16\times$ compression loses critical boundary information. Third, performance degrades significantly on **out-of-distribution domains** such as medical imaging (CT/MRI slices with different contrast statistics), remote sensing (small geospatial objects), and industrial inspection (texture-based defect detection), because SA-1B is composed exclusively of consumer photographs.

**Composing SAM with open-vocabulary detectors** extends its semantic reach. Grounding DINO can localize objects specified by free-form text descriptions; SAM then generates precise masks for the detected boxes. This combination achieves open-vocabulary instance segmentation (any object described in natural language) without any task-specific fine-tuning.

SAM's temporal limitation—no memory of previous frames—was addressed by SAM 2, which introduces a memory bank and conditioned prompt propagation for video object segmentation. Architecturally, SAM 2 extends the mask decoder with a streaming memory encoder/attention module, enabling consistent instance tracking across video without re-prompting every frame. The combination of promptable segmentation, memory, and hierarchical temporal features represents the current frontier for general-purpose segmentation models.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Anchor matching, regression targets, NMS | Detection Foundations |
| Q2 | Basic | Feature Pyramid Networks | Detection Foundations |
| Q3 | Advanced | GIoU, DIoU, CIoU loss family | Detection Foundations |
| Q4 | Advanced | Focal Loss and class imbalance | Detection Foundations |
| Q5 | Basic | R-CNN → Fast R-CNN → Faster R-CNN | Two-Stage Detection |
| Q6 | Advanced | RPN design and proposal generation | Two-Stage Detection |
| Q7 | Advanced | RoI Align vs RoI Pooling | Two-Stage Detection |
| Q8 | Advanced | Cascade R-CNN progressive IoU | Two-Stage Detection |
| Q9 | Basic | FCOS anchor-free detection | One-Stage and Transformer-Based Detection |
| Q10 | Basic | YOLO evolution and design principles | One-Stage and Transformer-Based Detection |
| Q11 | Advanced | DETR bipartite matching and convergence | One-Stage and Transformer-Based Detection |
| Q12 | Advanced | DINO-DETR: anchor queries and denoising | One-Stage and Transformer-Based Detection |
| Q13 | Basic | Mask R-CNN mask branch | Instance Segmentation |
| Q14 | Advanced | SOLOv2 dynamic kernel prediction | Instance Segmentation |
| Q15 | Advanced | CondInst conditional convolutions | Instance Segmentation |
| Q16 | Advanced | Instance segmentation paradigm comparison | Instance Segmentation |
| Q17 | Basic | FCN and DeepLabv3+ ASPP | Semantic, Panoptic Segmentation, and Foundation Models |
| Q18 | Advanced | Mask2Former universal segmentation | Semantic, Panoptic Segmentation, and Foundation Models |
| Q19 | Basic | SAM promptable segmentation | Semantic, Panoptic Segmentation, and Foundation Models |
| Q20 | Advanced | SAM capabilities, failures, and composition | Semantic, Panoptic Segmentation, and Foundation Models |

## Resources

- Girshick et al., [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](https://arxiv.org/abs/1311.2524) (2014)
- Girshick, [Fast R-CNN](https://arxiv.org/abs/1504.08083) (2015)
- Ren et al., [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) (2015)
- Lin et al., [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) (2017a)
- Lin et al., [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) (2017b)
- Rezatofighi et al., [Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression](https://arxiv.org/abs/1902.09630) (2019)
- Zheng et al., [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287) (2020)
- Cai & Vasconcelos, [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726) (2018)
- He et al., [Mask R-CNN](https://arxiv.org/abs/1703.06870) (2017)
- Tian et al., [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) (2019)
- Redmon & Farhadi, [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) (2018)
- Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) (2020)
- Liu et al., [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329) (2022)
- Zhang et al., [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605) (2022)
- Wang et al., [SOLOv2: Dynamic and Fast Instance Segmentation](https://arxiv.org/abs/2003.10152) (2020)
- Tian et al., [Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664) (2020)
- Long et al., [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) (2015)
- Chen et al., [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) (2018)
- Cheng et al., [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) (2022)
- Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643) (2023)
