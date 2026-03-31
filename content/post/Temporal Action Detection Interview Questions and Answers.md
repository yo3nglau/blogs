---
title: "Temporal Action Detection: Interview Questions and Answers"
author: yo3nglau
date: '2026-03-31'
categories:
  - Interview
tags:
  - Deep Learning
  - Temporal Action Detection
  - Video Understanding
toc: true
---

## TAD Fundamentals & Detection Pipelines

### Q1 [Basic] How does Temporal Action Detection differ from Action Recognition?

**Q:** What is Temporal Action Detection, and how does it differ from action recognition in both task formulation and difficulty?

**A:** Action Recognition operates on trimmed video clips where a single action occupies the entire clip. Given a short clip as input, the model outputs a single class label. The task assumes the action boundaries are already known and provided externally. This makes it primarily a classification problem over a fixed-duration input.

Temporal Action Detection (TAD) operates on untrimmed videos of arbitrary length containing multiple action instances separated by background segments. Given an untrimmed video, the model must simultaneously localize and classify all action instances, producing a set of output tuples (t_start, t_end, class, confidence) for each detected action. The model receives no prior information about where or how many actions occur.

This dual requirement makes TAD substantially harder than recognition. The model must handle background suppression (most frames are non-action), variable action duration (from seconds to minutes), multiple concurrent or overlapping actions, and precise boundary estimation. Evaluation therefore uses tIoU-based metrics (temporal intersection over union) rather than simple classification accuracy, rewarding both correct classification and accurate temporal localization.

---

### Q2 [Basic] What is the two-stage TAD pipeline?

**Q:** Describe the classical two-stage pipeline for temporal action detection and explain why this decomposition is useful.

**A:** The two-stage pipeline decomposes TAD into two sequential subproblems. The first stage is class-agnostic temporal proposal generation: a proposal network scans the untrimmed video and generates a set of candidate temporal segments (t_start, t_end) that are likely to contain any action, without predicting which action. The goal is high recall at the cost of precision — generate enough proposals to cover all true action instances. Classic proposal methods include TAG (Temporal Actionness Grouping), TURN, and SST.

The second stage performs per-proposal classification and boundary regression. Each proposal segment is re-encoded (either by re-extracting features over the proposal window or pooling from precomputed features), and a classifier assigns a class label and confidence score. A boundary regressor optionally refines the proposal boundaries. This stage trades recall for precision. After classification, non-maximum suppression (NMS) removes duplicate detections.

The decomposition is useful because it separates two conceptually distinct tasks — temporal localization and action recognition — allowing each to be optimized independently. Proposal networks can be trained with generic actionness labels rather than fine-grained class labels, making them more generalizable. However, the pipeline is slower than single-stage methods due to the sequential processing of potentially hundreds of proposals, and errors in stage one (missed proposals) cannot be recovered in stage two.

---

### Q3 [Basic] What evaluation metrics are used for TAD, and what are their limitations?

**Q:** How is temporal action detection evaluated, and what are the shortcomings of standard metrics?

**A:** The primary metric is mean Average Precision (mAP) at a specified temporal Intersection over Union (tIoU) threshold. tIoU measures the temporal overlap between a predicted segment and a ground truth segment: tIoU = |prediction ∩ ground truth| / |prediction ∪ ground truth|. A predicted segment is considered a true positive only if its tIoU with a ground truth segment exceeds the threshold and the predicted class is correct. Average Precision (AP) is computed per class by ranking all predictions by confidence and computing the area under the precision-recall curve. mAP averages AP over all classes.

Different benchmarks use different threshold conventions. THUMOS14 reports mAP at individual thresholds (0.3, 0.4, 0.5, 0.6, 0.7), with higher thresholds requiring tighter localization. ActivityNet reports the average mAP over thresholds from 0.5 to 0.95 in steps of 0.05, following the COCO convention for object detection. Models are typically evaluated at tIoU = 0.5 as the primary operating point.

Standard mAP has several limitations. It does not penalize over-complete predictions that correctly contain the ground truth but extend significantly beyond it — such predictions can receive full credit if tIoU exceeds the threshold. It also weights all classes equally regardless of frequency. For videos with densely overlapping actions (e.g., sports highlights), the tIoU-based matching may fail to credit correct detections when two predicted segments both overlap one ground truth segment. Finally, mAP does not measure detection latency, which matters in online/streaming scenarios.

---

### Q4 [Advanced] How do anchor-based and anchor-free temporal detectors differ?

**Q:** What are the fundamental differences between anchor-based and anchor-free approaches to temporal action detection, and what are their respective trade-offs?

**A:** Anchor-based detectors pre-define a set of temporal anchors — fixed-duration segments placed densely at each temporal location, covering a range of durations (e.g., 16, 32, 64, 128 frames). For each anchor, the model predicts classification scores and offset corrections (Δt_start, Δt_end) to shift the anchor to the true action boundaries. The anchor design introduces prior knowledge about expected action durations and reduces the search space. Representative methods include R-C3D, SSAD, and SSN. The main limitation is that the anchor set must be carefully designed for each dataset: if typical action durations lie outside the anchor range, recall suffers. Additionally, anchors are fixed-size at each scale, making them less suited for actions with extreme aspect ratios in the temporal dimension.

Anchor-free detectors avoid pre-defined templates and instead directly predict action boundaries or center-duration pairs from video features without anchors. Methods such as AFSD (Lin et al., 2021) predict per-snippet boundary offsets using dense regression, and ActionFormer predicts at each feature pyramid level the class and duration of any action centered there. This eliminates the need for anchor engineering and handles arbitrary durations naturally. The trade-off is that anchor-free training is generally harder to stabilize: without anchors providing spatial priors, the model must learn to associate each temporal location with the correct action instance purely from features, requiring careful loss design (e.g., focal loss for classification, IoU loss for regression).

The broader trend in TAD mirrors object detection: anchor-based methods dominated early work for their stability, but anchor-free and query-based methods have progressively overtaken them as training techniques have matured, offering better performance and simpler pipeline design.

---

### Q5 [Advanced] How do query-based detectors like ActionFormer work?

**Q:** What is the design of query-based or DETR-style temporal action detectors, and what advantages do they offer over proposal-based methods?

**A:** Query-based temporal detectors adapt the set-prediction paradigm introduced by DETR (Detection Transformer) to the temporal domain. Instead of generating proposals and classifying them, the model uses a fixed set of learnable query vectors, each responsible for detecting one action instance. Each query attends over the video feature sequence via cross-attention and outputs a predicted (class, t_center, t_width) tuple. Hungarian matching between predictions and ground truth action instances during training ensures each ground truth is assigned to exactly one query, eliminating the need for NMS post-processing.

ActionFormer (Shi et al., 2022) is one of the most effective implementations. It builds a feature pyramid from the input video features using a transformer encoder with local self-attention (each token attends to a fixed-size temporal window rather than the full sequence, reducing complexity from O(T^2) to O(T·w)). At each scale of the pyramid, a classification head and a regression head predict action classes and durations independently. Predictions from all pyramid levels are aggregated and suppressed with NMS (ActionFormer is not strictly NMS-free despite its transformer backbone; fully NMS-free models require the set-prediction loss exclusively). TemporalMaxer (2023) demonstrates that replacing self-attention with simple MaxPooling for temporal aggregation achieves competitive results with far lower computational cost, suggesting that the feature pyramid structure matters more than the attention mechanism itself.

The advantages of query-based designs include end-to-end training without anchors, natural multi-scale detection via feature pyramids, and compatibility with modern pre-trained video backbones. The primary remaining challenge is handling very long videos: the self-attention in the encoder scales poorly with temporal length, addressed by local attention windows or linear attention approximations.
