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

## Temporal Proposal Generation

### Q6 [Basic] What is Temporal Action Proposal generation and what are its classical approaches?

**Q:** What is the Temporal Action Proposal task, why is it useful, and what are the key classical methods?

**A:** Temporal Action Proposal (TAP) generation is the task of producing a ranked set of class-agnostic temporal segments (t_start, t_end) that are likely to contain an action of any category. Unlike full detection, proposals carry no class label — the goal is purely temporal localization with high recall. TAP is used as the first stage of two-stage TAD pipelines, and proposal quality directly bounds the recall of the full detector: any action instance missed at the proposal stage cannot be recovered.

The main quality criteria for proposals are recall (the fraction of ground truth actions covered by at least one proposal above a tIoU threshold) and average number of proposals (efficiency). An ideal proposal generator produces few proposals with very high recall. Classical approaches include sliding window (dense uniform windows at multiple scales, simple but produces many redundant proposals), actionness scoring (assign a per-snippet actionness probability, group consecutive high-scoring snippets), and deep learning methods. SST (Single Stream Temporal) uses an LSTM to model the video stream and score fixed-duration candidate segments at each step. BSN (Boundary Sensitive Network, Lin et al., 2018) was a significant advance: it decouples boundary detection from proposal scoring, first predicting per-snippet probabilities for being an action start, an action end, or within an action, then combining high-probability start/end pairs into proposals with a temporal evaluation module.

The key insight behind BSN is that temporal boundaries are the most informative cue for proposal quality. By training the network to precisely localize boundary positions before forming proposals, BSN achieves substantially better boundary precision than methods that regress from anchors or group action snippets directly.

---

### Q7 [Basic] How does BMN (Boundary-Matching Network) work?

**Q:** What is BMN, and how does it improve upon BSN for temporal action proposal generation?

**A:** BMN (Boundary-Matching Network, Lin et al., 2019) extends BSN by jointly modeling the probability of all possible start-end combinations in a single forward pass. Where BSN independently predicts start and end probabilities and then combines them heuristically, BMN constructs a two-dimensional confidence map of size T×T (where T is the number of temporal locations), in which cell (i, j) represents the model's confidence that the segment from position i to position j is a valid action proposal. This allows the network to directly capture the relationship between specific start and end boundaries rather than treating them as independent predictions.

The confidence map is computed by a boundary-matching module that samples features along each candidate proposal using bilinear interpolation, creating a fixed-length feature representation for each (start, end) pair. A lightweight convolutional network processes these sampled features to predict the confidence score. The boundary probabilities from the BSN-style head and the confidence map scores are then combined into a final proposal score for each (i, j) pair. After thresholding the confidence map, soft-NMS is applied to generate the final ranked proposal list.

BMN is trained with a multi-task loss: a temporal evaluation loss (binary cross-entropy on per-snippet start/end/actionness probabilities) and a proposal evaluation loss (mean squared error on the confidence map against IoU-based ground truth scores). The joint training ensures that boundary detection and proposal scoring are mutually consistent. Compared to BSN, BMN achieves higher recall at the same number of proposals and generates more precise boundaries, making it a widely adopted proposal backbone in subsequent TAD systems.

---

### Q8 [Advanced] How does GTAD use graph neural networks for temporal action detection?

**Q:** What is the GTAD framework, and how does modeling proposals as a graph improve detection quality?

**A:** G-TAD (Sub-Graph Localization for Temporal Action Detection, Xu et al., 2020) models the temporal proposal graph problem as a graph neural network (GNN) inference task. After generating initial proposals (using a BSN/BMN-style boundary detector), GTAD represents each proposal as a node in a graph. Two types of edges connect nodes: semantic edges (connecting proposals with similar visual content, determined by feature similarity) and temporal edges (connecting proposals that overlap or are adjacent in time). The GNN propagates information along these edges, allowing each proposal to incorporate context from semantically related or temporally neighboring proposals before making its classification and boundary prediction.

The motivation is that actions rarely occur in isolation. Concurrent actions (e.g., a player kicking a ball while another is running) and temporally adjacent actions (e.g., a preparation step followed by the main action) share context that can improve detection of individual instances. Standard per-proposal classifiers treat each proposal independently, missing these dependencies. By explicitly modeling inter-proposal relationships through graph edges, GTAD achieves better discrimination between semantically similar actions and more accurate boundary estimation in cluttered scenes.

GTAD uses a graph convolutional network where node features are updated through weighted aggregation of neighbor features, with edge weights learned from the semantic and temporal similarity of connected nodes. The entire pipeline — boundary detection, graph construction, GNN, and classification — is trained end-to-end. On ActivityNet and THUMOS14, GTAD outperforms BMN-based baselines, particularly on videos with multiple simultaneous action instances where inter-proposal context provides the most benefit.

---

### Q9 [Advanced] How do snippet-level and segment-level features affect proposal and detection quality?

**Q:** What is the difference between snippet-level and segment-level video features, and how does this choice affect temporal proposal generation and action classification?

**A:** Snippet-level features are extracted densely at every short temporal unit — typically a 16-frame or 32-frame clip — by a video backbone (C3D, I3D, TSN, SlowFast). The output is a feature sequence of length proportional to the video duration, preserving fine-grained temporal resolution. Segment-level features are extracted over an entire proposal or action segment, either by processing the whole segment as input to the backbone or by temporally pooling the snippet-level features within the segment into a single fixed-dimensional vector.

For temporal proposal generation, snippet-level features are essential. Boundary detection requires per-snippet actionness scores and precise start/end probability predictions — operations that need the full temporal resolution of the snippet sequence. Methods like BSN and BMN explicitly operate on snippet-level feature sequences to predict per-snippet probabilities. Using segment-level features at this stage would destroy the temporal resolution needed for precise boundary localization.

For action classification, the choice involves a precision-coverage trade-off. Segment-level features aggregate information across the full action extent and are more robust to per-frame noise, making them better suited for recognizing the holistic pattern of an action. However, they discard temporal structure within the segment. Snippet-level features preserve the internal temporal dynamics (e.g., the sub-motion sequence within a complex activity) and allow the classifier to use attention mechanisms to focus on the most discriminative moments. Modern two-stage detectors often compute segment-level features from the snippet-level sequence using attention pooling or ROI-Align-style temporal pooling rather than global average pooling, balancing coverage with discriminability.

---

### Q10 [Advanced] How do NMS and Soft-NMS operate in the temporal domain, and what are their limitations?

**Q:** How are NMS and Soft-NMS applied to temporal action detection outputs, and what improvements have been proposed to address their limitations?

**A:** In temporal action detection, after scoring all candidate proposals, multiple predictions often correspond to the same underlying action instance. Non-Maximum Suppression (NMS) resolves this by iterating over proposals in decreasing confidence order: the highest-scoring proposal is retained, and all remaining proposals with tIoU above a threshold θ with the retained proposal are hard-suppressed (their scores are set to zero and they are discarded). This process repeats until no proposals remain. The result is a sparse set of non-overlapping detections. The key hyperparameter is θ: a small θ aggressively suppresses overlapping proposals (low false positives, but may miss distinct actions with high temporal overlap), while a large θ allows more proposals to survive (higher recall, but more duplicates).

Soft-NMS (Bodla et al., 2017), originally proposed for image detection and readily applicable to TAD, replaces hard suppression with a continuous score decay. Instead of zeroing out overlapping proposals, Soft-NMS reduces their scores proportionally to their tIoU with the retained proposal: either linearly (score × (1 - tIoU)) or with a Gaussian decay (score × exp(-tIoU²/σ)). Proposals with low overlap receive minimal penalty, while highly overlapping proposals are down-weighted but not eliminated. This is particularly beneficial for TAD because actions can genuinely overlap in time (e.g., "drinking" and "talking" occurring simultaneously), and hard suppression would incorrectly eliminate valid detections.

Both NMS and Soft-NMS share a fundamental limitation: they are non-differentiable post-processing steps that prevent end-to-end gradient flow from the final detection set to the proposal scoring network. Several approaches have been proposed to address this. Learning-based suppression methods train a separate network to select the optimal subset of proposals. Set-prediction approaches (DETR-style) replace NMS entirely with bipartite matching during training, constraining each query to detect at most one instance. Despite these advances, NMS and Soft-NMS remain the dominant post-processing strategy in practice due to their simplicity and strong empirical performance.

## Temporal Sentence Grounding

### Q11 [Basic] What is Temporal Sentence Grounding and how does it differ from TAD?

**Q:** What is the Temporal Sentence Grounding task, and what are its fundamental differences from Temporal Action Detection?

**A:** Temporal Sentence Grounding (TSG), also called Natural Language Video Grounding or Temporal Moment Retrieval, is the task of localizing a specific moment in an untrimmed video described by a natural language query. Given a video and a sentence such as "a person opens the refrigerator and takes out a bottle," the model must predict the temporal segment (t_start, t_end) that corresponds to this description. The output is typically a single segment per query, reflecting that the query describes one specific moment.

The core difference from TAD lies in the nature of the queries. TAD uses a fixed, closed vocabulary of predefined action categories (e.g., "throwing," "high jump") and detects all instances of these categories in a video without external conditioning. TSG uses open-vocabulary natural language queries that can describe arbitrary events in compositional language, and the localization is conditioned on the specific query — the same video yields different segments for different queries. This makes TSG fundamentally a cross-modal task requiring both visual understanding and language comprehension.

TSG is also typically evaluated differently: the primary metric is "R@1, IoU≥θ" (the fraction of queries for which the top-1 predicted segment achieves tIoU ≥ θ with the ground truth), reported at multiple thresholds (θ = 0.3, 0.5, 0.7). Mean IoU over all queries is also reported. Standard benchmark datasets include Charades-STA (first-person indoor activities with sentence annotations), ActivityNet-Captions (diverse activities from YouTube), DiDeMo (Flickr videos with descriptive sentences), and QVHighlights (YouTube videos with both grounding and highlight detection annotations).

---

### Q12 [Basic] What are the trade-offs between two-stage and one-stage grounding approaches?

**Q:** How do two-stage and one-stage methods for temporal sentence grounding differ, and what are the trade-offs?

**A:** Two-stage grounding methods first generate a set of temporal proposals (using any TAP method) independently of the query, then score each proposal by measuring the cross-modal similarity between the proposal's visual features and the query's language features. The proposal with the highest similarity score is selected as the prediction. Representatives include CTRL (Cross-modal Temporal Regression Localizer) and ACRN (Attentive Cross-modal Retrieval Network). The advantage is modularity: any proposal generator can be paired with any cross-modal matching head, and the proposal quality sets an upper bound on localization precision. The disadvantage is efficiency: the cross-modal matching must be computed for every proposal (O(N) operations per query), and proposals are generated without query conditioning, potentially producing candidates that are poorly aligned with query semantics.

One-stage grounding methods bypass explicit proposal generation and directly predict the target segment from joint video-query representations in a single forward pass. The model encodes both the video and the query, fuses their representations, and applies a regression head to predict (t_start, t_end) directly. Methods include LGI (Local-Global Video-Text Interactions), VSLNet (Video Span Localizing Network), and SeqPAN (Sequence-to-Sequence Proposal-Aware Network). One-stage methods are faster at inference (O(1) per query) and allow the query to condition the temporal search from the start, potentially focusing attention on relevant video regions before any localization. The trade-off is optimization difficulty: the model must simultaneously learn cross-modal alignment and temporal regression without the intermediate proposal signal providing training guidance.

The field has largely shifted toward one-stage and query-based (DETR-style) methods as the dominant paradigm, driven by improved pre-trained vision-language backbones (CLIP, BLIP) that provide strong initial cross-modal alignment.

---

### Q13 [Advanced] What are the main mechanisms for cross-modal alignment in temporal sentence grounding?

**Q:** How do language-video feature interaction mechanisms work in temporal sentence grounding, and what are the dominant design choices?

**A:** Cross-modal alignment in TSG requires computing a shared representation where semantically matching video segments and language descriptions are close in feature space. The first design choice is the encoding backbone. Early work used C3D or I3D features for video and LSTMs for language encoding. Modern methods use pre-trained vision-language models, most commonly CLIP, whose contrastive pre-training on image-text pairs provides strong zero-shot alignment that transfers well to video when applied at the snippet level. Fine-tuned CLIP features substantially reduce the cross-modal alignment burden for the grounding head.

The interaction mechanism determines how video and language features are combined. Early fusion concatenates or element-wise adds the sentence representation to each snippet feature before any temporal reasoning, giving every temporal computation access to the query. Cross-attention (transformer-based) allows each video snippet to attend over all query word tokens and vice versa, computing query-aware video features and video-aware query features simultaneously. This bidirectional interaction is now standard. Feature-wise Linear Modulation (FiLM) conditions the visual feature processing on affine transformations derived from the query, modulating both scale and shift of visual features by query-derived parameters — a lightweight alternative to full cross-attention that works well when the query is short.

Recent methods increasingly use pre-trained video-language models (InternVideo, VideoCLIP, BLIP-2) as unified encoders that jointly process video and text with interleaved cross-attention from the start, rather than encoding separately and fusing later. This pre-trained joint encoding yields richer cross-modal representations and reduces the amount of task-specific fine-tuning needed for grounding.

---

### Q14 [Advanced] How does weakly-supervised temporal sentence grounding work?

**Q:** What are the main strategies for temporal sentence grounding without timestamp annotations, and what makes this setting challenging?

**A:** In the weakly-supervised setting, the training data consists only of (video, sentence) pairs without temporal annotations indicating where in the video the described moment occurs. The model must learn to localize moments from the indirect supervision signal that the sentence should describe something that actually appears in the video — a highly underspecified constraint.

Multiple Instance Learning (MIL) is the most common approach. The video is divided into temporal segments (proposals or sliding windows), forming a "bag" of candidate moments. The learning objective is that at least one moment in the bag should match the sentence. This is typically implemented as a max-pooling loss: compute a matching score between each proposal and the sentence, take the maximum score as the bag score, and train the bag-level binary classification (does this video contain the described moment?) with cross-entropy. At inference, the highest-scoring proposal is selected. The challenge is the "false positive" problem: many proposals may superficially match the query without being the true moment, and MIL training cannot distinguish them.

Reconstruction-based methods use a different indirect signal: encode the sentence, locate a moment, and then decode the sentence from the located moment's visual features. The reconstruction loss (e.g., cross-entropy over the sentence tokens) indirectly supervises temporal localization — only the correct moment will contain sufficient information to reconstruct the specific query. Contrastive learning provides a complementary signal: pull matched (video segment, sentence) pairs close in embedding space and push mismatched pairs apart, with hard negative mining selecting challenging mismatches from within the same video or from similar-content videos.

The fundamental challenge is temporal bias: models often learn spurious correlations (e.g., most described moments start in the first half of the video) rather than genuine visual-language alignment. This requires explicit de-biasing strategies during training, such as negatively mining in-video negatives (same query, different temporal location) to force the model to localize based on content rather than position.

---

### Q15 [Advanced] How does Moment-DETR adapt the DETR paradigm to temporal sentence grounding?

**Q:** What is Moment-DETR's architecture and training approach, and how does set-prediction change the grounding pipeline?

**A:** Moment-DETR (Yuan et al., 2021) adapts DETR's set-prediction framework to temporal sentence grounding. The model uses N learnable moment queries, each intended to detect one relevant temporal moment in the video. The architecture encodes the video with a visual backbone (C3D or CLIP features at the snippet level) and the query sentence with a text encoder (RoBERTa or CLIP text encoder). Both are projected to a shared embedding dimension and concatenated along the sequence axis to form a joint visual-language token sequence. A standard transformer encoder processes this joint sequence with full self-attention, producing contextualized tokens for both video snippets and query words.

The N learnable moment queries then attend over the encoded joint sequence via transformer decoder cross-attention, each query aggregating information relevant to one potential moment. Each query's output is passed to a prediction head consisting of a linear classifier (foreground vs. background) and a linear regressor predicting (t_center, t_width) in normalized coordinates. During training, predictions are matched to ground truth moments using the Hungarian algorithm — bipartite matching that assigns each ground truth to exactly one query at minimum cost — eliminating the need for NMS. The training loss combines a classification loss, an L1 regression loss on moment coordinates, and a generalized IoU loss on predicted and ground truth segments. For the QVHighlights benchmark, an additional saliency loss is added to score each video clip for highlight detection.

The set-prediction formulation offers several advantages: it handles multiple relevant moments (one per query) naturally, avoids the non-differentiable NMS post-processing, and trains the full pipeline end-to-end. The main limitation is sensitivity to initialization and the need for sufficient capacity in the N queries to cover all possible moments — if N is too small, some moments go undetected. Subsequent work (QD-DETR, EaTR) extends Moment-DETR with query denoising and efficient attention to improve convergence and precision.
