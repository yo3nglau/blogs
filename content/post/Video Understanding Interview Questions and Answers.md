---
title: "Video Understanding: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-09'
categories:
  - Interview
tags:
  - Deep Learning
  - Video Understanding
  - Computer Vision
toc: true
---

## Temporal Modeling Foundations

### Q1 [Basic] Explain why processing video frames independently fails to capture temporal structure

**Q:** What information is lost when a model applies image-level processing to each frame in isolation, and which tasks expose this limitation most clearly?

**A:** A 2D CNN or image-based ViT applied frame-by-frame treats a video as an unordered bag of still images. Since each frame is processed independently, the model has no mechanism to relate activation patterns across time — motion direction, speed, deformation, and causal event sequences are entirely invisible. Two frames that appear at different temporal positions produce representations that never interact, so any information requiring comparison across time is discarded.

The practical consequence is **appearance bias**: the model classifies clips based on the objects and scenes present in individual frames rather than the dynamics between them. On Kinetics-400, many classes — "playing guitar," "riding a horse," "cooking" — carry strong single-frame appearance signatures that allow high accuracy without temporal reasoning. This masks the limitation on standard leaderboards.

Something-Something V2 (Goyal et al., 2017) was designed explicitly to expose this bias. Its 174 categories describe relative motion — "pushing something from left to right," "moving something away from the camera," "pretending to pour something into something" — where the action category changes if the motion direction is reversed, yet the objects and scene remain identical. On SSv2, models that achieve strong Kinetics-400 results via appearance features perform substantially worse relative to their capacity, revealing that their temporal modeling is superficial. Resolving this bias requires either explicit motion representations (optical flow), 3D convolutions that span time, or attention mechanisms that relate tokens across frames.

---

### Q2 [Basic] Describe the Two-Stream architecture and what each stream contributes

**Q:** How does the Two-Stream network achieve temporal modeling, and what signal does each stream provide that the other cannot?

**A:** The **Two-Stream network** (Simonyan & Zisserman, 2014) processes each video clip through two parallel CNN pathways with separate weights that are fused at the prediction layer. The **spatial stream** receives a single RGB frame sampled from the clip and recognizes objects, scenes, and contextual appearance. The **temporal stream** receives a stack of $L = 10$ consecutive optical flow fields — $L$ horizontal-displacement maps concatenated with $L$ vertical-displacement maps, forming a $2L$-channel input — and captures the pattern and magnitude of motion. Class scores from both streams are averaged at inference time.

The division of labor is intentional: the spatial stream excels at identifying what is in the scene, while the temporal stream recognizes how things are moving. Actions that are appearance-ambiguous (e.g., "brushing hair" vs. "brushing teeth") benefit from motion disambiguation; actions that are motion-ambiguous (e.g., multiple activities occurring in a distinctive location) benefit from spatial context. Simonyan & Zisserman (2014) reported 88.0% top-1 accuracy on UCF-101 and 59.4% on HMDB-51 by fusing the two streams, with the temporal stream alone outperforming the spatial stream on both benchmarks — demonstrating that explicit motion representation provides complementary and often dominant information.

The principal limitation is the preprocessing cost of dense optical flow, which must be computed offline for each video and stored as additional data. This cost motivated subsequent research into architectures that learn implicit motion representations from RGB input alone using 3D convolutions or temporal attention.

---

### Q3 [Advanced] Explain how 3D convolutions extend spatial CNNs to video and how factorized variants reduce their cost

**Q:** What does extending 2D convolutions to 3D enable in video representation learning, and why do factorized architectures such as R(2+1)D recover the performance of full 3D convolutions more efficiently?

**A:** A **3D convolution** applies a filter of shape $k_t \times k_h \times k_w$ to a spatiotemporal volume, learning features jointly over time and space within a local neighborhood. Unlike 2D convolutions followed by temporal pooling, 3D filters can detect spatiotemporally local patterns — the direction and speed of motion within a patch, the evolution of a deforming object — and compose them hierarchically across layers. C3D (Tran et al., 2015) demonstrated that $3 \times 3 \times 3$ filters applied uniformly across all layers on 16-frame clips learn general spatiotemporal features; fine-tuned from Sports-1M pretraining, C3D achieves 85.2% top-1 on UCF-101.

I3D (Carreira & Zisserman, 2017) introduced **inflation**: all 2D filters and pooling kernels in a pretrained Inception model are expanded to 3D by repeating the weights $N$ times along the temporal axis and dividing by $N$ to preserve the average activation magnitude. This allows initializing 3D models from ImageNet-pretrained weights, dramatically improving training efficiency. Two-stream I3D — RGB and optical flow streams each inflated independently — achieves 98.0% on UCF-101 and 80.7% on HMDB-51 after Kinetics pretraining, establishing 3D CNNs as the dominant video backbone of the era.

The core drawback of 3D convolutions is their parameter count: a $3 \times 3 \times 3$ filter has $3\times$ the parameters of a $3 \times 3$ filter and mixes spatial and temporal learning in ways that may be suboptimal. **Factorized** approaches decompose the 3D operation into a $1 \times k \times k$ spatial convolution followed by a $k \times 1 \times 1$ temporal convolution. R(2+1)D (Tran et al., 2018) showed that this decomposition introduces an additional nonlinearity between the spatial and temporal operations — effectively increasing the representational capacity without adding parameters — and achieves higher accuracy than full 3D convolutions at the same parameter count, while also enabling separate pretraining of spatial weights from image data.

---

### Q4 [Advanced] Describe the SlowFast network and how dual temporal pathways model different temporal granularities

**Q:** What motivates processing video at two different frame rates simultaneously, and how do lateral connections between the two pathways enable information exchange?

**A:** **SlowFast** (Feichtenhofer et al., 2019) is motivated by neurophysiology: primate visual cortex has M (magnocellular) cells that respond at high temporal frequency to coarse motion and P (parvocellular) cells that respond more slowly but encode fine spatial detail and color. The network instantiates this asymmetry with two pathways operating on the same video at different temporal sampling rates.

The **Slow pathway** samples 8 frames at low temporal resolution ($\tau = 16$ frame stride) and processes them with a large channel capacity ($C$ channels), learning rich semantic and appearance features. The **Fast pathway** samples 32 frames at high temporal resolution ($\tau = 2$) with small channel capacity ($C/8$), specializing in temporal dynamics and motion. Because the Fast pathway has only $1/8$ the channels of the Slow pathway, processing $4\times$ more frames adds only approximately $20\%$ additional computation — the asymmetry is computationally efficient.

Information flows from Fast to Slow through **lateral connections** at multiple stages: the Fast pathway's feature maps are time-strided to match the Slow pathway's temporal resolution and then concatenated or summed. This one-directional transfer allows the Slow pathway's semantic features to be informed by the Fine pathway's motion features without requiring the Slow pathway to process high-frame-rate input. Feichtenhofer et al. (2019) reported SlowFast 8×8 ResNet-101 with non-local blocks (Wang et al., 2018) achieves 79.8% top-1 on Kinetics-400 and 45.2% mAP on AVA v2.2, outperforming I3D two-stream and establishing the SlowFast design as a strong baseline for both action recognition and detection.

---

## Video Transformers

### Q5 [Basic] Explain how TimeSformer adapts ViT to video using factorized space-time attention

**Q:** What is the computational challenge of applying standard ViT attention to video, and how does TimeSformer's divided space-time attention resolve it?

**A:** Applying standard ViT to a $T$-frame video with $N$ spatial patches per frame produces $TN$ tokens. A standard multi-head self-attention block computes pairwise attention over all $TN$ tokens, with complexity $O(T^2 N^2)$ — quadratic in both frame count and spatial resolution. For $T = 8$ and $N = 196$ (14×14 grid on a 224×224 image), this is already $\sim 2.4\text{M}$ attention pairs per layer, growing to tens of millions for longer clips or higher resolution.

**TimeSformer** (Bertasius et al., 2021) introduces **divided space-time attention**: within each transformer block, attention is computed in two sequential steps. First, each patch token attends only to the other $N-1$ tokens at the same spatial position in the current frame — **spatial attention**, $O(N^2)$ per frame. Then, each patch token attends only to the $T-1$ counterparts at the same spatial position in all other frames — **temporal attention**, $O(T^2)$ per spatial location. Total complexity: $O(T \cdot N^2 + N \cdot T^2)$, which is asymptotically lower than $O(T^2 N^2)$ and practically much cheaper for large $N$.

Bertasius et al. (2021) ablated five attention designs — local, global, divided space-time, sparse local + global, and axial — and found divided space-time consistently best under matched compute budgets. TimeSformer-L (large model, 96 frames with stride 4) achieves 80.7% top-1 on Kinetics-400 and 62.5% on Something-Something V2, the latter demonstrating that temporal attention across many frames improves fine-grained motion understanding.

---

### Q6 [Basic] Describe ViViT's factorised encoder model and what its architectural choices achieve

**Q:** What are ViViT's main architectural variants for video, and how does the factorised encoder balance temporal coverage with computational feasibility?

**A:** **ViViT** (Video Vision Transformer, Arnab et al., 2021) adapts ViT to video by proposing four model variants with different approaches to spatiotemporal attention. Model 1 (*Spatio-temporal tokens*) tokenizes all patches from all frames jointly and applies standard ViT attention — straightforward but quadratically expensive in $TN$. Model 3 and 4 factorize attention within each block (similar to TimeSformer). Model 2 (*Factorised encoder*) achieves the best accuracy-efficiency trade-off and is the canonical ViViT.

In the **Factorised encoder** (Model 2), a ViT spatial encoder processes each frame independently, extracting a [CLS] token per frame representing the frame's global summary. These $T$ per-frame [CLS] tokens are then passed to a second, smaller **temporal encoder** (ViT) that performs self-attention across the $T$ tokens to aggregate temporal information. This two-stage design reduces the temporal attention cost from $O(T^2 N^2)$ to $O(T \cdot N^2) + O(T^2)$ — the spatial encoder sees $N$ tokens per frame, and the temporal encoder sees only $T$ tokens total.

Arnab et al. (2021) trained ViViT with ViT-L/16 initialized from JFT-300M pretraining. Model 2 achieves 81.3% top-1 on Kinetics-400, 65.9% on Something-Something V2, and 83.0% on Kinetics-600 — results competitive with state-of-the-art 3D CNNs while using a pure transformer architecture. The factorised encoder's separation of spatial and temporal processing also enables initializing the spatial encoder directly from pretrained image ViTs, with only the small temporal encoder trained from scratch.

---

### Q7 [Advanced] Explain how Video Swin Transformer extends shifted window attention to spatiotemporal volumes

**Q:** What enables Video Swin to achieve linear computational complexity in video token count, and how do shifted windows maintain global information flow across non-overlapping local windows?

**A:** The image **Swin Transformer** partitions the 2D feature map into non-overlapping windows and applies self-attention locally within each window, achieving linear complexity $O(HW)$ instead of $O((HW)^2)$ for full attention. Each token only attends to the $M^2$ tokens in its window, where $M$ is the fixed window size, making the attention cost $O(M^4 \cdot HW/M^2) = O(M^2 \cdot HW)$ — linear in the number of tokens for fixed $M$.

**Video Swin Transformer** (Liu et al., 2022) extends this to 3D by partitioning the spatiotemporal volume into non-overlapping 3D windows of size $P \times M \times M$ (temporal × height × width). Self-attention is applied within each 3D window. For a video of $T \times H \times W$ tokens with window size $P \times M \times M$, the cost is $O(P^2 M^4 \cdot THW / (PM^2)) = O(P M^2 \cdot THW)$ — linear in $T$, $H$, and $W$ for fixed window sizes. This achieves global coverage in terms of video length while maintaining local attention structure.

The challenge is that non-overlapping windows cannot communicate directly, limiting the model to local context. **Shifted window partitioning** addresses this by alternating between two window configurations every other layer: regular windows at positions $(0, 0, 0)$ and shifted windows at positions $(P/2, M/2, M/2)$. Tokens near window boundaries in one layer fall into the center of windows in the next — information flows across window boundaries through this shifting mechanism without explicit cross-window attention. Liu et al. (2022) showed Video Swin-L achieves 84.9% top-1 on Kinetics-400 and 86.1% on Kinetics-600, at the time substantially outperforming TimeSformer and ViViT at comparable or lower FLOPs, demonstrating the advantage of preserving local spatial inductive biases in video through windowed attention.

---

### Q8 [Advanced] Describe MViTv2's multiscale feature hierarchy and pooling attention mechanism

**Q:** What architectural principle does MViTv2 introduce that standard video transformers lack, and how does pooling attention enable spatiotemporal resolution changes within a transformer?

**A:** Standard ViT-based video models process all tokens at a single fixed resolution throughout all layers — every layer operates on the same $T \times H \times W$ token grid. This is wasteful: early layers that capture local low-level features need high resolution, while later layers that aggregate semantic information could operate more efficiently on fewer tokens. CNNs have long used feature pyramids (progressively coarser resolution, larger channels) to match representational needs to architectural depth.

**MViTv2** (Multiscale Vision Transformers v2, Li et al., 2022) introduces this multiscale hierarchy to video transformers through **Pooling Attention**. At each attention layer, the queries $Q$, keys $K$, and values $V$ are independently pooled using learned convolutional kernels with stride $s$ before computing attention:

$$\hat{Q} = \text{Pool}(Q, s_q), \quad \hat{K} = \text{Pool}(K, s_k), \quad \hat{V} = \text{Pool}(V, s_v)$$

At stage transitions (resolution reduction points), applying stride $s_q > 1$ to $Q$ reduces the output token count by $s_q^2$ per spatial dimension, halving spatial resolution while the channel count doubles. This creates a feature pyramid analogous to ResNet stages: fine-grained high-resolution tokens in early layers, coarse low-resolution tokens in deep layers.

MViTv2 further introduces **decomposed relative position embeddings**, factorizing the 3D positional bias into separate spatial and temporal components to reduce parameter count, and **residual pooling connections** that add the input tokens (before pooling) to the pooled output to improve gradient flow. Li et al. (2022) showed MViTv2-L on 40×3 clips achieves 86.1% top-1 on Kinetics-400 and 70.5% on Something-Something V2, with competitive efficiency relative to Video Swin at comparable model size. The strong SSv2 result reflects that the multiscale hierarchy captures fine local motion features in early high-resolution layers while aggregating temporal context in later coarse layers.

---

## Video-Language Understanding

### Q9 [Basic] Explain how CLIP-based models are adapted for video-text alignment

**Q:** What strategies enable image-pretrained CLIP to process video sequences, and how do temporal fusion approaches compare in retrieval performance?

**A:** CLIP produces a single embedding per image. To encode a video clip, the simplest adaptation samples $T$ frames uniformly, encodes each independently with CLIP's image encoder, and aggregates the frame embeddings — typically by **mean pooling** along the temporal dimension. The resulting video embedding is compared to a text query embedding via cosine similarity. This approach requires no temporal-specific parameters and allows zero-shot video retrieval using the same image-text alignment CLIP learned.

**CLIP4Clip** (Luo et al., 2022) systematically studied temporal fusion strategies for CLIP-based video retrieval. Three designs were evaluated: (1) mean pooling over frame embeddings; (2) a sequential transformer that applies temporal self-attention over frame embeddings after adding temporal position encodings; (3) a tight-type design that uses a weighted sum over frames. On MSR-VTT-1kA text-to-video retrieval, mean pooling achieves 43.1% R@1, while sequential transformer and tight-type both reach approximately 44.5% R@1 — mean pooling is surprisingly competitive despite ignoring temporal order entirely.

This result reveals both a strength and a limitation of CLIP for video: its rich visual-semantic features transfer effectively to video retrieval on datasets where temporal order provides limited additional signal (retrieval from clip-level semantics). However, CLIP's temporal insensitivity becomes a problem for tasks requiring motion understanding — video QA, temporal grounding, or fine-grained action recognition on SSv2-style tasks. Addressing these requires either temporal modules added on top of CLIP frame features, joint video-text contrastive pretraining that explicitly encodes temporal structure, or architectures like InternVideo that combine masked video modeling with contrastive alignment.

---

### Q10 [Basic] Describe the standard benchmarks for video understanding and what different benchmarks measure

**Q:** How do the major video understanding benchmarks differ in what temporal and semantic skills they evaluate?

**A:** **Kinetics-400/600/700** provides large-scale action recognition with ~300K trimmed 10-second clips. Most Kinetics categories — "playing soccer," "cutting hair," "cooking spaghetti" — are strongly appearance-driven and can be recognized from a single frame at high accuracy. Kinetics measures whether a model has learned visual semantics broadly enough to discriminate 400–700 human activities, but does not require genuine temporal reasoning.

**Something-Something V2** (Goyal et al., 2017) deliberately tests temporal reasoning: its 174 categories describe object–motion relationships — "pushing something from left to right," "covering something with something," "pretending to take something from somewhere" — where the action identity changes if the motion direction is reversed. A model that processes frames independently cannot distinguish symmetric motions. SlowFast 8×8 achieves 63.1% top-1 on SSv2 vs. 79.8% on Kinetics-400 — the gap between a model's Kinetics and SSv2 performance is a diagnostic of how much its temporal modeling is genuine vs. appearance-based.

**ActivityNet-1.3** contains ~20K untrimmed videos (2–10 minutes) with 200 activity categories, primarily used for temporal action detection and dense video captioning. It tests temporal localization: the model must determine not just what happens but exactly when within a long, multi-event video. **THUMOS-14** similarly tests temporal action detection on 200 untrimmed videos with 20 classes, evaluated by mean Average Precision (mAP) at multiple temporal IoU thresholds.

**AVA v2.2** evaluates spatio-temporal action detection: person bounding boxes are provided at 1 FPS over 430 video clips, and the model must classify 80 atomic visual actions (e.g., "stand," "eat," "talk to") for each person. This benchmark tests whether models can ground fine-grained actions to specific people at precise temporal positions rather than clip-level predictions.

---

### Q11 [Advanced] Explain InternVideo's unified pretraining strategy for a general video foundation model

**Q:** What pretraining objectives does InternVideo combine, and what gap does each component address that the other alone cannot close?

**A:** **InternVideo** (Wang et al., 2022) is motivated by a key observation: masked video modeling and multimodal contrastive learning address complementary weaknesses. Masked video modeling (VideoMAE-style) learns rich spatiotemporal features by predicting masked regions — the model must understand motion, object part structure, and temporal continuity. However, it produces representations that are not aligned with text or abstract semantic categories; the learned features are optimized for pixel-level prediction rather than task-level classification or retrieval. Conversely, CLIP-style contrastive learning aligns video and text representations effectively for retrieval and zero-shot tasks, but the visual encoder processes frames independently, failing to capture temporal structure — CLIP with mean pooling and CLIP with temporal ordering are nearly equivalent on retrieval benchmarks.

InternVideo addresses both limitations through a two-stage training pipeline. In stage 1, a ViT-L backbone is pretrained with masked video modeling on large-scale video corpora, learning spatiotemporally rich features. In stage 2, the pretrained backbone is fine-tuned with video-text contrastive learning using a cross-modal attention module (based on UniFormer) that enables dense video-text interactions. The contrastive stage aligns the temporally-aware representations from stage 1 with language semantics — a combination unavailable in either approach independently.

The resulting model achieves state-of-the-art results across 39 video understanding datasets spanning action recognition, video-text retrieval, temporal action detection, and video question answering (Wang et al., 2022). This breadth demonstrates that the combination of temporal feature learning and semantic alignment produces more general representations than either objective alone.

---

### Q12 [Advanced] Describe the architecture and key challenges of video large language models

**Q:** How do video LLMs connect visual encoders with language generation, and what specific challenges arise from the temporal dimension of video that do not appear in image-based LLMs?

**A:** **VideoLLaMA** (Zhang et al., 2023) follows the LLaVA paradigm for video: a frozen visual encoder extracts per-frame features, a module compresses them into a manageable token count, and a language model (LLaMA) generates responses conditioned on the visual tokens and a text instruction. Specifically, each frame is processed by BLIP-2's Q-Former, which cross-attends between learned query tokens and the dense frame features to produce a fixed set of 32 query tokens per frame. A **Video Q-Former** then applies temporal attention across the per-frame query tokens to integrate temporal information before passing them to the LLM. The LLM sees the compressed video tokens as prefix tokens in its context, enabling open-ended question answering, description, and instruction following over video content.

The temporal dimension introduces challenges not present in image LLMs. **Token count** scales with clip length: 8 frames × 32 Q-Former tokens = 256 visual tokens; 32 frames × 32 = 1024 tokens. The LLM attends quadratically over context, so long videos with many frames quickly become computationally prohibitive. Solutions include aggressive temporal subsampling (limiting input to 8–16 keyframes), spatial token compression (reducing from 256 to 32 tokens per clip via Q-Former pooling), or using LLMs with extended context windows.

**Temporal grounding** presents a fundamental challenge: when the user asks "what happened at timestamp $t$?" or "what does the person do before dropping the object?", the model must relate absolute or relative timestamps to visual content. Standard Q-Former compression loses temporal resolution information — the 32 output tokens summarize the entire clip without explicit temporal indexing. Enabling precise temporal localization in video LLMs requires either temporal position encodings on visual tokens, specialized temporal grounding heads, or formulations that predict timestamps as text tokens alongside event descriptions.

---

## Temporal Action Localization

### Q13 [Basic] Describe the temporal action detection task and how evaluation differs from action recognition

**Q:** What does a temporal action detection model need to output, and how do the evaluation metrics reflect the joint difficulty of localization and classification?

**A:** **Temporal action detection** (TAD) takes an untrimmed video — often several minutes long containing multiple events, background segments, and transitions — and produces a set of predictions, each consisting of a start time $t_s$, end time $t_e$, action class, and confidence score. Unlike action recognition, which classifies a pre-segmented clip, TAD requires simultaneously determining what happened and when, without pre-specified boundaries.

The evaluation metric is **mean Average Precision** (mAP) over classes, computed at one or more temporal Intersection-over-Union (tIoU) thresholds. A predicted segment $[\hat{t}_s, \hat{t}_e]$ matches a ground truth $[t_s^*, t_e^*]$ if:

$$\text{tIoU} = \frac{|\hat{I} \cap I^*|}{|\hat{I} \cup I^*|} \geq \theta$$

where $\theta$ is typically $0.5$ for ActivityNet-1.3 and swept across $\{0.3, 0.4, 0.5, 0.6, 0.7\}$ for THUMOS-14 (with the average reported). The joint requirement — correct class label and temporal overlap above threshold — means that a perfectly classified but temporally misaligned prediction receives zero recall, and a well-localized prediction of the wrong class receives zero precision. This forces models to be simultaneously precise in time and accurate in class.

Two main paradigms exist: **two-stage** methods that first generate class-agnostic temporal proposals (candidate intervals likely to contain any action) and then classify each; and **one-stage** methods that directly predict (class, start, end) at each temporal anchor point without separate proposal generation. Two-stage methods typically achieve higher accuracy by allowing the classifier to focus on foreground proposals; one-stage methods are faster and more end-to-end.

---

### Q14 [Advanced] Explain how BMN generates temporal proposals using the boundary-matching mechanism

**Q:** What is the Boundary-Matching confidence map, and how does it enable evaluating all possible temporal proposals simultaneously?

**A:** **BMN** (Boundary-Matching Network, Lin et al., 2019) frames temporal proposal generation as a two-component prediction problem: (1) where are action boundaries, and (2) how confident is each possible (start, duration) pair as a complete action proposal?

The **Temporal Evaluation Module** (TEM) processes the video feature sequence $\{h_t\}$ and predicts, for each frame $t$, two probabilities: start probability $p^s_t$ (the likelihood that frame $t$ is the beginning of an action) and end probability $p^e_t$ (the likelihood that it is the end). These boundary probabilities are trained with ground-truth annotations indicating which frames lie at action boundaries.

The **Boundary-Matching Module** (BMM) constructs a 2D confidence map $M$ of shape $T \times D_{\max}$, where entry $M[s][d]$ represents the confidence of the proposal starting at frame $s$ with duration $d$ being a complete action. To fill this map, each proposal $(s, d)$ is characterized by features sampled at $N$ uniformly spaced points between $s$ and $s + d$ — this **BM sampling** collapses the varying-length proposal into a fixed-length representation. A lightweight network then produces $M[s][d]$ from this sampled feature. All $O(T \cdot D_{\max})$ proposals are evaluated in parallel via the shared BM sampling mechanism.

Final proposal scores combine all three signals: $\text{score}(s, d) = p^s_s \cdot p^e_{s+d} \cdot M[s][d]$. High-scoring proposals have both high boundary confidence and high content confidence. Lin et al. (2019) reported BMN achieves 67.10% AUC on ActivityNet-1.3 temporal proposal generation, and the two-stage pipeline (BMN proposals + action classifier) established the boundary-matching framework as a strong baseline for temporal action detection.

---

### Q15 [Advanced] Describe ActionFormer's transformer-based approach to one-stage temporal action detection

**Q:** How does ActionFormer structure multi-scale temporal features for dense prediction, and what role does local window attention play in its design?

**A:** **ActionFormer** (Zhang et al., 2022) is a one-stage, anchor-free temporal action detector built on a multi-scale transformer encoder with dense prediction heads. The model takes pre-extracted video features (e.g., from I3D, SlowFast, or C3D) at a fixed temporal stride as input and produces per-location predictions without requiring separate proposal generation.

The architecture consists of three stages. First, a lightweight 1D temporal convolution backbone projects input features and expands temporal depth to produce an initial feature sequence. Second, a **multi-scale feature pyramid** is constructed by applying temporal strided convolutions to create representations at multiple temporal resolutions — coarser levels capture long-range context while finer levels localize precise boundaries. At each scale, **multi-head local temporal attention** (window-based, window size $w$) is applied: each feature token attends only to its $w$ temporal neighbors rather than all positions, maintaining $O(w \cdot T)$ complexity per scale.

At each pyramid level, three prediction heads operate in parallel: a **classification head** that predicts action classes at each temporal location; a **regression head** that predicts the distance to the start and end boundaries of the action centered at each location (anchor-free formulation); and an **IoU head** that predicts the expected temporal IoU between the predicted and ground-truth segment, used for proposal ranking. Predictions across all scales are merged with NMS.

Zhang et al. (2022) reported ActionFormer achieves 82.1% average mAP at tIoU thresholds $[0.3{:}0.7]$ on THUMOS-14 and 36.56% average mAP on ActivityNet-1.3, with I3D features. The local window attention's ability to model temporal context beyond fixed receptive fields — while avoiding the cost of global attention — enables ActionFormer to capture both fine-grained boundary cues and long-range context simultaneously.

---

### Q16 [Advanced] Explain how Vid2Seq formulates dense video captioning as sequence generation with temporal tokens

**Q:** What makes dense video captioning harder than standard captioning, and how does Vid2Seq's temporal token approach enable joint event localization and description?

**A:** **Dense video captioning** requires generating a set of $\{(t_s^i, t_e^i, \text{caption}_i)\}$ tuples describing all events in an untrimmed video — unlike standard video captioning, which produces a single description for a pre-segmented clip. The joint difficulty is that the model must simultaneously determine how many events are present, when each event occurs, and what natural-language description fits each. Prior approaches handled this with separate proposal and captioning modules; **Vid2Seq** (Yang et al., 2023) unifies both into a single sequence-to-sequence model.

Vid2Seq's key insight is treating temporal timestamps as discrete vocabulary tokens. The time axis is divided into $T_B$ bins, and each bin is assigned a special token $[\text{TIME}_0], [\text{TIME}_1], \ldots, [\text{TIME}_{T_B-1}]$ added to the language model's vocabulary. The model generates an interleaved sequence of time tokens and text tokens:

$$\langle \text{TIME}_{42} \rangle \text{ a person begins cutting vegetables} \langle \text{TIME}_{93} \rangle \, \langle \text{TIME}_{95} \rangle \text{ they add spices to the pan} \langle \text{TIME}_{142} \rangle \ldots$$

The sequential structure naturally handles variable numbers of events — the model generates as many event tuples as needed by producing time-token pairs interleaved with text — and enables end-to-end training with a standard language modeling loss.

Vid2Seq is pretrained on $\sim$1M narrated YouTube videos where Automatic Speech Recognition (ASR) transcripts provide free temporal supervision: each ASR sentence has a timestamp from subtitle timing, enabling the model to learn to associate spoken descriptions with visual content at no annotation cost. Yang et al. (2023) showed this large-scale noisy pretraining substantially improves performance on ActivityNet Captions and YouCook2 dense captioning benchmarks over models trained from scratch, demonstrating that narrated video provides a scalable source of structured temporal supervision.

---

## Video Self-Supervised Learning

### Q17 [Basic] Explain optical flow estimation and why deep learning methods outperform classical approaches

**Q:** What does optical flow represent, and what fundamental limitations of classical algorithms does RAFT's iterative correlation-based approach overcome?

**A:** **Optical flow** estimates the apparent 2D displacement of each pixel between two consecutive frames: given frames $I_1$ and $I_2$, the flow field $\mathbf{f} = (u, v)$ at pixel $(x, y)$ encodes where that pixel moves, such that $I_1(x, y) \approx I_2(x + u, y + v)$. Accurate flow requires handling large displacements, occluded regions, motion boundaries, and fine-grained texture.

Classical methods — Lucas-Kanade (patch-based, small-motion assumption) and Horn-Schunck (global variational regularization) — rely on the brightness constancy and spatial smoothness assumptions. Both fail on large displacements (violating the linearization required by gradient-based methods), repetitive textures (which produce ambiguous matches), and motion boundaries (where smoothness regularization over-smooths the sharp transition between foreground and background motion).

**RAFT** (Recurrent All-Pairs Field Transforms, Teed & Deng, 2020) overcomes these limitations through an explicit **all-pairs correlation volume**: the dot product between encoder features at every pair of positions in the two frames is precomputed — $C(i, j) = \langle f_1(i), f_2(j) \rangle$ for all source position $i$ and target position $j$. This 4D correlation volume captures match quality at all displacements without linearization. Flow is then estimated by an iterative GRU-based **update operator** that, at each step, looks up the correlation volume at positions offset by the current flow estimate and its neighborhood, and predicts a flow correction. Multiple iterations progressively refine the estimate, enabling both accuracy on small details and robustness to large displacements via the multi-scale correlation pyramid.

RAFT achieves 1.43 EPE (End-Point Error) on Sintel Clean and 2.71 EPE on Sintel Final (Teed & Deng, 2020), substantially outperforming prior learned methods. For video understanding, precomputed RAFT flow can replace hand-crafted optical flow in two-stream architectures; however, the computational cost of computing dense flow per frame pair motivates architectures that implicitly learn motion representations from RGB input.

---

### Q18 [Basic] Describe masked video modeling and how tube masking prevents trivial temporal shortcuts

**Q:** Why does masked image modeling's approach require modification for video, and what property of tube masking forces the model to learn genuine spatiotemporal reasoning?

**A:** **Masked image modeling** (MAE-style) randomly masks spatial patches and trains the model to reconstruct them from visible context. Applied naively to video, masking each frame's patches independently at the same $75\%$ rate creates a shortcut: for any masked patch at position $(t, h, w)$, the same patch position in adjacent frames $t\pm 1$ is likely unmasked. The model can fill in masked regions by copying content from neighboring frames — this copy-paste shortcut requires no semantic understanding, only temporal proximity.

**Tube masking** eliminates this shortcut by masking the same spatial positions across all frames: if patch $(h, w)$ is masked in frame $t = 0$, it is also masked in frames $t = 1, 2, \ldots, T-1$. A tube of masked content spans the entire temporal dimension at consistent locations, making temporal copying impossible. The model must infer masked content from the visible spatial context within the same timestamp and adjacent visible tubes — a task requiring genuine spatial reasoning about object structure and appearance.

The masking ratio is also raised from MAE's $75\%$ to $90\%$ for video. Video frames are highly temporally redundant: adjacent frames differ primarily in slow motion and minor deformation, sharing most visual content. At $75\%$ masking with random per-frame masks, many patches remain unmasked in at least one nearby frame, providing sufficient low-level context for reconstruction without global understanding. At $90\%$, only $10\%$ of spatiotemporal patches are visible — the model must reason about global image structure, object semantics, and motion patterns across the sparse visible context to reconstruct large connected masked regions.

---

### Q19 [Advanced] Explain VideoMAE's design choices and what its data efficiency results reveal about in-domain pretraining

**Q:** What specific design elements make VideoMAE effective, and why does pretraining on a small in-domain dataset outperform pretraining on a large generic dataset for fine-tuning on that domain?

**A:** **VideoMAE** (Tong et al., 2022) combines tube masking at $90\%$ ratio with a **joint space-time encoder**: all visible tokens from all $T$ frames are concatenated and processed together by a standard ViT, enabling direct spatiotemporal attention between any two visible patches regardless of their frame origin. There is no factorization of spatial and temporal attention — the model can discover whatever spatiotemporal dependencies are most useful for reconstruction, without architectural constraints on how time and space interact.

The architecture is asymmetric: the encoder processes only the $\sim 10\%$ visible tokens, producing a compact set of latent representations. A lightweight decoder (smaller ViT with fewer blocks) receives these encoder outputs interleaved with mask tokens and reconstructs the $16\times 16$ pixel values of all masked patches. The encoder's cost is low due to the small visible token count ($\approx 10\%$) despite joint space-time processing; the decoder's cost is moderate but it is discarded at inference time.

The most surprising result concerns **data efficiency**: Tong et al. (2022) showed that VideoMAE ViT-S pretrained on UCF-101 (3,537 videos, 13 action classes) achieves higher fine-tuning accuracy on UCF-101 than VideoMAE ViT-S pretrained on Kinetics-400 ($\sim$240K videos). This reversal of the usual "more data is better" trend suggests that the masked video modeling pretext task strongly benefits from in-domain visual content: a model that learns to reconstruct human action videos develops spatiotemporal priors specific to human body motion, scene context, and object interaction patterns that transfer efficiently to action recognition with few labeled examples. Generic large-scale data introduces visual diversity that may not align with the target domain's distribution. VideoMAE ViT-H pretrained on Kinetics-400 and fine-tuned achieves 86.6% top-1 on Kinetics-400, demonstrating strong results when pretraining and target domains align.

---

### Q20 [Advanced] Describe how VideoMAE V2's dual masking enables scaling masked video pretraining to billion-parameter models

**Q:** What memory bottleneck prevents directly scaling VideoMAE to ViT-g, and how does dual masking address it while maintaining pretraining quality?

**A:** **VideoMAE V2** (Wang et al., 2023) scales masked video pretraining to ViT-g ($\sim$1B parameters) — an architecture previously too large for practical video pretraining. The barrier is memory: even with $90\%$ encoder masking, a ViT-g encoding $T \times H/16 \times W/16$ visible tokens produces intermediate activations that, when combined with the decoder's reconstruction of all masked tokens, exceed GPU memory limits at training-viable batch sizes. The decoder alone must process $0.9 \cdot T \cdot (H/16) \cdot (W/16)$ mask tokens — for a 16-frame, 224² input, approximately 2,300 masked tokens — each requiring ViT-g-scale computation.

**Dual masking** introduces a second masking stage at the decoder: after the encoder processes the small visible set, the decoder is only asked to reconstruct a random subset ($\sim 50\%$) of the masked tokens rather than all of them. The encoder mask selects $10\%$ of tokens to keep visible (as in VideoMAE); the decoder mask further subsamples the $90\%$ masked tokens to the $50\%$ that will be reconstructed. Only these $\sim 45\%$ of total tokens need to be processed by the decoder, reducing decoder computation by approximately $2\times$.

Wang et al. (2023) showed that selecting which masked tokens to reconstruct randomly — rather than systematically biasing toward hard or easy patches — preserves reconstruction quality with negligible accuracy degradation on downstream tasks. The gradient signal from partial reconstruction remains well-distributed because any token is equally likely to be reconstructed across training steps.

VideoMAE V2 also introduces **mixed pretraining**: supervised labeled data (Kinetics) and unlabeled web video data (WEB-VIDEO, $\sim$10M clips) are mixed into the same masked pretraining batch. The labeled videos contribute richer semantic content alongside the unlabeled diversity, improving both representation quality and data utilization. VideoMAE V2 ViT-g achieves 90.0% top-1 on Kinetics-400 (Wang et al., 2023) — the first video model to surpass $90\%$ on this benchmark — and sets new state-of-the-art results across multiple action recognition and video understanding datasets.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Why frame-independent processing fails temporal modeling | Temporal Modeling Foundations |
| Q2 | Basic | Two-Stream spatial and temporal streams | Temporal Modeling Foundations |
| Q3 | Advanced | 3D convolutions: C3D, I3D, and factorized variants | Temporal Modeling Foundations |
| Q4 | Advanced | SlowFast dual temporal pathway design | Temporal Modeling Foundations |
| Q5 | Basic | TimeSformer divided space-time attention | Video Transformers |
| Q6 | Basic | ViViT factorised encoder model | Video Transformers |
| Q7 | Advanced | Video Swin shifted 3D window attention | Video Transformers |
| Q8 | Advanced | MViTv2 pooling attention and multiscale hierarchy | Video Transformers |
| Q9 | Basic | Adapting CLIP for video-text alignment | Video-Language Understanding |
| Q10 | Basic | Key benchmarks for temporal reasoning evaluation | Video-Language Understanding |
| Q11 | Advanced | InternVideo unified masked modeling + contrastive pretraining | Video-Language Understanding |
| Q12 | Advanced | Video LLM architecture and temporal token challenges | Video-Language Understanding |
| Q13 | Basic | Temporal action detection task and mAP evaluation | Temporal Action Localization |
| Q14 | Advanced | BMN boundary-matching proposal generation | Temporal Action Localization |
| Q15 | Advanced | ActionFormer one-stage transformer-based detection | Temporal Action Localization |
| Q16 | Advanced | Vid2Seq dense video captioning via temporal tokens | Temporal Action Localization |
| Q17 | Basic | RAFT optical flow with all-pairs correlation volume | Video Self-Supervised Learning |
| Q18 | Basic | Tube masking for masked video modeling | Video Self-Supervised Learning |
| Q19 | Advanced | VideoMAE data efficiency and joint space-time encoding | Video Self-Supervised Learning |
| Q20 | Advanced | VideoMAE V2 dual masking for billion-parameter pretraining | Video Self-Supervised Learning |

## Resources

- Simonyan & Zisserman, [Two-Stream Convolutional Networks for Action Recognition in Videos](https://arxiv.org/abs/1406.2199) (2014)
- Tran et al., [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/abs/1412.0767) (2015)
- Carreira & Zisserman, [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) (2017)
- Goyal et al., [The Something Something Video Database for Learning and Evaluating Visual Common Sense](https://arxiv.org/abs/1706.04261) (2017)
- Wang et al., [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) (2018)
- Feichtenhofer et al., [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982) (2019)
- Lin et al., [BMN: Boundary-Matching Network for Temporal Action Proposal Generation](https://arxiv.org/abs/1907.09702) (2019)
- Teed & Deng, [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) (2020)
- Bertasius et al., [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) (2021)
- Arnab et al., [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) (2021)
- Fan et al., [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227) (2021)
- Li et al., [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526) (2022)
- Liu et al., [Video Swin Transformer](https://arxiv.org/abs/2106.13230) (2022)
- Luo et al., [CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval and Question Answering](https://arxiv.org/abs/2104.08860) (2022)
- Tong et al., [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) (2022)
- Wang et al., [InternVideo: General Video Foundation Models via Generative and Discriminative Learning](https://arxiv.org/abs/2212.03191) (2022)
- Zhang et al., [ActionFormer: Localizing Moments of Actions with Transformers](https://arxiv.org/abs/2202.07925) (2022)
- Wang et al., [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://arxiv.org/abs/2303.16727) (2023)
- Yang et al., [Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning](https://arxiv.org/abs/2302.14115) (2023)
- Zhang et al., [Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding](https://arxiv.org/abs/2306.02858) (2023)
