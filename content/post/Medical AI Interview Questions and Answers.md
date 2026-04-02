---
title: "Medical AI: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-02'
categories:
  - Interview
tags:
  - Deep Learning
  - Medical Imaging
  - Medical AI
toc: true
---

## Medical Imaging Fundamentals

### Q1 [Basic] Characterize the unique challenges of medical imaging compared to natural image analysis

**Q:** What properties of medical image data make it fundamentally more challenging to analyze than natural images?

**A:** Medical imaging differs from natural image analysis across several interrelated dimensions.

**Modality diversity and physics-based image formation**: Unlike RGB photographs, medical images arise from distinct physical acquisition processes—X-ray attenuation, MRI spin relaxation, ultrasound acoustic reflection, histological light microscopy—each producing images with different noise models, contrast mechanisms, and spatial properties. Features that transfer across natural images rarely transfer across modalities, requiring modality-specific architectures or careful pretraining strategies.

**Annotation scarcity and expertise requirements**: Labeling natural images (classification, bounding boxes) requires minimal domain knowledge; labeling medical images requires years of clinical training. A radiologist may spend 20–30 minutes contouring a tumor on a CT scan, and inter-annotator agreement for complex structures is often substantially below 1.0. This limits dataset scale—medical imaging datasets rarely exceed hundreds of thousands of labeled examples, compared to billions in natural image benchmarks.

**Class imbalance and rare pathology**: Clinically significant findings (small lung nodules, early-stage lesions) are rare relative to normal anatomy. A chest radiograph dataset may contain fewer than 1% positive cases for pneumothorax, requiring specialized loss functions (focal loss, Dice loss) and sampling strategies to avoid degenerate solutions.

**Distribution shift from acquisition heterogeneity**: MRI scanner manufacturers (Siemens, GE, Philips), field strengths (1.5 T vs. 3 T), and imaging protocols produce images that look visually distinct despite containing identical anatomical information. A model trained on one scanner can lose substantial performance when evaluated on another—a particularly dangerous failure mode in clinical deployment, where no labeled target-domain data is available for validation.

---

### Q2 [Basic] Explain why U-Net became the dominant architecture for medical image segmentation

**Q:** What architectural properties made U-Net so effective for medical image segmentation?

**A:** **U-Net** (Ronneberger et al., 2015) was designed explicitly for biomedical image segmentation under extreme label scarcity. Its success rests on two core architectural choices. First, the **encoder-decoder with skip connections**: the contracting path (encoder) progressively halves spatial resolution while doubling feature channels, capturing semantic context; the expanding path (decoder) upsamples back to input resolution. Critically, skip connections concatenate encoder feature maps directly to corresponding decoder layers, preserving fine-grained spatial information—cell boundaries, thin vessel walls—that would otherwise be lost during downsampling.

Second, **full-resolution output and overlap-tile inference**: U-Net produces segmentation maps at the same resolution as the input, enabling pixel-wise predictions without post-processing. For large images exceeding GPU memory, Ronneberger et al. (2015) introduced the overlap-tile strategy: overlapping patches are processed with mirrored padding at borders, enabling arbitrarily large inputs.

These properties addressed the core challenge of medical segmentation: learning from very few images (as few as 30 annotated images in the original publication) while localizing fine structures with high spatial precision. The architecture proved so broadly effective that it forms the basis of nearly all subsequent medical segmentation systems, including nnU-Net (Isensee et al., 2021), TransUNet (Chen et al., 2021), and SwinUNETR (Tang et al., 2022).

---

### Q3 [Advanced] Assess how self-supervised pretraining addresses label scarcity in medical imaging

**Q:** How do self-supervised learning approaches reduce dependence on expert annotations in medical imaging, and what forms have proven most impactful?

**A:** Self-supervised learning (SSL) reduces annotation dependence by pretraining on unlabeled medical images or naturally occurring paired data before fine-tuning on small labeled sets. The core insight is that medical data contains rich supervisory structure—anatomical consistency, cross-modal correspondences, paired reports—that can serve as a training signal without manual labels.

**Contrastive image-only methods** (SimCLR-style) apply data augmentations to each image and train a network to produce similar representations for augmented views of the same image while repelling representations from different images. Medical imaging requires careful augmentation design: intensity augmentations simulating scanner variability (random windowing, brightness shifts) are more effective than heavy geometric augmentations, which may alter diagnostically meaningful anatomy (e.g., flipping a chest X-ray laterally inverts left-right orientation, which is clinically significant).

**Paired image-text pretraining** exploits naturally occurring paired data in clinical practice: radiology reports accompany nearly every imaging study. ConVIRT (Zhang et al., 2020) demonstrated that contrastive learning on chest X-ray/report pairs significantly outperforms ImageNet pretraining for downstream classification, achieving competitive performance with only 1% of labeled data. The paired text provides semantic supervision that image-only contrastive methods lack—a report mentioning "bilateral infiltrates" teaches the model to associate global visual patterns with clinical concepts rather than just image-level similarities.

**Masked image modeling (MAE-style)** masks random patches and reconstructs them, forcing the encoder to learn local structural representations. Adapting this approach to 3D volumes (CT, MRI) requires volumetric positional embeddings and patch sizes spanning anatomically meaningful units. A key advantage over contrastive methods is that masked reconstruction does not require careful augmentation invariance design—the self-supervised signal is inherent in image structure.

The fundamental limitation of image-only SSL for medical imaging is that augmentation invariances must be domain-appropriate: invariances that help natural image models (color jitter, horizontal flip) can destroy medically relevant signals (laterality in chest X-rays, intensity units in CT Hounsfield values).

---

### Q4 [Advanced] Analyze nnU-Net's automated pipeline configuration for medical segmentation

**Q:** How does nnU-Net automatically configure an optimal segmentation pipeline, and why does this approach outperform manually designed architectures across diverse medical datasets?

**A:** **nnU-Net** (Isensee et al., 2021) is a self-configuring framework that automatically determines all architectural and training hyperparameters from a dataset's statistical properties, without manual tuning. Its central contribution is replacing manual architecture search with principled rule-based inference from a **dataset fingerprint**: properties computed prior to training, including image dimensionality (2D/3D), voxel spacing, image size, patch size requirements given GPU memory, class frequency distributions, and intensity statistics.

From the fingerprint, nnU-Net infers: (1) whether to train 2D, 3D full-resolution, or 3D cascade (low-to-high resolution) U-Net variants; (2) patch size and batch size under a fixed GPU memory budget; (3) normalization strategy—z-score normalization using foreground statistics for CT (which has a meaningful Hounsfield scale), no intensity normalization for MRI (which lacks an absolute scale); (4) data augmentation pipeline parameterized by expected spatial deformations; and (5) test-time post-processing based on connected component analysis of predicted masks.

The framework evaluates all inferred configurations via 5-fold cross-validation and selects the best-performing model or ensemble. Isensee et al. (2021) demonstrated that nnU-Net outperforms or matches all prior manually designed methods across 23 of 23 public datasets in the Medical Segmentation Decathlon—spanning brain tumors, cardiac structures, liver lesions, pancreas, colon cancer, and prostate—without any dataset-specific modification.

The key insight is that most historical architectural gains in medical segmentation came from dataset-specific tuning that most practitioners lack the time and expertise to reproduce. nnU-Net makes this tuning automatic and reproducible, serving as both a strong baseline and a deployment-ready system. Its design also reveals that architecture choice matters far less than proper normalization, patch size selection, and training regularization—a finding with broad implications for how medical segmentation research is conducted.

---

## Segmentation Architectures

### Q5 [Basic] Explain what motivated Transformer-based architectures for medical image segmentation

**Q:** What limitations of CNN-based segmentation architectures motivated the adoption of Transformers in medical imaging?

**A:** CNN-based segmentation architectures, including U-Net and its variants, process images through convolutional layers with receptive fields determined by kernel size and network depth. This **limited and implicitly local receptive field** creates two clinically important failure modes. First, global context is captured only through repeated pooling and downsampling, losing spatial precision. For structures whose boundaries depend on global anatomical configuration—e.g., the liver boundary shifts with the position of neighboring viscera—local convolutions provide insufficient long-range context. Second, structures extending across large spatial extents (elongated blood vessels, colon segments) require extremely deep networks with large effective receptive fields, which are expensive and prone to over-smoothing of fine boundaries.

Transformers address these limitations through **multi-head self-attention**, which directly computes pairwise attention weights between all spatial positions, establishing global dependencies in a single layer regardless of spatial distance. For medical image segmentation, this enables accurate delineation of elongated structures, disambiguation of organ boundaries in cluttered environments, and integration of contextual information across an entire anatomical region.

The primary trade-off is the quadratic complexity of self-attention with respect to sequence length: $O(n^2)$ where $n$ is the number of patch tokens. Medical volumes (e.g., $512 \times 512 \times 300$ CT) exacerbate this cost, motivating efficient attention variants such as shifted window attention (Swin Transformer) and hierarchical tokenization strategies.

---

### Q6 [Advanced] Examine TransUNet's hybrid CNN-Transformer design for segmentation

**Q:** How does TransUNet integrate convolutional and Transformer components, and what does this hybrid design achieve over purely CNN or purely Transformer models?

**A:** **TransUNet** (Chen et al., 2021) addresses the complementary strengths of CNNs and Transformers by placing them in series within a U-Net encoder-decoder framework. The encoder begins with a **ResNet-like CNN** that extracts feature maps at progressively lower spatial resolutions, producing strong low-level spatial features with high channel dimension. The final CNN feature map is tokenized—spatially flattened into a sequence of $n$ patch tokens—and fed into a stack of Vision Transformer (ViT) layers. The Transformer operates on this encoded sequence, building global pairwise dependencies across the entire feature map. The resulting sequence is reshaped back to a 2D feature map and passed to a CNN decoder with skip connections from the encoder (identical to U-Net), recovering full-resolution predictions.

This design captures both **local texture and edge features** (CNN) and **global structural context** (Transformer), which neither architecture achieves in isolation. Purely CNN-based models lack non-local context; purely Transformer-based models such as Swin-UNet (Cao et al., 2021) lack the low-level spatial precision provided by convolutional inductive biases, particularly for fine-grained structure delineation in small-data regimes.

Chen et al. (2021) demonstrated that TransUNet outperforms a strong U-Net baseline by 2.6% mean Dice on the Synapse multi-organ CT segmentation benchmark (8 abdominal organs). The key limitation is that the ViT component requires ImageNet or large medical corpus pretraining to converge reliably—training the Transformer blocks from scratch on small medical datasets is problematic because the Transformer lacks the convolutional inductive bias that enables CNNs to learn spatial hierarchies from limited data.

---

### Q7 [Advanced] Assess SAM's transferability to medical imaging and MedSAM's adaptation strategy

**Q:** How does the Segment Anything Model perform on medical images without adaptation, and what does MedSAM contribute beyond zero-shot SAM?

**A:** **Segment Anything Model (SAM)** (Kirillov et al., 2023) is a promptable segmentation foundation model trained on 1.1 billion masks from the SA-1B natural image dataset. Its architecture consists of a heavy image encoder (ViT-H, 307M parameters), a lightweight prompt encoder accepting points, bounding boxes, or masks, and a mask decoder using cross-attention between image embeddings and prompt tokens. SAM achieves strong zero-shot interactive segmentation on natural images.

Applied directly to medical images without adaptation, SAM shows substantial performance degradation. Three factors explain this. First, **domain gap**: SA-1B contains no CT, MRI, histological, or ultrasound images; the visual statistics of medical modalities—grayscale intensity distributions, Hounsfield units in CT, staining patterns in pathology—are entirely outside SAM's training distribution. Second, **granularity mismatch**: SAM segments objects at natural image granularity, while medical segmentation requires sub-millimeter boundary precision for structures like tumor margins and vessel walls. Third, **prompt domain dependence**: effective prompts for medical structures require anatomical knowledge unavailable from natural image interaction experience.

**MedSAM** (Ma et al., 2024) addresses these limitations by fine-tuning SAM on 1.5 million medical image-mask pairs spanning 10 imaging modalities (CT, MRI, X-ray, ultrasound, pathology, fundus photography, dermoscopy, endoscopy, mammography, and optical coherence tomography) and 30 cancer types. Ma et al. (2024) froze the patch embedding layer and fine-tuned the remaining image encoder and all decoder parameters, preserving low-level feature extraction while adapting higher-level representations to medical image statistics. MedSAM achieved competitive performance across all 10 modalities with bounding-box prompts. The persistent limitation is that MedSAM inherits SAM's interactive prompting requirement—it does not provide automatic (prompt-free) segmentation, which is necessary for routine clinical screening workflows.

---

### Q8 [Advanced] Explain how SwinUNETR extends Swin Transformers for 3D volumetric analysis

**Q:** How does SwinUNETR adapt the Swin Transformer to volumetric medical images, and what role does self-supervised pretraining play in its performance?

**A:** **SwinUNETR** (Tang et al., 2022) integrates the Swin Transformer into a U-Net encoder-decoder to handle 3D volumetric medical images (CT, MRI) with $O(10^7)$ voxels per volume. Standard ViT self-attention is intractable at this scale. SwinUNETR uses **shifted window attention** computed within non-overlapping 3D windows (e.g., $7 \times 7 \times 7$ voxels), with windows shifted between alternating layers to achieve cross-window information exchange. This reduces attention complexity from $O(n^2)$ global to $O(n \cdot w^3)$ local, where $w$ is the window size—enabling practical training on standard GPU hardware.

The encoder processes the volumetric input at four hierarchical resolutions via patch merging (analogous to 3D pooling), producing feature maps at 1/4, 1/8, 1/16, and 1/32 of input resolution. A CNN decoder with skip connections—structurally identical to U-Net—recovers full-resolution segmentation predictions.

Critically, Tang et al. (2022) showed that **self-supervised pretraining** on unlabeled medical volumes is essential for SwinUNETR to generalize from limited labeled data. The encoder is pretrained on 5,050 unlabeled CT volumes using four complementary proxy tasks: masked patch inpainting (a 3D analogue of MAE), rotation prediction, contrastive learning between two augmented views, and masked volume inpainting at coarser resolution. This produces encoder weights encoding 3D anatomical structure before any task-specific supervision. Fine-tuned SwinUNETR achieves state-of-the-art results on the Medical Segmentation Decathlon and BTCV multi-organ segmentation benchmarks (Tang et al., 2022), demonstrating that 3D self-supervised pretraining transfers effectively to multiple segmentation targets.

---

## Medical Foundation Models

### Q9 [Basic] Define what distinguishes a medical foundation model from fine-tuning a general model

**Q:** What distinguishes a medical foundation model from a general-purpose large model fine-tuned on medical data?

**A:** A **medical foundation model** is pretrained at scale on diverse medical data—radiology reports, clinical notes, pathology images, biomedical literature, ECG traces, genomic sequences—with the explicit design goal of broad medical capability transferable across downstream clinical tasks. The distinction from fine-tuning a general model on medical data is one of pretraining specificity, data scale, and emergent capability breadth.

General-purpose models (GPT-4, CLIP) are pretrained on internet data that incidentally includes some medical text and images, constituting a small fraction of the training distribution. Medical foundation models allocate pretraining compute specifically to medical knowledge: Med-PaLM 2 (Singhal et al., 2023) starts from a general language model but undergoes medical instruction tuning with physician-generated chain-of-thought reasoning, calibrating factual accuracy and clinical reasoning style simultaneously.

The key capabilities that emerge at medical foundation model scale and are absent in fine-tuned general models include: (1) **multi-task medical generalization**—a single model that can answer clinical questions, generate radiology reports, and explain differential diagnoses; (2) **clinical concept grounding**—deep understanding of medical terminology, drug-disease interactions, anatomical relationships, and diagnostic criteria; (3) **uncertainty-aware reasoning**—medical foundation models can express calibrated uncertainty ("this finding is more consistent with X than Y given the clinical context") rather than overconfident point predictions.

---

### Q10 [Advanced] Examine how Med-PaLM achieves a passing score on USMLE-style benchmarks

**Q:** What training strategy and evaluation methodology does Med-PaLM employ, and what does its performance on medical QA reveal about LLM clinical reasoning?

**A:** **Med-PaLM** (Singhal et al., 2022) is instruction-tuned from the 540B-parameter PaLM language model using **MultiMedQA**, a benchmark curated from seven medical QA datasets: MedQA (USMLE-style), MedMCQA, PubMedQA, LiveQA, MedicationQA, MMLU clinical topics, and HealthSearchQA. The instruction tuning employs both standard few-shot prompting and **chain-of-thought (CoT)** prompting, where the model generates step-by-step clinical reasoning before producing a final answer—mimicking the differential diagnosis process.

Singhal et al. (2022) achieved 67.6% accuracy on MedQA, surpassing the approximate 60% threshold considered equivalent to a passing USMLE score—the first time any AI system had demonstrated this capability on a standardized medical licensing exam. Alongside automated metrics, the authors conducted a **physician evaluation** using a detailed rubric assessing factuality, evidence-based reasoning, possible harm, and completeness; Med-PaLM responses were rated favorably against a web-search baseline but below the standard of expert physicians.

A key methodological finding is that **prompt engineering substantially drives performance**: carefully crafted few-shot CoT examples improved accuracy by approximately 9 percentage points over standard prompting on MedQA. This dependency on prompt design is a significant limitation: performance is not robust to prompt variation, which is problematic for deployment contexts where non-expert users formulate queries without optimized prompting.

---

### Q11 [Advanced] Analyze the advances Med-PaLM 2 introduces over the original Med-PaLM

**Q:** What architectural and methodological improvements allow Med-PaLM 2 to substantially exceed Med-PaLM on medical benchmarks, and what gaps remain?

**A:** **Med-PaLM 2** (Singhal et al., 2023) builds on the PaLM 2 base model and introduces three methodological advances that collectively increase MedQA accuracy from 67.6% to 86.5%—surpassing the average score of practicing physicians (~76%) on the same benchmark.

First, **stronger base model**: PaLM 2 introduces improved pretraining data mixture, architectural improvements, and training recipe refinements over PaLM. The substantially stronger general reasoning capability of PaLM 2 translates directly to medical reasoning performance, reflecting that medical problem-solving shares deep structure with general multi-step reasoning.

Second, **ensemble refinement via self-critique**: Med-PaLM 2 employs an **ensemble refinement** procedure where multiple independent model responses to the same question are generated, and a separate refinement pass synthesizes these responses into a final answer. This is analogous to physician consensus rounds: disagreement among independent responses signals uncertainty, and the refinement step identifies and corrects errors. Singhal et al. (2023) show that ensemble refinement substantially outperforms simple majority voting across independently sampled responses.

Third, **adversarial evaluation and detailed physician annotation**: the evaluation set includes adversarially constructed cases targeting factual errors and unsafe recommendations, annotated across 9 axes including factual accuracy, evidence basis, potential for patient harm, and appropriateness for clinical decision support.

A persistent gap identified by Singhal et al. (2023) is long-form clinical reasoning (case summaries, multi-step diagnostic chains) where factual errors compound and hallucinated citations are hard to detect without domain expertise—particularly concerning for clinical deployment.

---

### Q12 [Advanced] Evaluate CheXagent as a unified foundation model for chest X-ray analysis

**Q:** How does CheXagent unify chest X-ray analysis tasks under a single foundation model, and what benchmarks validate its performance?

**A:** **CheXagent** (Chen et al., 2024) is a vision-language foundation model designed to unify classification, report generation, visual question answering, and report summarization for chest X-ray (CXR) analysis under a single instructable model. This contrasts with the prior paradigm of training task-specific models for each CXR analysis task.

CheXagent's architecture comprises a CXR-specific vision encoder (initialized from a visual foundation model and fine-tuned on radiology images), a BioMedBERT text encoder for aligning radiology-specific terminology, and an instruction-following language model backbone. The model is trained on **CheXinstruct**, a dataset of 6 million instruction-response pairs constructed from existing radiology datasets—including CheXpert (Irvin et al., 2019) with its 224,316 chest radiographs from 65,240 patients—augmented with GPT-4-generated instruction phrasings covering diverse clinical query types.

The unified training enables **task-generalization at inference time**: a single model can generate a full radiology report, answer a binary finding query ("Is there pleural effusion?"), or provide a differential diagnosis, with response format conditioned on instruction phrasing. Chen et al. (2024) evaluated CheXagent on 18 downstream tasks using **CheXbench**, a comprehensive evaluation suite spanning report generation quality (RadGraph F1, CheXbert label alignment), clinical VQA accuracy, and cross-finding consistency. CheXagent outperforms prior single-task and multi-task CXR models on the majority of tasks (Chen et al., 2024); report generation for rare findings remains a limitation due to the long-tail distribution of pathologies in CheXpert.

---

### Q13 [Advanced] Analyze LLaVA-Med's data strategy for biomedical visual question answering

**Q:** How does LLaVA-Med adapt a general multimodal LLM for biomedical VQA, and what does its data construction strategy contribute?

**A:** **LLaVA-Med** (Li et al., 2023) adapts the LLaVA (Large Language-and-Vision Assistant) framework for biomedical images by constructing a high-quality instruction-tuning dataset from PMC (PubMed Central) figure-caption pairs, with GPT-4 serving as an instruction data generator.

The data pipeline works as follows: PMC provides 1.5M+ biomedical figure-caption pairs from open-access scientific publications. Each caption is passed to GPT-4, which generates diverse instruction-response pairs for each image type: classification questions ("What cell type is depicted in this microscopy image?"), comparison questions ("How do the morphological features in this MRI differ from a normal presentation?"), and mechanistic reasoning questions ("What pathological process is evidenced by the histological findings?"). This creates **grounded conversational data** without requiring specialist annotation—the cost is dominated by GPT-4 API calls rather than clinical expertise.

Li et al. (2023) train LLaVA-Med in two stages: first, the vision encoder (CLIP ViT-L) is aligned with the Vicuna LLM backbone on biomedical figure-caption pairs via next-token prediction; second, the model is instruction fine-tuned on the GPT-4-generated instruction data with visual inputs. Full training completes in approximately one day on 8 A100 GPUs.

On PathVQA (pathology), VQA-RAD (radiology), and SLAKE (medical knowledge), LLaVA-Med substantially outperforms general VLMs on closed-ended questions, demonstrating that domain-specific instruction tuning transfers even when training images differ from test modalities. The central limitation is **hallucination for rare conditions**: the model generates plausible but factually incorrect descriptions for conditions underrepresented in PMC, a failure mode that is particularly risky in clinical contexts where rare conditions disproportionately benefit from AI-assisted review.

---

## Computational Pathology

### Q14 [Basic] Characterize the unique challenges of whole-slide image analysis

**Q:** What properties of whole-slide images make computational pathology analysis fundamentally different from standard medical image analysis?

**A:** **Whole-slide images (WSIs)** are digitized tissue specimens scanned at 20× or 40× magnification, producing images of 100,000×100,000 pixels (10–20 GB per slide). This scale distinguishes WSI analysis from all other medical imaging modalities and creates three interrelated challenges.

**Memory infeasibility for end-to-end learning**: No GPU can hold a full WSI in memory. Models must process the slide as a collection of small patches (typically 256×256 or 512×512 pixels), aggregating patch-level representations into a slide-level prediction. The aggregation strategy—how to combine thousands of patch features into a single diagnostic output—is a central research question in computational pathology.

**Weak supervision**: Pathology labels (cancer subtype, grade, molecular marker status) are annotated at the slide level during routine diagnostic reporting. Pixel-level or region-level annotations are prohibitively expensive for large cohorts. Models must therefore train with **multiple instance learning (MIL)** frameworks where the slide is a "bag" of patch instances and only the bag-level label is available—without knowing which specific patches contain the diagnostic evidence.

**Multi-scale diagnostic reasoning**: Pathological diagnosis integrates information across magnification levels: at 4×, overall tissue architecture is visible; at 20×, individual cells and gland formations are resolved; at 40×, nuclear morphology and mitotic figures are identifiable. Effective models must integrate features across scales—a fundamental challenge for architectures designed for single-scale inputs with fixed patch sizes.

---

### Q15 [Advanced] Examine CLAM's attention-based weakly supervised classification of whole-slide images

**Q:** How does CLAM perform weakly supervised WSI classification, and what does its clustering-constrained attention mechanism contribute beyond standard MIL?

**A:** **CLAM** (Clustering-constrained Attention Multiple Instance Learning, Lu et al., 2021) is a weakly supervised framework extending attention-based MIL with two domain-specific innovations.

In standard attention-based MIL, a slide is represented as a bag of patch embeddings $\{h_1, \ldots, h_n\}$, and slide-level classification uses an attention-pooled representation $z = \sum_k a_k h_k$, where attention weights $a_k$ derive from a learnable gated attention mechanism. The slide-level cross-entropy loss provides no direct signal about which patches are diagnostically relevant—attention weights are learned purely as a by-product of improving slide-level predictions.

CLAM's first innovation is **instance-level clustering supervision**: after computing attention weights, CLAM identifies the top-$K$ and bottom-$K$ attended patches (high attention presumed to encode positive evidence, low attention to encode negative evidence) and applies a contrastive clustering loss pushing high-attention patches toward the slide's class prototype and low-attention patches toward an out-of-class prototype in feature space. This creates pseudo-supervised patch-level signal from bag-level labels, improving the discriminativeness of learned patch representations without requiring patch-level annotations.

CLAM's second innovation is a **multi-branch architecture** maintaining separate attention branches for each pathology class, enabling class-specific attention patterns rather than sharing a single attention mechanism across all classes. Lu et al. (2021) demonstrated that CLAM achieves competitive or superior performance to fully supervised methods on lung cancer subtype classification (TCGA-LUAD vs. TCGA-LUSC) and renal cell carcinoma subtype classification using only 10–20% of the labeled slides required by fully supervised alternatives.

---

### Q16 [Advanced] Analyze HIPT's hierarchical self-supervised learning for gigapixel pathology images

**Q:** How does HIPT decompose the gigapixel WSI representation problem, and what does hierarchical self-supervised pretraining achieve over flat patch-level pretraining?

**A:** **HIPT** (Hierarchical Image Pyramid Transformer, Chen et al., 2022) addresses WSI representation learning through a three-level hierarchical decomposition aligned with the natural structural organization of tissue images.

At the lowest level, $256 \times 256$ pixel patches are processed by a ViT-S/16 pretrained with **DINO** (self-distillation with no labels) on 10,678 WSIs from TCGA, producing 384-dimensional patch tokens that encode local cell and gland morphology. At the second level, $4096 \times 4096$ pixel regions (each containing $16 \times 16 = 256$ patches) are represented by a second ViT-S/16 that takes pre-extracted patch tokens as input and applies self-attention across them—again pretrained with DINO. This second-level encoder learns tissue architecture at the region scale. At the slide level, an MIL aggregation (mean or attention pooling) combines all region tokens for the full WSI.

The hierarchical design is motivated by the observation that pathological features are scale-specific: individual cell morphology is relevant at $256 \times 256$ resolution, gland and stromal arrangement at $4096 \times 4096$, and tumor heterogeneity and spatial patterns at the whole-slide level. Pretraining separate encoders at each scale with DINO allows each encoder to develop scale-appropriate representations independently.

Chen et al. (2022) demonstrated that HIPT achieves strong performance on 6 cancer survival prediction tasks and 4 subtype classification tasks across TCGA cancer types. Qualitatively, the region-level ViT's attention maps identify diagnostically relevant tissue regions—tumor nests, tertiary lymphoid structures, necrotic regions—without any supervision, providing interpretable evidence that the learned representations capture tissue-level diagnostic semantics.

---

### Q17 [Advanced] Evaluate design considerations for pathology-specific foundation models

**Q:** What data, architectural, and pretraining choices distinguish an effective computational pathology foundation model from a general vision model fine-tuned on pathology data?

**A:** Computational pathology foundation models face domain-specific challenges that fine-tuning a general vision model on pathology images does not adequately address.

**Staining variability as a systematic distribution shift**: Hematoxylin and eosin (H&E) staining varies significantly across labs, scanners, and tissue processing protocols. A foundation model pretraining on diverse multi-institution WSIs is inherently exposed to this variation, learning staining-invariant tissue representations. A general model fine-tuned on single-institution data instead embeds institution-specific staining as a spurious feature, degrading cross-site generalization.

**Patch-level pretraining vs. slide-level clinical tasks**: Computational pathology foundation models are pretrained at the patch level due to memory constraints but deployed for slide-level clinical predictions. The pretraining objective must produce patch features discriminative for tissue phenotype—not generic texture—to support effective downstream aggregation. DINO-based self-distillation (as used in HIPT, Chen et al., 2022) produces features with strong spatial attention properties aligned with tissue structure, outperforming ImageNet-pretrained ViT features for downstream MIL aggregation.

**Scale of pretraining data and cancer type coverage**: Effective foundation models require pretraining across diverse cancer types, tissue types, and institutions. TCGA provides approximately 30,000 diagnostic WSIs across 33 cancer types and is the primary public pretraining corpus; models supplemented with institutional data consistently outperform TCGA-only pretrained baselines on out-of-distribution cancer types.

**Magnification encoding**: Pathological diagnosis is magnification-dependent—the same tissue region has different diagnostic relevance at 4× vs. 40×. Foundation models that do not encode magnification level (via explicit positional embedding metadata) cannot distinguish diagnostically relevant from irrelevant features at each scale, limiting their utility as universal pathology encoders.

---

## Evaluation and Clinical Translation

### Q18 [Basic] Identify the key metrics for evaluating medical image segmentation

**Q:** What quantitative metrics are used to evaluate segmentation models in medical imaging, and why is no single metric sufficient?

**A:** Medical image segmentation is evaluated with two complementary classes of metrics targeting different aspects of quality.

**Overlap-based metrics** measure volumetric agreement between predicted and ground-truth masks. The **Dice similarity coefficient (DSC)** is the most widely used: $\text{DSC} = \frac{2|P \cap G|}{|P| + |G|}$, where $P$ and $G$ are predicted and ground-truth masks. DSC corresponds directly to the F1 score and is robust to class imbalance (unlike pixel accuracy). However, DSC is insensitive to boundary precision: a prediction that captures the bulk of a tumor with a slightly eroded boundary may achieve $\text{DSC} > 0.95$ while failing to delineate the tumor margin accurately—which is critical for radiotherapy planning where millimeter-level errors directly affect dose coverage of the tumor and sparing of adjacent organs.

**Surface distance metrics** measure boundary quality. The **95th percentile Hausdorff distance (HD95)** computes the 95th percentile of symmetric surface distances between prediction and ground-truth contours, discarding the worst 5% of surface points to reduce sensitivity to isolated outliers from annotation inconsistencies. **Normalized Surface Distance (NSD)** measures the fraction of predicted surface points within a clinically specified tolerance $\tau$ (typically 1–2 mm) of the ground-truth surface, directly encoding the clinical boundary precision requirement.

In practice, medical segmentation papers report DSC as the primary metric (interpretable as combined precision-recall on volumetric overlap) and HD95 as secondary (capturing boundary quality). Reporting DSC alone can be particularly misleading for small or elongated structures—pancreatic ducts, spinal cord, retinal vessels—where small volumetric errors can correspond to catastrophically large surface distance violations that affect clinical utility.

---

### Q19 [Advanced] Characterize how distribution shift undermines medical AI performance in deployment

**Q:** How do the forms of distribution shift encountered in clinical deployment differ from those in standard computer vision benchmarks, and what strategies mitigate their impact?

**A:** Medical AI models face systematic forms of distribution shift that are more consequential and harder to detect than the natural image distribution shifts studied in standard robustness benchmarks.

**Scanner and protocol shift** is the dominant challenge in deployment. Models trained on Siemens FLASH CT with specific reconstruction kernels may fail when deployed on GE Revolution CT with different noise characteristics and slice thicknesses. Lung nodule detection models trained at 1.25 mm slice thickness systematically miss nodules at 5 mm slice thickness due to partial volume averaging. Unlike natural image shift (lighting, viewpoint), scanner shift is invisible to human observers—images look equivalent to a radiologist—making it impossible to detect shift from visual inspection alone and requiring statistical performance monitoring.

**Site and population shift** arises from demographic differences between training and deployment populations. Diabetic retinopathy screening models trained predominantly on specific ethnic groups show lower specificity on underrepresented groups due to differences in fundus pigmentation and baseline disease prevalence. This form of shift is well-documented in retrospective audits of deployed systems and drove FDA guidance requiring algorithmic bias assessment across demographic subgroups.

**Temporal shift** (concept drift) occurs as disease epidemiology, treatment practices, and imaging protocols evolve. A pneumonia detector trained before the COVID-19 pandemic shows performance degradation on COVID-19 pneumonia despite visual overlap with other viral presentations—the disease prevalence shift alone changes the operating point even if visual features are similar.

Mitigation strategies include: (1) **federated learning**, training models across multiple institutions without centralizing patient data, exposing models to multi-site variation during training; (2) **domain adaptation** using unlabeled target-domain images to align feature distributions via adversarial training or instance re-weighting; (3) **model cards and data sheets** documenting training data demographics so deployment teams can prospectively assess shift risk; (4) **continuous prospective monitoring** with rolling performance dashboards that detect when model confidence and measured accuracy diverge.

---

### Q20 [Advanced] Assess the principal barriers to regulatory approval and clinical deployment of medical AI

**Q:** What requirements and failure modes separate a high-performing research prototype from a clinically deployed medical AI system?

**A:** Clinical translation of medical AI fails at multiple stages between peer-reviewed publication and patient use, each imposing requirements that retrospective model development does not address.

**Regulatory pathway and adaptive model governance**: In the US, AI-based medical devices are regulated by the FDA as Software as a Medical Device (SaMD). Most imaging AI systems seek Class II clearance via the 510(k) pathway, requiring demonstration of substantial equivalence to a predicate device. The FDA's 2021 AI/ML-based SaMD action plan introduced **predetermined change control plans**: manufacturers must prospectively specify how model updates will be validated before deployment, since models that retrain on new data can shift performance characteristics after initial clearance. In the EU, the Medical Device Regulation (MDR) imposes analogous post-market clinical follow-up requirements.

**Prospective vs. retrospective performance gap**: Retrospective studies on curated test sets consistently overestimate real-world performance. A radiology AI model achieving 95% sensitivity in a retrospective cohort may show 80% sensitivity in prospective deployment due to case-mix differences, alert fatigue from workflow integration, and label quality differences between research annotations and clinical ground truth. Randomized controlled trials evaluating AI-assisted vs. unassisted clinical workflows are the gold standard for demonstrating clinical benefit but require multi-year timelines and substantial infrastructure.

**Interpretability and liability**: Clinicians bear legal liability for diagnostic errors. Adopting an AI recommendation without understanding its basis creates accountability gaps that are both medically and legally untenable. Saliency maps and attention-based explanations show where the model attends, not why that region is associated with the output—insufficient for clinical acceptance. Trust calibration requires that the model accurately represent its own uncertainty, routing low-confidence cases to human review rather than forcing a prediction.

**Annotation and validation cost**: Validating a model across the full deployment population requires large, representative, expertly labeled validation datasets. For rare conditions, such datasets may not exist outside tertiary care centers. This creates a circular dependency: deployment at the scale needed to accumulate rare-condition cases is precluded by the validation requirements that must be satisfied before deployment.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Medical imaging challenges | Medical Imaging Fundamentals |
| Q2 | Basic | U-Net architectural significance | Medical Imaging Fundamentals |
| Q3 | Advanced | Self-supervised pretraining for label scarcity | Medical Imaging Fundamentals |
| Q4 | Advanced | nnU-Net automated configuration | Medical Imaging Fundamentals |
| Q5 | Basic | Motivation for Transformer-based segmentation | Segmentation Architectures |
| Q6 | Advanced | TransUNet hybrid CNN-Transformer design | Segmentation Architectures |
| Q7 | Advanced | SAM transferability and MedSAM adaptation | Segmentation Architectures |
| Q8 | Advanced | SwinUNETR for 3D volumetric analysis | Segmentation Architectures |
| Q9 | Basic | Medical foundation model definition | Medical Foundation Models |
| Q10 | Advanced | Med-PaLM on USMLE-style benchmarks | Medical Foundation Models |
| Q11 | Advanced | Med-PaLM 2 improvements | Medical Foundation Models |
| Q12 | Advanced | CheXagent for chest X-ray analysis | Medical Foundation Models |
| Q13 | Advanced | LLaVA-Med for biomedical VQA | Medical Foundation Models |
| Q14 | Basic | Whole-slide image analysis challenges | Computational Pathology |
| Q15 | Advanced | CLAM weakly supervised WSI classification | Computational Pathology |
| Q16 | Advanced | HIPT hierarchical self-supervised learning | Computational Pathology |
| Q17 | Advanced | Pathology foundation model design | Computational Pathology |
| Q18 | Basic | Segmentation evaluation metrics | Evaluation and Clinical Translation |
| Q19 | Advanced | Distribution shift in medical AI deployment | Evaluation and Clinical Translation |
| Q20 | Advanced | Clinical translation barriers | Evaluation and Clinical Translation |

## Resources

- Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) (2015)
- Zhang et al., [Contrastive Learning of Medical Visual Representations from Paired Images and Text](https://arxiv.org/abs/2010.00747) (2020)
- Cao et al., [Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation](https://arxiv.org/abs/2105.05537) (2021)
- Chen et al., [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306) (2021)
- Isensee et al., [nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation](https://arxiv.org/abs/1809.10486) (2021)
- Lu et al., [Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images](https://arxiv.org/abs/2004.09666) (2021)
- Chen et al., [Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning](https://arxiv.org/abs/2206.02647) (2022)
- Tang et al., [Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis](https://arxiv.org/abs/2201.01266) (2022)
- Irvin et al., [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031) (2019)
- Singhal et al., [Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138) (2022)
- Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643) (2023)
- Li et al., [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890) (2023)
- Singhal et al., [Towards Expert-Level Medical Question Answering with Large Language Models](https://arxiv.org/abs/2305.09617) (2023)
- Chen et al., [CheXagent: Towards a Foundation Model for Chest X-Ray Analysis](https://arxiv.org/abs/2401.12208) (2024)
- Ma et al., [Segment Anything in Medical Images](https://arxiv.org/abs/2304.12306) (2024)
