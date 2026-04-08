---
title: "Vision-Language Models: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-08'
categories:
  - Interview
tags:
  - Deep Learning
  - Vision-Language Models
  - Multimodal Learning
toc: true
---

## Contrastive Vision-Language Pretraining

### Q1 [Basic] Explain CLIP's contrastive training objective

**Q:** What is the core training objective of CLIP, and how does it enable image representations that generalize across tasks?

**A:** **CLIP** (Radford et al., 2021) trains an image encoder and a text encoder jointly by maximizing the cosine similarity between matching image-text pairs while minimizing it for non-matching pairs. Given a batch of $N$ image-text pairs, the symmetric InfoNCE loss treats the $N$ same-index pairs as positives and the remaining $N^2 - N$ cross-pairs as negatives:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\!\left[\log\frac{\exp(\mathrm{sim}(v_i, t_i)/\tau)}{\sum_{j}\exp(\mathrm{sim}(v_i, t_j)/\tau)} + \log\frac{\exp(\mathrm{sim}(t_i, v_i)/\tau)}{\sum_{j}\exp(\mathrm{sim}(t_i, v_j)/\tau)}\right]$$

where $\mathrm{sim}(\cdot,\cdot)$ is cosine similarity and $\tau$ is a learnable temperature. This objective pushes both encoders toward a shared embedding space where semantically related image-text pairs cluster together.

CLIP was trained on 400M image-text pairs (WebImageText) using either a ResNet or Vision Transformer as the image encoder and a Transformer as the text encoder. Unlike prior work relying on curated datasets, CLIP leverages natural language supervision implicit in web-crawled image-caption pairs. The result is a visual encoder that learns semantically rich features aligned with language, which transfer across classification, retrieval, and generative tasks without task-specific fine-tuning. CLIP ViT-L/14 achieves 76.2% zero-shot top-1 accuracy on ImageNet (Radford et al., 2021).

---

### Q2 [Basic] Describe CLIP's zero-shot transfer mechanism

**Q:** How does CLIP convert a downstream classification task into a similarity matching problem at inference, and what design choices influence zero-shot accuracy?

**A:** CLIP's zero-shot classification works by encoding all class names using the text encoder and selecting the class whose text embedding is most similar to the image embedding. For $C$ classes:

$$p(y = c \mid x) \propto \exp\!\left(\mathrm{sim}(f_v(x),\, f_t(\text{``a photo of a \{class\}_c''}))/\tau\right)$$

No gradient updates occur at inference — the pre-trained similarity function is applied directly.

**Prompt engineering** substantially affects zero-shot accuracy. Using "a photo of a {class}" outperforms bare class names, because web text rarely contains isolated labels. Ensembling over 80 prompt templates (e.g., "a photo of many {class}", "a photo of the large {class}") improves ImageNet zero-shot accuracy by up to 3.5 percentage points relative to a single template (Radford et al., 2021).

**Distribution shift robustness** is another key property: because no labeled data from the downstream dataset is used, CLIP does not overfit to dataset-specific spurious correlations. On distribution-shifted ImageNet variants (ImageNet-V2, -Sketch, -A, -R), CLIP maintains substantially higher accuracy than fine-tuned ResNets, suggesting that linguistic supervision encourages learning semantically meaningful features rather than dataset-specific shortcuts.

---

### Q3 [Basic] Explain how ALIGN scales to noisy web data

**Q:** What distinguishes ALIGN's pretraining approach from CLIP, and what does its result reveal about the role of data scale versus curation?

**A:** **ALIGN** (Jia et al., 2021) scales contrastive vision-language pretraining to 1.8 billion noisy image-text pairs from the internet with minimal curation — only simple rule-based filtering to remove very short alt-texts, duplicates, and explicit content. Unlike CLIP, which applied substantial filtering to produce its 400M WebImageText dataset, ALIGN relies on scale to compensate for noise.

ALIGN uses EfficientNet-L2 as the image encoder and BERT as the text encoder, trained with a normalized softmax loss equivalent to InfoNCE. Despite training on noisier data, ALIGN achieves strong zero-shot performance on ImageNet and competitive cross-modal retrieval performance on MSCOCO and Flickr30k (Jia et al., 2021), matching or exceeding CLIP at comparable scale.

The key finding is that the bottleneck in vision-language pretraining is data scale rather than data quality at the scale studied. Web-crawled data, despite its noise, provides an essentially unlimited supply of image-text co-occurrences encoding the natural diversity of visual concepts. This shifted subsequent work toward harvesting very large datasets with lightweight filtering rather than expensive manual curation, influencing the data strategy of nearly all large multimodal models that followed.

---

### Q4 [Advanced] Analyze SigLIP's sigmoid loss and its batch-size implications

**Q:** How does SigLIP's sigmoid loss differ from the softmax-based InfoNCE used in CLIP, and what practical advantages does this decoupling bring?

**A:** The InfoNCE loss in CLIP applies a softmax over all $N$ pairs in a batch for each anchor, normalizing each positive pair against all $N-1$ negatives:

$$\mathcal{L}_\text{InfoNCE} = -\frac{1}{N}\sum_i \log \frac{\exp(\mathrm{sim}(v_i, t_i)/\tau)}{\sum_j \exp(\mathrm{sim}(v_i, t_j)/\tau)}$$

This global normalization couples all examples within a batch: the gradient for one pair depends on the similarities of all other pairs. Consequently, InfoNCE is highly sensitive to batch size — larger batches provide more negatives and a sharper contrastive signal, which is why CLIP required extremely large batch sizes (up to 32,768) to train effectively.

**SigLIP** (Zhai et al., 2023) replaces this with an independent sigmoid binary cross-entropy applied to each pair:

$$\mathcal{L}_\text{SigLIP} = -\frac{1}{N^2}\sum_{i,j} \log \sigma\!\left(z_{ij} \cdot y_{ij} \cdot t - b\right)$$

where $z_{ij} = \mathrm{sim}(v_i, t_j)$, $y_{ij} = +1$ for matching pairs and $-1$ for non-matching, $t$ is a learnable temperature, and $b$ is a learnable bias. Each pair is treated independently — no normalization over the batch is required.

The practical advantages are twofold. First, SigLIP decouples batch size from normalization: because there is no softmax denominator requiring all pairs to be compared simultaneously, the method is less batch-size sensitive and can work with smaller batches without performance collapse. Second, positive pairs from multiple sources (captions from different datasets) can be incorporated more naturally since they do not affect each other's normalization. SigLIP with ViT-So400M achieves 83.1% zero-shot ImageNet top-1 (Zhai et al., 2023), outperforming CLIP at comparable model scale. Its representations have since become a widely used visual backbone for downstream multimodal LLMs including LLaVA-1.5 and InternVL.

---

## Architecture Design for Multimodal Models

### Q5 [Basic] Describe Flamingo's frozen-model architecture

**Q:** What components does Flamingo introduce to bridge a pre-trained vision encoder and a large language model without fine-tuning either?

**A:** **Flamingo** (Alayrac et al., 2022) keeps both the vision encoder (a pre-trained NFNet) and the language model (Chinchilla 70B) frozen and inserts two learnable components between them.

The first is the **Perceiver Resampler**: a cross-attention module that maps a variable number of vision tokens — from any number of input images or video frames — to a fixed set of 64 visual tokens. This provides a compact, fixed-size visual representation regardless of input resolution or number of frames, which the LLM can efficiently process at a predictable cost.

The second is a set of **cross-attention layers** interleaved with the frozen LLM's self-attention layers. At each interleaved position, the LLM's hidden states cross-attend to the 64 visual tokens, injecting visual information into the language generation process. These layers are gated with a learned $\tanh$ coefficient initialized to zero, ensuring the visual contribution starts neutral and is gradually integrated during training.

The frozen approach preserves the language model's existing capabilities while training only the cross-attention layers and Perceiver Resampler — roughly 10B out of 80B total parameters. Flamingo 80B achieves 82.0% on VQAv2 with 32 in-context examples, surpassing fine-tuned models of the era without updating any task-specific weights (Alayrac et al., 2022). The architecture also inherits the LLM's few-shot learning ability: providing example image-question-answer triples in the context prompt is sufficient to steer the model without gradient updates.

---

### Q6 [Basic] Explain the Q-Former in BLIP-2

**Q:** What problem does Q-Former solve in BLIP-2, and how does it act as a bridge between frozen vision and language models?

**A:** **BLIP-2** (Li et al., 2023a) addresses the challenge of connecting a high-dimensional frozen image encoder (e.g., ViT-G with 1.4B parameters producing hundreds of tokens per image) to a frozen LLM without fine-tuning either. Directly feeding all vision tokens into the LLM is computationally prohibitive and requires the LLM to interpret raw visual features it was never exposed to during language pretraining.

The **Q-Former** (Querying Transformer) is a 188M-parameter module containing 32 learnable query vectors. These queries interact with image features via cross-attention and with each other via self-attention. The output is always 32 query vectors regardless of input image size, providing a compact visual representation that is then projected to the LLM's input dimension via a linear layer.

BLIP-2 trains Q-Former in two stages. In the first stage, with the image encoder frozen, Q-Former is trained jointly on image-language matching, image-text contrastive learning, and image-grounded text generation — three objectives jointly applied via separate attention masks that control which query tokens can attend to text. In the second stage, the Q-Former output is connected to a frozen LLM and trained to convert visual features into soft visual prompts the LLM generates from.

The Q-Former acts as an information bottleneck that extracts only the visual information most relevant to language, filtering out low-level image statistics that would confuse the LLM. BLIP-2 with FlanT5-XXL achieves 65.0% on VQAv2 (Li et al., 2023a) without any downstream fine-tuning of the LLM.

---

### Q7 [Advanced] Analyze LLaVA's minimal architecture and its data-driven insight

**Q:** What design choices distinguish LLaVA from Flamingo or BLIP-2, and what do its results reveal about the relative importance of architecture versus data?

**A:** **LLaVA** (Liu et al., 2023a) adopts a deliberately minimal architecture: a CLIP ViT-L/14 image encoder, a single linear projection layer, and a Vicuna language model. There is no Perceiver Resampler, no Q-Former, and no cross-attention layers inserted into the LLM. The image encoder outputs a fixed sequence of visual tokens that are projected by the linear layer and prepended to the text token sequence; the LLM then processes visual and text tokens together with its standard causal self-attention.

The critical innovation is in the **training data**. Rather than training on large-scale image-text pairs from the web, LLaVA uses GPT-4 to generate a small but high-quality instruction-following dataset of 158K samples. GPT-4 receives image captions and bounding box descriptions (not the image itself) and generates multi-turn conversations, reasoning chains, and detailed descriptions. This leverages GPT-4's reasoning capability to produce richer supervision signal than image-caption pairs alone.

Training proceeds in two stages: a pretraining stage that trains only the linear projection on image-text alignment data (LLM frozen), followed by a visual instruction tuning stage that fine-tunes both the projection and the LLM. LLaVA achieves 90.92% on ScienceQA, surpassing GPT-4 (83.99%) despite a far simpler architecture than BLIP-2 or Flamingo (Liu et al., 2023a).

The implication is that CLIP visual features contain sufficient information for many multimodal tasks, and the real bottleneck is the quality of instruction-following data, not the sophistication of the vision-language bridge. This finding motivated a generation of simplified architectures that prioritize data quality.

---

### Q8 [Advanced] Compare cross-attention fusion and projection-based fusion

**Q:** How do cross-attention fusion and projection-based fusion differ in information routing, computational cost, and architectural flexibility?

**A:** **Cross-attention fusion** (Flamingo; Alayrac et al., 2022) inserts cross-attention layers at multiple depths within the language model. At each cross-attention layer, the LLM's hidden states query the visual tokens, dynamically retrieving visual information at multiple abstraction levels. Visual features thus condition the language model's representations throughout its full depth, enabling fine-grained visual integration at every processing stage.

**Projection-based fusion** (LLaVA, Liu et al., 2023a; MLP variant in LLaVA-1.5, Liu et al., 2023b) maps all visual tokens through a single projection and prepends them to the text sequence. The LLM processes visual and text tokens together through its standard self-attention, treating visual tokens as a fixed prefix. Visual features are available at every layer through self-attention, but their representation is fixed after projection — the LLM cannot query visual features dynamically as it can through cross-attention.

Three trade-offs are key. **Computational cost**: cross-attention at every LLM layer adds $O(L_\text{vis} \cdot L_\text{text})$ attention operations per layer. Projection fusion incurs no extra operations per layer, but the concatenated sequence length grows with the number of visual tokens, increasing quadratic self-attention cost.

**Frozen LLM compatibility**: cross-attention enables keeping the LLM strictly frozen (as in Flamingo), because new inserted parameters absorb all visual information. Projection-based fusion typically requires fine-tuning the LLM, because visual tokens must be interpretable in the LLM's native token space.

**Multi-image and video flexibility**: cross-attention handles variable numbers of images or frames gracefully through the Perceiver Resampler, which compresses any number of visual tokens to a fixed $K$. Projection-based methods must control visual token count by limiting resolution or using explicit token compression, making long video or multi-image inputs more challenging.

---

## Multimodal Instruction Tuning and Capabilities

### Q9 [Basic] Describe visual instruction tuning and its data requirements

**Q:** How does visual instruction tuning differ from standard visual question answering fine-tuning, and what types of data enable it?

**A:** **Visual instruction tuning** trains a multimodal model to follow free-form natural language instructions about images, rather than answering questions in a fixed short-answer format. Standard VQA fine-tuning optimizes accuracy on a narrow distribution (short-answer VQAv2-style QA); visual instruction tuning trains the model to handle diverse instructions — generating detailed descriptions, reasoning step-by-step, comparing objects, answering in different styles and lengths — making the model conversationally useful rather than benchmark-specialized.

The data format consists of multi-turn conversations where the user provides instructions about an image and the model generates variable-length responses. Liu et al. (2023a) demonstrated that GPT-4 can generate such conversations from image metadata (captions, bounding boxes) without direct image access, producing 158K high-quality instruction-following samples covering three output types: conversational QA, detailed description, and complex reasoning chains.

This is analogous to how instruction-tuned LLMs extend base models for general-purpose use: the instruction data teaches the model to apply its visual understanding in open-ended ways. The training signal — teacher-forced generation on GPT-4-produced responses — is standard cross-entropy, but the diversity of the instruction dataset shapes the model's conversational and reasoning capabilities far more than the loss function itself.

---

### Q10 [Advanced] Explain InstructBLIP's instruction-aware Q-Former

**Q:** What limitation in BLIP-2 does InstructBLIP address, and how does making the Q-Former instruction-aware change what is extracted from the image?

**A:** **InstructBLIP** (Dai et al., 2023) identifies a key limitation in BLIP-2's Q-Former: the 32 learned queries extract a fixed visual summary regardless of which instruction is being processed. For different instructions about the same image — "describe the scene," "count the objects," "read the text on the sign" — identical visual tokens are provided to the LLM, even though each instruction requires attention to entirely different visual regions and properties.

InstructBLIP resolves this by feeding the instruction text into the Q-Former alongside the 32 learnable queries via self-attention. Instruction token embeddings are concatenated with the queries and participate in the same self-attention mechanism; through the subsequent cross-attention to image features, the queries can then selectively extract the visual information most relevant to the current instruction. An attention mask ensures queries can attend to instruction tokens but not vice versa — the instruction text conditions the extraction process without being modified.

This modification does not add significant parameters: it reuses the Q-Former's existing text encoder. The output remains 32 visual tokens, but they are now instruction-conditioned: the same image yields different visual tokens depending on whether the instruction asks for counting, spatial reasoning, or OCR.

InstructBLIP is fine-tuned on 26 datasets spanning 11 task categories (Dai et al., 2023), covering VQA, image captioning, reasoning, and OCR tasks. The instruction-aware features improve performance on held-out benchmarks in a zero-shot setting, demonstrating that richer per-instruction visual extraction generalizes beyond the training tasks.

---

### Q11 [Advanced] Identify what LLaVA-1.5 changes and why it matters

**Q:** What architectural and data modifications does LLaVA-1.5 introduce, and how do these changes reflect lessons from the original LLaVA?

**A:** **LLaVA-1.5** (Liu et al., 2023b) makes two targeted changes informed by ablations on LLaVA's design.

The first change replaces the linear projection connector with a two-layer **MLP connector** (GELU activation between layers). The linear projection performs a purely affine mapping from visual to language token space; the MLP adds nonlinearity that enables more expressive alignment between the two modalities. This single change consistently improves performance across benchmarks without any other modifications, confirming that the linear projection was a representational bottleneck.

The second change augments the instruction-following data with **academic task instruction data** — VQAv2, GQA, and TextVQA converted to instruction format — alongside the original GPT-4-generated conversational data. The original LLaVA dataset is strong for open-ended reasoning but lacks coverage of tasks requiring precise visual reading (OCR, charts, dense text). Adding task-specific data in instruction format expands coverage while maintaining the conversational style.

LLaVA-1.5 also uses a higher-resolution image encoder (CLIP ViT-L/14 at 336×336 vs. 224×224), providing finer-grained spatial tokens that improve performance on spatially demanding tasks.

LLaVA-1.5-13B achieves 85.9% on VQAv2 and 67.7% on MMBench (Liu et al., 2023b), outperforming models like Qwen-VL-Chat and InstructBLIP-13B on multiple benchmarks while using a simpler architecture and fully publicly available training data. The results established that MLP connectors and broader instruction data were consistently beneficial defaults for projection-based multimodal LLMs.

---

### Q12 [Advanced] Identify where multimodal LLMs fail relative to human ability

**Q:** Where does current VLM performance most sharply diverge from human-level visual understanding, and what structural properties of these tasks explain the gap?

**A:** Current multimodal LLMs perform well on coarse-grained visual understanding — scene description, VQA about salient objects, commonsense inference about depicted situations. The gap relative to human performance is concentrated in tasks requiring **fine-grained spatial reasoning, precise counting, and compositional attribute binding**.

**Spatial reasoning**: Questions like "is the red book to the left of the blue mug?" require encoding precise spatial relationships between multiple co-present objects. ViT-based encoders process images with global self-attention and do not explicitly encode absolute spatial coordinates; positional information is encoded coarsely relative to the precision required for grounded directional claims. Models frequently confuse left/right and near/far relationships even when correctly identifying all objects.

**Counting**: VLMs often fail to accurately count objects beyond small numbers ($\leq 4$), even when they correctly describe object identity and categories. The failure is not in recognizing instances but in aggregating count information across distributed spatial positions — a form of structured, iterative attention that uniform global pooling does not naturally support.

**Attribute binding under compositional descriptions**: "What color is the object to the right of the cube?" requires associating an attribute with a specific instance identified via a relational spatial chain. VLMs exhibit **compositional failures** in such cases, reporting the most visually prominent attribute rather than the one correctly bound to the referenced object.

The MMMU benchmark (Yue et al., 2024) reveals a related dimension: tasks requiring domain-specific expert knowledge (medical imaging, engineering diagrams, mathematical proof figures) where visual and textual reasoning must be tightly integrated with disciplinary grounding. GPT-4V achieves 56.8% on MMMU, substantially below expert human performance, indicating that current VLMs lack the structured domain knowledge required for expert-level multimodal reasoning.

---

### Q13 [Advanced] Compare closed-source and open-source VLMs on multimodal benchmarks

**Q:** What capabilities distinguish models like GPT-4V from open-source VLMs, and what does benchmark evidence reveal about the nature of these gaps?

**A:** Closed-source VLMs, including GPT-4V (OpenAI, 2023), represent the current performance ceiling on most comprehensive multimodal benchmarks. On MMMU, GPT-4V achieves 56.8% versus LLaVA-1.5-13B at 36.4% (Yue et al., 2024) — a gap exceeding 20 percentage points on expert-level knowledge tasks. On MMBench, GPT-4V scores 77.0% compared to LLaVA-1.5-13B at 67.7% (Liu et al., 2023b).

The capabilities where the gap is most pronounced are **multi-image cross-referencing**, **OCR and document understanding**, **fine-grained spatial reasoning**, and **multi-step visual reasoning chains with intermediate verification**. GPT-4V demonstrates stronger compositional generalization — handling novel combinations of visual and linguistic concepts — and produces more calibrated explanations with explicit intermediate reasoning steps.

However, direct comparisons are complicated by undisclosed training details. Closed-source models are trained with undisclosed datasets, RLHF pipelines, and chain-of-thought supervision; attributing performance differences to architecture versus data versus scale is not possible. Open-source models such as InternVL (Chen et al., 2023) have progressively closed the gap by combining larger vision encoders (ViT-6B scale), stronger instruction tuning datasets, and higher resolution inputs, suggesting that much of the gap reflects training recipe differences rather than fundamental architectural limits.

A practical concern in benchmark comparisons is **score inflation from data contamination**: instruction-tuned models trained after benchmark release may have seen test-set-adjacent data, particularly for benchmarks derived from publicly available datasets. Benchmarks designed to minimize contamination risk — such as MMMU — show larger and more persistent gaps, suggesting that the capability differences are real but partly masked by contamination in other evaluations.

---

## Visual Grounding and Dense Visual Understanding

### Q14 [Basic] Explain GLIP's grounding-based formulation of open-vocabulary detection

**Q:** What is the core insight that allows GLIP to detect arbitrary object categories by reformulating detection as a grounding problem?

**A:** **GLIP** (Li et al., 2022) reformulates object detection as phrase grounding: given an image and a text description of categories, the model predicts bounding boxes by grounding each noun phrase to image regions. Standard detection classifies visual features against a fixed set of $C$ category indices; GLIP instead computes a similarity between each region proposal and each phrase token:

$$S = \text{ImageEncoder}(I) \cdot \text{TextEncoder}(P)^\top$$

where $S \in \mathbb{R}^{N_\text{boxes} \times N_\text{tokens}}$ is the alignment score matrix. Detection proceeds by finding high-similarity (region, phrase) pairs, treating localization as an alignment problem rather than a classification problem.

The advantage is **vocabulary generalization**: because the text encoder can embed any noun phrase, GLIP detects categories expressible in language, not only those seen during training. The model can be prompted with phrases like "a person in a blue jacket carrying a suitcase" and localize the matching region without being trained on that description. GLIP-L achieves 60.8 AP zero-shot on COCO without any COCO training data (Li et al., 2022), by training on grounding datasets (COCO Captions, Visual Genome, CC3M) where bounding boxes are matched to noun phrases in captions.

The reformulation also enables grounding supervision from large-scale image-caption pairs with bounding box annotations, vastly expanding available training data beyond what curated detection datasets provide.

---

### Q15 [Advanced] Explain Grounding DINO's tight feature fusion strategy

**Q:** How does Grounding DINO integrate language into the detection pipeline more deeply than GLIP, and why does earlier fusion improve performance?

**A:** **Grounding DINO** (Liu et al., 2023c) extends the DINO detection framework with three components that enable tight image-text feature fusion throughout the detection pipeline, rather than only at the final alignment step as in GLIP.

The first component is a **feature enhancer** that applies cross-modal attention between image backbone features and text encoder features at multiple scales. Image features attend to text features (and vice versa) through interleaved cross-attention and deformable self-attention layers, so that spatial visual features are explicitly conditioned on the text query before any proposal selection occurs. The resulting image features are language-aware from the earliest stages of processing.

The second component is **language-guided query selection**: DINO's standard top-$K$ region proposal selection is replaced by a selection based on image-text alignment scores. Proposals with high spatial overlap with the described phrase are selected as decoder queries, ensuring that only description-relevant regions enter the decoder.

The third component is a **cross-modality decoder** where object queries cross-attend to both visual features and text features at each decoder layer, jointly refining box predictions and phrase classification scores.

The advantage of early fusion over late fusion (GLIP's post-hoc alignment) is that image features are conditioned on language before box coordinates are regressed, allowing the detector to focus spatially on described regions rather than extracting generic region features that are later compared to text. Grounding DINO achieves 63.0 AP zero-shot on COCO (no COCO training) and outperforms GLIP on both zero-shot and few-shot grounding benchmarks (Liu et al., 2023c).

---

### Q16 [Advanced] Contrast referring expression comprehension with visual question answering

**Q:** What makes referring expression comprehension structurally harder than standard VQA, and what architectural requirements does it impose?

**A:** **Referring expression comprehension** (REC) requires localizing the image region corresponding to a natural language description of a specific instance: given "the man standing on the left wearing a red shirt," the model must output a bounding box identifying exactly that person. VQA typically requires generating a short textual answer (class label, count, attribute) about the image.

The structural difference is **output modality and spatial precision requirements**. VQA's output is discrete text, handled with standard cross-entropy over a fixed vocabulary. REC's output is a continuous spatial coordinate, requiring the model to precisely localize instances based on relational descriptions involving multiple properties and inter-object relations simultaneously.

REC imposes a distinct set of architectural demands. A successful model must: (1) ground the described instance's visual properties (attributes, category) in the image, (2) evaluate relational spatial claims ("to the left of the table") that require comparing candidate regions to other scene elements, and (3) resolve referential ambiguity when multiple objects satisfy some but not all description constraints. TransVG (Deng et al., 2021) handles this with a transformer-based grounding head that cross-attends language features to visual region proposals, predicting box coordinates from the attended representation.

A key failure mode specific to REC is **referential chain failure**: the model correctly identifies each word's visual referent in isolation but fails to chain multiple constraints. "The dog sitting on the left side of the red couch, not the one by the window" requires composing a color constraint (red couch), a spatial constraint (left side), and a negation constraint (not by the window) — a compositional grounding problem that VLMs trained primarily on VQA-style single-constraint questions handle poorly.

---

### Q17 [Advanced] Describe how VLMs extend to pixel-level segmentation tasks

**Q:** What mechanisms connect language understanding to dense spatial prediction in vision-language segmentation systems?

**A:** Standard VLMs output text or bounding boxes; segmentation requires per-pixel predictions. Two main strategies connect language to pixel-level output.

**Modular composition** combines a language-grounded detector with a prompt-based segmentation model. **SAM** (Kirillov et al., 2023) was trained on 1.1 billion masks (SA-1B) with a promptable mask decoder that accepts box, point, or text prompts and generates high-quality instance masks. Combining Grounding DINO (language-grounded detection) with SAM (mask generation) creates a description-to-mask pipeline without additional training: a natural language description is first localized to a bounding box by Grounding DINO, which is then passed to SAM as a spatial prompt for segmentation. This enables open-vocabulary segmentation of arbitrary described objects.

**End-to-end reasoning segmentation** integrates language and pixel-level prediction within a multimodal LLM. **LISA** (Lai et al., 2023) extends an LLM with a special `<SEG>` token: when generated, the hidden state at the `<SEG>` position is passed to a lightweight mask decoder (built on SAM's decoder architecture) to produce a segmentation mask. This enables **reasoning segmentation** — given "segment the object most useful for cutting the rope," LISA reasons that scissors are the answer, generates a response with `<SEG>`, and the decoder masks the scissors. The LLM is responsible for multi-step reasoning; the mask decoder handles spatial precision.

The central challenge in end-to-end approaches is that the LLM must encode sufficient spatial localization information in the `<SEG>` hidden state for the mask decoder to succeed, while also maintaining coherent high-level semantic reasoning. The two objectives — spatial precision and semantic reasoning — operate at different levels of abstraction and are not naturally aligned in standard LLM training.

---

## Hallucination, Benchmarks, and Frontiers

### Q18 [Basic] Define object hallucination and describe how it is measured

**Q:** How do VLMs produce objects not present in an image, and what evaluation protocols have been developed to quantify this?

**A:** **Object hallucination** occurs when a multimodal LLM's output refers to objects, attributes, or relationships absent from the given image. A model describing an image containing a dog bowl may confidently assert "a dog eating from the bowl" because "dog" and "dog bowl" co-occur frequently in training captions. The language model generates tokens that are statistically plausible given the context, rather than strictly grounded in the visual input.

The root cause is the **language prior dominating the visual signal**: VLMs are trained on image-caption pairs where captions reflect common co-occurrence statistics from the web. The LLM component's strong prior over likely next tokens can override weak or uncertain visual features, producing confident-sounding descriptions of absent objects.

Two evaluation protocols are widely used. **CHAIR** (Rohrbach et al., 2018) — Caption Hallucination Assessment with Image Relevance — measures what fraction of generated caption nouns refer to objects not in the image, using COCO object labels as ground truth. It quantifies hallucination at the caption level (CHAIRs) and sentence level (CHAIRi).

**POPE** (Li et al., 2023b) — Polling-based Object Probing Evaluation — asks binary yes/no questions: "Is there a {object} in the image?" Three sampling strategies of increasing difficulty test different failure modes: **random** (any COCO object), **popular** (frequently occurring objects), and **adversarial** (objects that commonly co-occur with the image's actual content but are absent). Models with strong language priors score substantially lower on the adversarial split than the random split, isolating the degree to which responses are driven by statistical association rather than visual grounding.

---

### Q19 [Advanced] Explain the gap between benchmark performance and real-world reliability

**Q:** What systematic factors cause VLMs that score well on standard benchmarks to fail in deployment, and how have the research community's evaluation practices evolved in response?

**A:** Standard VLM benchmark scores reflect average performance over fixed test sets derived from specific image distributions and question formats, which may not capture the variance and failure modes evident in real deployments. Several systematic factors contribute to the gap.

**Training distribution shift**: Most benchmarks draw from MS COCO, Flickr, or Wikipedia images — professionally taken, well-lit, salient-object-centered photographs. Real deployment involves screenshots, medical images, satellite imagery, low-light captures, and unusual viewpoints. VLM performance degrades substantially on out-of-distribution visual styles not well-represented in pretraining, because the visual encoder's feature distribution shifts in ways that misalign with what the LLM was instruction-tuned on.

**Shortcut exploitation**: Several benchmark analyses found that high VQA accuracy can be achieved by exploiting language biases without visual understanding — answering "2" to most "how many?" questions or "yes" to most binary questions is sufficient for above-chance accuracy on biased datasets. Models optimizing these metrics score well while failing qualitatively on a redistributed question set.

**Compositional generalization failure**: Models tested on held-out combinations of concepts — "a purple elephant in a living room" — frequently fail even when handling each component in isolation, indicating that the model has memorized common training combinations rather than learning compositional visual semantics.

**Metric-task misalignment for generation tasks**: BLEU and CIDEr for captioning reward lexical overlap with reference captions and do not penalize hallucinated objects that use high-frequency words, nor reward correctly described rare objects. The MMMU benchmark (Yue et al., 2024) was designed to minimize these effects through expert-level, knowledge-intensive questions that require visual and disciplinary reasoning jointly — the gap between GPT-4V and open-source models on MMMU is substantially larger than on COCO-derived benchmarks, suggesting contamination and shortcut effects inflate reported performance on older benchmarks.

---

### Q20 [Advanced] Analyze the architectural challenges of video-language models

**Q:** What computational and reasoning challenges arise when extending image-based VLMs to video, and how have current approaches addressed them?

**A:** Video understanding introduces two challenges absent from image VLMs: **token scale** and **temporal reasoning**. A video of $T$ frames processed by a ViT backbone produces $T \times N_\text{patch}$ tokens: at 8 frames per second with ViT-L (256 tokens/frame), a 10-second clip yields 20,480 visual tokens — far exceeding the context length of most LLMs and quadratically inflating self-attention cost.

The primary architectural response is **temporal sampling and compression**. Most video-LLMs select a fixed number of frames (typically 8–16) and process them independently through the image encoder, then aggregate via temporal pooling or concatenation before passing to the LLM. **Video-LLaMA** (Zhang et al., 2023) adds a dedicated Video Q-Former that applies temporal self-attention across frame-level features, compressing the video into a fixed token count before the LLM. **VideoChat** (Li et al., 2023c) adopts a similar Q-Former-based design with video-specific instruction tuning data for conversational video understanding.

The deeper challenge is **temporal reasoning**: correctly ordering events, inferring causal relationships between actions, and maintaining consistent entity tracking across frames. Sparse frame sampling may miss key moments; models must integrate evidence from a sequence of static visual snapshots rather than a continuous signal. Current video-LLMs handle short-clip event recognition reasonably but struggle with long-form video where critical information is distributed across frames minutes apart — the fixed token budget forces lossy compression rather than selective retrieval.

An emerging direction is **streaming video understanding**, where the model processes frames as they arrive rather than over pre-segmented clips. This requires memory mechanisms that efficiently summarize previously observed content while remaining sensitive to newly arriving frames — combining recurrent compression with selective attention — a design space that static VLM architectures do not naturally support and that remains an active area of research.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | CLIP contrastive objective | Contrastive Vision-Language Pretraining |
| Q2 | Basic | Zero-shot transfer and prompt engineering | Contrastive Vision-Language Pretraining |
| Q3 | Basic | ALIGN and noisy large-scale data | Contrastive Vision-Language Pretraining |
| Q4 | Advanced | SigLIP sigmoid loss and batch-size sensitivity | Contrastive Vision-Language Pretraining |
| Q5 | Basic | Flamingo frozen-model architecture | Architecture Design for Multimodal Models |
| Q6 | Basic | BLIP-2 Q-Former | Architecture Design for Multimodal Models |
| Q7 | Advanced | LLaVA minimal architecture | Architecture Design for Multimodal Models |
| Q8 | Advanced | Cross-attention vs. projection fusion | Architecture Design for Multimodal Models |
| Q9 | Basic | Visual instruction tuning | Multimodal Instruction Tuning and Capabilities |
| Q10 | Advanced | InstructBLIP instruction-aware Q-Former | Multimodal Instruction Tuning and Capabilities |
| Q11 | Advanced | LLaVA-1.5 MLP connector and data expansion | Multimodal Instruction Tuning and Capabilities |
| Q12 | Advanced | Fine-grained visual understanding failures | Multimodal Instruction Tuning and Capabilities |
| Q13 | Advanced | Closed-source vs. open-source VLMs | Multimodal Instruction Tuning and Capabilities |
| Q14 | Basic | GLIP open-vocabulary detection | Visual Grounding and Dense Visual Understanding |
| Q15 | Advanced | Grounding DINO feature fusion | Visual Grounding and Dense Visual Understanding |
| Q16 | Advanced | Referring expression comprehension | Visual Grounding and Dense Visual Understanding |
| Q17 | Advanced | Language-driven pixel-level segmentation | Visual Grounding and Dense Visual Understanding |
| Q18 | Basic | Object hallucination and POPE | Hallucination, Benchmarks, and Frontiers |
| Q19 | Advanced | Benchmark vs. real-world reliability | Hallucination, Benchmarks, and Frontiers |
| Q20 | Advanced | Video-language model architecture | Hallucination, Benchmarks, and Frontiers |

## Resources

- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021)
- Jia et al., [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) (2021)
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) (2023)
- Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (2022)
- Li et al., [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) (2023a)
- Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (2023a)
- Liu et al., [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) (2023b)
- Dai et al., [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) (2023)
- Yue et al., [MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI](https://arxiv.org/abs/2311.16502) (2024)
- Li et al., [Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857) (2022)
- Liu et al., [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499) (2023c)
- Deng et al., [TransVG: End-to-End Visual Grounding with Transformers](https://arxiv.org/abs/2104.08567) (2021)
- Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643) (2023)
- Lai et al., [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692) (2023)
- Rohrbach et al., [Object Hallucination in Image Captioning](https://arxiv.org/abs/1809.02156) (2018)
- Li et al., [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355) (2023b)
- Chen et al., [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238) (2023)
- Zhang et al., [Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding](https://arxiv.org/abs/2306.02858) (2023)
- Li et al., [VideoChat: Chat-Centric Video Understanding](https://arxiv.org/abs/2305.06355) (2023c)
