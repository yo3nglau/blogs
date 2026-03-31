# Design: Temporal Action Detection Interview Questions

**Date:** 2026-03-31  
**Status:** Approved

## Overview

A blog post presenting 20 interview questions and recommended answers covering Temporal Action Detection (TAD) and related video understanding tasks, targeting researchers preparing for deep learning / computer vision job interviews. Questions span four topics: TAD fundamentals & detection pipelines, temporal proposal generation, temporal sentence grounding, and action segmentation & frontiers. Each is tagged [Basic] or [Advanced]. A quick index table at the end enables targeted review. Style matches the existing Transformer/ViT, Mamba, and LLM interview posts.

## Frontmatter

```yaml
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
```

## Article Structure

### Section 1 — TAD Fundamentals & Detection Pipelines (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q1 | Basic    | TAD task definition: difference from Action Recognition |
| Q2 | Basic    | Two-stage pipeline: proposal + classification |
| Q3 | Basic    | Evaluation metrics: tIoU, mAP@tIoU, and limitations |
| Q4 | Advanced | Anchor-based vs anchor-free temporal detectors |
| Q5 | Advanced | Query-based / DETR-style detectors (ActionFormer, TemporalMaxer) |

### Section 2 — Temporal Proposal Generation (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q6  | Basic    | Temporal Action Proposal (TAP): objectives and classical methods (SST, BSN) |
| Q7  | Basic    | BMN (Boundary-Matching Network): design and confidence map |
| Q8  | Advanced | Graph-based proposals: GTAD and temporal graph modeling |
| Q9  | Advanced | Snippet-level vs segment-level features for proposal quality |
| Q10 | Advanced | NMS and Soft-NMS in the temporal domain |

### Section 3 — Temporal Sentence Grounding (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q11 | Basic    | TSG task definition: core difference from TAD (query-conditioned localization) |
| Q12 | Basic    | Two-stage vs one-stage grounding: trade-offs |
| Q13 | Advanced | Cross-modal alignment: language-video feature interaction mechanisms |
| Q14 | Advanced | Weakly-supervised temporal grounding: learning without timestamp annotations |
| Q15 | Advanced | Moment-DETR: query-based grounding design and advantages |

### Section 4 — Action Segmentation & Frontiers (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q16 | Basic    | Temporal Action Segmentation vs TAD: task differences and applications |
| Q17 | Basic    | MS-TCN: multi-stage temporal convolutional network design |
| Q18 | Advanced | Transformer-based segmentation: ASFormer improvements |
| Q19 | Advanced | Online / streaming TAD: key differences from offline methods and challenges |
| Q20 | Advanced | Video-LLM + TAD: how large models enable temporal localization |

### Quick Reference Table

Columns: # / Difficulty / Topic / Section (all 20 rows).

### Resources

- Lin et al., "BSN: Boundary Sensitive Network for Temporal Action Proposal Generation" (2018)
- Lin et al., "BMN: Boundary-Matching Network for Temporal Action Proposal Generation" (2019)
- Xu et al., "G-TAD: Sub-Graph Localization for Temporal Action Detection" (2020)
- Shou et al., "CDC: Convolutional-De-Convolutional Networks for Precise Temporal Action Localization" (2017)
- Shi et al., "ActionFormer: Localizing Moments of Actions with Transformers" (2022)
- Yuan et al., "Moment-DETR: End-to-End Temporal Sentence Grounding with Transformers" (2021)
- Li et al., "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation" (2019)
- Yi et al., "ASFormer: Transformer for Action Segmentation" (2021)
- Ren et al., "TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding" (2024)

## Per-Question Format

```markdown
### Q1 [Basic] How does Temporal Action Detection differ from Action Recognition?

**Q:** ...

**A:** [2-3 paragraphs for Basic, 3-5 for Advanced]
```

## Constraints

- Language: English
- Output file: `content/post/Temporal Action Detection Interview Questions and Answers.md`
- Hugo-compatible Markdown with YAML frontmatter
- Site rebuild required after writing: `hugo` command
