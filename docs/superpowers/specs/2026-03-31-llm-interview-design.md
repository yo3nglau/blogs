# Design: Large Language Models Interview Questions

**Date:** 2026-03-31  
**Status:** Approved

## Overview

A blog post presenting 20 interview questions and recommended answers covering Large Language Models (LLMs), targeting candidates preparing for deep learning / NLP job interviews. Questions span four topics: pretraining & architecture, alignment & fine-tuning, inference & efficiency, and applications & engineering. Each is tagged [Basic] or [Advanced]. A quick index table at the end enables targeted review. Style matches the existing Transformer/ViT and Mamba interview posts.

## Frontmatter

```yaml
title: "Large Language Models: Interview Questions and Answers"
author: yo3nglau
date: '2026-03-31'
categories:
  - Interview
tags:
  - Deep Learning
  - Large Language Model
  - NLP
toc: true
```

## Article Structure

### Section 1 — Pretraining & Architecture (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q1 | Basic    | LLM pretraining objective (next token prediction) |
| Q2 | Basic    | Scaling Laws: model size, data, and performance |
| Q3 | Basic    | Positional encoding: RoPE vs ALiBi vs absolute PE |
| Q4 | Advanced | Long-context extension: 4K to 128K |
| Q5 | Advanced | Mixture of Experts (MoE): architecture and benefits |

### Section 2 — Alignment & Fine-tuning (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q6  | Basic    | Supervised Fine-Tuning (SFT) |
| Q7  | Basic    | RLHF: principles and pipeline |
| Q8  | Basic    | DPO: bypassing RL for alignment |
| Q9  | Advanced | RLHF challenges and limitations |
| Q10 | Advanced | Alignment tax and RLHF vs DPO trade-offs |

### Section 3 — Inference & Efficiency (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q11 | Basic    | KV Cache: role and mechanics |
| Q12 | Basic    | Model quantization (INT8/INT4) |
| Q13 | Basic    | Speculative decoding |
| Q14 | Advanced | Continuous batching for throughput |
| Q15 | Advanced | FlashAttention in LLM inference |

### Section 4 — Applications & Engineering (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q16 | Basic    | RAG: principles and advantages |
| Q17 | Basic    | Prompt engineering core techniques |
| Q18 | Advanced | LLM agents: tool use and planning |
| Q19 | Advanced | RAG limitations and long context vs RAG |
| Q20 | Advanced | Hallucination: causes and mitigation |

### Quick Reference Table

Columns: # / Difficulty / Topic / Section (all 20 rows).

### Resources

- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT/RLHF, 2022)
- Rafailov et al., "Direct Preference Optimization" (DPO, 2023)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE, 2021)
- Press et al., "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022)
- Dao et al., "FlashAttention-2" (2023)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, 2020)

## Per-Question Format

```markdown
### Q1 [Basic] What is the pretraining objective of LLMs?

**Q:** ...

**A:** [2-3 paragraphs for Basic, 3-5 for Advanced]
```

## Constraints

- Language: English
- Output file: `content/post/Large Language Models Interview Questions and Answers.md`
- Hugo-compatible Markdown with YAML frontmatter
- Site rebuild required after writing: `hugo` command
