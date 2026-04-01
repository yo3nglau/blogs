# Design: LLM-based Agents Interview Questions

**Date:** 2026-04-01  
**Status:** Approved

## Overview

A blog post presenting 20 interview questions and recommended answers covering LLM-based Agents, targeting candidates preparing for deep learning / AI research job interviews. Questions span five sections: Agent Fundamentals, Planning & Reasoning, Memory & Retrieval, Multi-Agent Systems, and Evaluation & Benchmarks. Each question is tagged [Basic] or [Advanced] and anchored to a key paper or method. A quick reference table at the end enables targeted review. Style and structure match the existing LLM, Mamba, and TAD interview posts.

## Frontmatter

```yaml
title: "LLM-based Agents: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-01'
categories:
  - Interview
tags:
  - Deep Learning
  - Large Language Model
  - Agent
toc: true
```

## Article Structure

### Section 1 — Agent Fundamentals (4 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q1 | Basic    | LLM-based agent definition: components (perception, memory, planning, action) |
| Q2 | Basic    | ReAct: interleaving reasoning traces and actions |
| Q3 | Basic    | Tool use: function calling, tool selection, and execution loop |
| Q4 | Advanced | Agent loop failure modes: hallucination, infinite loops, error propagation |

### Section 2 — Planning & Reasoning (5 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q5 | Basic    | Chain-of-Thought and its relationship to agent planning |
| Q6 | Advanced | Tree of Thoughts (ToT): search tree structure and BFS/DFS strategies |
| Q7 | Advanced | Reflexion: verbal reinforcement learning and self-reflection |
| Q8 | Advanced | MCTS applied to LLM planning (RAP, AlphaCode 2) |
| Q9 | Advanced | Task decomposition strategies: MRKL, HuggingGPT, TaskMatrix |

### Section 3 — Memory & Retrieval (4 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q10 | Basic    | Three memory types: sensory buffer, working memory (context window), long-term memory |
| Q11 | Basic    | RAG as external agent memory: mechanism and limitations |
| Q12 | Advanced | MemGPT: hierarchical memory management and context window extension |
| Q13 | Advanced | Episodic vs semantic memory: implementation and retrieval strategies |

### Section 4 — Multi-Agent Systems (4 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q14 | Basic    | Motivation for multi-agent systems over single-agent |
| Q15 | Basic    | AutoGen: ConversableAgent architecture and conversation-driven collaboration |
| Q16 | Advanced | Multi-agent communication topologies: star, bus, hierarchical trade-offs |
| Q17 | Advanced | MetaGPT: role specialization, SOP-driven workflows, and structured outputs |

### Section 5 — Evaluation & Benchmarks (3 questions)

| # | Difficulty | Topic |
|---|------------|-------|
| Q18 | Basic    | Key agent benchmarks: ALFWorld, WebArena, SWE-bench — design and evaluation dimensions |
| Q19 | Advanced | Agent failure mode taxonomy: grounding, planning, memory, safety |
| Q20 | Advanced | Open problems and research frontiers in LLM agents |

### Quick Reference Table

Columns: # / Difficulty / Topic / Section (all 20 rows).

### Resources

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023)
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
- Hao et al., "Reasoning with Language Model is Planning with World Model" (RAP, 2023)
- Nakano et al., "WebGPT: Browser-assisted question-answering with human feedback" (2022)
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023)
- Packer et al., "MemGPT: Towards LLMs as Operating Systems" (2023)
- Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)
- Hong et al., "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework" (2023)
- Shen et al., "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace" (2023)
- Koh et al., "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks" (2024)
- Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2024)

## Per-Question Format

```markdown
### Q1 [Basic] What defines an LLM-based agent?

**Q:** ...

**A:** [2-3 paragraphs for Basic, 3-5 for Advanced]
```

## Constraints

- Language: English
- Output file: `content/post/LLM-based Agents Interview Questions and Answers.md`
- Hugo-compatible Markdown with YAML frontmatter
- LaTeX math in `$...$` delimiters where applicable (Goldmark passthrough extension enabled)
- Site rebuild required after writing: `hugo --minify`
