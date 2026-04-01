---
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
---

## Agent Fundamentals

### Q1 [Basic] What is an LLM-based agent and what are its core components?

**Q:** How is an LLM-based agent defined, and what are the four components that distinguish it from a standalone language model?

**A:** An LLM-based agent is a system that uses a large language model as its central reasoning engine to autonomously pursue goals by taking sequences of actions in an environment. The critical distinction from a standalone LLM is that an agent operates in a loop: it perceives inputs, plans actions, executes them via tools or code, and integrates the results into subsequent reasoning steps — rather than producing a single response to a single prompt.

Four components characterize LLM agents. First, **perception** (or the sensory buffer): the agent receives inputs from its environment, which may include text, tool outputs, memory retrievals, observations from prior steps, or structured data. Second, **memory**: the agent maintains state across steps. Working memory is the context window — all tokens currently in the prompt. Long-term memory extends beyond the context window via external stores such as vector databases or key-value stores. Third, **planning**: the agent decomposes tasks, generates candidate action sequences, and may revise plans in light of intermediate results. The LLM itself provides this capability through prompting strategies such as chain-of-thought or more structured search methods. Fourth, **action**: the agent executes operations that affect the environment — calling APIs, running code, querying databases, browsing the web, or invoking specialized sub-models. The action space is defined by the tools available to the agent.

The agent loop — observe, think, act, observe — repeats until the goal is achieved or a termination condition is met. This loop is what makes agents qualitatively different from prompted LLMs: the model's output is not the final product but an intermediate step in an ongoing process.

---

### Q2 [Basic] What is the ReAct framework and how does it combine reasoning and acting?

**Q:** Describe the ReAct prompting strategy, its key insight, and how it improves upon chain-of-thought and action-only baselines.

**A:** ReAct (Reasoning + Acting, Yao et al., 2023) is a prompting framework that interleaves natural language reasoning traces with discrete actions in a single sequence. Each agent step consists of a *Thought* (a free-text reasoning trace in which the model updates its understanding and decides what to do next), an *Action* (a structured command such as `Search[query]` or `Lookup[term]`), and an *Observation* (the environment's response, appended to the context). This Thought–Action–Observation triplet repeats until the task is complete.

The key insight is that reasoning and acting benefit from tight coupling. Chain-of-thought (CoT) prompting generates reasoning but cannot interact with the environment to verify or update beliefs — it hallucinates facts it cannot look up. Pure action baselines execute tool calls but lack interpretable intermediate reasoning, making it hard to diagnose failures or course-correct. ReAct combines both: the Thought step allows the model to interpret tool outputs, reassess its plan, and recover from errors before committing to the next action.

On knowledge-intensive question answering (HotpotQA, FEVER) and interactive decision-making tasks (ALFWorld, WebShop), ReAct substantially outperforms both CoT and action-only baselines, and also outperforms imitation-learning and reinforcement-learning baselines on ALFWorld. Critically, ReAct's reasoning traces make the agent's behavior interpretable and support human-in-the-loop correction: a human can read the Thought steps, identify where the agent went wrong, and intervene by editing the context.

---

### Q3 [Basic] How does tool use work in LLM agents, and what are the key design challenges?

**Q:** What mechanisms enable LLMs to call external tools, how is tool selection implemented, and what are the principal challenges in building reliable tool-use systems?

**A:** Tool use in LLM agents works by representing available tools as structured function signatures (name, description, parameter schema) and including them in the model's prompt or system message. The model generates a structured output — either a JSON object matching a function schema (OpenAI function calling / tool use API) or a free-text action following a fixed template (as in ReAct) — that the agent runtime parses and executes. The result is appended to the context as an observation, and the model continues.

Tool selection is the model's implicit decision about which tool to invoke given the current state. This is driven purely by in-context reasoning: the model reads the tool descriptions and infers which is most appropriate. Toolformer (Schick et al., 2023) showed that LLMs can be fine-tuned to decide when and how to call tools by training on self-supervised examples where tool calls are inserted at positions where they reduce future token prediction loss — the model learns both the utility and the correct syntax of each tool without explicit supervision.

Three principal challenges arise in tool-use systems. First, **tool selection errors**: models may invoke the wrong tool, pass malformed arguments, or invoke tools unnecessarily. This worsens as the tool set grows, because the model must distinguish between many similar tools from descriptions alone. Second, **error propagation**: tool errors (network failures, API rate limits, invalid queries) must be handled gracefully; a naive agent stalls or hallucinate results. Third, **grounding**: models tend to generate tool arguments that look plausible but are semantically wrong (e.g., a search query that is grammatically correct but fails to retrieve the relevant document). Robust tool-use systems require careful prompt engineering, error-handling wrappers around tool execution, retry logic, and often fine-tuning on domain-specific tool-use traces.

---

### Q4 [Advanced] What are the principal failure modes of the LLM agent loop?

**Q:** Categorize and explain the main ways LLM agent loops fail, and describe the mechanisms by which errors compound across steps.

**A:** LLM agent failures fall into four categories, each with distinct mechanisms and compounding dynamics.

**Hallucination and confabulation** occur when the model generates plausible-sounding but factually wrong reasoning traces or tool arguments. Unlike in single-turn generation, hallucinations in agents persist: a wrong belief stated in a Thought step is carried forward as a premise in all subsequent Thought steps, and tool calls based on hallucinated facts produce nonsensical observations that the model often rationalizes rather than corrects. This is the compounding dynamic unique to agentic settings.

**Infinite loops and repetition** arise when the model fails to make progress: it re-invokes the same tool with the same arguments, generating the same observation repeatedly, or alternates between two states without converging. This happens because the LLM's next-token prediction has no explicit termination signal — it lacks a learned notion of "I have tried this already." Mitigation requires explicit loop detection in the agent runtime (tracking action history and blocking duplicate calls) or a maximum step budget.

**Error propagation from early mistakes** is the sequential dependency problem: each step conditions on all prior steps. A wrong tool selection in step 2 changes the observation in step 3, which may lead to a coherent but entirely incorrect plan thereafter. The agent has no mechanism to backtrack to a pre-error state unless the architecture explicitly supports branching (as in Tree of Thoughts or Reflexion).

**Context length overflow** is a practical failure mode as trajectories grow long: relevant early context (task instructions, initial observations) is pushed out of the effective attention window by accumulating Thought–Action–Observation triplets, causing the model to lose track of the original goal. This motivates memory compression strategies and hierarchical context management (as in MemGPT).

---
