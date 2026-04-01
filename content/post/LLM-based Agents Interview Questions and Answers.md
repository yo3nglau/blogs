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

**Q:** How is an LLM-based agent defined, and what components distinguish it from a standalone language model?

**A:** An LLM-based agent is a system that uses a large language model as its central reasoning engine to autonomously pursue goals by taking sequences of actions in an environment. The critical distinction from a standalone LLM is that an agent operates in a loop: it perceives inputs, plans actions, executes them via tools or code, and integrates the results into subsequent reasoning steps — rather than producing a single response to a single prompt.

Four components characterize LLM agents (Wang et al., 2023; Weng, 2023). First, **perception** (or the sensory buffer): the agent receives inputs from its environment, which may include text, tool outputs, memory retrievals, observations from prior steps, or structured data. Second, **memory**: the agent maintains state across steps. Working memory is the context window — all tokens currently in the prompt. Long-term memory extends beyond the context window via external stores such as vector databases or key-value stores. Third, **planning**: the agent decomposes tasks, generates candidate action sequences, and may revise plans in light of intermediate results. The LLM itself provides this capability through prompting strategies such as chain-of-thought or more structured search methods. Fourth, **action**: the agent executes operations that affect the environment — calling APIs, running code, querying databases, browsing the web, or invoking specialized sub-models. The action space is defined by the tools available to the agent.

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

Tool selection is the model's implicit decision about which tool to invoke given the current state. This is driven purely by in-context reasoning: the model reads the tool descriptions and infers which is most appropriate. Toolformer (Schick et al., 2023) demonstrates one training-based approach to tool awareness: LLMs are fine-tuned on self-supervised examples where tool calls are inserted at positions where they reduce future token prediction loss, teaching the model both the utility and the correct syntax of each tool without explicit supervision. This complements the prompting-based approach above; most production frameworks (OpenAI function calling, Anthropic tool use) rely on the latter.

Three principal challenges arise in tool-use systems. First, **tool selection errors**: models may invoke the wrong tool, pass malformed arguments, or invoke tools unnecessarily. This worsens as the tool set grows, because the model must distinguish between many similar tools from descriptions alone. Second, **error propagation**: tool errors (network failures, API rate limits, invalid queries) must be handled gracefully; a naive agent stalls or hallucinates results. Third, **grounding**: models tend to generate tool arguments that look plausible but are semantically wrong (e.g., a search query that is grammatically correct but fails to retrieve the relevant document). Robust tool-use systems require careful prompt engineering, error-handling wrappers around tool execution, retry logic, and often fine-tuning on domain-specific tool-use traces.

---

### Q4 [Advanced] What are the principal failure modes of the LLM agent loop?

**Q:** Categorize and explain the main ways LLM agent loops fail, and describe the mechanisms by which errors compound across steps.

**A:** LLM agent failures fall into four categories, each with distinct mechanisms and compounding dynamics.

**Hallucination and confabulation** occur when the model generates plausible-sounding but factually wrong reasoning traces or tool arguments. Unlike in single-turn generation, hallucinations in agents persist: a wrong belief stated in a Thought step is carried forward as a premise in all subsequent Thought steps, and tool calls based on hallucinated facts produce nonsensical observations that the model often rationalizes rather than corrects. This is the compounding dynamic unique to agentic settings.

**Infinite loops and repetition** arise when the model fails to make progress: it re-invokes the same tool with the same arguments, generating the same observation repeatedly, or alternates between two states without converging. This happens because the LLM's next-token prediction has no explicit termination signal — it lacks a learned notion of "I have tried this already." Mitigation requires explicit loop detection in the agent runtime (tracking action history and blocking duplicate calls) or a maximum step budget.

**Error propagation from early mistakes** is the sequential dependency problem: each step conditions on all prior steps. A wrong tool selection in step 2 changes the observation in step 3, which may lead to a coherent but entirely incorrect plan thereafter. The agent has no mechanism to backtrack to a pre-error state unless the architecture explicitly supports branching (as in Tree of Thoughts (Yao et al., 2023) or Reflexion (Shinn et al., 2023)).

**Context length overflow** is a practical failure mode as trajectories grow long: relevant early context (task instructions, initial observations) is pushed out of the effective attention window by accumulating Thought–Action–Observation triplets, causing the model to lose track of the original goal. This motivates memory compression strategies and hierarchical context management (as in MemGPT (Packer et al., 2023)).

---

## Planning & Reasoning

### Q5 [Basic] What is Chain-of-Thought prompting and how does it relate to agent planning?

**Q:** What is chain-of-thought prompting, what makes it effective, and how does it serve as the foundation for more sophisticated agent planning?

**A:** Chain-of-thought (CoT) prompting (Wei et al., 2022) elicits intermediate reasoning steps from LLMs by including examples in the prompt where the solution is accompanied by a natural language derivation rather than just a final answer. Given a few such exemplars, the model generalizes the pattern and produces its own reasoning traces before producing an answer. Zero-shot CoT (Kojima et al., 2022) achieves a similar effect simply by appending "Let's think step by step" to the prompt, without any examples.

CoT is effective because multi-step reasoning problems require intermediate computations that cannot be computed in a single forward pass through the model's feed-forward layers. By writing intermediate steps into the output (and thus into future input via the autoregressive generation process), the model effectively extends its working memory — each generated token is available as context for generating the next. This makes tasks requiring arithmetic, symbolic manipulation, or logical chaining tractable for models that fail on the same problems when prompted for a direct answer.

In the agent context, CoT is the basic planning mechanism: the Thought step in ReAct, the reasoning in Reflexion, and the node expansion in Tree of Thoughts are all instantiations of chain-of-thought. More advanced planning methods extend CoT by adding search (ToT selects which reasoning paths to pursue), iteration (Reflexion revises plans based on feedback), or grounding (ReAct interleaves CoT with tool calls). Understanding CoT is therefore prerequisite to understanding all higher-level agent planning approaches.

---

### Q6 [Advanced] How does Tree of Thoughts extend chain-of-thought for deliberate planning?

**Q:** Describe the Tree of Thoughts framework, how it structures the search over reasoning paths, and in what settings it outperforms standard chain-of-thought.

**A:** Tree of Thoughts (ToT, Yao et al., 2023) reframes the LLM's generation process as a search over a tree of coherent reasoning steps ("thoughts"), where each node is a partial solution state. Unlike linear CoT (a single path from problem to answer) or self-consistency (multiple independent paths, aggregated by majority vote), ToT explicitly maintains and explores multiple branches simultaneously, using the LLM itself as both a generator of candidate thoughts and an evaluator of their promise.

The framework requires defining three components: a **thought decomposition** (what constitutes a meaningful intermediate step for the task — a sentence, an equation, a plan action), a **thought generator** (either sampling multiple completions from the LLM, or proposing candidates with a separate "propose" prompt), and a **state evaluator** (the LLM judges each partial state as "sure", "likely", or "impossible" to lead to a correct solution, using a scalar value prompt). With these components, standard tree search algorithms — BFS or DFS with pruning — can be applied.

ToT is most beneficial for tasks requiring exploration: when the correct path is hard to identify at the first step and early commitments frequently lead to dead ends. The paper demonstrates this on Game of 24 (arithmetic reasoning requiring backtracking), Creative Writing (multi-step coherence planning), and mini Crosswords. Standard CoT (even with self-consistency) nearly fails on Game of 24 ($4\%$ success), while ToT with BFS reaches $74\%$. The trade-off is cost: ToT requires many LLM calls per problem ($O(b \times d)$ where $b$ is the branching factor and $d$ is the tree depth), making it expensive for tasks where linear CoT already works well.

---

### Q7 [Advanced] What is Reflexion and how does it use verbal reinforcement learning?

**Q:** Describe the Reflexion framework's architecture, its verbal reinforcement mechanism, and how it differs from standard RL fine-tuning.

**A:** Reflexion (Shinn et al., 2023) is a framework that improves an LLM agent's performance over multiple trials on the same task through self-generated verbal feedback, without updating model weights. The architecture has three components. First, an **Actor** (the base LLM agent) generates actions and trajectories as in ReAct. Second, an **Evaluator** scores the completed trajectory — this may be an external reward signal (task success), a heuristic (unit test pass/fail), or another LLM acting as a judge. Third, a **Self-Reflection** model (the same LLM, prompted to reflect) takes the trajectory and its evaluation, and generates a natural language summary of what went wrong and how to do better next time. This verbal reflection is stored in an **episodic memory buffer** and prepended to the agent's context in subsequent attempts.

The key mechanism is that natural language feedback is a richer, more targeted learning signal than scalar rewards. A reflection such as "I searched for the wrong entity because I misread the question — next time I should re-read the question before searching" directly guides the agent's next attempt in a way that a binary failure signal cannot. This is "verbal reinforcement learning" in the sense that the feedback accumulates across trials and shapes behavior, but through in-context conditioning rather than gradient descent.

Reflexion differs from standard RL fine-tuning in three ways: (1) it requires no gradient updates and works with black-box LLM APIs; (2) its memory is ephemeral — verbal reflections persist only within a session, not across unrelated tasks; (3) it depends on the LLM's ability to generate accurate self-diagnoses, which can fail when the model lacks the capability to identify its own errors. Reflexion with GPT-4 as the Actor achieves $91\%$ pass@1 on HumanEval and ALFWorld sequential decision-making by combining the strengths of CoT, ReAct, and trial-and-error learning.

---

### Q8 [Advanced] How is Monte Carlo Tree Search applied to LLM planning?

**Q:** Describe how MCTS integrates with LLM agents for planning (as in RAP), and what advantages this offers over greedy or beam-search planning.

**A:** Monte Carlo Tree Search (MCTS) applied to LLM planning (most directly in RAP — Reasoning via Planning, Hao et al., 2023) treats the reasoning process as a Markov Decision Process (MDP) where states are partial reasoning traces, actions are the next reasoning step generated by the LLM, and the reward is a signal from the LLM acting as a world model (estimating the likelihood that the current state leads to a correct answer). MCTS iterates four phases: **selection** (traverse the tree using UCT to balance exploitation of high-value nodes and exploration of less-visited ones), **expansion** (generate new child states by sampling the next reasoning step from the LLM), **simulation** (roll out the trajectory to completion and obtain a reward), and **backpropagation** (update value estimates along the path).

The key contribution of RAP is using the LLM simultaneously as the **policy** (generating candidate reasoning steps) and the **world model** (evaluating the expected utility of each state by prompting it to answer "how likely is this to lead to a correct solution?"). This avoids the need for a separately trained value function or external reward model for domains where the LLM's own beliefs about solution quality are reliable.

MCTS offers advantages over greedy decoding (single path, no backtracking) and beam search (fixed-width parallel paths without value estimation or pruning by quality): it allocates search budget to the most promising branches, can recover from early mistakes by backtracking, and produces diverse candidate solutions. On Blocksworld planning and mathematical reasoning benchmarks (including GSM8K), RAP with MCTS substantially outperforms CoT-SC (self-consistency) and ToT-DFS. The cost is the same as ToT — many LLM calls — but the UCT selection policy is more principled than ToT's heuristic pruning.

---

### Q9 [Advanced] How do task decomposition strategies differ across MRKL, HuggingGPT, and TaskMatrix?

**Q:** Compare the task decomposition and tool orchestration approaches of MRKL, HuggingGPT, and TaskMatrix, and identify the design trade-offs each makes.

**A:** These three systems represent an evolutionary progression in how LLM agents decompose tasks and route sub-tasks to specialized tools or models.

**MRKL** (Modular Reasoning, Knowledge, and Language, Karpas et al., 2022) is the earliest formulation. It defines a neuro-symbolic architecture with a central LLM router that receives a user query, identifies which discrete "expert module" (a calculator, a database lookup, a search engine) is appropriate, and routes the query there. The LLM serves as a natural language interface between the user and a fixed, manually curated set of symbolic modules. The decomposition is shallow (typically one routing step) and the module set is static — the system cannot discover or compose new tools.

**HuggingGPT** (Shen et al., 2023) extends the routing paradigm to a vast, dynamically queryable model hub (Hugging Face). Given a user request, the LLM generates a multi-step task plan (a sequence of structured JSON tasks, each with a task type, dependencies, and arguments), then dispatches each sub-task to a specialized model selected by matching the task description to model metadata. Results from completed tasks are returned to the LLM, which synthesizes the final response. The key advance is multi-step sequential and parallel task planning: a request like "describe this image and translate the caption to French" is decomposed into image captioning → translation, with explicit dependency tracking.

**TaskMatrix.AI** (Liang et al., 2023) scales this further by proposing a universal API platform where millions of APIs (software functions, cloud services, physical devices) are uniformly described using an API schema. The LLM uses an **action executor** that selects APIs from this registry via semantic search on their descriptions, generates the API call, and chains calls across steps. The design trade-off is discoverability vs. reliability: a large, automatically curated API set enables breadth but makes it harder for the LLM to select the right API from many similar candidates, increasing tool selection error rates compared to small, manually curated tool sets.

---

## Memory & Retrieval

### Q10 [Basic] What are the three types of memory in LLM agents?

**Q:** How is memory categorized in LLM agent systems, and what are the storage mechanism, capacity, and access pattern of each type?

**A:** LLM agent memory is categorized into three types by analogy to human cognitive memory systems (Weng, 2023; Wang et al., 2023).

**Sensory memory** (or sensory buffer) corresponds to the raw, unprocessed inputs received in the current step: the user's request, the latest tool output, a retrieved document chunk. It is transient — only the current observation is in this buffer, and it must be integrated into working memory to influence reasoning.

**Working memory** is the model's context window: all tokens currently visible to the model during inference. This is where active reasoning happens — the current task, recent Thought–Action–Observation history, retrieved memory snippets, and system instructions all compete for context space. Working memory is fast (constant-time access — everything is in-context) but severely capacity-limited. Current LLMs have context windows of 8K–1M tokens, but effective attention degrades for very long contexts, and cost scales with context length. Working memory is also volatile: it resets between sessions unless persisted externally.

**Long-term memory** is stored outside the context window and retrieved selectively. It is subdivided into: (1) **episodic memory** — records of specific past events or trajectories (what the agent did in a prior session, past tool outputs); (2) **semantic memory** — factual knowledge (a vector database of documents, a knowledge graph); (3) **procedural memory** — learned skills or workflows, often encoded as few-shot examples or fine-tuned model weights. Access requires an explicit retrieval step (embedding similarity search, keyword lookup, or structured query), which introduces latency and retrieval errors. Long-term memory is persistent across sessions and scales to billions of tokens, but the retrieval step is a bottleneck and a source of failure.

---

### Q11 [Basic] How does Retrieval-Augmented Generation function as agent memory, and what are its limitations?

**Q:** Describe the RAG pipeline as an agent memory mechanism and explain the cases where it fails.

**A:** Retrieval-Augmented Generation (RAG, Lewis et al., 2020) adds external semantic memory to an LLM by prepending retrieved documents to the generation context. The pipeline has three stages: (1) **indexing** — documents are split into chunks, encoded into dense vectors by an embedding model, and stored in a vector database; (2) **retrieval** — given the current query (or a rewritten version of it), an encoder (which may be a separate query encoder as in DPR, or a shared encoder in modern implementations) encodes the query and the $k$ most similar chunks are retrieved via approximate nearest-neighbor search; (3) **generation** — retrieved chunks and the original query are concatenated into the prompt, and the LLM generates a response grounded in the retrieved context.

In the agent context, RAG serves as the agent's long-term semantic memory: tool call results, prior conversation summaries, external knowledge bases, and task-relevant documents can all be indexed and retrieved on demand, allowing the agent to access information beyond its context window.

RAG fails in four principal scenarios. First, **retrieval failure**: if the query is ambiguous, misleading, or the relevant document uses different terminology (vocabulary mismatch), the correct chunk is not retrieved. Second, **multi-hop reasoning**: questions requiring synthesis across multiple documents may not be answerable from a fixed-$k$ retrieval, because the first retrieved document may not contain the bridging entity needed to formulate a retrieval query for the second. Third, **temporal staleness**: the index is a snapshot; if the underlying data changes, retrieval returns outdated information. Fourth, **context length pressure**: retrieving many chunks quickly fills the context window, forcing a trade-off between retrieval breadth and remaining space for reasoning.

---

### Q12 [Advanced] How does MemGPT address the context window limitation for long-term agent memory?

**Q:** Describe MemGPT's hierarchical memory architecture and the mechanism by which it manages information across context window boundaries.

**A:** MemGPT (Packer et al., 2023) reimagines the LLM agent as an operating system process with managed memory tiers, directly analogous to OS virtual memory. The core problem it addresses is that long-running agents accumulate more information than fits in the context window, but naive truncation discards potentially critical earlier context.

MemGPT defines two memory tiers. **In-context memory** is divided into a fixed system prompt region (agent identity, core instructions, and a "persona" description), a working context region (the last $n$ Thought–Action–Observation steps), and a FIFO message queue (recent user–agent exchanges). **External memory** stores all prior events, documents, and conversation history outside the context window, indexed for retrieval.

The key mechanism is **memory management functions** — a set of tools exposed to the agent that allow it to explicitly manage its own memory: `core_memory_append` and `core_memory_replace` update the in-context persona and facts; `archival_memory_insert` and `archival_memory_search` write to and retrieve from external storage; `conversation_search` retrieves prior conversation segments by keyword or recency. The agent is prompted to call these functions proactively — before an important piece of context scrolls out of the window, the agent writes it to archival memory, and retrieves it later when relevant.

This is analogous to OS page swapping: the agent controls what occupies its limited "RAM" (context window) by explicitly moving information between fast in-context storage and slow external storage. The overhead is additional LLM calls for memory management, but the benefit is that the agent can maintain coherent long-horizon tasks across unlimited context without losing critical state.

---

### Q13 [Advanced] How are episodic and semantic memory implemented in LLM agents, and what retrieval strategies does each require?

**Q:** Distinguish episodic from semantic memory in the agent setting, describe concrete implementations, and explain why each requires a different retrieval approach.

**A:** In cognitive science, episodic memory stores specific experienced events (what happened, when, and in what context), while semantic memory stores general factual knowledge independent of any specific episode. This distinction maps cleanly to LLM agent architectures and has important consequences for implementation and retrieval.

**Episodic memory** in agents records specific past trajectories: full Thought–Action–Observation sequences from prior sessions, past errors and corrections (as in Reflexion), or records of specific user interactions. The critical property is temporal and contextual specificity — the agent needs to retrieve "what I did the last time I encountered a task like this" rather than "what is generally true about this domain." Retrieval is therefore best served by **recency-weighted similarity search** (recent episodes are more relevant than old ones), **task similarity matching** (embedding the current task and finding episodes with similar embeddings), or **structured lookup** (if episodes are tagged with task type, entity names, or success/failure labels). Generative Agents (Park et al., 2023) implement episodic memory as a stream of natural language "memories," retrieved by weighting recency, importance (scored by the LLM), and relevance to the current context. The three signals are each normalized to [0, 1] and combined as a weighted sum with equal weights to produce a final retrieval score — a recent but unimportant memory can be outranked by an older but highly relevant one.

**Semantic memory** stores factual knowledge: a vector database of domain documents, a knowledge graph of entities and relations, or a code library of reusable functions. The content is independent of when or how it was acquired. Retrieval is best served by **dense retrieval** (embedding similarity between the query and document chunks, as in RAG), **sparse retrieval** (BM25 keyword matching, effective when terminology is domain-specific and exact terms matter), or **hybrid retrieval** (combining dense and sparse scores, the current best practice for production RAG systems). Knowledge graphs additionally support **multi-hop retrieval**: structured graph traversal can answer relational queries that embedding similarity cannot, by following entity links across the graph.

The practical challenge is that most agents need both: they must recall prior experiences (episodic) and look up factual knowledge (semantic), and the two should be stored and retrieved separately to avoid interference. A single undifferentiated vector store conflating episodes and facts degrades retrieval precision for both.

---

## Multi-Agent Systems

### Q14 [Basic] What motivates multi-agent systems over single-agent designs?

**Q:** Why are multi-agent systems used, and what specific limitations of single-agent architectures do they address?

**A:** Single-agent architectures face three fundamental limitations that motivate multi-agent designs. First, **context length saturation**: a single agent handling a large, complex task must carry all task state — instructions, tool outputs, intermediate results, domain knowledge — in one context window. As the task grows, this window fills, degrading effective attention on earlier content and eventually making the task intractable. Multi-agent systems decompose the task across multiple context windows, each carrying only the state relevant to its sub-problem.

Second, **lack of specialization**: a single generalist agent must perform all subtasks — code generation, web search, document summarization, mathematical reasoning — with the same prompt configuration. Different subtasks benefit from different system prompts, few-shot examples, and tool sets. Multi-agent systems assign specialized agents to subtasks, with each agent's prompt, tools, and context optimized for its role (analogous to the division of labor in human organizations).

Third, **absence of cross-validation**: a single agent cannot independently verify its own outputs. Multi-agent systems enable **adversarial collaboration** — one agent generates a solution while another critiques it (as in multi-agent debate frameworks). This debate or peer-review dynamic improves factual accuracy and robustness on tasks where self-critique by the same model is limited by shared failure modes. Experiments in Du et al. (2023) show that multi-agent debate consistently improves factual accuracy and mathematical reasoning over single-model self-consistency.

---

### Q15 [Basic] How does AutoGen implement multi-agent collaboration?

**Q:** Describe AutoGen's ConversableAgent abstraction, how conversations between agents are orchestrated, and what types of workflows it supports.

**A:** AutoGen (Wu et al., 2023) is a framework for building LLM applications as networks of **ConversableAgents** — agents that can send and receive messages and trigger callable functions (tools) in response. Every agent in AutoGen is a ConversableAgent, configured with a name, a system prompt defining its role, a list of callable tools, and a policy for when to terminate the conversation or hand off to another agent.

The central orchestration primitive is a **two-agent conversation**: a UserProxyAgent (representing the human or a programmatic trigger) initiates a conversation with an AssistantAgent (an LLM-backed agent). Messages flow back and forth as chat turns. Tool calls from the AssistantAgent are intercepted by the UserProxyAgent, executed locally, and the result is returned as the next message. This simple two-agent loop handles a wide range of agentic tasks, including code generation, execution, and debugging.

For more complex workflows, AutoGen supports **GroupChat**: multiple ConversableAgents are placed in a shared conversation managed by a GroupChatManager, which selects which agent speaks next (via round-robin, a speaker selection LLM, or a custom policy). This enables parallel specialization: a PlannerAgent decomposes a task, a CoderAgent implements it, a ReviewerAgent checks correctness, and a CriticAgent challenges assumptions — all within one conversation thread. AutoGen also supports nested chats (an agent's tool call triggers a sub-conversation with another agent), enabling hierarchical task decomposition. The framework's key design choice is treating conversation as the universal interface between agents, which makes it flexible but also means all state is passed through natural language messages rather than structured APIs.

---

### Q16 [Advanced] What are the principal multi-agent communication topologies and their trade-offs?

**Q:** Describe the main topologies used to organize inter-agent communication in multi-agent systems, and analyze the trade-offs in scalability, reliability, and task suitability.

**A:** Multi-agent communication topologies determine how information flows between agents and which agents can initiate communication with which others.

**Centralized (star) topology** routes all communication through a single orchestrator agent: every worker agent sends outputs to the orchestrator, which synthesizes them and issues the next instruction. This is the most common topology in practice (AutoGen's GroupChat with a manager, LangGraph's supervisor pattern). Advantages: the orchestrator maintains a global view of task state, enabling coherent multi-step plans; bottlenecks and redundant work are easy to detect. Disadvantages: the orchestrator is a single point of failure (a reliability risk — if it misunderstands or loses track of task state, the entire workflow fails), its context window carries the entire task state, and all latency paths go through it.

**Decentralized (peer-to-peer) topology** allows any agent to communicate directly with any other. This enables parallel processing without a bottleneck and supports emergent coordination patterns. Generative Agents (Park et al., 2023) use this topology for social simulation. The challenge is consistency: without a central coordinator, agents can develop conflicting beliefs about task state, producing redundant work or contradictory outputs. This topology suits simulations and open-ended exploration where loose coordination is acceptable, not tight goal-directed tasks.

**Hierarchical topology** organizes agents in a tree: a top-level orchestrator delegates to mid-level managers, which delegate to specialist workers. MetaGPT and HuggingGPT use hierarchical designs. This scales naturally — each layer's context window only needs to track its sub-tree — and mirrors how human organizations decompose complex projects. The trade-off is communication latency (information must traverse multiple levels) and rigidity: the hierarchy must be designed in advance, and tasks that don't fit the predefined decomposition are handled poorly.

**Shared message bus** topology has all agents subscribe to and publish on a common channel (e.g., a shared scratchpad or blackboard). Any agent can read all prior messages and post new ones. This maximizes information sharing but scales poorly (all agents must process all messages) and produces unstructured communication that is hard to audit.

---

### Q17 [Advanced] How does MetaGPT use role specialization and structured outputs for multi-agent software development?

**Q:** Describe MetaGPT's architecture, its SOP-driven workflow, and how structured outputs reduce communication errors between agents.

**A:** MetaGPT (Hong et al., 2023) frames software development as a multi-agent collaboration governed by a **Standard Operating Procedure (SOP)** — a predefined workflow that mirrors how human software teams work. The system instantiates specialized agents for canonical software development roles: ProductManager (writes a Product Requirements Document from the user's one-sentence idea), Architect (produces a system design and API specifications), ProjectManager (decomposes the design into tasks and assigns them to engineers), Engineer (writes code for assigned tasks), QAEngineer (writes and executes tests). Each agent's system prompt encodes the responsibilities and deliverables of its role.

The critical innovation is **structured output schemas**. Each agent communicates exclusively through predefined document templates: the ProductManager outputs a PRD in a fixed Markdown schema, the Architect outputs a design document with mandatory sections (tech stack, data structures, API signatures), and so on. Downstream agents parse these structured documents rather than interpreting free-form prose, dramatically reducing the ambiguity and hallucination in inter-agent communication. This is analogous to typed function interfaces in software: well-defined schemas catch mismatches early and make the communication contract explicit.

**Action-observation subscriptions** further structure the information flow: each agent subscribes to the outputs of the agents whose work it depends on, and is triggered when those outputs are available. This creates a data-flow graph (ProductManager → Architect → ProjectManager → Engineer → QAEngineer) with explicit dependencies and no unnecessary communication. On MetaGPT's internal SoftwareDev benchmark (measuring executability and completeness of multi-file software projects), the structured SOP workflow produces substantially more complete software artifacts than single-agent GPT-4, demonstrating that role specialization and structured communication meaningfully improve output quality for complex multi-step software generation.

---
