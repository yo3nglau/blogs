---
title: "LLM Inference & Serving: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-10'
categories:
  - Interview
tags:
  - Deep Learning
  - Large Language Models
  - Systems
toc: true
---

## KV Cache and Memory Management

### Q1 [Basic] Explain why KV cache is necessary for autoregressive decoding

**Q:** Why does autoregressive generation require caching, and what exactly is stored in a KV cache?

**A:** In autoregressive decoding the model generates one token at a time, each conditioned on all previous tokens. Without caching, generating token $t$ requires recomputing keys and values for all $t-1$ preceding tokens—an $O(t^2)$ cost over the full sequence. The KV cache stores the key and value tensors produced by each attention layer for every previously generated token so that, at step $t$, only the new token's query, key, and value are computed; cached keys and values are retrieved and concatenated before attention is applied. Memory footprint per token per layer is $2 \cdot d_{head} \cdot n_{heads}$ in float16, which for a 7B-parameter model with 32 layers and 32 heads at $d_{head} = 128$ amounts to roughly $0.5$ MB per token—enough that long sequences or large batches can exhaust GPU memory.

The KV cache enables autoregressive decoding to run in $O(t)$ time per step, turning the quadratic total cost into linear in the generated length. The trade-off is memory: as batch size $B$ and sequence length $T$ grow, the cache consumes $O(B \cdot T \cdot L \cdot d_{model})$ bytes, often rivaling or exceeding the model weights themselves.

---

### Q2 [Advanced] Describe how PagedAttention manages KV cache memory

**Q:** What is PagedAttention, and how does it address the memory fragmentation problem in LLM serving?

**A:** Traditional KV cache implementations pre-allocate a contiguous memory block of size $\text{max\_seq\_len}$ for each request at the start of inference. Because actual generation lengths vary and are unknown in advance, this causes two types of waste: internal fragmentation (padding within the reserved block) and external fragmentation (gaps between blocks that cannot be reused). Studies of production systems found that over 60% of KV cache memory was wasted this way (Kwon et al., 2023).

PagedAttention, introduced in vLLM (Kwon et al., 2023), borrows the virtual memory and paging abstraction from operating systems. KV cache memory is divided into fixed-size physical blocks (pages), each holding the keys and values for a small number of tokens (e.g., 16). A per-request logical-to-physical block table maps contiguous logical positions to arbitrary physical pages, exactly like a page table in virtual memory. Pages are allocated on demand as the sequence grows and reclaimed when the request finishes. The modification to attention computation is minor: within-block attention is contiguous; cross-block attention follows the indirection through the block table.

This approach reduces memory waste to under 4% in experiments on LLaMA and OPT models (Kwon et al., 2023). A secondary benefit is copy-on-write sharing: parallel sampling (beam search, best-of-$N$) or prefix caching can share physical pages without duplication until a divergence point.

---

### Q3 [Advanced] Explain prefix caching and the radix attention mechanism

**Q:** How does prefix caching work in LLM serving systems, and what data structure does SGLang use to manage shared prefixes efficiently?

**A:** When multiple requests share a common prompt prefix—system prompts in chatbots, few-shot examples, or repeated document context in RAG—recomputing the KV cache for the shared prefix is wasteful. Prefix caching stores the computed KV blocks for known prefixes so that a new request can skip the prefill phase for the cached portion and start generation from where the cache ends. The challenge is organizing cached entries so that partial matches, branching continuations, and eviction under memory pressure are handled efficiently.

SGLang (Zheng et al., 2023) introduces **radix attention**, which organizes the KV cache as a radix tree (compressed trie) keyed on token sequences. Each internal node represents a shared prefix; each edge corresponds to a sequence of tokens. When a new request arrives, SGLang traverses the tree to find the longest cached prefix, reuses those KV blocks directly, and extends the tree with newly computed blocks. Eviction under LRU policy removes leaf nodes first, preserving longer shared prefixes as long as possible.

Radix attention achieves high cache hit rates across diverse workload types: multi-turn chat (each turn appends to the previous), batch inference with shared system prompts, and tree-of-thought sampling with branching continuations. Experiments show 1.7–8$\times$ throughput improvement over systems without prefix caching depending on the workload (Zheng et al., 2023).

---

### Q4 [Advanced] Compare KV cache eviction strategies for long-context inference

**Q:** When the KV cache cannot fit the full context into GPU memory, what eviction strategies exist, and how do they decide which tokens to drop?

**A:** Full-context KV caches for sequences exceeding tens of thousands of tokens can exhaust GPU memory. The key question is: which past tokens' KV entries can be dropped with least impact on generation quality?

**H2O** (Heavy-Hitter Oracle; Zhang et al., 2023) observes that a small fraction of tokens accumulate disproportionately large cumulative attention scores—"heavy hitters"—and that these tokens are consistently attended to across layers and generation steps. H2O maintains a fixed budget of KV slots by evicting the entry with the lowest cumulative attention score whenever the cache is full. On tasks such as text summarization and question answering, H2O with 20% cache retention matches full-cache quality while reducing peak memory by $5\times$ (Zhang et al., 2023).

**SnapKV** (Li et al., 2024) takes a different approach based on the observation that attention patterns over long documents stabilize after the initial "observation window" of tokens immediately preceding the generated answer. SnapKV clusters the key vectors of the observation window, identifies representative key positions via pooled attention, and retains only the KV entries for those positions plus the full observation window. This avoids rerunning the model: the compression decision is made once in one forward pass, making it suitable for prefill-time caching.

Both strategies accept a quality-memory trade-off. Full eviction of low-importance tokens can degrade performance on tasks requiring needle-in-a-haystack retrieval; combining eviction with compression (quantizing retained entries to INT4) is an emerging direction.

---

## Efficient Attention Mechanisms

### Q5 [Basic] Describe how FlashAttention achieves memory efficiency

**Q:** What is the core algorithmic insight behind FlashAttention that allows it to avoid materializing the full attention matrix?

**A:** Standard attention computes the $N \times N$ score matrix $QK^T / \sqrt{d}$, applies softmax, and multiplies by $V$—requiring $O(N^2)$ memory to store the intermediate matrix. For long sequences this quickly exceeds SRAM capacity and forces round-trips between HBM (GPU global memory) and the compute units, making attention memory-bandwidth-bound rather than compute-bound.

FlashAttention (Dao et al., 2022) applies **tiling**: query, key, and value matrices are split into blocks that fit in SRAM. For each query block, FlashAttention iterates over all key/value blocks, computing partial attention outputs and accumulating a numerically stable running softmax using the online softmax algorithm (tracking the running maximum and normalizer). The final output for each query tile is emitted to HBM once, without ever writing the full $N \times N$ matrix. Memory complexity drops from $O(N^2)$ to $O(N)$, and the number of HBM reads/writes falls by a factor of $N / d$. On A100 GPUs, FlashAttention achieves 2–4$\times$ wall-clock speedup over PyTorch standard attention for sequence lengths of 1K–16K, and enables training with sequences up to 64K tokens (Dao et al., 2022).

---

### Q6 [Advanced] What improvements do FlashAttention-2 and FlashAttention-3 introduce?

**Q:** How do FlashAttention-2 and FlashAttention-3 improve upon the original FlashAttention, and what hardware features do they exploit?

**A:** **FlashAttention-2** (Dao, 2023) targets two bottlenecks in the original: excessive non-matmul operations and poor parallelism. The original FA1 performed redundant rescaling of partial sums at every key block; FA2 eliminates this by delaying the final scaling to the output write. Parallelism is improved by partitioning work across query blocks (not just KV blocks) so that forward and backward passes on multi-head attention fully utilize GPU thread blocks with no idle warps. FA2 achieves roughly $2\times$ the throughput of FA1 on A100 and is integrated into PyTorch 2.0 as `F.scaled_dot_product_attention`.

**FlashAttention-3** (Shah et al., 2024) targets the Hopper (H100/H200) GPU architecture specifically. H100 introduces Warp Group Matrix Multiply-Accumulate (WGMMA) instructions and the Tensor Memory Accelerator (TMA) for asynchronous data movement. FA3 overlaps GEMM computation with softmax using a two-stage pipeline: while one warp group computes partial matrix products, another concurrently executes the softmax rescaling for the previous tile. Additionally, FA3 exploits H100's FP8 tensor cores, achieving near-peak hardware throughput. On H100 SXM5, FA3 reaches approximately $75\%$ of theoretical peak FLOPs for FP16 attention, compared to roughly $35\%$ for FA2 (Shah et al., 2024).

---

### Q7 [Basic] Compare Multi-Head Attention, Multi-Query Attention, and Grouped-Query Attention

**Q:** What are Multi-Query Attention and Grouped-Query Attention, and why are they preferred in modern LLMs for inference?

**A:** Standard **Multi-Head Attention** (MHA) uses $H$ independent query, key, and value projections. During inference, all $H$ KV heads must be cached, giving KV cache size proportional to $H$. This is memory-bandwidth-intensive: for each decode step, the GPU must load the full KV cache from HBM.

**Multi-Query Attention** (MQA; Shazeer, 2019) reduces this by sharing a single KV head across all $H$ query heads. KV cache memory shrinks by a factor of $H$, and memory bandwidth for decoding decreases proportionally. The trade-off is quality: with only one KV representation, expressiveness is reduced, and models sometimes need retraining to recover accuracy.

**Grouped-Query Attention** (GQA; Ainslie et al., 2023) interpolates between MHA and MQA by grouping query heads into $G$ groups, each sharing one KV head. With $G = 1$ this is MQA; with $G = H$ this is MHA. GQA with $G = 8$ recovers nearly all of MHA's quality while achieving MQA-level memory bandwidth savings. It is the default in LLaMA-2, LLaMA-3, Mistral, Gemma, and most modern LLMs. Notably, Ainslie et al. (2023) show that a pretrained MHA model can be converted to GQA by mean-pooling the original KV heads within each group, followed by brief fine-tuning—avoiding training from scratch.

---

### Q8 [Advanced] How do sparse and linear attention methods scale beyond quadratic cost?

**Q:** What approaches reduce attention complexity from $O(N^2)$ to sub-quadratic, and what are their practical trade-offs?

**A:** Standard attention is $O(N^2)$ in both time and memory, which becomes prohibitive for contexts of hundreds of thousands of tokens. Two major directions address this.

**Sparse attention** restricts each query to attend only to a subset of keys. Patterns include local sliding windows (each token attends to its $w$ neighbors), global tokens (a small set of special tokens attend everywhere), and strided patterns. Sliding-window attention achieves $O(N \cdot w)$ complexity and is effective for tasks with local structure; global tokens recover long-range dependencies for task-critical positions. In practice, sparse attention is most useful when document structure is locally organized but may miss long-range dependencies that full attention captures.

**Linear attention** approximates softmax attention by decomposing the kernel: $\text{softmax}(QK^T)V \approx \phi(Q)(\phi(K)^T V)$, reordering matrix multiplication to compute $\phi(K)^T V \in \mathbb{R}^{d \times d}$ first, yielding $O(N \cdot d^2)$ cost. Linear attention generalizes naturally to a recurrent formulation, enabling $O(1)$ per-step decoding at inference time. The key limitation is approximation quality: random feature maps used to construct $\phi$ introduce variance, and for tasks requiring precise key matching the gap from exact softmax attention can be significant.

State space models such as Mamba achieve similar sub-quadratic inference cost through selective recurrence rather than approximating attention, and have shown competitive quality at scale—though they require training from scratch rather than extending existing Transformer checkpoints.

---

## Speculative Decoding

### Q9 [Basic] Explain the draft-and-verify framework in speculative decoding

**Q:** How does speculative decoding accelerate autoregressive generation without changing the output distribution?

**A:** In standard autoregressive decoding, each new token requires a full forward pass through the large target model, making generation latency proportional to the number of tokens. Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) leverages the observation that much of the autoregressive output is predictable: a smaller, faster **draft model** proposes $k$ candidate tokens in sequence, then the large **target model** evaluates all $k$ tokens in a single parallel forward pass.

The target model's parallel pass produces logits at each of the $k$ positions. Each draft token is accepted or rejected by comparing the draft model's probability $q_i$ with the target model's probability $p_i$ at that position: token $i$ is accepted with probability $\min(1, p_i / q_i)$. If accepted, the next draft token is evaluated; if rejected, a corrected token is sampled from an adjusted distribution $(p_i - q_i)^+$ and generation continues from that point. This **rejection sampling** procedure guarantees that the marginal distribution over accepted tokens is identical to what the target model would have produced (Leviathan et al., 2023). Efficiency gain arises because target model evaluation is parallelized over $k$ positions while costing approximately the same compute as a single decode step.

---

### Q10 [Advanced] Analyze the conditions under which speculative decoding provides the largest speedup

**Q:** What factors determine the acceptance rate in speculative decoding, and under what conditions does the method fail to accelerate generation?

**A:** The expected speedup of speculative decoding with $k$ draft tokens scales with $\mathbb{E}[\text{accepted tokens per target forward pass}]$. This expectation depends on the **acceptance rate** $\alpha = \mathbb{E}[\min(1, p_i/q_i)]$ at each position. If $\alpha \approx 1$ (draft and target distributions closely agree), nearly all $k$ tokens are accepted and throughput approaches $k$-fold over single-token decoding. If $\alpha \approx 0$, the draft is rejected almost immediately, and speculative decoding adds overhead without benefit.

Acceptance rate is high when (1) the output is predictable or repetitive (boilerplate text, structured formats, code with repeated patterns), (2) the draft model is a closely distilled or architecturally similar smaller version of the target, and (3) temperature is low (near-greedy decoding). Acceptance rate is low when (1) the task requires creative or diverse generation (high temperature sampling), (2) the draft model is architecturally dissimilar or poorly matched to the target's distribution, or (3) the context requires long-range reasoning the draft model cannot replicate.

Another consideration is the batching regime. Speculative decoding helps most in the **latency-optimized, small-batch setting** where the target model is memory-bandwidth-bound. At large batch sizes, the target model is already compute-bound, and the overhead of verifying $k$ tokens per target pass can reduce overall throughput. Systems like vLLM support speculative decoding but recommend it primarily for batch size 1–4 (Kwon et al., 2023).

---

### Q11 [Advanced] Describe Medusa's multi-head draft approach

**Q:** How does Medusa generate draft tokens without a separate draft model, and what is tree attention?

**A:** Medusa (Cai et al., 2024) eliminates the separate draft model by attaching multiple lightweight **decoding heads** directly to the frozen LLM. Each additional head $i$ is a two-layer MLP that predicts the token at offset $+i$ from the current hidden state. Head 1 predicts the next token (the standard LM head), head 2 predicts the token after that, and so on. These extra heads are trained on a small fine-tuning dataset with cross-entropy loss while the base model weights remain fixed.

Because each Medusa head independently predicts its offset token, there is no sequential dependency between heads: all $k$ predictions are produced in a single forward pass of the base model. The combined predictions form a **candidate tree**: head 1 produces top-$s_1$ candidates, head 2 produces top-$s_2$ candidates for each head-1 candidate, and so on, generating $\prod s_i$ candidate sequences. To evaluate all candidates in parallel, Medusa constructs a **tree attention mask**—a causal mask that allows each leaf node to attend only to its own ancestor path. The same base model (with its standard LM head acting as the verifier) evaluates the entire tree in one forward pass and accepts the longest prefix consistent with its greedy or sampled choices.

Medusa achieves 2–3$\times$ speedup on Vicuna-7B/13B (Cai et al., 2024) without draft model alignment overhead, at the cost of slightly lower acceptance rates compared to a well-matched external draft model.

---

### Q12 [Advanced] Compare lookahead decoding with retrieval-based and self-speculative approaches

**Q:** What alternatives to draft-model speculative decoding exist, and how do they avoid the need for a separate model?

**A:** Several approaches decouple draft generation from a separate model entirely.

**Lookahead decoding** (Fu et al., 2024) adapts the Jacobi iterative method to autoregressive generation. Instead of generating tokens one at a time, it maintains a $W \times N$ "lookahead window" of speculative future tokens that are refined in parallel Jacobi iterations. In each step, the model evaluates all positions in the window simultaneously, updating each with the token it would predict given its current neighbors. Newly consistent $n$-gram sequences ("Jacobi trajectories") are committed to an $n$-gram pool. When a generated prefix matches an $n$-gram in the pool, that sequence is proposed as a draft and verified in one pass. Lookahead decoding requires no additional parameters or training and is especially effective for long-form generation where $n$-gram reuse is high (Fu et al., 2024).

**Self-speculative decoding** reduces the target model itself for drafting by skipping certain layers during the draft phase. The same model, run with a subset of layers, produces a reasonable approximation of the full model's distribution sufficient to generate draft tokens with non-trivial acceptance rates. This avoids any additional model storage or alignment cost. The trade-off is that layer-skipping perturbs internal representations, limiting acceptance rates to moderate values.

**Retrieval-based speculation** indexes previously decoded sequences and retrieves the most likely continuation from a datastore rather than running any model for the draft, reducing draft latency to near zero. Acceptance depends on exact or near-exact matches in the datastore, making this effective for repetitive workloads (code generation, document completion) but unreliable for open-ended generation.

---

## Batching, Scheduling, and Serving

### Q13 [Basic] Explain continuous batching and why it outperforms static batching

**Q:** What is continuous batching, and how does it improve GPU utilization compared to static batching for LLM serving?

**A:** In **static batching**, the serving system groups a fixed set of requests into a batch, runs the batch until all sequences complete, and only then accepts new requests. Because different sequences generate different numbers of tokens, short requests finish early but their GPU slots remain occupied (filled with padding) until the last sequence in the batch terminates. GPU utilization is therefore bounded by the longest sequence, and newly arriving short requests must queue even when compute is available.

**Continuous batching** (also called iteration-level scheduling; Yu et al., 2022) allows completed sequences to be evicted and new requests inserted at every decode step rather than at batch boundaries. The serving system maintains an active set of sequences; after each token generation step, any sequence that has produced an end-of-sequence token is removed and a waiting request is immediately admitted. Because each decode step is independent (the KV cache carries forward state), the batch composition can change at every iteration with minimal overhead.

Orca (Yu et al., 2022), the system that popularized this technique, demonstrated up to $23\times$ higher throughput and significantly lower tail latency compared to a static batching baseline on LLaMA-scale models. Modern serving systems including vLLM, TGI (Hugging Face), and TensorRT-LLM all implement continuous batching as a core feature.

---

### Q14 [Advanced] Describe chunked prefill and prefill-decode disaggregation

**Q:** What problem do chunked prefill and disaggregated prefill-decode architectures solve, and how do they differ in their approach?

**A:** In standard serving, **prefill** (processing the input prompt in parallel) and **decode** (generating output tokens one at a time) share the same GPU. Prefill for a long prompt is compute-intensive and can take hundreds of milliseconds, during which all ongoing decode requests are blocked—creating "prefill stalls" that inflate time-to-first-token (TTFT) and increase decode latency for co-located requests.

**Chunked prefill** (Agrawal et al., 2024) splits a long prefill into smaller chunks and interleaves these chunks with decode steps from other requests. Each iteration processes one prefill chunk plus the decode step for all active sequences. This limits the maximum blocking time per step to the chunk duration, smoothing TTFT without sacrificing decode throughput. Chunked prefill is particularly effective when TTFT SLO is tight but all computation remains on the same GPU pool.

**Disaggregated prefill-decode** (Zhong et al., 2024) goes further by physically separating prefill and decode onto different GPU instances. Prefill machines receive requests, compute KV caches, and transfer the resulting cache tensors to dedicated decode machines that run continuous batching. This separation exploits the fact that prefill is arithmetic-intensity-bound (benefits from high-FLOP GPUs) while decode is memory-bandwidth-bound (benefits from large HBM capacity), allowing independent scaling and SLO targeting. DistServe reports $2.4\times$ higher goodput at the same TTFT SLO on LLaMA-65B compared to co-located serving (Zhong et al., 2024), with the trade-off of KV cache transfer latency over the interconnect.

---

### Q15 [Basic] Describe the architecture of vLLM and its key design decisions

**Q:** What are the main components of the vLLM serving system, and how do they interact to achieve high throughput?

**A:** vLLM (Kwon et al., 2023) is organized around three interacting components: a **scheduler**, a **KV cache manager**, and a set of **model execution workers**.

The **scheduler** implements continuous batching with a priority queue. At each iteration it selects from waiting, running, and swapped request queues. If GPU memory is insufficient for a new request, the scheduler can preempt a running request by swapping its KV blocks to CPU memory and resuming it later. The **KV cache manager** implements PagedAttention: it maintains a pool of fixed-size physical KV blocks and a per-request block table. On request arrival, blocks are allocated on demand; on completion, blocks are freed. For parallel sampling (e.g., best-of-$N$), copy-on-write shares prompt KV blocks across outputs.

**Model execution workers** are spawned as separate processes (one per GPU for tensor-parallel inference). The scheduler sends a `SchedulerOutput` containing the current batch and physical block mappings; each worker runs a forward pass using custom PagedAttention CUDA kernels that dereference the block table during attention computation. Results are gathered and returned to the scheduler for token sampling. This architecture achieves 2–4$\times$ higher throughput than HuggingFace text generation with naïve KV management (Kwon et al., 2023), primarily by eliminating memory fragmentation.

---

### Q16 [Advanced] Analyze scheduling policies for LLM serving under latency SLOs

**Q:** How should a serving system schedule requests to meet latency SLOs, and what are the trade-offs between FCFS, SJF, and preemptive scheduling?

**A:** LLM serving must balance two objectives: maximizing throughput (requests per second) and meeting per-request latency SLOs, typically expressed as time-to-first-token (TTFT) and time-per-output-token (TPOT).

**First-Come-First-Served (FCFS)** admits requests in arrival order. It is fair but suboptimal under load: a long-prefill request can block many short requests behind it, inflating their TTFT. FCFS is common in practice for its simplicity and predictability.

**Shortest Job First (SJF)** prioritizes requests with shorter estimated prefill length or total output length, reducing average waiting time. The challenge is that output length is unknown at admission time. Some systems use a length predictor trained on request features; others approximate with prefill length. SJF reduces average TTFT but can starve long requests under heavy load.

**Preemptive scheduling** in vLLM allows the system to evict an active sequence (swapping its KV blocks to CPU) when a higher-priority request arrives, enabling low-latency response for short requests. The swap cost—KV transfer to CPU HBM over PCIe—can reach several hundred milliseconds for long contexts, so preemption is only beneficial if the arriving request is substantially shorter than the evicted one.

The chunked prefill mechanism from Sarathi-Serve (Agrawal et al., 2024) interacts favorably with scheduling: by bounding prefill chunk size, even FCFS achieves predictable TTFT bounds, reducing the need for complex preemptive policies.

---

## Quantization and Distributed Inference

### Q17 [Basic] Describe the main post-training quantization approaches for LLM inference

**Q:** What are GPTQ and AWQ, and how do they achieve accurate 4-bit weight quantization for large language models?

**A:** Post-training quantization (PTQ) reduces model weights from 16-bit floating point to lower precision (typically INT4) without full retraining, enabling larger models to fit in GPU memory and reducing memory bandwidth during decode.

**GPTQ** (Frantar et al., 2023) applies the Optimal Brain Quantization (OBQ) framework layer by layer. For each linear layer, GPTQ solves a weight quantization problem that minimizes the increase in layer output error using the Hessian of the layer's input distribution. The key approximation is processing weights column by column and propagating quantization error to remaining unquantized columns via a closed-form update, making the procedure efficient enough to apply to 175B-parameter models in a few GPU-hours. GPTQ achieves near-FP16 perplexity at 4 bits and near-INT8 quality at 3 bits.

**AWQ** (Activation-aware Weight Quantization; Lin et al., 2024) observes that channels corresponding to large-magnitude activations are much more sensitive to quantization error. AWQ identifies the top 1% of weight channels by activation scale and applies a per-channel scaling transformation that effectively shifts quantization difficulty from the sensitive weights to less-sensitive ones, without changing the mathematical output. The result is a hardware-friendly INT4 quantization (no mixed precision) that outperforms GPTQ on most benchmarks at equivalent bit-width, particularly for instruction-following and reasoning tasks (Lin et al., 2024).

---

### Q18 [Advanced] Explain the challenges of weight-activation quantization and how SmoothQuant addresses them

**Q:** Why is quantizing activations harder than quantizing weights in LLMs, and what is the SmoothQuant transformation?

**A:** Weight quantization (W4A16 or W4A8) is relatively tractable because weight distributions are approximately bell-shaped and per-channel scaling is straightforward. **Activation quantization** to INT8 or INT4 is harder because LLM activations exhibit **outliers**: a small fraction of channels have magnitudes $10\text{–}100\times$ larger than typical channels. Naïve per-tensor INT8 quantization clips these outliers severely or wastes bits on a large dynamic range, causing significant accuracy degradation.

**SmoothQuant** (Xiao et al., 2023) addresses this with a mathematically equivalent migration of quantization difficulty. For a linear layer $Y = XW$, activation outliers are concentrated in specific channels. SmoothQuant introduces a per-channel scaling vector $s$ and rewrites the computation as:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X}\hat{W}$$

where $\hat{X} = X / s$ has reduced outlier magnitude and $\hat{W} = sW$ has correspondingly larger values in those channels. The scaling factor $s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$ with $\alpha \in [0, 1]$ balances the migration. At $\alpha = 0.5$ activation and weight difficulty are equalized. Because $\hat{W}$ is fixed after calibration, the per-channel scaling is absorbed into the weight matrix with no runtime overhead. SmoothQuant enables W8A8 quantization on LLaMA-65B with less than 1% perplexity degradation and achieves $1.56\times$ speedup and $2\times$ memory reduction compared to FP16 inference (Xiao et al., 2023).

---

### Q19 [Basic] Describe tensor parallelism and pipeline parallelism for distributed LLM inference

**Q:** How are tensor parallelism and pipeline parallelism applied to serve large language models across multiple GPUs?

**A:** When a model is too large to fit on a single GPU, or when serving latency requirements demand distributing computation, two primary strategies partition the model across devices.

**Tensor parallelism** (Shoeybi et al., 2019) splits individual weight matrices across GPUs. In a Transformer layer, the column-parallel–row-parallel pattern is standard: the first linear layer (e.g., $W_Q$, $W_K$, $W_V$, or the MLP up-projection) is split column-wise across $P$ GPUs, each computing a partition of the output; the second layer (output projection or MLP down-projection) is split row-wise, and an all-reduce synchronizes partial sums across GPUs before the result is passed to the next layer. This requires two all-reduce operations per Transformer layer and therefore benefits from high-bandwidth interconnects (NVLink). Tensor parallelism reduces per-GPU memory proportionally to $P$ and cuts latency roughly by $P$ plus communication overhead.

**Pipeline parallelism** assigns different layers to different GPUs; activations flow sequentially through stages. Naïve pipeline parallelism creates "bubbles" (idle time) while earlier stages wait; this is mitigated by micro-batching (sending multiple micro-batches through the pipeline concurrently). For inference at low batch sizes, pipeline parallelism is less efficient because the bubble fraction is $(P-1)/P$ for a single request. In practice, tensor parallelism is preferred for small-batch low-latency serving, while pipeline parallelism is used when the model exceeds NVLink-connected GPU memory—often combined as, e.g., 8-way TP $\times$ 4-way PP for a 70B model on 32 GPUs.

---

### Q20 [Advanced] Discuss disaggregated inference and the evolving landscape of LLM serving at scale

**Q:** What are the key system-level trade-offs when scaling LLM inference to production, and how does disaggregated inference change hardware provisioning strategy?

**A:** Production LLM serving must simultaneously optimize for TTFT (latency to first token, dominated by prefill), TPOT (per-token decode latency, dominated by memory bandwidth), and throughput (requests per second, dominated by batching efficiency). These objectives conflict: large batches improve throughput but increase queuing latency; long sequences stress KV cache memory; low-latency requirements limit batch sizes.

Disaggregated prefill-decode (Zhong et al., 2024) recognizes that prefill and decode have fundamentally different resource profiles. Prefill is arithmetic-intensity-bound (high FLOP/byte ratio, benefits from high-FLOP GPUs with large SRAM), while decode is memory-bandwidth-bound (low FLOP/byte ratio, benefits from large HBM capacity). Colocating both phases forces over-provisioning on one dimension. DistServe provisions separate prefill and decode fleets sized independently to match request arrival rates, transferring KV caches over NVLink or InfiniBand when a request transitions phases. This allows matching hardware to workload—e.g., H100 SXM for prefill (high FLOP throughput) and A100 80GB for decode (large HBM)—while independently scaling each fleet.

As context lengths extend to 128K–1M tokens, KV cache management becomes the dominant engineering challenge: even a 7B model with 128K-token context at batch size 32 requires over 1 TB of KV cache at FP16, necessitating hierarchical storage (GPU HBM → CPU DRAM → NVMe SSD → remote memory) with prefetching. The interplay of scheduling, memory tiering, disaggregation, and speculative decoding defines the frontier of production LLM serving research.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | KV cache fundamentals | KV Cache and Memory Management |
| Q2 | Advanced | PagedAttention memory paging | KV Cache and Memory Management |
| Q3 | Advanced | Prefix caching and radix attention | KV Cache and Memory Management |
| Q4 | Advanced | KV cache eviction (H2O, SnapKV) | KV Cache and Memory Management |
| Q5 | Basic | FlashAttention IO-aware tiling | Efficient Attention Mechanisms |
| Q6 | Advanced | FlashAttention-2 and FlashAttention-3 | Efficient Attention Mechanisms |
| Q7 | Basic | MQA and GQA | Efficient Attention Mechanisms |
| Q8 | Advanced | Sparse and linear attention | Efficient Attention Mechanisms |
| Q9 | Basic | Speculative decoding framework | Speculative Decoding |
| Q10 | Advanced | Acceptance rate analysis | Speculative Decoding |
| Q11 | Advanced | Medusa multi-head drafts | Speculative Decoding |
| Q12 | Advanced | Lookahead and self-speculative decoding | Speculative Decoding |
| Q13 | Basic | Continuous batching | Batching, Scheduling, and Serving |
| Q14 | Advanced | Chunked prefill and disaggregation | Batching, Scheduling, and Serving |
| Q15 | Basic | vLLM architecture | Batching, Scheduling, and Serving |
| Q16 | Advanced | Scheduling policies under SLOs | Batching, Scheduling, and Serving |
| Q17 | Basic | GPTQ and AWQ | Quantization and Distributed Inference |
| Q18 | Advanced | SmoothQuant W8A8 quantization | Quantization and Distributed Inference |
| Q19 | Basic | Tensor and pipeline parallelism | Quantization and Distributed Inference |
| Q20 | Advanced | Disaggregated inference at scale | Quantization and Distributed Inference |

## Resources

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (2022)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023)
- Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (2024)
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (2023)
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (2023)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2023)
- Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (2023)
- Cai et al., [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) (2024)
- Fu et al., [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) (2024)
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (2022)
- Agrawal et al., [Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2403.02310) (2024)
- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (2019)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (2023)
- Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) (2023)
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2023)
- Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (2024)
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019)
- Zhong et al., [DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) (2024)
- Zhang et al., [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048) (2023)
- Li et al., [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469) (2024)
