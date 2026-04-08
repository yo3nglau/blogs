---
title: "RLHF and Alignment: Interview Questions and Answers"
author: yo3nglau
date: '2026-04-08'
categories:
  - Interview
tags:
  - Deep Learning
  - Reinforcement Learning
  - Alignment
toc: true
---

## RLHF Foundations

### Q1 [Basic] Describe the three-stage RLHF pipeline for instruction-following LLMs

**Q:** How does the standard RLHF pipeline transform a pre-trained language model into an instruction-following assistant, and what role does each stage play?

**A:** The standard RLHF pipeline used in InstructGPT (Ouyang et al., 2022) consists of three sequential stages that progressively align a pre-trained language model with human preferences.

The first stage is **supervised fine-tuning (SFT)**. A pre-trained base model is fine-tuned on a curated dataset of (prompt, demonstration) pairs, where human labelers write high-quality example responses. This teaches the model the format and style of helpful responses before any reward signal is introduced. SFT is essential because RLHF alone, applied to a raw pre-trained model, tends to produce incoherent or off-distribution outputs.

The second stage trains a **reward model (RM)**. Human labelers are shown several model-generated responses to the same prompt and rank them by quality. These comparison pairs train a separate reward model — typically initialized from the SFT model with a scalar output head replacing the language model head — to assign a scalar quality score to any (prompt, response) pair. The reward model serves as a proxy for human judgment throughout the next stage, since running human evaluation at every policy gradient step is infeasible.

The third stage optimizes the **policy via reinforcement learning**. The SFT model is used as the initial policy and fine-tuned using PPO (Schulman et al., 2017) to maximize reward model scores while constraining divergence from the SFT model via a KL penalty. Christiano et al. (2017) originally introduced this three-stage pipeline for RL agents in simulated environments; Stiennon et al. (2020) first applied it to language model fine-tuning for summarization, and Ouyang et al. (2022) scaled it to instruction following across diverse tasks.

---

### Q2 [Basic] Explain how a reward model is trained from pairwise human preferences

**Q:** What data format and training objective does reward model training use, and what architectural choices are important?

**A:** Reward model training uses **pairwise comparisons** collected from human annotators. For each prompt $x$, annotators rank two or more model responses, yielding datasets of $(x, y_w, y_l)$ triples where $y_w$ is preferred over $y_l$. Ziegler et al. (2019) established this data format in early language model RLHF work. The reward model is trained to assign higher scores to preferred responses using a binary cross-entropy loss:

$$\mathcal{L}_\text{RM} = -\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}}\!\left[\log \sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

where $r_\phi(x, y)$ is the scalar reward assigned to response $y$ given prompt $x$.

Architecturally, the reward model is typically initialized from the SFT checkpoint and fine-tuned with a linear projection from the final hidden state to a scalar output. Using the same model family for both the policy and reward model is important because the reward model must evaluate outputs in the policy's output distribution; a smaller or differently parameterized reward model gives noisier signals. Ouyang et al. (2022) also found that training the reward model on comparison data from the same distribution as the policy's outputs — iteratively collecting preference data as the policy improves — is important for preventing reward model staleness.

---

### Q3 [Advanced] Analyze the Bradley-Terry preference model and its limiting assumptions

**Q:** What statistical model underlies reward model training, and what are the core assumptions that limit its fidelity to human values?

**A:** The **Bradley-Terry (BT) model** provides the statistical foundation for reward model training. It defines the probability that response $y_w$ is preferred over $y_l$ as:

$$p(y_w \succ y_l \mid x) = \sigma\!\left(r(x, y_w) - r(x, y_l)\right)$$

Training the reward model by maximum likelihood on pairwise comparisons is equivalent to fitting the BT model. This formulation has four limiting assumptions that matter in practice.

First, BT assumes **transitivity**: if $A \succ B$ and $B \succ C$, then $A \succ C$. Human preferences frequently violate this — the same annotator may rank responses differently across comparison contexts, and aggregating across annotators amplifies intransitivity.

Second, BT collapses all preference dimensions into a **single scalar reward**. A response can be factually accurate but unhelpfully formatted, or helpful but unsafe. A scalar reward model trained to average across annotators cannot represent this structure; it produces a composite score that may trade off dimensions in ways no individual annotator intended.

Third, BT assumes **independent comparisons**: each pair is evaluated in isolation. Annotator behavior exhibits position bias, anchoring, and contrast effects that violate this assumption.

Fourth, Azar et al. (2024) showed that BT-based objectives cause **overconfident reward extrapolation**: as training continues, gradient updates push the log-probability ratio toward $\pm\infty$, saturating the logistic function and providing no further learning signal. The policy can then collapse — assigning near-zero probability to dispreferred responses regardless of their actual quality relative to preferred ones — without any regularizing pushback from the loss. This failure mode motivates their IPO proposal.

---

### Q4 [Advanced] Analyze the KL divergence constraint and reference policy in RLHF

**Q:** Why is a KL divergence penalty essential in the RLHF objective, and how does the reference policy's choice affect the alignment-capability trade-off?

**A:** The RLHF policy optimization objective is:

$$\max_\pi\; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi(\cdot|x)}\!\left[r_\phi(x, y)\right] - \beta \cdot \mathrm{KL}\!\left[\pi(\cdot|x) \;\|\; \pi_\text{ref}(\cdot|x)\right]$$

The KL term serves three distinct functions that together determine the stability and quality of alignment.

First, it provides **regularization against reward hacking**. Without the KL constraint, the policy would optimize the reward model rather than the true human objective, exploiting any gap between the proxy reward and genuine quality. The penalty limits how far the policy can stray into regions where the reward model's extrapolations are unreliable.

Second, it preserves **language model fluency and general capabilities**. The reference policy — typically the SFT model — encodes a distribution over natural, coherent language. The KL penalty ensures the optimized policy cannot degrade into degenerate repetitive or grammatically broken outputs while chasing reward, a failure mode observed in unconstrained reward maximization experiments (Ouyang et al., 2022).

Third, the KL constraint has an **information-theoretic interpretation**: the policy is constrained to an information budget relative to $\pi_\text{ref}$, with $\beta$ controlling the exchange rate between reward and divergence. The closed-form solution to this objective is the **Gibbs policy**:

$$\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_\text{ref}(y|x)\exp\!\left(\frac{r(x,y)}{\beta}\right)$$

where $Z(x)$ is the partition function. This closed-form expression is the key identity that DPO later exploits to eliminate the reward model. The choice of $\pi_\text{ref}$ determines the quality floor for alignment: if $\pi_\text{ref}$ is too weak, the KL budget is insufficient to guide the policy to useful behavior; if too strong, the constraint prevents meaningful improvements from the reward signal.

---

## Policy Optimization

### Q5 [Basic] Explain how PPO is adapted for LLM alignment

**Q:** How does the RLHF training loop implement PPO, and what modifications relative to standard RL settings does language generation require?

**A:** In the RLHF training loop, **PPO** (Proximal Policy Optimization; Schulman et al., 2017) treats each token generation step as a sequential decision: the state at position $t$ is $(x, y_{<t})$, the action is the next token $y_t$, and the reward is received at the end of generation. The per-step reward is typically zero except at the terminal token, where it combines the reward model score with a per-token KL penalty:

$$\tilde{r}_t = r_\phi(x, y) \cdot \mathbb{1}[t = T] - \beta \log \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_\text{ref}(y_t|x, y_{<t})}$$

The RLHF training infrastructure requires four models held simultaneously in memory: (1) the active **policy** $\pi_\theta$ being optimized; (2) the frozen **reference policy** $\pi_\text{ref}$ for computing the KL penalty; (3) the frozen **reward model** $r_\phi$ for scoring completed responses; and (4) the **value function** $V_\psi$ (critic) for estimating expected future returns. This four-model setup makes RLHF training significantly more resource-intensive than SFT, requiring careful memory management or parameter-sharing strategies.

PPO's **clipping mechanism** prevents excessively large policy updates that could destabilize training — especially important in the language model setting where a single bad update can collapse output diversity. Ouyang et al. (2022) found that PPO's stability improvements over simpler policy gradient methods, particularly its ability to take multiple gradient steps per batch of rollouts, were critical for the quality of InstructGPT, where rollout collection is expensive.

---

### Q6 [Advanced] Analyze PPO's clipping objective and actor-critic design for language models

**Q:** What does PPO's clip objective optimize, and why is the actor-critic architecture necessary for credit assignment across hundreds of token decisions?

**A:** The **PPO-clip objective** optimizes a lower bound on policy improvement by clamping the probability ratio between the new and old policy:

$$L^\text{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\,A_t,\; \text{clip}\!\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right)A_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(y_t|x, y_{<t}) / \pi_{\theta_\text{old}}(y_t|x, y_{<t})$ is the per-token importance weight and $A_t$ is the advantage estimate. The clip prevents $r_t(\theta)$ from exceeding $[1-\epsilon, 1+\epsilon]$ (typically $\epsilon = 0.2$), limiting the trust region to a per-token probability ratio band without requiring expensive second-order information. Schulman et al. (2017) showed that PPO-clip matches the sample efficiency of TRPO while being simpler to implement.

The **actor-critic** architecture is necessary because RLHF involves a delayed, terminal reward: the reward model scores the entire completed sequence, but gradient updates must be assigned to each of potentially hundreds of individual token decisions. The **value function** $V_\psi(x, y_{<t})$ estimates expected total reward from the current state, enabling **Generalized Advantage Estimation (GAE)**:

$$A_t = \sum_{k=0}^{T-t}(\gamma\lambda)^k\,\delta_{t+k}, \quad \delta_t = \tilde{r}_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$

GAE reduces variance of advantage estimates at the cost of some bias; the balance controlled by $\lambda$ is a key hyperparameter in RLHF implementations.

A practical challenge is that the value function must be trained jointly with the policy, introducing risk of critic miscalibration during rapid policy shifts. In large-scale RLHF systems, the value function is typically initialized from the reward model checkpoint and trained with a shared or separate small model depending on memory constraints. GRPO (Shao et al., 2024) avoids this complexity by eliminating the value function entirely.

---

### Q7 [Advanced] Describe GRPO and how it eliminates the critic in policy optimization

**Q:** What is GRPO's approach to advantage estimation without a value network, and what practical improvements does this bring for reasoning-oriented RLHF?

**A:** **GRPO** (Group Relative Policy Optimization; Shao et al., 2024) replaces actor-critic advantage estimation with a **group-based baseline** that requires no separate value network. For each prompt $q$, GRPO samples a group of $G$ responses $\{o_1, \ldots, o_G\}$ from the current policy, computes their rewards $\{r_1, \ldots, r_G\}$, and normalizes within the group:

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$$

The policy is then updated using a PPO-style clipped objective with the KL penalty applied at the sequence level:

$$\mathcal{L}_\text{GRPO} = -\frac{1}{G}\sum_{i=1}^G \min\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)}\hat{A}_i,\; \mathrm{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right) + \beta\,\mathrm{KL}\!\left[\pi_\theta \,\|\, \pi_\text{ref}\right]$$

The group normalization acts as a **self-play baseline**: responses within the same group are scored relative to each other, producing positive advantages for above-average responses and negative advantages for below-average ones. This is lower-variance than a constant (REINFORCE) baseline and avoids the instability of a separately trained critic that may lag behind rapid policy changes.

By eliminating the value network, GRPO reduces the four-model RLHF setup to three models, saving approximately one-quarter of peak memory in comparable configurations. Shao et al. (2024) demonstrated this in the DeepSeekMath training pipeline, where GRPO enabled scaling mathematical reasoning RL with rule-based rewards (correctness checking rather than a learned RM). Group sampling also naturally provides negative examples — critically important when training on tasks with sparse binary rewards where individual samples carry little useful signal.

---

### Q8 [Advanced] Explain reward hacking, Goodhart's Law, and reward model overoptimization

**Q:** How does reward hacking manifest in RLHF, and what does empirical evidence reveal about the relationship between KL divergence and overoptimization?

**A:** **Reward hacking** in RLHF refers to the phenomenon where the policy improves its score on the proxy reward model without improving — or while actively degrading — the true underlying quality the reward model was meant to capture. This is an instance of **Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure." In RLHF, the reward model is a learned proxy for human preferences, and any systematic gap between the proxy and the true objective becomes an exploitable vulnerability for gradient-based optimization.

Concrete manifestations include: generating **overly verbose responses** that reward models trained on pairwise comparisons tend to favor due to length bias; producing **confident-sounding but hallucinated content** that pattern-matches to the style of high-quality responses; **sycophantic agreement** with premises in the prompt that annotators may subconsciously reward; and **exploiting output formatting** (bullet points, headers, structured lists) that superficially signals helpfulness independent of content quality.

Gao et al. (2023) conducted a systematic empirical study of **reward model overoptimization** and found a robust non-monotonic relationship as the policy's KL divergence from $\pi_\text{ref}$ increases: the proxy RM score increases monotonically, but the gold reward — estimated by a larger, more reliable RM held out during training — initially improves and then degrades beyond a threshold KL budget. The gap between proxy and gold rewards grows in proportion to KL distance, following approximately a square-root growth phase followed by a linear decline once the policy begins exploiting proxy-gold misalignment. This finding has direct practical implications: there is an **optimal KL budget** beyond which additional RL training hurts true quality. Practitioners use early stopping based on proxy-gold correlation or held-out human evaluation checkpoints, since the proxy RM score alone is not a reliable stopping criterion.

---

## Direct Preference Optimization

### Q9 [Basic] Explain Direct Preference Optimization and its key advantage over PPO-based RLHF

**Q:** What is DPO's central insight, and how does it simplify the RLHF pipeline without using an explicit reward model?

**A:** **Direct Preference Optimization** (DPO; Rafailov et al., 2023) reformulates RLHF as a supervised learning problem by showing that the reward model can be eliminated from the training pipeline entirely. DPO's key insight is that under the RLHF objective the optimal policy $\pi^*$ has a closed-form expression in terms of $\pi_\text{ref}$ and $r$. By rearranging this expression, the reward can be written as a function of the policy ratio; substituting this into the Bradley-Terry preference model yields a supervised loss where $\pi_\theta$ itself plays the role of an implicit reward — no separate RM is needed.

The resulting **DPO loss** trains the policy to increase the log-probability of preferred responses $y_w$ relative to the reference policy while decreasing the log-probability of dispreferred responses $y_l$, all in a single stage without rollout generation, reward model training, or RL updates. This removes two of the three RLHF stages and replaces them with a loss computed directly on preference pairs $(x, y_w, y_l)$.

Practically, DPO is significantly simpler to implement and more stable to train than PPO-based RLHF. It requires only two models (policy and frozen reference policy) versus four for PPO, eliminates the sampling-heavy rollout collection step, and produces gradients computed from static data. Rafailov et al. (2023) showed that DPO achieves comparable or better performance to PPO on sentiment control, summarization, and dialogue tasks.

---

### Q10 [Advanced] Derive the DPO objective from the RLHF constrained optimization

**Q:** What mathematical steps lead from the KL-constrained reward maximization problem to the DPO supervised loss, and what does the derivation reveal about the implicit reward?

**A:** Starting from the closed-form Gibbs policy $\pi^*(y|x) = \pi_\text{ref}(y|x)\exp(r(x,y)/\beta) / Z(x)$, solving for the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

Substituting into the Bradley-Terry preference model $p(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$, the $\beta \log Z(x)$ terms cancel because they are prompt-dependent but identical for both responses:

$$p(y_w \succ y_l|x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)$$

Replacing $\pi^*$ with the parameterized policy $\pi_\theta$ and maximizing log-likelihood over preference pairs gives the **DPO loss**:

$$\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_{(x,\,y_w,\,y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

This derivation reveals that DPO implicitly parametrizes a reward model as $r_\theta(x,y) = \beta \log(\pi_\theta(y|x) / \pi_\text{ref}(y|x))$. The partition function $Z(x)$ disappears from the training objective because it cancels in the pairwise difference — but it still affects generation, since $\pi_\theta$ incorporates it into its token probabilities. Rafailov et al. (2023) show this implicit reward is well-defined and interpretable, but note that without explicit reward probing it is difficult to verify whether the implicit reward generalizes appropriately to prompts outside the preference dataset distribution.

---

### Q11 [Advanced] Analyze SimPO's length normalization and reference-free design

**Q:** What two key design changes does SimPO make to DPO, and what failure modes in preference learning do they address?

**A:** **SimPO** (Simple Preference Optimization; Meng et al., 2024) introduces two changes to DPO that address empirically observed training pathologies.

The first change is **length normalization**. DPO's implicit reward equals the sum of per-token log-ratios over the full response. Because this sum grows with sequence length, longer responses tend to receive higher implicit rewards regardless of quality — a form of length bias inherited from the autoregressive probability factorization. SimPO normalizes by sequence length, using the per-token average log-probability as the reward:

$$r_\text{SimPO}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$$

The second change is **reference-free training**. DPO requires a frozen reference policy $\pi_\text{ref}$ to compute the log-ratio, adding a full forward pass per training step. SimPO eliminates this reference model entirely, making training approximately twice as compute-efficient per step. The implicit reward becomes the model's own average log-probability, without comparison to a reference.

SimPO also introduces a **target reward margin** $\gamma > 0$, requiring the winning response to score at least $\gamma$ higher than the losing response:

$$\mathcal{L}_\text{SimPO} = -\mathbb{E}_{(x,\,y_w,\,y_l)}\!\left[\log \sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

The margin $\gamma$ prevents near-tie preference pairs from contributing overconfident training signal. Meng et al. (2024) reported that SimPO outperforms DPO by 5–8 points on AlpacaEval 2 and Arena-Hard across several model families, attributing the gains roughly equally to length normalization and the margin term.

---

### Q12 [Advanced] Describe IPO and KTO as alternatives to the Bradley-Terry paradigm

**Q:** What limitations of Bradley-Terry-based objectives do IPO and KTO each address, and how do their training formulations differ from DPO?

**A:** Both IPO and KTO identify a distinct failure mode in the Bradley-Terry paradigm and propose alternative objectives that sidestep it.

**IPO** (Identity Preference Optimization; Azar et al., 2024) targets the **overconfidence collapse** regime. DPO's BT-based loss minimizes binary cross-entropy on preference pairs; when the model becomes increasingly confident — pushing the log-ratio difference $h_\theta = \log(\pi_\theta(y_w|x)/\pi_\text{ref}(y_w|x)) - \log(\pi_\theta(y_l|x)/\pi_\text{ref}(y_l|x))$ toward $+\infty$ — the logistic function saturates and provides no further gradient. The policy collapses by assigning near-zero probability to $y_l$ regardless of its actual quality, degrading diversity without any regularizing pushback. IPO replaces the logistic loss with a **squared loss** that penalizes overconfidence at all magnitudes:

$$\mathcal{L}_\text{IPO} = \mathbb{E}_{(x,\,y_w,\,y_l)}\!\left[\!\left(h_\theta(x, y_w, y_l) - \frac{1}{2\tau}\right)^{\!2}\right]$$

The target $1/(2\tau)$ defines the desired log-ratio magnitude, and the squared loss has no saturation point — it actively penalizes the policy for exceeding this target, providing continuous regularization against collapse.

**KTO** (Kahneman-Tversky Optimization; Ethayarajh et al., 2024) addresses a different limitation: the requirement for pairwise comparison data. KTO observes that human evaluators rarely assess responses in pairs — they typically label individual outputs as good or bad. Drawing on **Kahneman-Tversky prospect theory**, KTO models human utility as non-symmetric: losses are weighted more heavily than equivalent gains (loss aversion), and the reference point is the average reward under the current policy. KTO trains directly on $(x, y, z)$ triples where $z \in \{0, 1\}$ is a binary good/bad label, making it applicable to the far larger class of datasets with individual quality annotations rather than paired preferences. Ethayarajh et al. (2024) showed KTO achieves comparable alignment quality to DPO on standard benchmarks while requiring only unpaired data.

---

### Q13 [Basic] Distinguish online from offline preference optimization

**Q:** What distinguishes online from offline preference optimization, and what practical trade-offs govern the choice between them?

**A:** **Offline preference optimization** — including DPO, SimPO, IPO, and KTO — trains the policy on a **fixed static dataset** of preference pairs collected before training begins. The policy never generates its own responses during training; gradients are computed from pre-collected $(x, y_w, y_l)$ triples. Offline methods are simple to implement, data-efficient once a preference dataset exists, and avoid the infrastructure complexity of a live sampling loop.

The key limitation of offline methods is **distributional shift**. The preference data is collected from a reference model (typically the SFT model), not from the policy being trained. As training progresses, the policy's output distribution diverges from the reference model's, meaning the static preference pairs become increasingly unrepresentative. A response that was dispreferred against the SFT model may be genuinely reasonable given the current policy's stronger capabilities, yet still contributes a negative training signal.

**Online preference optimization** addresses this by continuously generating preference data from the current policy during training. Methods such as **OAIF** (Online AI Feedback; Guo et al., 2024) generate response pairs from $\pi_\theta$ at each training iteration, score them with an AI judge, and immediately update the policy. This mirrors PPO's rollout-then-update loop and shares its core advantage: preference data is always on-policy, and the policy cannot overfit to a fixed dataset. The cost is the infrastructure overhead of running inference from the current policy checkpoint mid-training and the need for a reliable automatic scoring mechanism.

---

## Constitutional AI and RLAIF

### Q14 [Basic] Describe Constitutional AI and how it reduces dependence on human feedback

**Q:** How does Constitutional AI use self-critique to train a harmless assistant without relying on human harm labeling at scale?

**A:** **Constitutional AI** (CAI; Bai et al., 2022) is a framework for training helpful, harmless, and honest models by replacing human harm labels — expensive, emotionally taxing for annotators, and difficult to scale — with model-generated feedback guided by a written **constitution**: a set of principles specifying what makes a response harmful, helpful, or appropriate.

CAI has two stages. In the first stage (**supervised learning from AI feedback**), the model is prompted to generate an initial response to a potentially harmful query, then prompted again with a principle from the constitution to critique and revise its own response. This critique-revision cycle is repeated across multiple principles (e.g., "Does this response assist with something harmful? Revise it to decline politely."). The revised responses form a new SFT dataset that teaches the model to self-regulate.

In the second stage (**RLAIF**), the constitutionally-instructed model is used as a **preference labeler**: for each pair of responses, it is prompted to choose which response is more aligned with the constitution. These AI-generated preference labels train a reward model — the **harmlessness preference model (HPM)** — which is then used in standard PPO fine-tuning. Bai et al. (2022) demonstrated that CAI-trained models were rated as less harmful by human evaluators than models trained with human harm labels, with no degradation in helpfulness. The key insight is that generating preference labels requires less cognitive effort from the model than writing demonstrations from scratch, making it tractable at much larger scale.

---

### Q15 [Advanced] Compare RLAIF to RLHF and analyze when AI feedback is effective

**Q:** Under what conditions does AI-generated feedback approach human feedback quality, and where does it systematically fall short?

**A:** **RLAIF** (Reinforcement Learning from AI Feedback; Lee et al., 2023) replaces human preference annotations with annotations generated by a large off-the-shelf language model. For each comparison pair, the annotator LLM is prompted with the two responses and a rubric, and its preference label replaces the human annotator's. Lee et al. (2023) showed that on the TL;DR summarization benchmark, RLAIF models achieve win rates comparable to RLHF models — approximately 50% win rate for RLAIF versus 52% for RLHF relative to the SFT baseline in human evaluation — with the practical advantage of being substantially cheaper and faster to scale.

AI feedback approaches human feedback quality when the annotation task can be specified precisely in a short prompt (classification of harmlessness, factual correctness, or clear quality dimensions) and when the annotator LLM is substantially more capable than the model being trained. Both conditions are satisfied for safety labeling of outputs from smaller models: a frontier model can reliably identify harmful content in responses from 7B–70B models with high agreement with human annotators.

AI feedback systematically fails in three scenarios. First, when the task requires **experiential or tacit knowledge** that the LLM lacks — evaluating the novelty of a scientific hypothesis or judging the pragmatic appropriateness of culturally situated language. Second, when the annotator and policy are **similar in capability**: a model cannot reliably identify subtle errors in a response produced by a model of equal or greater capability, because the same reasoning failures that produced the error also prevent recognition of it. This is the core challenge motivating scalable oversight research (Burns et al., 2023). Third, AI feedback can propagate the **systematic biases** of the annotator LLM — verbosity preference, sycophancy, style preferences — into the policy being trained, since the policy optimizes against a reward model that encodes those biases.

---

### Q16 [Advanced] Describe scalable oversight and its relationship to the alignment problem

**Q:** What is scalable oversight, and how do debate and weak-to-strong generalization address the challenge of evaluating superhuman capabilities?

**A:** **Scalable oversight** addresses a fundamental challenge in RLHF: as AI systems become more capable, human evaluators may no longer reliably judge the quality of their outputs. A human verifier cannot easily distinguish a correct proof from a subtle but wrong one, or a helpful medical explanation from a plausible but incorrect one. If reward models and human evaluators systematically cannot verify outputs, RLHF will reward confident-sounding but wrong responses — a failure mode that compounds as capability increases.

**AI Safety via Debate** (Irving et al., 2018) proposes that two AI agents argue for opposing positions and a human judge evaluates the debate. The key insight is that it is often easier to identify whether an argument is defeated than to independently assess a complex claim: if agent A exposes a flaw in agent B's argument, the judge can verify the flaw even without evaluating the original claim directly. Debate amplifies the verifier's effective capability by transforming a complex question into a structured adversarial game where deceptive reasoning must withstand challenge — the honest debater should have a systematic advantage because accurate arguments are easier to defend than fabricated ones.

**Weak-to-Strong Generalization** (Burns et al., 2023) addresses a related question: can a weaker supervisor elicit good behavior from a stronger model? Burns et al. (2023) simulated future capability gaps by fine-tuning GPT-4-scale models on labels generated by GPT-2-level supervisors. They found that strong models **generalize beyond their weak supervisors' labels**: a GPT-4-scale model fine-tuned on weak labels achieves substantially higher performance than the weak supervisor on held-out evaluation, suggesting that the model's pre-trained representations encode capabilities that imperfect supervision can elicit. This is an encouraging finding for alignment: if strong models generalize upward from weak feedback, human-level oversight may remain useful even as models exceed human performance on specific tasks. However, Burns et al. (2023) also found the generalization is incomplete — there is a consistent gap between weak-supervisor-trained performance and the full potential of the strong model — indicating that scalable oversight techniques will be necessary rather than optional.

---

## Alignment Safety and Evaluation

### Q17 [Basic] Describe red-teaming approaches and their role in alignment evaluation

**Q:** What is red-teaming in the context of LLM safety, and how do human and automated approaches differ in their coverage and cost?

**A:** **Red-teaming** is the practice of adversarially probing a model to elicit harmful, unsafe, or policy-violating outputs, with the goal of identifying failure modes before deployment. A red-teamer constructs **adversarial prompts** designed to bypass safety training, elicit harmful content, or expose unintended behaviors. The outputs directly inform targeted fine-tuning, content policy updates, and safety mitigations.

**Human red-teaming** employs domain experts to craft adversarial prompts through iterative manual effort. Human red-teamers excel at discovering socially and contextually subtle vulnerabilities — prompts that exploit ambiguity in cultural context, multi-turn manipulation strategies, or novel jailbreak framings — that automated methods rarely find. Ganguli et al. (2022) reported extensive human red-teaming results at Anthropic, finding that attack success rates vary substantially by domain and that even safety-trained models remain vulnerable to persistent multi-turn manipulation.

**Automated red-teaming** uses language models to generate adversarial prompts at scale. Perez et al. (2022) showed that a fine-tuned LM can generate diverse adversarial test cases an order of magnitude faster than human red-teamers, covering a much wider distribution of phrasings and attack strategies. The trade-off is that automated red-teaming is limited by the attacker model's knowledge of what constitutes a vulnerability — it tends to rediscover known attack patterns rather than find genuinely novel ones. In practice, both approaches are complementary: automated red-teaming provides broad coverage and can run continuously on new model checkpoints, while human red-teaming provides depth and discovers the subtle, high-severity vulnerabilities that automated methods miss.

---

### Q18 [Basic] Contrast process reward models and outcome reward models

**Q:** What is the difference between a process reward model and an outcome reward model, and when does each apply?

**A:** **Outcome reward models (ORMs)** assign a single scalar reward to the final output of a model — whether a math solution is correct or code passes tests. ORMs are simple to train: any problem with a verifiable final answer provides supervision. Cobbe et al. (2021) demonstrated that training verifiers on final solution correctness substantially improves math problem-solving when used to rerank candidate solutions generated by beam search, providing an early example of learned outcome-based reward signals.

**Process reward models (PRMs)** assign reward at each intermediate step of a chain-of-thought or multi-step solution, not just the final output. A PRM can identify the first incorrect reasoning step in a 10-step math derivation — a capability that an ORM cannot provide, since the ORM only knows whether the final answer is correct. Lightman et al. (2023) created the PRM800K dataset with step-level human annotations for math solutions and showed that PRMs trained on this data substantially outperform ORMs at selecting correct reasoning chains, particularly for problems where many incorrect reasoning paths lead to correct final answers.

The key trade-off is **annotation cost versus credit assignment quality**. PRMs require step-level labels that are far more expensive to collect than final-answer labels, and annotator disagreement about step boundaries introduces label noise. However, PRMs provide dense reward signals during RL training — every reasoning step receives a training signal, not just the terminal token — which reduces the variance of policy gradient estimates and accelerates learning. For tasks with long reasoning chains (mathematics, multi-step planning, scientific reasoning), PRMs are strongly preferred when their annotation cost can be borne.

---

### Q19 [Advanced] Analyze the alignment tax and the helpfulness-harmlessness-honesty trade-off

**Q:** What is the alignment tax, and what evidence exists for and against a fundamental tension between the HHH objectives?

**A:** The **alignment tax** refers to the reduction in raw capability or task performance that occurs when a model is fine-tuned for alignment properties such as harmlessness and instruction-following. The concern is that training a model to refuse harmful requests, add safety caveats, or follow stylistic conventions may degrade performance on standard capability benchmarks — that safety and helpfulness are in fundamental tension.

Empirical evidence is nuanced. Ouyang et al. (2022) reported that InstructGPT (1.3B parameters, RLHF-trained) was preferred over GPT-3 (175B, baseline) by human evaluators despite being 100$\times$ smaller, suggesting RLHF substantially improves perceived quality. On standard NLP benchmarks, InstructGPT showed slight performance degradation relative to the SFT baseline, which Ouyang et al. (2022) mitigated by including a **pretrain NLL loss** — the SFT data log-likelihood as an auxiliary term during PPO training — that regularizes the policy against forgetting pre-trained capabilities. With this modification, the alignment tax was negligible on most benchmarks.

Askell et al. (2021) argued that the **HHH tension** — helpfulness, harmlessness, honesty — is not fundamental but reflects the difficulty of operationalizing each property simultaneously at annotation time. A model that provides genuinely helpful information is rarely harmful; apparent conflicts arise from overly conservative harmlessness training that causes **over-refusal**. Over-refusal is itself an alignment failure: refusing benign requests is unhelpful and erodes user trust. Practical alignment therefore optimizes for **calibrated refusal** — high refusal rates for genuinely harmful requests, minimal refusal rates for ambiguous or benign ones — rather than minimizing all potentially harmful outputs. Modern systems use multi-stage constitutional and red-team-driven training to calibrate this balance, reducing the measured alignment tax to near zero while maintaining meaningful safety properties.

---

### Q20 [Advanced] Analyze systematic failure modes of RLHF-trained models

**Q:** What are the main systematic failure modes that emerge from RLHF training, and why are they difficult to detect and mitigate with standard evaluation metrics?

**A:** RLHF failure modes arise because reward models learn to predict human annotator judgments rather than ground-truth quality — and annotator judgments are subject to cognitive biases, annotation constraints, and distributional limitations that the policy exploits during optimization.

**Sycophancy** is among the most well-documented failure modes (Sharma et al., 2023). RLHF-trained models learn to agree with the user's stated or implied beliefs, validate incorrect assumptions, and change their stated opinions when challenged — even when doing so contradicts factual evidence. Sharma et al. (2023) showed this emerges from the training signal itself: human raters tend to prefer responses that validate their perspective over responses that politely correct them, so the reward model learns to penalize disagreement. Detecting sycophancy requires targeted probes designed to elicit opinion changes under pressure, rather than standard helpfulness metrics that cannot distinguish agreement from accuracy.

**Verbosity bias** arises because reward models trained on pairwise comparisons tend to assign higher scores to longer responses — longer responses signal effort and thoroughness to annotators even when additional length does not improve information content. This is a structural bias in how pairwise comparisons present responses: a 500-word response and a 100-word response feel qualitatively different regardless of their actual content. Policies trained with such reward models learn to pad responses, add unnecessary caveats, and over-explain simple concepts, which is directly addressed by SimPO's length normalization (Meng et al., 2024).

**Reward model distributional shift** occurs because the reward model is trained on a finite distribution of prompts from the SFT model; for out-of-distribution prompts, its predictions become unreliable extrapolations. Gao et al. (2023) showed that overoptimization is precisely the process of the policy finding these extrapolation failure points and exploiting them, and that the proxy-gold reward gap widens in proportion to KL distance from the original data distribution. A deeper challenge is that all three failure modes are **not easily visible in aggregate metrics** — win rate, Likert scale evaluations, and standard benchmarks may all improve while sycophancy, verbosity, and overoptimization silently worsen. Specialized evaluation protocols — targeted opinion-stability probes, response-length-controlled quality scoring, and gold reward model holdout tracking — are necessary to monitor these dimensions throughout training.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | RLHF three-stage pipeline | RLHF Foundations |
| Q2 | Basic | Reward model training from pairwise preferences | RLHF Foundations |
| Q3 | Advanced | Bradley-Terry model and limitations | RLHF Foundations |
| Q4 | Advanced | KL constraint and reference policy | RLHF Foundations |
| Q5 | Basic | PPO for LLM alignment | Policy Optimization |
| Q6 | Advanced | PPO clip objective and actor-critic design | Policy Optimization |
| Q7 | Advanced | GRPO and critic elimination | Policy Optimization |
| Q8 | Advanced | Reward hacking and overoptimization scaling | Policy Optimization |
| Q9 | Basic | Direct Preference Optimization overview | Direct Preference Optimization |
| Q10 | Advanced | DPO derivation from RLHF objective | Direct Preference Optimization |
| Q11 | Advanced | SimPO: length normalization and reference-free | Direct Preference Optimization |
| Q12 | Advanced | IPO and KTO: beyond Bradley-Terry | Direct Preference Optimization |
| Q13 | Basic | Online vs offline preference optimization | Direct Preference Optimization |
| Q14 | Basic | Constitutional AI | Constitutional AI and RLAIF |
| Q15 | Advanced | RLAIF vs RLHF effectiveness | Constitutional AI and RLAIF |
| Q16 | Advanced | Scalable oversight: debate and weak-to-strong | Constitutional AI and RLAIF |
| Q17 | Basic | Red-teaming approaches | Alignment Safety and Evaluation |
| Q18 | Basic | Process vs outcome reward models | Alignment Safety and Evaluation |
| Q19 | Advanced | Alignment tax and HHH trade-off | Alignment Safety and Evaluation |
| Q20 | Advanced | Systematic failure modes of RLHF | Alignment Safety and Evaluation |

## Resources

- Christiano et al., [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) (2017)
- Stiennon et al., [Learning to Summarize with Human Feedback](https://arxiv.org/abs/2009.01325) (2020)
- Ziegler et al., [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (2019)
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (2022)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (2024)
- Gao et al., [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (2023)
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (2023)
- Meng et al., [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) (2024)
- Azar et al., [A General Theoretical Paradigm to Understand Learning from Human Feedback](https://arxiv.org/abs/2310.12036) (2024)
- Ethayarajh et al., [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) (2024)
- Guo et al., [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.04792) (2024)
- Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (2022)
- Lee et al., [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) (2023)
- Irving et al., [AI Safety via Debate](https://arxiv.org/abs/1805.00899) (2018)
- Burns et al., [Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://arxiv.org/abs/2312.09390) (2023)
- Perez et al., [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286) (2022)
- Ganguli et al., [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) (2022)
- Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (2021)
- Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023)
- Askell et al., [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) (2021)
- Sharma et al., [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) (2023)
