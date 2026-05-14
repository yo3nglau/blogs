---
title: "Model-Free Reinforcement Learning: Interview Questions and Answers"
author: yo3nglau
date: '2026-05-14'
categories:
  - Interview
tags:
  - Deep Learning
  - Reinforcement Learning
  - Model-Free RL
toc: true
---

## Foundations

### Q1 [Basic] What is model-free reinforcement learning and how does it differ from model-based RL?

**Q:** How is model-free reinforcement learning defined, and what does the absence of an environment model imply for learning and sample efficiency?

**A:** **Model-free reinforcement learning** is the family of RL methods that learn a policy or value function directly from environment interaction, without constructing or querying an explicit model of the environment's transition dynamics $p(s' | s, a)$ or reward function $r(s, a)$. The agent learns purely from the raw sequence of $(s_t, a_t, r_t, s_{t+1})$ tuples it experiences, using this stream to update either a value function, a policy, or both.

The absence of a transition model has two immediate consequences. First, the agent cannot plan ahead: it cannot imagine what would happen if it took an action before executing it, so every policy improvement must be grounded in real — or previously collected — experience. Second, data efficiency is constrained: unlike model-based methods that can generate synthetic rollouts from a learned model to augment training, model-free methods must rely solely on real interactions. DreamerV3 (Hafner et al., 2023), for example, achieves competitive performance with 5–20× fewer real environment steps than model-free SAC on the DeepMind Control Suite.

The compensating advantage is simplicity and robustness: model-free methods introduce no model approximation error, and their value or policy estimates, while potentially noisy, are not systematically misled by inaccurate transition dynamics. This makes model-free methods the default choice for tasks where the environment is cheap to interact with (simulation, games) or where the transition dynamics are too complex or stochastic to model accurately.

---

### Q2 [Basic] What is the Bellman equation and how does it form the basis of temporal difference learning?

**Q:** State the Bellman expectation equation for the state-value function and explain how temporal difference learning derives from it.

**A:** The **Bellman expectation equation** expresses the value function recursively: the value of a state under policy $\pi$ equals the expected immediate reward plus the discounted value of the next state,

$$V^\pi(s) = \mathbb{E}_{a \sim \pi,\; s' \sim p}\!\left[r(s,a) + \gamma V^\pi(s')\right]$$

where $\gamma \in [0, 1)$ is the discount factor. The corresponding Bellman optimality equation replaces the policy expectation with a max over actions: $V^*(s) = \max_a \mathbb{E}_{s'}[r(s,a) + \gamma V^*(s')]$.

**Temporal difference (TD) learning** turns this recursive identity into an online update rule. Rather than waiting for the full return from an episode (as Monte Carlo methods do), TD methods use a one-step bootstrap target: the current reward plus the discounted estimated value of the next state, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, called the **TD error**. The value function is updated by $V(s_t) \leftarrow V(s_t) + \alpha \delta_t$. Because the target $r_t + \gamma V(s_{t+1})$ is available after a single transition, TD methods can update online and do not require episodes to terminate.

The key tradeoff introduced by bootstrapping is bias: the TD target depends on the current (imperfect) estimate $V(s_{t+1})$, so early in training the target is inaccurate. However, bootstrapping dramatically reduces variance compared to Monte Carlo, whose return estimates average over the stochasticity of all future time steps. This bias-variance tradeoff is a central organizing principle of model-free RL.

---

### Q3 [Basic] How does the choice between Monte Carlo and TD estimation affect bias and variance?

**Q:** Compare Monte Carlo returns and TD bootstrapping as value estimation strategies, and explain how $n$-step returns and $\lambda$-returns interpolate between them.

**A:** **Monte Carlo (MC) estimation** computes the value of a state as the actual discounted return $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$ from that state. Because $G_t$ is the true return under the current policy, MC estimation is unbiased: $\mathbb{E}[G_t] = V^\pi(s_t)$. However, $G_t$ sums over all future stochastic rewards, so its variance grows with horizon length — in long-horizon or sparse-reward tasks, the variance of MC estimates can be so large that convergence is extremely slow.

**TD(0) bootstrapping** substitutes the one-step target $r_t + \gamma V(s_{t+1})$ for the full return. This introduces bias (the target depends on the current value estimate, which may be wrong) but dramatically reduces variance by cutting off the sum after one step. For most practical tasks, the variance reduction more than compensates for the bias in the early phase of training.

**$n$-step returns** interpolate by summing $n$ actual rewards before bootstrapping: $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$. As $n$ increases, bias decreases and variance increases. **$\lambda$-returns** (TD($\lambda$)) form a geometrically-weighted average over all $n$-step returns: $G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$, with $\lambda = 0$ recovering TD(0) and $\lambda = 1$ recovering MC. Generalized Advantage Estimation (Schulman et al., 2016) applies this same $\lambda$-return structure to advantage estimation in policy gradient methods, providing a tunable bias-variance tradeoff for actor-critic training.

---

### Q4 [Advanced] What is the policy gradient theorem and why does it enable direct policy optimization?

**Q:** Derive the policy gradient theorem and explain how it allows the gradient of expected return with respect to policy parameters to be estimated from sampled trajectories without knowing the environment dynamics.

**A:** The **policy gradient theorem** gives an analytically tractable expression for $\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$ even though the return depends on the environment's unknown transition dynamics. The key identity, established by Williams (1992) via the log-derivative trick, is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right]$$

The derivation replaces $\nabla_\theta p(\tau | \theta)$ with $p(\tau | \theta) \nabla_\theta \log p(\tau | \theta)$ using $\nabla \log f = \nabla f / f$. The log probability of a trajectory factorizes as $\log p(\tau | \theta) = \sum_t \log \pi_\theta(a_t | s_t) + \sum_t \log p(s_{t+1} | s_t, a_t)$; the transition log-probabilities $\log p(s_{t+1} | s_t, a_t)$ vanish from the gradient because they do not depend on $\theta$. Crucially, this cancellation means the dynamics $p$ never appear in the gradient estimator — no model is needed.

The resulting estimator is unbiased: actions with higher $Q$-values receive stronger positive gradient signal, increasing their probability; low-value actions receive negative signal. In practice, $Q^{\pi_\theta}(s_t, a_t)$ is replaced by an estimate — either the Monte Carlo return $G_t$ (REINFORCE) or a learned critic — and a **baseline** $b(s_t)$ is subtracted to reduce variance: $\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b(s_t))$. The baseline does not introduce bias because $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = 0$.

---

## Value-Based Methods

### Q5 [Basic] How does Q-learning work and what are its convergence guarantees?

**Q:** Describe the Q-learning update rule, explain why it is off-policy, and state the conditions under which it is guaranteed to converge to $Q^*$.

**A:** **Q-learning** maintains a table $Q(s, a)$ estimating the optimal action-value function $Q^*(s,a) = \max_\pi \mathbb{E}[G_t | s_t = s, a_t = a]$. At each step, after observing $(s_t, a_t, r_t, s_{t+1})$, it applies:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

The target $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ is the Bellman optimality operator applied to the current estimate. Critically, the $\max$ in the target is taken over all actions regardless of what action the behavior policy executed, making Q-learning **off-policy**: the agent can follow any behavior policy (e.g., $\epsilon$-greedy for exploration) while the update still targets the optimal policy. This off-policy property allows Q-learning to reuse data collected by earlier, less capable policies without bias.

Q-learning converges to $Q^*$ in the tabular case under three conditions: (1) all state-action pairs are visited infinitely often; (2) the step sizes satisfy the Robbins-Monro conditions $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$; and (3) rewards are bounded. These conditions ensure the Bellman operator is a $\gamma$-contraction in the $\ell^\infty$ norm and that the iterates converge to its fixed point. In practice, a fixed $\alpha$ (violating condition 2) is combined with experience replay, and the formal convergence guarantee no longer applies, though empirical performance is typically good.

---

### Q6 [Advanced] What innovations did DQN introduce to make Q-learning work with deep neural networks?

**Q:** Describe DQN's two core stabilization techniques, explain the instability each addresses, and summarize its Atari performance.

**A:** **DQN** (Mnih et al., 2015) extended Q-learning to raw pixel observations using a convolutional neural network as the Q-function approximator. Naïve application of gradient-based Q-learning with a neural network destabilizes training due to two correlated failure modes, each addressed by a dedicated mechanism.

**Experience replay** stores transitions $(s_t, a_t, r_t, s_{t+1})$ in a fixed-size circular buffer and samples random minibatches for gradient updates instead of using the most recent transition. This breaks the temporal correlations in the data stream: sequential observations share overlapping visual features and are nearly identical, creating strongly autocorrelated gradients that cause the network to overfit to recent experience and catastrophically forget earlier learning. Random sampling from the buffer restores the approximately i.i.d. assumption that stochastic gradient descent requires and reuses each transition multiple times, improving data efficiency.

**Target networks** maintain a separate copy $Q_{\theta^-}$ of the Q-network whose parameters are frozen for $C$ steps and then hard-copied from the online network $Q_\theta$. The Bellman target $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$ is computed using the frozen copy. Without this, the target and the network being updated share the same parameters, creating a moving regression target: each gradient step shifts both the prediction and the target, analogous to chasing a moving goalposts and causing oscillation or divergence. The frozen target provides a stable regression objective for $C$ gradient steps.

On 49 Atari games played from raw pixels with identical hyperparameters across all games, DQN surpassed human-level performance on 29 games and exceeded all prior methods on 43 of 49 (Mnih et al., 2015), demonstrating for the first time that a single end-to-end deep RL agent could achieve human-level control across a diverse set of tasks.

---

### Q7 [Advanced] What do Double DQN, Dueling DQN, and Prioritized Experience Replay each address?

**Q:** Identify the specific failure mode each extension corrects and explain the mechanism of each fix.

**A:** **Double DQN** (van Hasselt et al., 2016) addresses the **overestimation bias** inherent in the $\max$ operator. Standard DQN uses the same network to both select the greedy action and evaluate it: the target is $r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta^-}(s',a'))$. Because the same noisy estimates are used for both selection and evaluation, the target is systematically biased upward — the maximum of noisy values exceeds the true maximum in expectation. Double DQN decouples the two roles: the online network $Q_\theta$ selects the action, and the target network $Q_{\theta^-}$ evaluates it:

$$\text{target} = r + \gamma Q_{\theta^-}\!\left(s',\, \arg\max_{a'} Q_\theta(s', a')\right)$$

This cross-evaluation eliminates the upward bias because the action selected by one network is statistically independent of the value assigned by the other.

**Dueling DQN** (Wang et al., 2016) addresses **credit assignment inefficiency** in states where most actions have no effect on the outcome. The network architecture explicitly factorizes $Q(s,a) = V(s) + A(s,a)$, where $V(s)$ is the state-value function and $A(s,a) = Q(s,a) - V(s)$ is the **advantage function**. Two separate network streams estimate $V$ and $A$ and are combined with mean-centering for identifiability: $Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$. The value stream can update $V(s)$ from any transition regardless of which action was taken, improving generalization when actions are irrelevant to immediate reward.

**Prioritized Experience Replay (PER)** (Schaul et al., 2016) addresses **uniform sampling inefficiency**: not all stored transitions are equally informative. PER assigns each transition a priority proportional to the magnitude of its TD error $|\delta_t|$, sampling high-error transitions more frequently. Importance sampling weights $w_i = (N \cdot P(i))^{-\beta}$ correct for the resulting distribution shift, annealing $\beta \to 1$ over training. PER substantially accelerates learning by focusing gradient updates on transitions where the current model is most wrong.

---

### Q8 [Advanced] How does Rainbow combine multiple DQN improvements and which contribute most?

**Q:** Describe Rainbow's six components, identify the most impactful improvements from the ablation study, and explain why combining them is non-trivial.

**A:** **Rainbow** (Hessel et al., 2018) integrates six extensions to DQN: Double Q-learning, Dueling networks, Prioritized Experience Replay, multi-step returns, **distributional RL** (C51, Bellemare et al., 2017), and **Noisy Nets** (Fortunato et al., 2017). On 57 Atari games at 200M environment frames, Rainbow substantially outperforms every individual component and all prior methods at the time of publication.

**Distributional RL** (C51) replaces the scalar Q-value estimate with a full distribution over returns $Z(s,a)$, represented as a discrete probability distribution over $N = 51$ atoms spanning a fixed range. Learning the full return distribution rather than its expectation provides a richer gradient signal, reduces variance in the TD target (the distributional target is more stable than a scalar bootstrap), and captures multi-modal return distributions that a scalar mean would collapse.

**Noisy Nets** replace $\epsilon$-greedy exploration with learned stochastic weights in the network's linear layers: $y = (\mu^w + \sigma^w \odot \varepsilon^w)\,x + (\mu^b + \sigma^b \odot \varepsilon^b)$, where $\varepsilon$ is sampled noise and $(\sigma^w, \sigma^b)$ are learned parameters. Exploration is driven by the network's internal uncertainty rather than an external schedule, enabling state-dependent exploration that adapts as training progresses.

The Rainbow ablation study (Hessel et al., 2018) shows that the two most impactful components on median human-normalized score are **Prioritized Experience Replay** and **multi-step returns**, followed by distributional RL. Combining the components is non-trivial because they interact structurally: distributional RL changes the form of the Bellman target, which affects how PER priorities are computed; multi-step returns require adjusting the distributional projection; and Noisy Nets alter the exploration mechanism that PER was designed to complement.

---

## Policy Gradient Methods

### Q9 [Basic] What is REINFORCE and how do baselines reduce its variance?

**Q:** Describe the REINFORCE algorithm, identify its principal weakness, and explain how a baseline corrects it without introducing bias.

**A:** **REINFORCE** (Williams, 1992) is the prototypical policy gradient algorithm. It collects complete episodes under the current policy $\pi_\theta$, computes the Monte Carlo return $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}$ for each timestep, and updates the policy parameters:

$$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t$$

The update increases the log-probability of actions taken in proportion to their observed return. REINFORCE is unbiased and requires no model of the environment.

Its principal weakness is **high variance**. The return $G_t$ depends on all future rewards in the trajectory, accumulating stochasticity from every subsequent timestep. In long-horizon or sparse-reward tasks, returns vary enormously between trajectories, causing large gradient fluctuations and slow, unstable learning.

A **baseline** $b(s_t)$ reduces variance by subtracting a state-dependent constant from the return: the update becomes $\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (G_t - b(s_t))$. The baseline does not bias the gradient because $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot 0 = 0$. The optimal baseline in terms of variance reduction is approximately $V^\pi(s_t)$; subtracting it leaves the **advantage** $A(s_t, a_t) = G_t - V^\pi(s_t)$, which is centered at zero for average actions and positive or negative only for better or worse-than-average ones. Learning this baseline requires a critic, transitioning REINFORCE into the actor-critic framework.

---

### Q10 [Advanced] How do TRPO and PPO constrain policy updates to prevent destructive gradient steps?

**Q:** Explain the optimization problem TRPO solves, identify its practical limitation, and describe how PPO approximates the same constraint with a first-order method.

**A:** Unconstrained gradient ascent on $J(\theta)$ can take steps that collapse the policy: the gradient is estimated under the old policy distribution but applied to new parameters that induce a very different distribution, causing catastrophically poor performance. **Trust Region Policy Optimization (TRPO)** (Schulman et al., 2015) formalizes this as a constrained optimization: maximize the surrogate objective

$$\mathcal{L}^{CPI}(\theta) = \mathbb{E}_t\!\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t\right]$$

subject to $\mathbb{E}_t[D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t))] \leq \delta$. The surrogate is a first-order approximation to the true objective under the old policy distribution, and the KL constraint ensures the new policy does not deviate far enough to invalidate that approximation.

TRPO solves this constrained problem via conjugate gradient to compute the natural gradient direction, followed by a line search enforcing the KL constraint. This requires computing **Fisher information matrix** vector products, costing multiple backward passes per update and making TRPO impractical for large models with millions of parameters.

**Proximal Policy Optimization (PPO)** (Schulman et al., 2017) achieves comparable policy stability with a first-order optimizer by replacing the hard KL constraint with a **clipped surrogate objective**:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\;\operatorname{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$. The clip removes the gradient incentive to push $r_t$ outside $[1-\epsilon, 1+\epsilon]$, preventing large policy updates without a second-order calculation. PPO applies multiple epochs of minibatch gradient ascent per data collection phase, amortizing sample collection across gradient steps. With $\epsilon = 0.2$, PPO outperforms TRPO on most continuous control and Atari benchmarks while being substantially simpler to implement (Schulman et al., 2017).

---

### Q11 [Basic] How do A3C and A2C use parallel workers to improve policy gradient training?

**Q:** Describe A3C's asynchronous training mechanism, explain why parallelism decorrelates experience, and contrast it with the synchronous A2C variant.

**A:** **A3C** (Asynchronous Advantage Actor-Critic, Mnih et al., 2016) addresses the data correlation problem in on-policy policy gradient methods without experience replay, by running multiple independent actor-learner threads in parallel. Each thread maintains a local copy of the policy and value network, interacts with its own environment instance for $t_{max}$ steps, computes advantage-weighted policy gradient updates locally, and asynchronously applies those gradients to a shared global network. Because each worker operates from a different environment state and follows the same policy independently, their gradients reflect different parts of the environment simultaneously, providing a decorrelated gradient signal analogous to replay buffer sampling but without storing past transitions.

The **advantage estimate** in each worker's update is the $k$-step advantage $\hat{A}_t = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V(s_{t+k}) - V(s_t)$, which interpolates between TD and MC. Both the actor (policy) and critic (value function) are trained: the actor maximizes $\sum_t \log \pi(a_t|s_t) \hat{A}_t$ and the critic minimizes $\sum_t (G_t^{(k)} - V(s_t))^2$. A3C achieves competitive performance with DQN on Atari and Mujoco tasks using only a CPU with 16 threads (Mnih et al., 2016), replacing the GPU and experience replay that DQN requires.

**A2C** is the synchronous variant: all workers collect experience simultaneously and a single update aggregates gradients across workers before applying to the shared network. This eliminates the gradient staleness of asynchronous updates (where workers may compute gradients using parameters that have since been updated by other workers), at the cost of synchronization — each update waits for the slowest worker. On GPU hardware, A2C's synchronous batching typically matches or exceeds A3C's performance because the batched matrix operations benefit more from GPU parallelism.

---

### Q12 [Advanced] What is maximum entropy reinforcement learning and why does entropy regularization help?

**Q:** Define the maximum entropy RL objective, explain the practical benefits of the entropy bonus, and describe how the temperature parameter is automatically tuned.

**A:** **Maximum entropy reinforcement learning** augments the standard RL objective with a policy entropy bonus at every timestep:

$$J_{MaxEnt}(\pi) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t \left(r(s_t, a_t) + \alpha\,\mathcal{H}(\pi(\cdot|s_t))\right)\right]$$

where $\mathcal{H}(\pi(\cdot|s)) = -\sum_a \pi(a|s) \log \pi(a|s)$ is the **policy entropy** and $\alpha > 0$ is the **temperature** hyperparameter. The optimal policy under this objective is stochastic, assigning probability to all actions that achieve near-optimal returns rather than collapsing to a deterministic argmax.

Entropy regularization provides three practical benefits. First, it prevents **premature policy collapse**: a deterministic policy that performs well early in training may overfit to a local optimum; the entropy bonus forces the policy to maintain a minimum spread over actions, keeping it recoverable via gradient updates. Second, it improves **exploration** by encouraging the policy to visit a broader range of state-action pairs, providing richer data for both policy and value learning. Third, it provides a **robustness to reward misspecification** benefit: a maximum-entropy policy retains information about all near-optimal behaviors, making it a better initialization for fine-tuning on related tasks with modified rewards.

The temperature $\alpha$ controls the exploration-exploitation balance: high $\alpha$ forces near-uniform action distributions, while $\alpha \to 0$ recovers the standard RL objective. Haarnoja et al. (2018b) show that $\alpha$ can be automatically tuned by treating it as a Lagrange multiplier enforcing a minimum target entropy $\mathcal{H}^*$: $\min_\alpha \mathbb{E}[-\alpha \log \pi(a|s) - \alpha \mathcal{H}^*]$. Gradient descent on $\alpha$ increases temperature when entropy is below $\mathcal{H}^*$ and decreases it when entropy exceeds the target, eliminating the most sensitive hyperparameter in practice.

---

## Actor-Critic Methods

### Q13 [Basic] What is the actor-critic framework and what problem does it solve compared to REINFORCE?

**Q:** Describe the roles of the actor and critic, explain how the critic replaces the Monte Carlo return, and identify the variance-bias tradeoff this introduces.

**A:** The **actor-critic framework** maintains two function approximators simultaneously: an **actor** $\pi_\theta(a|s)$ (the policy) and a **critic** $V_\psi(s)$ or $Q_\psi(s,a)$ (a value function). The actor is updated via policy gradient using the critic's estimates as a variance-reducing baseline; the critic is updated via temporal difference to track the current policy's value.

The core motivation is variance reduction over REINFORCE. In REINFORCE, the policy gradient uses the Monte Carlo return $G_t$ — a high-variance, unbiased estimate of $Q^\pi(s_t, a_t)$. The actor-critic substitutes the TD advantage $\hat{A}_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$, which uses only one step of actual reward and bootstraps the remainder from the critic. This replaces high-variance cumulative sums with a single-step residual, dramatically reducing gradient variance at the cost of the bias introduced by bootstrapping with an imperfect critic.

The separation of actor and critic parameters enables each component to specialize independently: the critic can use any value learning algorithm (TD, $\lambda$-returns, off-policy data) without being constrained by the actor's update rule, while the actor focuses on policy improvement using the critic's gradient signal. This separation is what enables off-policy actor-critic methods like SAC and TD3 — the critic is trained on experience from any behavior policy, while the actor is updated by differentiating through the critic rather than by policy gradient samples.

---

### Q14 [Advanced] What is Generalized Advantage Estimation and how does it interpolate between TD and MC?

**Q:** Derive the GAE formula, explain how $\lambda$ controls the bias-variance tradeoff, and state the default values used in PPO.

**A:** **Generalized Advantage Estimation (GAE)** (Schulman et al., 2016) provides a family of advantage estimators parameterized by $\lambda \in [0, 1]$ that trade off bias and variance continuously. The starting point is the $k$-step TD error sum: defining $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, the $k$-step advantage can be written as

$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t) = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$$

GAE takes a geometric average over all $k$-step estimators:

$$\hat{A}_t^{GAE(\gamma,\lambda)} = (1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

The simplified form — a discounted sum of TD errors with effective discount $\gamma\lambda$ — is computed recursively in $O(T)$. With $\lambda = 0$, only $\delta_t$ contributes: GAE reduces to the one-step TD advantage (low variance, high bias). With $\lambda = 1$ and $\gamma < 1$, all future TD errors are summed with discount $\gamma$: GAE reduces to the Monte Carlo advantage (high variance, zero bias if $V$ is exact). Values $\lambda \in (0, 1)$ interpolate smoothly between these extremes, allowing the estimator to be tuned as a hyperparameter rather than committing to either endpoint.

In practice, GAE with $\lambda = 0.95$ and $\gamma = 0.99$ is the widely-used default in PPO implementations (Schulman et al., 2017), providing a favorable bias-variance balance across a wide range of continuous control tasks without task-specific tuning.

---

### Q15 [Advanced] How does Soft Actor-Critic incorporate maximum entropy RL for off-policy continuous control?

**Q:** Describe SAC's soft Bellman equation, the role of twin critics, and how it achieves state-of-the-art sample efficiency.

**A:** **Soft Actor-Critic (SAC)** (Haarnoja et al., 2018a) applies the maximum entropy RL objective to off-policy actor-critic training in continuous action spaces. The **soft Q-function** satisfies the soft Bellman equation:

$$Q(s_t, a_t) = \mathbb{E}\!\left[r_t + \gamma\!\left(Q(s_{t+1}, a_{t+1}) - \alpha \log \pi(a_{t+1}|s_{t+1})\right)\right]$$

where the entropy term $-\alpha \log \pi(a_{t+1}|s_{t+1})$ acts as an intrinsic per-step reward for stochasticity. The actor is trained to maximize the soft Q-value minus a KL penalty: $\mathcal{L}_\pi = \mathbb{E}_s[\mathbb{E}_a[\alpha \log \pi(a|s) - Q(s,a)]]$. Because the policy is a **squashed Gaussian** $a = \tanh(\mu + \sigma \varepsilon)$ with $\varepsilon \sim \mathcal{N}(0, I)$, the actor loss is differentiated through the reparameterization trick, providing low-variance gradient estimates.

SAC uses **twin critics** $Q_{\psi_1}$ and $Q_{\psi_2}$ trained independently on the same replay data, taking the minimum of their outputs for both the actor loss and the Bellman target. This clipped double-Q estimate addresses Q-value overestimation, which is particularly damaging in the continuous actor-critic setting because the actor is directly trained to maximize the critic's output — any upward bias in the critic is amplified by the actor update.

Automatic temperature tuning (Haarnoja et al., 2018b) treats $\alpha$ as a Lagrange multiplier on a minimum entropy constraint $\mathbb{E}[\mathcal{H}(\pi(\cdot|s))] \geq \mathcal{H}^*$, updating $\alpha$ by gradient descent alongside the actor and critic. On MuJoCo continuous control benchmarks, SAC with automatic temperature achieves state-of-the-art sample efficiency among off-policy methods, matching or outperforming TD3 on several tasks while requiring fewer hyperparameter decisions (Haarnoja et al., 2018b).

---

### Q16 [Advanced] How does TD3 address the overestimation bias and instability of DDPG?

**Q:** Identify the three specific failure modes in DDPG that TD3 addresses, and describe each corresponding fix.

**A:** **TD3** (Twin Delayed Deep Deterministic policy gradient, Fujimoto et al., 2018) identifies three correlated instabilities in DDPG (Deep Deterministic Policy Gradient) and introduces a targeted fix for each.

**Clipped double-Q learning** addresses overestimation bias from the deterministic policy maximizing over a continuous action space. DDPG uses a single critic; the actor is trained to maximize $Q_\psi(s, \pi_\theta(s))$, amplifying any overestimation in $Q_\psi$ because the actor gradient points directly toward the critic's maximum. TD3 maintains two critics $Q_{\psi_1}$ and $Q_{\psi_2}$, using their minimum for the Bellman target: $y = r + \gamma \min_{i=1,2} Q_{\psi_i}(s', \tilde{a}')$. Taking the minimum provides a conservative, lower-biased estimate that is less susceptible to actor exploitation.

**Delayed policy updates** address the instability from coupling the actor update to an inaccurate critic. In DDPG, both actor and critic update at every step, but the critic converges slower because it regresses on a moving target defined by the actor. TD3 updates the actor and target networks every $d = 2$ critic updates, allowing the critic to stabilize before the actor acts on its estimates. This decoupling substantially reduces the variance of actor gradients.

**Target policy smoothing** addresses Q-function overfitting to narrow peaks: a deterministic policy that maximizes a sharply peaked Q-function exploits inaccurate high-value regions that happen to be concentrated at specific action values. TD3 adds clipped Gaussian noise to the target action $\tilde{a}' = \pi_{\theta^-}(s') + \varepsilon$, $\varepsilon \sim \operatorname{clip}(\mathcal{N}(0, \sigma), -c, c)$, regularizing the critic to assign similar values to nearby actions and smoothing out spurious Q-peaks. On six MuJoCo locomotion and manipulation benchmarks, TD3 outperforms DDPG and contemporaneous methods on five tasks (Fujimoto et al., 2018).

---

## Exploration and Practical Considerations

### Q17 [Basic] What are the main exploration strategies in model-free RL?

**Q:** Compare $\epsilon$-greedy, Upper Confidence Bound, and entropy-based exploration, and explain when each is most appropriate.

**A:** **$\epsilon$-greedy exploration** selects a uniformly random action with probability $\epsilon$ and the greedy action otherwise. Typically $\epsilon$ is annealed from 1 to a small final value (e.g., 0.01) over training. $\epsilon$-greedy is universal and easy to implement, working well in value-based methods like DQN, but it is undirected — random exploration does not adapt to the agent's uncertainty and is inefficient in large state spaces where the informative states are sparse.

**Upper Confidence Bound (UCB)** methods quantify uncertainty about each action's value and add an exploration bonus: $a^* = \arg\max_a \left[Q(s,a) + c\sqrt{\ln t / N(s,a)}\right]$, where $N(s,a)$ is the visit count and $c$ controls exploration strength. UCB implements the **optimism in the face of uncertainty** principle — prefer actions about which less is known. In tabular MDPs, UCB achieves regret bounds that are optimal up to logarithmic factors (Auer et al., 2002). In deep RL with continuous state spaces, exact visitation counts are intractable; approximate UCB methods estimate uncertainty via ensembles or Bayesian approximations.

**Entropy-based exploration** (used in maximum entropy RL and SAC) explicitly maximizes policy entropy at each state, incentivizing the agent to spread probability mass over multiple actions. Unlike $\epsilon$-greedy, the entropy bonus is state-dependent — the agent explores more in states where it is uncertain and less where it has converged — and is learned jointly with the value function. Entropy-based exploration is most effective in continuous action spaces where $\epsilon$-greedy random perturbations are insufficient for structured exploration, and where the policy can learn to be precisely stochastic in informative states.

---

### Q18 [Advanced] How do intrinsic motivation methods drive exploration in hard-exploration environments?

**Q:** Describe the curiosity-driven and count-based intrinsic reward frameworks, and summarize their results on hard-exploration Atari games.

**A:** **Intrinsic motivation** augments the agent's reward signal with a self-generated **intrinsic reward** $r^i_t$ that quantifies how novel or surprising a state transition is, independent of the external task reward. This enables progress in sparse-reward environments where the extrinsic signal provides no gradient until the goal is accidentally reached after a long sequence of precise actions.

**Count-based intrinsic rewards** (Bellemare et al., 2016) generalize the tabular visitation count $N(s)$ to continuous state spaces using a Context Tree Switching (CTS) density model to estimate **pseudo-counts** $\hat{N}(s)$: the intrinsic reward is $r^i_t = (\hat{N}(s_t) + 0.01)^{-1/2}$, decaying as states are revisited. This incentivizes the agent to seek genuinely novel states. On Montezuma's Revenge — an Atari game requiring a long sequence of precise actions before any external reward — count-based exploration achieves scores orders of magnitude above DQN's baseline.

**The Intrinsic Curiosity Module (ICM)** (Pathak et al., 2017) uses **prediction error** as the intrinsic reward: a forward model predicts the next latent state $\hat{\phi}(s_{t+1})$ from the current latent state and action, and the intrinsic reward is $r^i_t = \frac{\eta}{2}\|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|^2$. An inverse model jointly trains $\phi$ to encode only aspects of the environment controllable by the agent's actions, filtering out irrelevant environmental stochasticity (e.g., swaying background). ICM enables progress in VizDoom and Super Mario Bros. without any external reward (Pathak et al., 2017).

**Random Network Distillation (RND)** (Burda et al., 2019) avoids the "noisy TV problem" of prediction-error methods — where a stochastic uncontrollable stimulus provides infinite curiosity — by measuring novelty as the prediction error of a network trained to match a fixed random target network: $r^i_t = \|f(s_t) - \hat{f}(s_t)\|^2$. Because the random target is deterministic, the prediction error reflects only visit frequency, not environmental stochasticity. RND achieves a mean score exceeding 10,000 on Montezuma's Revenge (Burda et al., 2019), far above prior exploration methods.

---

### Q19 [Advanced] What are the convergence guarantees and theoretical limits of model-free RL with function approximation?

**Q:** State the convergence result for tabular Q-learning, explain the deadly triad, and describe which combinations of approximation, bootstrapping, and off-policy data lead to divergence.

**A:** **Tabular Q-learning** converges to $Q^*$ with probability 1 under the Robbins-Monro conditions: all state-action pairs visited infinitely often, step sizes satisfying $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$, and bounded rewards. The proof relies on the Bellman optimality operator being a $\gamma$-contraction in the $\ell^\infty$ norm, ensuring the iterates converge to the unique fixed point. This guarantee also extends to on-policy TD with linear function approximation under mild conditions on the feature representation.

**The deadly triad** is the combination of three elements — off-policy training data, function approximation, and bootstrapping (TD targets) — which together can cause value estimates to diverge to infinity even with linear function approximators. The canonical demonstration is a simple six-state MDP where TD(0) with linear features and an off-policy data distribution is not a contraction: the semi-gradient update rule follows a gradient step in a direction that does not correspond to any loss function, and the iterates can grow without bound. This is not a pathological edge case; it represents the generic behavior when the three elements combine without special structure.

Modern deep RL algorithms mitigate but do not eliminate the triad through practical heuristics. Experience replay reduces the degree of off-policy-ness by sampling from a buffer with moderate temporal diversity rather than a completely different policy. Target networks stabilize bootstrapping by providing a slowly-moving regression target. Gradient clipping limits weight update magnitudes. Despite these heuristics, deep off-policy Q-learning can diverge on certain environments with poor hyperparameters, and no general convergence proofs exist for the nonlinear case. On-policy methods (PPO, A3C) avoid the deadly triad entirely by training only on experience collected under the current policy, at the cost of sample efficiency.

---

### Q20 [Advanced] What are the principal failure modes of model-free RL in practice?

**Q:** Identify the most common failure modes when deploying model-free RL on real tasks, and describe the mechanism and mitigation for each.

**A:** **Reward hacking and specification gaming** occurs when the policy finds behaviors that maximize the specified reward function but violate the designer's intent. Because model-free agents optimize a scalar signal without any built-in representation of task semantics, they exploit any gap between the specified reward and the true objective — a robot learning to exploit physics simulator artifacts to achieve a high position rather than walking, or an agent pausing to avoid accumulating negative reward rather than completing a task. Mitigation requires careful reward design, domain-specific reward shaping, or reward learning from human demonstrations or preferences.

**Distributional shift in off-policy learning** arises when the data distribution in the replay buffer differs substantially from the distribution induced by the current policy. The Q-network trained on past-policy data may overestimate values for state-action pairs that the current, more capable policy frequently visits but that were underrepresented in the buffer — a form of extrapolation error. This is most severe in offline RL where the buffer is completely fixed; conservative methods like CQL address it through pessimistic value estimation.

**Hyperparameter sensitivity** is a pervasive practical limitation: model-free algorithms, especially policy gradient methods, are brittle across hyperparameter configurations. PPO's performance varies by orders of magnitude across choices of learning rate, entropy coefficient, clip ratio, and network architecture (Schulman et al., 2017). This fragility makes reliable deployment difficult without extensive tuning or automated hyperparameter optimization. Methods like SAC reduce sensitivity somewhat through off-policy reuse and automatic temperature tuning, but deep RL remains substantially more hyperparameter-sensitive than supervised learning.

**Sample inefficiency** remains the fundamental constraint relative to model-based methods: solving continuous locomotion from pixels typically requires tens of millions of environment steps for model-free methods versus thousands for Dreamer-class world models. In real-world robotics where each trial consumes physical time and hardware wear, model-free methods are often prohibitively expensive without a high-fidelity simulator for pretraining.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Model-free RL definition: no environment model, sample efficiency tradeoff | Foundations |
| Q2 | Basic | Bellman equation and temporal difference learning | Foundations |
| Q3 | Basic | MC vs TD: bias-variance tradeoff, $n$-step and $\lambda$-returns | Foundations |
| Q4 | Advanced | Policy gradient theorem: log-derivative trick, unbiased gradient estimator | Foundations |
| Q5 | Basic | Q-learning: off-policy update rule and tabular convergence conditions | Value-Based Methods |
| Q6 | Advanced | DQN: experience replay and target networks — stabilization mechanisms | Value-Based Methods |
| Q7 | Advanced | Double DQN, Dueling DQN, PER: specific failure modes and fixes | Value-Based Methods |
| Q8 | Advanced | Rainbow: six components, ablation results, non-trivial interactions | Value-Based Methods |
| Q9 | Basic | REINFORCE: Monte Carlo policy gradient, variance, baselines | Policy Gradient Methods |
| Q10 | Advanced | TRPO and PPO: trust region constraint and clipped surrogate objective | Policy Gradient Methods |
| Q11 | Basic | A3C and A2C: asynchronous and synchronous parallel workers | Policy Gradient Methods |
| Q12 | Advanced | Maximum entropy RL: entropy bonus and automatic temperature tuning | Policy Gradient Methods |
| Q13 | Basic | Actor-critic framework: actor and critic roles, variance vs REINFORCE | Actor-Critic Methods |
| Q14 | Advanced | GAE: $\lambda$-weighted TD errors, interpolation between TD and MC | Actor-Critic Methods |
| Q15 | Advanced | SAC: soft Bellman equation, twin critics, automatic temperature | Actor-Critic Methods |
| Q16 | Advanced | TD3: clipped double-Q, delayed updates, target policy smoothing | Actor-Critic Methods |
| Q17 | Basic | Exploration strategies: $\epsilon$-greedy, UCB, entropy-based | Exploration and Practical Considerations |
| Q18 | Advanced | Intrinsic motivation: CTS, ICM, RND — hard-exploration benchmarks | Exploration and Practical Considerations |
| Q19 | Advanced | Convergence: deadly triad, tabular guarantees, deep RL limitations | Exploration and Practical Considerations |
| Q20 | Advanced | Failure modes: reward hacking, distributional shift, hyperparameter sensitivity | Exploration and Practical Considerations |

## Resources

- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (1992)
- Auer et al., [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352) (UCB, 2002)
- Mnih et al., [Human-level control through deep reinforcement learning](https://arxiv.org/abs/1312.5602) (DQN, 2015)
- Schulman et al., [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) (TRPO, 2015)
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (GAE, 2016)
- Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (A3C, 2016)
- van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (Double DQN, 2016)
- Wang et al., [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) (Dueling DQN, 2016)
- Schaul et al., [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (PER, 2016)
- Bellemare et al., [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868) (CTS, 2016)
- Fortunato et al., [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) (Noisy Nets, 2017)
- Bellemare et al., [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.09096) (C51, 2017)
- Pathak et al., [Curiosity-driven Exploration by Self-Supervised Prediction](https://arxiv.org/abs/1705.05363) (ICM, 2017)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO, 2017)
- Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (SAC, 2018a)
- Fujimoto et al., [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (TD3, 2018)
- Hessel et al., [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) (Rainbow, 2018)
- Haarnoja et al., [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) (SAC, 2018b)
- Burda et al., [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) (RND, 2019)
- Hafner et al., [Mastering Diverse Domains with World Models](https://arxiv.org/abs/2301.04104) (DreamerV3, 2023)
