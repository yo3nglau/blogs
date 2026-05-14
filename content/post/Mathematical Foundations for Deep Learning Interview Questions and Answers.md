---
title: "Mathematical Foundations for Deep Learning: Interview Questions and Answers"
author: yo3nglau
date: '2026-05-14'
categories:
  - Interview
tags:
  - Deep Learning
  - Mathematics
  - Optimization
toc: true
---

## Linear Algebra

### Q1 [Basic] Describe the geometric meaning of eigenvalues and eigenvectors

**Q:** What do eigenvalues and eigenvectors represent geometrically, and why are they useful for understanding linear transformations in machine learning?

**A:** For a square matrix $A$ and a nonzero vector $v$, the **eigenvector** equation $Av = \lambda v$ states that applying $A$ to $v$ produces the same vector scaled by the scalar **eigenvalue** $\lambda$. Geometrically, $v$ is a special direction that the transformation does not rotate — it only stretches (if $|\lambda| > 1$), compresses (if $|\lambda| < 1$), or reflects (if $\lambda < 0$). All other vectors are both rotated and scaled.

For symmetric positive semidefinite matrices — which arise naturally as covariance matrices and Gram matrices — all eigenvalues are non-negative and all eigenvectors are mutually orthogonal. This orthogonality means the eigenvectors form a natural coordinate system aligned with the transformation's principal directions. In practice, the covariance matrix of a dataset has eigenvectors pointing in the directions of greatest variance, and eigenvalues measuring the variance magnitude in each direction. Attention score matrices $QK^\top$ and weight matrices in linear layers are analyzed through their spectra to understand signal propagation and conditioning.

---

### Q2 [Basic] Explain SVD and how it generalizes eigendecomposition to non-square matrices

**Q:** What is the Singular Value Decomposition, and how does it relate to eigendecomposition when the matrix is not square?

**A:** The **Singular Value Decomposition** (SVD) factorizes any $m \times n$ matrix $A$ as $A = U\Sigma V^\top$, where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal matrices and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$ called **singular values**. The columns of $U$ are left singular vectors; the columns of $V$ are right singular vectors. Unlike eigendecomposition, SVD is defined for any matrix regardless of shape or whether it is diagonalizable.

The connection to eigendecomposition is exact: $A^\top A = V \Sigma^\top \Sigma V^\top$ is the eigendecomposition of the symmetric PSD matrix $A^\top A$, so the singular values of $A$ are the square roots of the eigenvalues of $A^\top A$. Equivalently, $AA^\top = U \Sigma \Sigma^\top U^\top$. The largest singular value $\sigma_1 = \|A\|_2$ is the operator norm, which controls how much $A$ can amplify an input vector.

In deep learning, SVD appears in weight matrix analysis, gradient analysis (Jacobians of layers), attention mechanism study, and low-rank adaptation methods such as LoRA, which represent weight updates as low-rank factors $\Delta W = BA$ where $\text{rank}(BA) \ll \min(m,n)$.

---

### Q3 [Advanced] Analyze PCA through SVD and the best low-rank approximation theorem

**Q:** How does PCA relate to SVD, and what theoretical guarantee justifies truncating to the top singular values?

**A:** **Principal Component Analysis** (PCA) seeks the $k$-dimensional linear subspace that retains maximum variance from a centered data matrix $X \in \mathbb{R}^{n \times d}$ (each row is a data point, columns are zero-mean). The sample covariance is $C = X^\top X / n$, which has eigendecomposition $C = V\Lambda V^\top$. The principal components are the columns of $V$, and the projected data is $Z = XV \in \mathbb{R}^{n \times k}$.

This is exactly the SVD of $X$: writing $X = U\Sigma V^\top$, the columns of $V$ are the right singular vectors (principal directions), the columns of $U\Sigma$ are the projections onto those directions, and $\sigma_i^2 / n$ equals the variance explained by the $i$-th principal component. PCA and SVD are therefore two views of the same computation.

The theoretical justification for keeping only the top $k$ singular values is the **Eckart–Young–Mirsky theorem**: among all rank-$k$ matrices $B$, the truncated SVD $A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top$ minimizes the Frobenius norm approximation error $\|A - B\|_F$. The residual is $\|A - A_k\|_F^2 = \sum_{i > k} \sigma_i^2$. This optimality guarantee means that discarding small singular values is the best possible rank-constrained compression, not just a heuristic. It also explains why SVD-based dimensionality reduction is information-optimal under a linear model and least-squares loss.

---

### Q4 [Advanced] Explain matrix conditioning and its effects on deep network training

**Q:** What is the condition number of a matrix, and how does poor conditioning manifest in deep network optimization?

**A:** The **condition number** of a matrix $A$ is $\kappa(A) = \sigma_{\max} / \sigma_{\min}$, the ratio of its largest to smallest singular value. For orthogonal matrices $\kappa = 1$ (perfectly conditioned); $\kappa \gg 1$ indicates ill-conditioning. Geometrically, the condition number measures how much $A$ distorts the shape of the unit sphere: a sphere becomes an ellipsoid with aspect ratio $\kappa$.

In optimization, the condition number of the Hessian $H = \nabla^2 L$ at a point determines how quickly gradient descent converges. For a quadratic loss, the optimal step size is $2/(\lambda_{\max} + \lambda_{\min})$ and convergence requires $O(\kappa(H) \log(1/\epsilon))$ steps, compared to $O(\log(1/\epsilon))$ if $\kappa = 1$. In a deep network, poorly conditioned weight matrices cause different gradient magnitudes along different directions of the parameter space, forcing a globally conservative learning rate and leading to slow training.

Practical mitigations attack conditioning at multiple levels. **Weight initialization** schemes (Glorot & Bengio, 2010; He et al., 2015) choose variance so that each layer approximately preserves signal magnitude, keeping the composition of Jacobians near the identity in spectral norm. **Batch normalization** (Ioffe & Szegedy, 2015) normalizes pre-activations at each layer, which smooths the loss landscape and reduces effective condition number, empirically enabling larger learning rates and faster convergence. Gradient clipping addresses exploding gradients (another manifestation of ill-conditioning in deep recurrent networks) by capping the gradient norm rather than the individual components.

---

### Q5 [Advanced] Describe how matrix calculus handles vector-valued and matrix-valued derivatives

**Q:** How do gradients, Jacobians, and the chain rule extend from scalar to vector and matrix inputs, and why does this matter for backpropagation?

**A:** For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** $\nabla_x f \in \mathbb{R}^n$ has components $(\nabla_x f)_i = \partial f / \partial x_i$. For a vector function $f: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** $J \in \mathbb{R}^{m \times n}$ has $J_{ij} = \partial f_i / \partial x_j$. Backpropagation propagates **vector-Jacobian products** (VJPs) rather than full Jacobians: given an upstream gradient $\bar{y} \in \mathbb{R}^m$ (the gradient of the loss w.r.t. the function output), the contribution to the input gradient is $\bar{x} = J^\top \bar{y} \in \mathbb{R}^n$, computed in $O(mn)$ without materializing $J$.

Several identities recur throughout deep learning. For a linear layer $y = Wx$:
- $\partial L / \partial x = W^\top (\partial L / \partial y)$
- $\partial L / \partial W = (\partial L / \partial y)\, x^\top$

For a quadratic form $f = x^\top A x$: $\nabla_x f = (A + A^\top)x$, which simplifies to $2Ax$ when $A$ is symmetric. For a trace $f = \mathrm{tr}(AB)$: $\partial f / \partial A = B^\top$.

The practical implication is that every layer in a network must implement two functions: a forward pass that computes the output given the input (and caches it), and a backward pass that computes the VJP given the upstream gradient. Reverse-mode automatic differentiation (the mechanism underlying all major deep learning frameworks) chains these VJPs through the computation graph, achieving the cost of two forward passes regardless of the number of parameters — a crucial property when the number of parameters vastly exceeds the output dimension.

---

## Probability and Statistics

### Q6 [Basic] Explain MLE and how it derives common loss functions

**Q:** How does maximum likelihood estimation motivate the choice of loss function, and what assumptions lead to MSE versus cross-entropy?

**A:** **Maximum Likelihood Estimation** (MLE) chooses model parameters $\theta$ to maximize the probability of the observed data: $\hat\theta = \arg\max_\theta \prod_i p_\theta(x_i)$. Taking logarithms converts the product to a sum and changes the sign to obtain an equivalent minimization: $\hat\theta = \arg\min_\theta \sum_i -\log p_\theta(x_i)$. The choice of the likelihood model $p_\theta$ directly determines the loss function.

If the target $y$ given input $x$ is modeled as **Gaussian**: $p_\theta(y|x) = \mathcal{N}(f_\theta(x),\, \sigma^2)$, then $-\log p_\theta(y|x) = (y - f_\theta(x))^2 / 2\sigma^2 + \text{const}$. Minimizing this negative log-likelihood w.r.t. $\theta$ is equivalent to **Mean Squared Error** regression. If $y$ is modeled as **Bernoulli**: $p_\theta(y|x) = \sigma(f_\theta(x))^y (1-\sigma(f_\theta(x)))^{1-y}$, the negative log-likelihood is the **binary cross-entropy**. For a $C$-class **categorical** model with softmax outputs $q_c$, it is the multiclass **cross-entropy** $-\log q_{y_\text{true}}$.

This framework also generalizes to more exotic loss functions. A Laplace likelihood $p_\theta(y|x) \propto \exp(-|y - f_\theta(x)| / b)$ gives the **L1 (MAE) loss**, which is more robust to outliers than MSE because Laplace tails are heavier than Gaussian tails. MLE thus provides a principled derivation of virtually every standard loss: the choice of distribution encodes an assumption about the noise structure of the data.

---

### Q7 [Basic] Describe the role of the Gaussian distribution in deep learning

**Q:** Why is the Gaussian distribution so prevalent in deep learning theory and practice?

**A:** The **Gaussian distribution** $\mathcal{N}(\mu, \sigma^2)$ with density $p(x) = (2\pi\sigma^2)^{-1/2}\exp(-(x-\mu)^2 / 2\sigma^2)$ appears throughout deep learning for two fundamental reasons. First, it is the **maximum entropy distribution** over $\mathbb{R}$ subject to a fixed mean and variance: among all distributions with the same first two moments, the Gaussian makes the fewest additional assumptions, making it the least-informative choice consistent with observed statistics. Second, the **Central Limit Theorem** guarantees that the sum of many independent random variables converges in distribution to a Gaussian regardless of their individual distributions, explaining why Gaussian noise models are appropriate whenever outcomes result from many small additive contributions.

In practice, Gaussians appear in weight initialization (small Gaussian noise prevents symmetry-breaking failures), the latent prior of VAEs ($p(z) = \mathcal{N}(0, I)$), Gaussian noise augmentation for robustness, and the theoretical analysis of infinitely wide networks (Neural Tangent Kernel; the function space of an infinitely wide randomly initialized network is a Gaussian process). The multivariate Gaussian $\mathcal{N}(\mu, \Sigma)$ has the additional property of **closure under linear transformations**: if $x \sim \mathcal{N}(\mu, \Sigma)$ and $A$ is a linear map, then $Ax \sim \mathcal{N}(A\mu, A\Sigma A^\top)$. This closure makes Gaussians analytically tractable throughout probabilistic machine learning.

---

### Q8 [Advanced] Analyze the bias-variance decomposition and the double descent phenomenon

**Q:** How does the classical bias-variance trade-off relate to the double descent phenomenon observed in modern overparameterized models?

**A:** The **bias-variance decomposition** expresses the expected test error of a model $\hat{f}$ at a point $x$ as:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f^*(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}} + \sigma^2_\text{noise}$$

Bias measures systematic error from model assumptions; variance measures sensitivity to the specific training set. The classical picture predicts a U-shaped test error curve: small models underfit (high bias, low variance), large models overfit (low bias, high variance), and an optimal complexity minimizes the sum.

The **double descent** phenomenon (Belkin et al., 2019) challenges this picture. At the **interpolation threshold** — the point where model capacity exactly matches the number of training samples — test error spikes because the model is forced to fit every training point but does so in a memorization-without-structure manner. As capacity continues to grow beyond this threshold, test error falls again and can match or beat the classical optimum. This second descent occurs because highly overparameterized models have many solutions that exactly fit the training data; the implicit bias of gradient descent selects the **minimum-norm** or **smoothest** such solution, which generalizes well.

Double descent has been observed in linear regression, kernel methods, random forests, and deep neural networks (Belkin et al., 2019). It implies that the classical bias-variance framing is incomplete for overparameterized regimes and that regularization via early stopping or explicit weight penalties may be unnecessary — or even harmful — when sufficient capacity is available. The phenomenon also depends on the choice of interpolating algorithm: minimum-norm least squares shows clean double descent, while other interpolators do not.

---

### Q9 [Advanced] Explain concentration inequalities and how they underpin generalization theory

**Q:** What are concentration inequalities, and how are they used to derive bounds on the generalization gap?

**A:** **Concentration inequalities** quantify how tightly a random variable concentrates around its mean. **Markov's inequality** states that for any non-negative $X$: $P(X \geq a) \leq \mathbb{E}[X] / a$ — a weak bound requiring only finite expectation. **Chebyshev's inequality** tightens this using variance: $P(|X - \mu| \geq k\sigma) \leq 1/k^2$, but still gives polynomial tail decay. **Hoeffding's inequality** is much stronger for bounded independent random variables $X_1, \ldots, X_n \in [a_i, b_i]$:

$$P\!\left(\left|\bar{X} - \mathbb{E}[\bar{X}]\right| \geq t\right) \leq 2\exp\!\left(\frac{-2n^2 t^2}{\sum_i (b_i - a_i)^2}\right)$$

The tail decays exponentially in $n$, making Hoeffding bounds much tighter than Chebyshev for averages over many samples.

Generalization theory applies these tools to bound the gap between training and test error. **VC-dimension** bounds show that the generalization gap of a hypothesis class of VC-dimension $d$ scales as $O(\sqrt{d/n})$. **Rademacher complexity** gives tighter instance-dependent bounds by measuring how well the class fits random noise. **PAC-Bayes bounds** express the generalization gap in terms of $\mathrm{KL}(Q \| P)$ where $Q$ is the posterior over learned models and $P$ is a prior, motivating Bayesian approaches and connecting to the regularization interpretation of weight priors. These bounds are rarely tight enough for practical neural networks, but they provide qualitative understanding of why more data, simpler models, or strong priors improve generalization.

---

### Q10 [Advanced] Describe how placing priors on weights connects Bayesian inference to regularization

**Q:** How does the MAP estimation framework unify regularization with probabilistic modeling, and how does the Bayesian view extend beyond penalized optimization?

**A:** **MAP estimation** maximizes the posterior $p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta)\, p(\theta)$. Taking the negative log:

$$\hat\theta_\text{MAP} = \arg\min_\theta \left[-\sum_i \log p(y_i | x_i, \theta) - \log p(\theta)\right]$$

The first term is the negative log-likelihood (the standard loss), and the second term $-\log p(\theta)$ acts as a regularizer determined by the prior. An **isotropic Gaussian prior** $p(\theta) = \mathcal{N}(0, \tau^2 I)$ gives $-\log p(\theta) = \|\theta\|_2^2 / 2\tau^2 + \text{const}$, which is exactly **L2 regularization** (weight decay) with strength $\lambda = 1/(2\tau^2)$. A **Laplace prior** $p(\theta) \propto \exp(-|\theta|/b)$ gives $-\log p(\theta) \propto \|\theta\|_1$, which is **L1 regularization**, inducing sparsity by setting many weights exactly to zero.

Beyond MAP, full **Bayesian inference** maintains a posterior distribution over parameters rather than a point estimate, averaging predictions over all plausible $\theta$: $p(y^* | x^*, \mathcal{D}) = \int p(y^* | x^*, \theta)\, p(\theta | \mathcal{D})\, d\theta$. This produces better-calibrated uncertainty estimates but is computationally intractable for neural networks. Approximate inference methods address this: **Dropout** at test time corresponds to Monte Carlo sampling under a specific approximate posterior (Gal & Ghahramani, 2016), connecting a simple regularization technique to a principled probabilistic interpretation. **Laplace approximation** fits a Gaussian to the posterior around the MAP estimate using the observed Hessian, enabling efficient posterior predictive distributions.

---

## Calculus and Optimization

### Q11 [Basic] Explain how backpropagation implements the chain rule on computational graphs

**Q:** What does backpropagation actually compute, and how does the computational graph structure make it efficient?

**A:** **Backpropagation** is reverse-mode automatic differentiation applied to a neural network's computation graph. It computes the gradient of the scalar loss $L$ with respect to every parameter by applying the chain rule: for any intermediate variable $z$ and its upstream successor $y$, the gradient satisfies $\partial L / \partial z = (\partial y / \partial z)^\top (\partial L / \partial y)$, a **vector-Jacobian product** (VJP).

The computation has two phases. The **forward pass** evaluates each operation in topological order, computing all intermediate values and caching those needed for the backward pass. The **backward pass** traverses the graph in reverse order, accumulating gradients: each node receives the upstream gradient $\partial L / \partial y$, computes the VJP $(\partial y / \partial z)^\top (\partial L / \partial y)$, and passes the result downstream. The gradient of $L$ with respect to any intermediate quantity is the sum of VJPs from all paths through the graph, computed without explicitly forming any Jacobian matrix.

The efficiency stems from reuse: reverse mode computes the gradient of a scalar output with respect to all inputs in a single backward pass with cost proportional to one forward pass (times a small constant). This is in contrast to forward-mode automatic differentiation, which computes the directional derivative for one input direction per pass and is efficient when the input is small but the output is large. Since deep learning losses are scalars and parameter counts are large, reverse mode is universally used.

---

### Q12 [Basic] Describe gradient descent convergence and how momentum improves it

**Q:** What determines how fast gradient descent converges, and why does momentum accelerate training?

**A:** For gradient descent $\theta \leftarrow \theta - \eta \nabla L(\theta)$ applied to an $L$-smooth (Lipschitz gradient) convex loss, convergence to a global minimum is guaranteed at rate $O(1/T)$ with step size $\eta = 1/L$. For $\mu$-strongly convex losses — those with a unique minimum and a curvature lower bound — convergence is **linear** (geometric): $L(\theta_T) - L(\theta^*) \leq (1 - \mu/L)^T [L(\theta_0) - L(\theta^*)]$. The condition number $\kappa = L/\mu$ governs the rate: large $\kappa$ means slow convergence, as gradients are large far from the minimum (where the loss is flat along some directions) and small near it (where the loss curves steeply along others).

**Momentum** (Polyak, 1964) maintains a velocity vector $v_t$ and updates as $v_t = \beta v_{t-1} + \nabla L(\theta_{t-1})$, $\theta_t = \theta_{t-1} - \eta v_t$. By accumulating past gradients, momentum dampens oscillations in directions where the gradient sign flips frequently (narrow valleys) and amplifies movement in directions with consistent gradient sign. **Nesterov's accelerated gradient** achieves the optimal convergence rate of $O(1/T^2)$ for smooth convex functions by computing the gradient at the look-ahead point $\theta - \beta v$. **Adam** combines momentum (first-moment estimate of the gradient) with **adaptive learning rates** (second-moment estimate of the squared gradient): $\theta \leftarrow \theta - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$, effectively rescaling each parameter's learning rate by the inverse root mean squared gradient, making it robust to poorly conditioned loss landscapes.

---

### Q13 [Advanced] Analyze the loss landscape of deep networks and the role of saddle points

**Q:** How does the non-convex loss landscape of a deep network differ from a convex one, and what does this mean for first-order optimization methods?

**A:** The loss landscape of a deep neural network is highly **non-convex**: it has exponentially many critical points $\nabla L = 0$. Critical points are classified by the Hessian's eigenvalues: a **local minimum** has all positive eigenvalues; a **local maximum** has all negative; a **saddle point** has both. For linear networks, all local minima are provably global minima (Baldi & Hornik, 1989 convention holds; every local minimum is a global one), but for deep nonlinear networks the structure is richer.

Dauphin et al. (2014) argued, drawing on random matrix theory results, that in high-dimensional networks the majority of critical points are saddle points rather than local minima — and crucially, the loss value at saddle points is often close to the global minimum loss. The key observation is that in high dimensions it is exponentially unlikely for all Hessian eigenvalues to be positive simultaneously: most degenerate critical points are near-flat (zero or near-zero negative curvature) rather than deep local traps. Gradient descent perturbed by minibatch noise escapes strict saddle points in polynomial time (Jin et al., 2017), making first-order stochastic methods effective in practice.

Overparameterized networks add a further structural feature: at interpolation (zero training loss), the set of global minima forms a high-dimensional manifold. SGD's noise introduces an **implicit bias** toward flatter regions of this manifold, which correspond to solutions with better generalization. Sharpness-aware minimization explicitly targets flat minima by perturbing weights in the direction of maximum loss increase before computing the gradient update, at the cost of doubled gradient evaluations per step.

---

### Q14 [Advanced] Explain second-order optimization and why it is rarely used in deep learning

**Q:** What does Newton's method offer over gradient descent, and what prevents it from being practical for deep networks?

**A:** **Newton's method** updates parameters as $\theta \leftarrow \theta - H^{-1} \nabla L$, where $H = \nabla^2 L$ is the Hessian. For strongly convex smooth functions, Newton's method converges **quadratically** near the solution: the number of correct digits doubles each iteration. It is also **scale-invariant** — it automatically accounts for curvature differences across parameter dimensions, making the effective condition number 1 near the optimum. On a quadratic loss it converges in a single step.

The practical barriers are severe. Storing $H \in \mathbb{R}^{p \times p}$ requires $O(p^2)$ memory, and inverting it costs $O(p^3)$ — both intractable for $p \gtrsim 10^6$ parameters. **Quasi-Newton methods** (BFGS, L-BFGS) approximate $H^{-1}$ using rank-2 updates from past gradient differences, reducing memory to $O(mp)$ for $m$ stored gradient pairs, and are widely used for small-to-medium networks or final fine-tuning steps.

The **natural gradient** replaces the Euclidean metric with the **Fisher information matrix** $F = \mathbb{E}[\nabla \log p_\theta \nabla \log p_\theta^\top]$, giving a gradient that is invariant to reparameterization of the model. This is theoretically appealing — the Fisher metric captures the intrinsic curvature of the model family — but $F$ has the same $O(p^2)$ storage issue as the Hessian. **K-FAC** (Martens & Grosse, 2015) approximates $F$ as a Kronecker product $F \approx A \otimes G$ per layer, where $A$ and $G$ are cheap-to-compute covariance matrices of activations and pre-activation gradients. This reduces inversion cost to $O(n^3 + m^3)$ per layer and has achieved practical speedups in image classification and reinforcement learning, though it requires careful implementation and is still significantly more expensive than Adam per step.

---

### Q15 [Advanced] Analyze vanishing and exploding gradients and how architectural choices address them

**Q:** What causes vanishing and exploding gradients in deep networks, and what mechanisms have proven most effective at resolving them?

**A:** In a network of depth $L$, the gradient of the loss with respect to parameters in layer 1 involves the product of Jacobians along the entire computation path:

$$\frac{\partial L}{\partial \theta^{(1)}} = \frac{\partial L}{\partial z^{(L)}} \prod_{\ell=2}^{L} \frac{\partial z^{(\ell)}}{\partial z^{(\ell-1)}} \cdot \frac{\partial z^{(1)}}{\partial \theta^{(1)}}$$

If the spectral radius $\rho\!\left(\partial z^{(\ell)} / \partial z^{(\ell-1)}\right) < 1$ at each layer, the product of $L-1$ Jacobians decays **exponentially** — vanishing gradients. If $\rho > 1$, gradients grow exponentially — exploding gradients. Either regime makes training early layers effectively impossible, which was a major obstacle before the mid-2010s.

Four complementary solutions have proven effective. **Careful initialization** (Glorot & Bengio, 2010) sets $\text{Var}(W) = 2/(\text{fan\_in} + \text{fan\_out})$ to preserve signal variance through linear layers (Xavier/Glorot initialization). He et al. (2015) corrected this for ReLU nonlinearities, setting $\text{Var}(W) = 2/\text{fan\_in}$ (Kaiming/He initialization), ensuring that the expected magnitude of activations remains stable at initialization. **Batch normalization** (Ioffe & Szegedy, 2015) normalizes pre-activations to zero mean and unit variance within each minibatch, preventing saturation of sigmoid/tanh activations and making gradient magnitudes relatively insensitive to the depth of the network. **Residual connections** (He et al., 2016) rewrite each block as $x^{(\ell+1)} = x^{(\ell)} + F(x^{(\ell)}, \theta^{(\ell)})$, so the Jacobian $\partial x^{(\ell+1)} / \partial x^{(\ell)} = I + \partial F / \partial x^{(\ell)}$ always includes the identity, providing a direct gradient path regardless of depth. **Gradient clipping** — rescaling the gradient when $\|\nabla L\| > c$ — handles exploding gradients in recurrent networks where the effective depth equals the sequence length.

---

## Information Theory

### Q16 [Basic] Explain entropy and cross-entropy and how they appear in classification training

**Q:** What do Shannon entropy and cross-entropy measure, and why is cross-entropy the natural loss for classification?

**A:** **Shannon entropy** $H(p) = -\sum_x p(x) \log p(x) = \mathbb{E}_p[-\log p(X)]$ measures the average uncertainty of a distribution $p$, or equivalently the minimum expected number of bits needed to encode a sample drawn from $p$. A uniform distribution over $C$ classes has maximum entropy $\log C$; a degenerate distribution placing all mass on one outcome has entropy $0$.

**Cross-entropy** $H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_p[-\log q(X)]$ measures the expected code length when using a code optimized for $q$ to encode samples from $p$. It is always at least $H(p)$, with equality iff $p = q$. In multiclass classification, the true label distribution $p$ is a one-hot vector concentrated at class $y$, and the model output $q$ is a softmax probability vector. The cross-entropy loss reduces to $-\log q_y$, the negative log-probability assigned to the correct class.

Minimizing cross-entropy is equivalent to MLE: maximizing $\log q_y$ maximizes the likelihood of the true label under the model. The deeper relationship is $H(p, q) = H(p) + \mathrm{KL}(p \| q)$: since $H(p) = 0$ for a one-hot $p$, minimizing cross-entropy over $q$ is identical to minimizing $\mathrm{KL}(p \| q)$, i.e., making the model distribution as close as possible to the empirical label distribution in the forward KL sense. This triple equivalence — MLE, cross-entropy minimization, KL minimization — is the foundation of supervised classification.

---

### Q17 [Basic] Describe KL divergence, its asymmetry, and how forward and reverse KL differ

**Q:** What does KL divergence measure, and why does the direction of the divergence matter in variational inference?

**A:** The **KL divergence** (Kullback–Leibler divergence) from distribution $q$ to $p$ is:

$$\mathrm{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\!\left[\log \frac{p}{q}\right]$$

It is always non-negative (Gibbs' inequality), equals zero iff $p = q$ almost everywhere, and is not symmetric: $\mathrm{KL}(p \| q) \neq \mathrm{KL}(q \| p)$ in general. KL divergence is not a metric, but it is an $f$-divergence and plays a central role in information theory and machine learning.

The asymmetry has important behavioral consequences. **Forward KL** $\mathrm{KL}(p \| q)$ is evaluated as an expectation over $p$: wherever $p(x) > 0$ but $q(x) = 0$, the term $p(x) \log(p(x)/q(x)) = +\infty$. Therefore minimizing forward KL forces $q$ to have support everywhere $p$ does — $q$ is **mass-covering** (also called inclusive). This is appropriate when the true distribution $p$ has multiple modes: $q$ must spread mass over all of them.

**Reverse KL** $\mathrm{KL}(q \| p)$ is evaluated under $q$: wherever $q(x) > 0$ but $p(x) \approx 0$, the term $q(x) \log(q(x)/p(x))$ is large. Minimizing reverse KL forces $q$ to concentrate only where $p$ is large — $q$ is **mode-seeking** (exclusive). In variational inference, the ELBO objective corresponds to minimizing $\mathrm{KL}(q_\phi(z|x) \| p_\theta(z|x))$ (reverse KL) because expectations under the variational distribution $q_\phi$ are tractable, while expectations under the true posterior $p_\theta$ are not. The resulting approximate posterior tends to undercover the true posterior's modes, a well-known limitation of mean-field variational inference.

---

### Q18 [Advanced] Explain mutual information and how it is estimated and maximized in representation learning

**Q:** What does mutual information capture that correlation does not, and how have recent methods made it practically useful for self-supervised learning?

**A:** **Mutual information** $I(X; Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y) = \mathrm{KL}(p(x,y) \| p(x)p(y))$ measures the reduction in uncertainty about $X$ given knowledge of $Y$, or equivalently, the information shared between the two random variables. Unlike Pearson correlation, MI captures arbitrary statistical dependencies — including nonlinear ones — and equals zero if and only if $X$ and $Y$ are independent. It is symmetric and always non-negative.

Direct MI computation requires the joint density $p(x,y)$, which is intractable in high dimensions. **MINE** (Belghazi et al., 2018) estimates MI via the **Donsker–Varadhan** variational representation:

$$I(X; Y) \geq \mathbb{E}_{p(x,y)}[T_\psi(x,y)] - \log\,\mathbb{E}_{p(x)p(y)}\!\left[e^{T_\psi(x,y)}\right]$$

A neural network $T_\psi$ is trained to maximize this lower bound, providing a differentiable MI estimator. **InfoNCE** / **CPC** (Oord et al., 2018) provides an alternative lower bound using noise-contrastive estimation: given one positive pair $(x, y)$ and $K$ negative pairs $(x, y_j^-)$, the InfoNCE objective is:

$$\mathcal{L}_\text{InfoNCE} = -\mathbb{E}\!\left[\log \frac{e^{f(x,y)}}{e^{f(x,y)} + \sum_{j=1}^K e^{f(x,y_j^-)}}\right]$$

which is an upper bound on $-I(X; Y) + \log(K+1)$. Maximizing InfoNCE increases a lower bound on MI. This principle underlies contrastive self-supervised methods (SimCLR, MoCo): by maximizing MI between representations of different augmented views of the same image, the model learns features that are invariant to the augmentation family and predictive of the view's identity — properties that transfer well to downstream tasks.

---

### Q19 [Advanced] Describe the information bottleneck principle and its connection to deep learning

**Q:** What objective does the information bottleneck minimize, and what does it predict about how deep networks should encode information?

**A:** The **information bottleneck** (IB; Tishby et al., 2000) formalizes the goal of finding a compact representation $Z$ of input $X$ that is maximally informative about a target $Y$. The Markov constraint $Y \to X \to Z$ requires $Z$ to be computed from $X$. The IB Lagrangian is:

$$\max_{p(z|x)}\; I(Z; Y) - \beta\, I(Z; X)$$

For $\beta = 0$, the solution is $Z = X$ (no compression). As $\beta \to \infty$, the solution collapses to a constant (full compression, zero relevant information retained). The **IB curve** — the Pareto frontier of $(I(Z;X),\, I(Z;Y))$ in the information plane — represents the optimal trade-off between representation complexity and task relevance.

Tishby & Schwartz-Ziv (2017) interpreted deep network training through this lens, claiming that networks first fit the training labels (increasing $I(Z;Y)$) and then undergo a **compression phase** (decreasing $I(Z;X)$ via diffusion-like dynamics in SGD). This attracted significant attention but was subsequently challenged: the apparent compression depends strongly on the activation function and on the MI estimator's binning hyperparameter. For networks with linear or ReLU activations, compression is not consistently observed. The IB interpretation remains an active area with disputed empirical support.

The IB framework nonetheless provides useful vocabulary and connects to established models: $\beta$-VAE optimizes a related objective where the ELBO reconstruction term plays the role of $I(Z;Y)$ and the KL penalty plays the role of $I(Z;X)$. Optimal representations under the IB objective are **sufficient statistics** for predicting $Y$ from $X$, making IB a formal criterion for what a representation learning system should achieve.

---

### Q20 [Advanced] Derive the equivalence between MLE, cross-entropy minimization, and KL divergence minimization

**Q:** Why does minimizing cross-entropy, maximizing likelihood, and minimizing KL divergence from the data distribution all reduce to the same optimization problem?

**A:** Let $p_\text{data}$ denote the true data distribution and $p_\theta$ the model. The **empirical MLE** objective on $n$ i.i.d. samples $\{x_i\}$ is:

$$\hat\theta_\text{MLE} = \arg\max_\theta \frac{1}{n}\sum_{i=1}^n \log p_\theta(x_i)$$

By the law of large numbers, as $n \to \infty$ this converges to $\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$, so asymptotically:

$$\hat\theta_\text{MLE} \approx \arg\max_\theta\; \mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$$

The **KL divergence** from the model to the data is:

$$\mathrm{KL}(p_\text{data} \| p_\theta) = \mathbb{E}_{p_\text{data}}\!\left[\log p_\text{data}(x)\right] - \mathbb{E}_{p_\text{data}}\!\left[\log p_\theta(x)\right]$$

Since $\mathbb{E}_{p_\text{data}}[\log p_\text{data}(x)] = -H(p_\text{data})$ is constant with respect to $\theta$:

$$\arg\min_\theta\; \mathrm{KL}(p_\text{data} \| p_\theta) = \arg\max_\theta\; \mathbb{E}_{p_\text{data}}[\log p_\theta(x)] = \hat\theta_\text{MLE}$$

The **cross-entropy** $H(p_\text{data}, p_\theta) = -\mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$ is precisely the negative of the MLE objective, so minimizing cross-entropy is identical. Putting these together: $H(p_\text{data}, p_\theta) = H(p_\text{data}) + \mathrm{KL}(p_\text{data} \| p_\theta)$, and since $H(p_\text{data})$ does not depend on $\theta$, minimizing cross-entropy minimizes the KL divergence from the empirical distribution to the model.

The practical implications are significant. Any neural network trained with a cross-entropy loss is implicitly fitting a statistical model $p_\theta$ to minimize divergence from the data-generating process — the choice of architecture and parameterization determines which family of distributions $\{p_\theta\}$ is searched. Replacing cross-entropy with a different divergence (e.g., reverse KL, $f$-divergences) yields different fitting behavior: reverse KL produces mode-seeking fits, while forward KL (cross-entropy) produces mass-covering fits. This unification also explains why softmax classifiers, language model next-token heads, and VAE decoders all share the same cross-entropy training objective despite solving superficially different tasks.

---

## Quick Reference

| # | Difficulty | Topic | Section |
|---|------------|-------|---------|
| Q1 | Basic | Eigenvalues and eigenvectors: geometric meaning | Linear Algebra |
| Q2 | Basic | SVD and its relationship to eigendecomposition | Linear Algebra |
| Q3 | Advanced | PCA via SVD and best low-rank approximation | Linear Algebra |
| Q4 | Advanced | Matrix conditioning and training stability | Linear Algebra |
| Q5 | Advanced | Matrix calculus, Jacobians, and VJPs | Linear Algebra |
| Q6 | Basic | MLE derivation of loss functions | Probability and Statistics |
| Q7 | Basic | Gaussian distribution and its prevalence in deep learning | Probability and Statistics |
| Q8 | Advanced | Bias-variance decomposition and double descent | Probability and Statistics |
| Q9 | Advanced | Concentration inequalities and generalization bounds | Probability and Statistics |
| Q10 | Advanced | Priors on weights and Bayesian regularization | Probability and Statistics |
| Q11 | Basic | Chain rule and backpropagation on computational graphs | Calculus and Optimization |
| Q12 | Basic | Gradient descent convergence and momentum | Calculus and Optimization |
| Q13 | Advanced | Non-convex loss landscapes and saddle points | Calculus and Optimization |
| Q14 | Advanced | Second-order optimization and K-FAC | Calculus and Optimization |
| Q15 | Advanced | Vanishing and exploding gradients: causes and fixes | Calculus and Optimization |
| Q16 | Basic | Shannon entropy and cross-entropy in classification | Information Theory |
| Q17 | Basic | KL divergence and its asymmetry | Information Theory |
| Q18 | Advanced | Mutual information estimation and contrastive learning | Information Theory |
| Q19 | Advanced | Information bottleneck principle | Information Theory |
| Q20 | Advanced | Equivalence of MLE, cross-entropy, and KL minimization | Information Theory |

## Resources

- Glorot & Bengio, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (2010)
- He et al., [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) (2015)
- Ioffe & Szegedy, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (2015)
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (2016)
- Belkin et al., [Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Off](https://arxiv.org/abs/1812.11118) (2019)
- Gal & Ghahramani, [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) (2016)
- Dauphin et al., [Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization](https://arxiv.org/abs/1406.2572) (2014)
- Jin et al., [How to Escape Saddle Points Efficiently](https://arxiv.org/abs/1703.00887) (2017)
- Martens & Grosse, [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671) (2015)
- Tishby et al., [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057) (2000)
- Tishby & Schwartz-Ziv, [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810) (2017)
- Belghazi et al., [Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062) (2018)
- Oord et al., [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (2018)
