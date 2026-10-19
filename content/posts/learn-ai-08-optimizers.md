---
title: "learn ai fundamentals, 6: optimizers"
date: 2026-10-18T12:00:00+07:00
description: "sgd to adamw to muon, and why the optimizer matters at scale"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 6 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/08-optimizers.html).

In Fundamentals you used AdamW with warmup and cosine decay as a black box with a few knobs. This chapter opens the box. We build up from SGD to Adam step by step, derive why each piece exists, then survey the optimizers that are actually changing how frontier models are trained: memory-efficient variants (8-bit Adam, Adafactor), sign-based and second-order methods (Lion, Sophia, Shampoo, SOAP), and Muon, which trained Kimi K2. We finish with schedules and with how to tell whether a new optimizer is actually better.

{{< fig "optimizers/01" >}}

## What you will learn

- The lineage SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam, and what problem each step fixed.
- Adam's update with bias correction derived, and why AdamW's decoupled weight decay is not the same as L2 regularization.
- The practical knobs: $\epsilon$, $\beta_2$, gradient clipping, and their link to loss spikes.
- Memory cost of optimizer state, and how 8-bit Adam and Adafactor cut it.
- Lion, Sophia, Shampoo and SOAP at the level of "what is the update and why".
- Muon: orthogonalized momentum via Newton-Schulz iteration, and how Moonlight / Kimi K2 scaled it.
- Schedule-free optimization and the warmup-stable-decay (WSD) schedule.
- How to compare optimizers without fooling yourself.
- How to write AdamW from scratch that matches `torch.optim.AdamW`, and a Muon step.

## Prerequisites

From LLM Fundamentals, review:

- **AdamW basics with warmup/cosine**: we assume you have used it and know the knobs exist.
- **Pretraining objective** and **scaling laws**: optimizer comparisons are only meaningful in terms of loss vs compute.
- **Distributed training (ZeRO/FSDP)**: optimizer state is the largest chunk of what ZeRO shards.
- **Transformer blocks**: we will distinguish 2D weight matrices from embeddings, norms and biases, because modern optimizers treat them differently.

---

## Optimizers

### 1. The setting

We minimize a loss $\mathcal{L}(\theta)$ over parameters $\theta \in \mathbb{R}^n$. At step $t$ we draw a minibatch and compute a stochastic gradient $g_t = \nabla_\theta \mathcal{L}_{\text{batch}}(\theta_{t-1})$. Every optimizer in this chapter is a rule

$$
\theta_t = \theta_{t-1} - \eta_t \, u_t(g_1, \dots, g_t)
$$

- $\theta_t$: parameters after step $t$.
- $\eta_t$: learning rate at step $t$ (the schedule).
- $u_t$: the update direction, built from current and past gradients (and sometimes curvature estimates).

The question each optimizer answers is: given a noisy gradient, which direction and how far?

Two problems make plain gradient descent hard for neural networks:

1. **Ill-conditioning.** The loss surface curves steeply in some directions and gently in others. A step size small enough not to diverge along the steep directions crawls along the flat ones.
2. **Noise.** $g_t$ is an estimate from a minibatch. Following it blindly wastes steps on noise.

{{< fig "optimizers/02" >}}

### 2. SGD, momentum, Nesterov

**SGD**: $u_t = g_t$. Simple, memoryless, one learning rate for everything.

**Momentum (heavy ball)** keeps an exponential moving average-like velocity:

$$
m_t = \mu \, m_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta \, m_t
$$

- $m_t$: momentum buffer (velocity).
- $\mu$: momentum coefficient, typically 0.9.

Intuition: in a narrow valley the gradient oscillates across the valley and points consistently along it. Averaging cancels the oscillation and accumulates the consistent component, so you effectively take steps up to $1/(1-\mu) = 10\times$ larger along directions where gradients agree. Distill's "Why Momentum Really Works" shows that on a quadratic, momentum improves the convergence rate's dependence on the condition number $\kappa$ from $\kappa$ to $\sqrt{\kappa}$.

{{< fig "optimizers/03" >}}

**Nesterov momentum** evaluates the gradient at the look-ahead point $\theta - \eta\mu m$ rather than at $\theta$. In the common "PyTorch" form it amounts to using $g_t + \mu m_t$ as the update instead of $m_t$. It corrects overshoot earlier and is used in Muon.

### 3. Adaptive methods: AdaGrad and RMSProp

Momentum fixes noise but still uses one learning rate for every parameter. In a transformer, gradients for embedding rows of rare tokens, LayerNorm gains and attention projections differ by orders of magnitude. Adaptive methods give each coordinate its own step size.

**AdaGrad** divides by the root of the *sum* of squared gradients:

$$
s_t = s_{t-1} + g_t^2, \qquad \theta_t = \theta_{t-1} - \eta \, \frac{g_t}{\sqrt{s_t} + \epsilon}
$$

All operations are elementwise. Coordinates with large historical gradients take small steps; rare features (sparse gradients) take big ones. Problem: $s_t$ only grows, so the effective learning rate decays to zero. Fine for convex problems, bad for long non-convex training.

**RMSProp** (Hinton's Coursera lecture, 2012) replaces the sum with an exponential moving average so old gradients are forgotten:

$$
v_t = \beta \, v_{t-1} + (1 - \beta) \, g_t^2, \qquad \theta_t = \theta_{t-1} - \eta \, \frac{g_t}{\sqrt{v_t} + \epsilon}
$$

- $v_t$: running estimate of the second moment $\mathbb{E}[g^2]$ per coordinate.
- $\beta$: decay, e.g. 0.99.

Dividing by $\sqrt{v_t}$ normalizes each coordinate to roughly unit scale. This is a crude diagonal preconditioner: it approximates rescaling by curvature using gradient magnitude as a proxy.

{{< fig "optimizers/04" >}}

### 4. Adam, derived

Adam (Kingma and Ba, 2014) = momentum on the numerator + RMSProp on the denominator + bias correction.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat m_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

- $m_t$: first-moment estimate (EMA of gradients), initialized to 0.
- $v_t$: second-moment estimate (EMA of squared gradients), initialized to 0.
- $\beta_1$: first-moment decay, typically 0.9.
- $\beta_2$: second-moment decay, 0.999 by default, 0.95 for most LLMs (see section 6).
- $\beta^t$: $\beta$ raised to the power $t$ (the step count), not a time index.
- $\hat m_t, \hat v_t$: bias-corrected estimates.
- $\epsilon$: small constant for numerical safety, default $10^{-8}$.

{{< fig "optimizers/05" >}}

**Why bias correction?** Unroll $v_t$ with $v_0 = 0$:

$$
v_t = (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} g_i^2
$$

Take expectations and assume the gradient distribution is stationary, $\mathbb{E}[g_i^2] = \mathbb{E}[g^2]$:

$$
\mathbb{E}[v_t] = \mathbb{E}[g^2] \, (1 - \beta_2) \sum_{i=1}^{t} \beta_2^{t-i} = \mathbb{E}[g^2] \, (1 - \beta_2^t)
$$

using the geometric series $\sum_{j=0}^{t-1} \beta^j = (1 - \beta^t)/(1 - \beta)$. So $v_t$ underestimates the second moment by the factor $(1 - \beta_2^t)$, which matters a lot early: with $\beta_2 = 0.999$, at $t = 1$ the factor is 0.001, meaning $v_1$ is 1000x too small. Without correction the denominator would be tiny and the first updates enormous. Dividing by $(1 - \beta_2^t)$ removes the bias; the same argument applies to $m_t$.

{{< fig "optimizers/06" >}}

**What the update looks like.** If the gradient were constant, $\hat m / \sqrt{\hat v} = \pm 1$, so each coordinate moves by about $\eta$ per step regardless of the gradient's scale. Adam's step size is *scale-invariant*: multiplying the loss by 1000 does not change the updates. That is why the same learning rate ranges (around $10^{-4}$ to $10^{-3}$) work across very different models. When gradients are noisy, $|\hat m| \ll \sqrt{\hat v}$ and Adam automatically takes smaller steps: a signal-to-noise ratio per coordinate.

### 5. AdamW: decoupled weight decay

Weight decay shrinks weights toward zero each step. With SGD, adding an L2 penalty $\frac{\lambda}{2}\|\theta\|^2$ to the loss and decaying weights directly are the same thing: the gradient of the penalty is $\lambda\theta$, and $\theta - \eta(g + \lambda\theta) = (1 - \eta\lambda)\theta - \eta g$.

With Adam they are not the same. If you add $\lambda\theta$ to the gradient (what `torch.optim.Adam(weight_decay=...)` does), the decay term goes into $m$ and $v$ and gets divided by $\sqrt{\hat v}$. Parameters with large gradient history (large $v$) get *less* decay; parameters with small gradients get *more*. The regularization strength now depends on gradient statistics, which is not what anyone intended.

{{< fig "optimizers/07" >}}

Loshchilov and Hutter's AdamW ("Decoupled Weight Decay Regularization", 2017) applies decay directly to the weights, outside the adaptive normalization:

$$
\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \, \theta_{t-1} \right)
$$

- $\lambda$: weight decay coefficient (0.1 is typical for LLM pretraining).
- $\eta_t$ multiplies both terms in PyTorch's implementation, so decay follows the schedule.

Why it matters in practice:

- Every parameter decays at the same relative rate $\eta\lambda$ per step, so $\lambda$ is a clean, tunable knob, and the optimal $\lambda$ is less entangled with the optimal $\eta$.
- Combined with Adam's roughly constant update size, decoupled decay creates an equilibrium weight norm, which keeps the effective learning rate of normalized layers in check over long runs.
- It is standard practice to **not** decay biases, LayerNorm/RMSNorm gains, and often embeddings. You do this with parameter groups.

### 6. The stability knobs: $\epsilon$, $\beta_2$, clipping

**$\epsilon$.** It prevents division by zero, but it also sets a floor: when $\sqrt{\hat v} \ll \epsilon$ the update becomes $\hat m / \epsilon$, i.e. SGD with momentum and learning rate $\eta / \epsilon$. If $\epsilon$ is too large relative to typical gradient magnitudes (which shrink as models grow and with gradient scaling in mixed precision), Adam silently degrades into momentum SGD for those parameters. If it is too small, coordinates with near-zero second moment (rare embedding rows, dead units) can receive huge updates when a gradient finally appears. Defaults of $10^{-8}$ are fine for most LLMs; some large-scale recipes use smaller values like $10^{-15}$ for very small gradients. Know that it is a knob.

{{< fig "optimizers/08" >}}

**$\beta_2$.** The default 0.999 averages over roughly $1/(1 - \beta_2) = 1000$ steps. If the gradient for some parameter suddenly grows (a rare token, a data spike, an attention logit growing), $v$ takes hundreds of steps to catch up, so $\hat m / \sqrt{\hat v}$ can be far larger than 1 and the update overshoots. Repeated, this produces the classic loss spike. LLM pretraining almost universally uses $\beta_2 = 0.95$ (GPT-3, Llama, and successors), averaging over about 20 steps, which reacts faster. The flip side is a noisier denominator; smaller batch sizes often want a larger $\beta_2$.

{{< fig "optimizers/09" >}}

**Gradient clipping.** Before the optimizer step, rescale the full gradient vector if its global L2 norm exceeds a threshold $c$ (1.0 is standard):

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right)
$$

Clipping protects against single bad batches. Note an interaction: Adam is scale-invariant in steady state, so clipping mostly affects the *relative* weight of the clipped step inside $m$ and $v$, which is still useful for preventing one batch from poisoning the moment estimates. Monitor the gradient norm: a steadily rising norm, or constant clipping, is a signal that something (LR, $\beta_2$, initialization, logit growth) needs attention.

### 7. Code: AdamW from scratch, matching PyTorch

The point of this exercise is to see that there is no magic. Note how PyTorch arranges the arithmetic: decay first, then $\epsilon$ is added *after* dividing $\sqrt{v}$ by $\sqrt{1 - \beta_2^t}$, which is exactly $\sqrt{\hat v} + \epsilon$.

{{< fig "optimizers/10" >}}

```python
import math
import torch


class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t, m, v, g = state["t"], state["m"], state["v"], p.grad

                # Decoupled decay: applied to the weights, never routed through m or v.
                p.mul_(1 - lr * wd)

                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)


if __name__ == "__main__":
    torch.manual_seed(0)
    make = lambda: torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
    a = make()
    b = make()
    b.load_state_dict(a.state_dict())

    opt_a = MyAdamW(a.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    opt_b = torch.optim.AdamW(b.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)

    for _ in range(200):
        x, y = torch.randn(64, 10), torch.randn(64, 1)
        for model, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            opt.step()

    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), (pa - pb).abs().max()
    print("MyAdamW matches torch.optim.AdamW")
```

Exercise: change `p.mul_(1 - lr * wd)` to `g = g + wd * p` (coupled L2, i.e. Adam with `weight_decay`), confirm it now matches `torch.optim.Adam(weight_decay=0.1)` instead, and plot the per-parameter-group weight norms of both over a longer run.

### 8. The memory bill

For each parameter in mixed-precision AdamW training you typically hold:

| Item | Bytes |
| --- | --- |
| bf16 weights (used in forward/backward) | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| fp32 $m$ | 4 |
| fp32 $v$ | 4 |
| **Total** | **16** |

This is the accounting from the ZeRO paper. A 70B model needs about 1.1 TB just for this, before activations. Optimizer state ($m$, $v$, master weights) is 12 of the 16 bytes, which is why ZeRO stage 1 shards it first. Two lines of work attack this cost directly.

{{< fig "optimizers/11" >}}

**8-bit Adam (Dettmers et al. 2021, "8-bit Optimizers via Block-wise Quantization").** Store $m$ and $v$ in 8 bits instead of 32. Naive quantization fails because optimizer states have outliers spanning many orders of magnitude. The fix: split each tensor into blocks (2048 values), quantize each block with its own scale (so an outlier only hurts its block), and use a non-linear *dynamic* quantization map that spends more codes near zero. States are dequantized, updated in 32 bits, and requantized every step. This cuts $m+v$ from 8 to 2 bytes per parameter with essentially no quality loss in their experiments. Available in `bitsandbytes`; widely used for fine-tuning.

**Adafactor (Shazeer and Stern 2018).** For a weight matrix $W \in \mathbb{R}^{r \times c}$, the second-moment matrix $V$ has $rc$ entries. Adafactor stores only its row sums $R \in \mathbb{R}^r$ and column sums $C \in \mathbb{R}^c$ (as EMAs of $g^2$ summed along each axis) and reconstructs a rank-1 approximation

$$
\hat V = \frac{R \, C^\top}{\mathbf{1}^\top R}
$$

- $R_i$: EMA of the sum of squared gradients in row $i$.
- $C_j$: EMA of the sum of squared gradients in column $j$.
- $\mathbf{1}^\top R$: total sum, so that $\hat V$ has the right overall scale.

Memory drops from $O(rc)$ to $O(r + c)$. Adafactor also optionally drops the first moment entirely, clips updates by their RMS, uses relative step sizes, and increases $\beta_2$ over time. It trained T5 and PaLM. The factored second moment is a useful idea on its own and reappears in later optimizers.

{{< fig "optimizers/12" >}}

### 9. Beyond Adam

{{< fig "optimizers/13" >}}

#### Lion (Chen et al. 2023)

Discovered by a program search over update rules ("Symbolic Discovery of Optimization Algorithms"):

$$
\begin{aligned}
c_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
\theta_t &= \theta_{t-1} - \eta \left( \mathrm{sign}(c_t) + \lambda \theta_{t-1} \right) \\
m_t &= \beta_2 m_{t-1} + (1 - \beta_2) g_t
\end{aligned}
$$

- $\mathrm{sign}(\cdot)$: elementwise sign, so every coordinate moves by exactly $\pm\eta$.
- $\beta_1 = 0.9$ (interpolation for the update), $\beta_2 = 0.99$ (for the stored momentum).

Only one state buffer (half of Adam's memory). Because sign updates have unit magnitude in every coordinate, Lion needs a learning rate about 3 to 10x smaller than AdamW and a correspondingly larger weight decay. Results were strong on vision and some language tasks; at LLM pretraining scale, independent comparisons show smaller and less consistent gains.

#### Sophia (Liu et al. 2023)

A lightweight second-order method. Every $k$ steps (e.g. 10) it estimates the diagonal of the Hessian $h$ (via Hutchinson's estimator or a Gauss-Newton-Bartlett estimator) and keeps an EMA of it. The update is a clipped Newton step:

$$
\theta_t = \theta_{t-1} - \eta \, \mathrm{clip}\left( \frac{m_t}{\max(\gamma h_t, \epsilon)}, \, 1 \right)
$$

- $m_t$: EMA of gradients.
- $h_t$: EMA of diagonal Hessian estimates.
- $\gamma$: scaling hyperparameter.
- $\mathrm{clip}(\cdot, 1)$: elementwise clip to $[-1, 1]$, guarding against tiny or negative curvature estimates.

The paper reported 2x fewer steps than AdamW on GPT-2-scale models. Later, more carefully tuned benchmarks (for example "Deconstructing What Makes a Good Optimizer for Language Models" and "Fantastic Pretraining Optimizers and Where to Find Them") found much smaller gaps. This is a recurring theme; see section 12.

#### Shampoo and SOAP

Diagonal methods (Adam and friends) treat every coordinate independently. But a weight matrix $W \in \mathbb{R}^{r \times c}$ has structure: rows and columns. **Shampoo** (Gupta et al. 2018) keeps two preconditioner matrices

$$
L_t = L_{t-1} + G_t G_t^\top \in \mathbb{R}^{r \times r}, \qquad R_t = R_{t-1} + G_t^\top G_t \in \mathbb{R}^{c \times c}
$$

and updates $W_t = W_{t-1} - \eta \, L_t^{-1/4} \, G_t \, R_t^{-1/4}$, where $G_t$ is the gradient reshaped as a matrix. This is a Kronecker-factored approximation to full-matrix AdaGrad. It captures correlations within rows and columns, at the cost of computing inverse fourth roots (eigendecompositions) periodically. A distributed Shampoo implementation won the external-tuning track of the AlgoPerf benchmark.

**SOAP** (Vyas et al. 2024) observes that Shampoo is closely related to running Adam in the eigenbasis of Shampoo's preconditioners. So it does exactly that: rotate the gradient into the eigenbasis of $L$ and $R$, run Adam there, rotate back, and only recompute the eigenbasis every $f$ steps. This is more stable and has fewer hyperparameters than Shampoo, and reported meaningful speedups over AdamW on language model pretraining.

### 10. Muon: orthogonalized momentum

Muon (Keller Jordan et al., 2024) is the most influential new optimizer of the last few years. It set records in the NanoGPT speedrun and was then scaled to large models by Moonshot AI (Moonlight, then Kimi K2).

**Intuition.** For a 2D weight matrix, the momentum buffer $B$ (same shape as $W$) tends to be dominated by a few directions: its singular values are very uneven, so a gradient step mostly pushes along a handful of "loud" directions and barely moves the rest. Muon replaces $B$ with the nearest *semi-orthogonal* matrix: if $B = U \Sigma V^\top$ is its SVD, use $O = U V^\top$. This keeps the directions but sets every singular value to 1, so all directions in the update get equal magnitude. One theoretical view (Jeremy Bernstein's work with Keller Jordan) is that this is steepest descent under the spectral norm, a natural norm for linear layers since it bounds how much the layer's output can change.

{{< fig "optimizers/14" >}}

**The update.** For each 2D hidden-layer weight:

$$
\begin{aligned}
B_t &= \mu B_{t-1} + G_t \\
O_t &= \mathrm{NewtonSchulz}_5(G_t + \mu B_t) \quad \text{(Nesterov variant)} \\
W_t &= W_{t-1} - \eta \, s \, O_t
\end{aligned}
$$

- $G_t$: gradient of $W$.
- $B_t$: momentum buffer, $\mu = 0.95$.
- $\mathrm{NewtonSchulz}_5$: 5 iterations of the polynomial below, approximating $UV^\top$.
- $s$: a shape-dependent scale (see below).

**Why Newton-Schulz instead of SVD?** SVD is slow and unfriendly to GPUs in bf16. Newton-Schulz iteration computes an approximate orthogonalization using only matrix multiplications. First normalize $X_0 = B / \|B\|_F$ so all singular values are at most 1. Then iterate

$$
X_{k+1} = a X_k + b (X_k X_k^\top) X_k + c (X_k X_k^\top)^2 X_k
$$

with $(a, b, c) = (3.4445, -4.7750, 2.0315)$. Each iteration applies the odd polynomial $\phi(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ to every singular value while leaving singular vectors untouched. The coefficients were tuned so that iterating $\phi$ pushes any $\sigma \in (0, 1]$ quickly into a band around 1 (roughly 0.7 to 1.2). That is not exact orthogonalization, but empirically it does not need to be. Five iterations in bf16 cost a few percent overhead.

{{< fig "optimizers/15" >}}

**What Muon applies to.** Only 2D weight matrices in the hidden layers (attention projections, MLP weights; conv filters flattened). Embeddings, the LM head, norm gains and biases are optimized with AdamW. Embeddings and the output layer behave differently (their rows correspond to tokens, not to a linear map between hidden spaces).

{{< fig "optimizers/16" >}}

**Scaling it up: Moonlight ("Muon is Scalable for LLM Training", Liu et al. 2025).** Two changes made Muon work at scale:

1. **Add weight decay.** Without it, weight and activation norms grew over long runs, hurting performance.
2. **Match update RMS to AdamW.** The RMS of $O_t$ depends on matrix shape. They scale the update by $s = 0.2\sqrt{\max(r, c)}$, which gives updates with RMS similar to typical AdamW updates, so AdamW learning rates and weight decay transfer directly.

With these, they reported Muon reaching AdamW's loss with roughly half the training FLOPs in their scaling-law experiments, and trained Moonlight (a 16B-total, 3B-active MoE) on 5.7T tokens. **Kimi K2** (1T total, 32B active MoE) was then trained with **MuonClip**: Muon plus "QK-clip", which rescales the query and key projection weights whenever the maximum attention logit exceeds a threshold. Muon made attention logits grow unboundedly at that scale; QK-clip fixed it, and K2 trained on 15.5T tokens with no loss spikes.

{{< fig "optimizers/17" >}}

### 11. Code: a Newton-Schulz Muon step

{{< fig "optimizers/18" >}}

```python
import torch


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    # Work in the wide orientation so X @ X.T is the smaller Gram matrix.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    # Frobenius normalization bounds every singular value by 1, inside the polynomial's basin.
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def muon_step(W, grad, buf, lr=0.02, momentum=0.95, weight_decay=0.1, nesterov=True):
    buf.mul_(momentum).add_(grad)
    update = grad.add(buf, alpha=momentum) if nesterov else buf
    O = newton_schulz5(update)
    # Moonlight scaling: makes update RMS comparable to AdamW so its lr/wd settings transfer.
    O = O * 0.2 * max(W.shape) ** 0.5
    W.mul_(1 - lr * weight_decay)
    W.add_(O, alpha=-lr)


if __name__ == "__main__":
    torch.manual_seed(0)
    G = torch.randn(256, 128) @ torch.diag(torch.logspace(0, -3, 128))  # badly conditioned
    O = newton_schulz5(G).float()
    s_before = torch.linalg.svdvals(G)
    s_after = torch.linalg.svdvals(O)
    print(f"singular values before: max/min = {(s_before.max() / s_before.min()).item():.1f}")
    print(f"singular values after:  min={s_after.min().item():.2f} max={s_after.max().item():.2f}")
    # Directions with tiny singular values converge slower; the bulk should land near 1.
    assert 0.5 < s_after.median() < 1.3

    W = torch.randn(256, 128) * 0.02
    buf = torch.zeros_like(W)
    target = torch.randn(64, 256)
    x = torch.randn(64, 128)
    for i in range(100):
        Wg = W.clone().requires_grad_(True)
        loss = ((x @ Wg.T - target) ** 2).mean()
        loss.backward()
        muon_step(W, Wg.grad, buf, lr=0.002, weight_decay=0.0)
        if i % 25 == 0:
            print(i, loss.item())
```

In a real model you would put all hidden 2D matrices in a Muon parameter group and everything else in an AdamW group. Reference implementations: Keller Jordan's `Muon` repository, and `torch.optim.Muon` in recent PyTorch releases.

### 12. Schedules revisited and schedule-free optimization

In Fundamentals you used linear warmup plus cosine decay. Two developments are worth knowing.

**Warmup-Stable-Decay (WSD).** Warm up, hold the learning rate constant for most of training, then decay rapidly (commonly the last 10 to 20% of steps) to near zero. Popularized by MiniCPM and now common (DeepSeek and others use step-based variants). Why it wins in practice:

- Cosine requires fixing the total number of steps in advance. Its intermediate checkpoints are not "finished", so you cannot evaluate a 2T-token model from a 4T-token cosine run.
- With WSD you can branch off from the stable phase at any point and run a short decay to get a finished model. That makes scaling-law sweeps much cheaper (one long run, many decay branches) and allows continuing training later.
- The decay phase is where loss drops sharply; the stable phase seems to make "progress" that the decay then cashes in. Final losses typically match or beat cosine.

{{< fig "optimizers/19" >}}

The decay phase is also a natural place to switch to high-quality data (annealing), which Llama 3 and others do.

**Schedule-free optimizers (Defazio et al. 2024, "The Road Less Scheduled").** Can we avoid a decay schedule entirely? The method keeps three sequences and evaluates gradients at an interpolation:

$$
\begin{aligned}
y_t &= (1 - \beta) z_t + \beta x_t \\
z_{t+1} &= z_t - \eta \nabla \mathcal{L}(y_t) \\
x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1}, \quad c_{t+1} = \frac{1}{t+1}
\end{aligned}
$$

- $z_t$: the "fast" SGD-like iterate that takes raw gradient steps.
- $x_t$: a running uniform average of the $z$ iterates; this is what you evaluate and ship.
- $y_t$: where gradients are computed, an interpolation that acts like momentum ($\beta \approx 0.9$).

Averaging iterates has long been known to mimic the effect of a decaying learning rate; schedule-free makes it practical, with an Adam-style variant. It won the self-tuning track of the AlgoPerf 2024 competition. Caveats: it still needs warmup, the model weights you train with ($y$) differ from those you evaluate ($x$), so you must switch the optimizer between `train()` and `eval()` modes, and at LLM pretraining scale WSD remains the more common choice.

{{< fig "optimizers/20" >}}

### 13. How to compare optimizers fairly

Most "X beats Adam" claims shrink under careful comparison. A checklist:

1. **Tune the baseline as hard as the new method.** At minimum sweep learning rate (log scale), weight decay and warmup for each optimizer. An under-tuned AdamW with $\beta_2 = 0.999$ is a straw man.
2. **Compare at matched compute and with finished schedules.** Compare final losses of runs whose schedules end at the budget, not intermediate checkpoints of a cosine run.
3. **Measure wall-clock and memory, not only steps.** Second-order methods take fewer steps but each costs more. Report time-to-target-loss on real hardware, as the AlgoPerf benchmark does.
4. **Check across scales.** Gains often shrink with model size. "Fantastic Pretraining Optimizers and Where to Find Them" (2025) found that with careful per-optimizer tuning, speedups over AdamW were around 1.1 to 1.4x at small scale and decreased with model size, and that matrix-based preconditioners (such as Muon and SOAP) performed best among the candidates.
5. **Use multiple seeds** and report variance; small differences in loss can be noise.
6. **Look at stability**: loss spikes, sensitivity to LR. An optimizer that is slightly worse at its best LR but far more robust across LRs may be the better engineering choice.

{{< fig "optimizers/21" >}}

### Common confusions

- **"Adam with weight_decay is AdamW."** No. `torch.optim.Adam(weight_decay=...)` adds L2 to the gradient, which then gets normalized; AdamW decays weights directly.
- **"Bias correction is a minor detail."** At $t=1$ with $\beta_2 = 0.999$ the uncorrected second moment is 1000x too small. It matters most at the start, which is also when instability is most likely.
- **"$\epsilon$ is just for numerical safety."** It also sets the scale below which Adam becomes momentum SGD.
- **"Lower $\beta_2$ is always more stable."** Lower $\beta_2$ reacts faster to gradient spikes, but its estimate is noisier; it is a trade-off tied to batch size.
- **"Muon replaces AdamW."** Muon handles hidden 2D matrices; embeddings, the LM head and 1D parameters still use AdamW.
- **"Newton-Schulz computes the exact orthogonal factor."** Muon's coefficients deliberately trade exactness for speed; singular values land roughly in 0.7 to 1.2.
- **"WSD and schedule-free are the same idea."** WSD is a schedule with a short final decay; schedule-free removes the decay by averaging iterates.

### Check yourself

1. Derive why $\mathbb{E}[v_t] = (1 - \beta_2^t)\,\mathbb{E}[g^2]$ under stationarity, and what happens to the first Adam update if you skip bias correction on $v$ but keep it on $m$.
2. Explain concretely why L2 regularization inside Adam decays a parameter with large historical gradients less than one with small historical gradients.
3. A 7B model is trained with mixed-precision AdamW. How many GB does model state (weights, grads, optimizer) take? How much would 8-bit Adam save?
4. Why does LLM pretraining use $\beta_2 = 0.95$ rather than 0.999?
5. What does Newton-Schulz do to the singular values and singular vectors of the momentum matrix, and why does that make a better update?
6. You read a paper showing optimizer X reaches AdamW's loss in 30% fewer steps on a 125M model. List three questions you would ask before adopting it.

<details>
<summary>Answers</summary>

1. Unroll: $v_t = (1-\beta_2)\sum_{i=1}^t \beta_2^{t-i} g_i^2$; take expectations and sum the geometric series to get $(1 - \beta_2^t)\mathbb{E}[g^2]$. Without correction on $v$, at $t=1$ we have $v_1 = 0.001 g_1^2$ (for $\beta_2 = 0.999$), $\sqrt{v_1} \approx 0.032|g_1|$, while $\hat m_1 = g_1$, so the update is about $31.6\eta$ per coordinate instead of $\eta$: a huge first step.
2. The L2 term $\lambda\theta$ is added to $g$, so the update for that coordinate is $\approx (g + \lambda\theta)/\sqrt{\hat v}$. A large $\hat v$ (large historical gradients) divides the decay term down; a small $\hat v$ amplifies it.
3. $16$ bytes/param $\times 7\times10^9 \approx 112$ GB. 8-bit Adam reduces $m$ and $v$ from 8 bytes to 2 bytes per parameter, saving 6 bytes/param, about 42 GB (leaving about 70 GB).
4. With $\beta_2 = 0.999$, $v$ averages over ~1000 steps and lags when a parameter's gradient suddenly grows, so updates become much larger than intended, causing loss spikes. $\beta_2 = 0.95$ averages over ~20 steps and tracks such changes.
5. Singular vectors are unchanged; every singular value is mapped to roughly 1. The update then moves equally in all directions of the momentum's column/row space instead of being dominated by a few large directions, which is steepest descent under the spectral norm and helps underrepresented directions learn.
6. Was AdamW's LR, weight decay, $\beta_2$ and warmup tuned as carefully? Does the gain hold in wall-clock time and at larger scales? Were finished schedules compared at equal budgets, with multiple seeds? (Also: memory overhead, stability across LRs.)

</details>

### Resources

**Read**

- [Why Momentum Really Works, Distill](https://distill.pub/2017/momentum/) (start here): interactive, rigorous intuition for momentum and conditioning.
- [Kingma and Ba, Adam (arXiv:1412.6980)](https://arxiv.org/abs/1412.6980): the original update and bias-correction derivation.
- [Loshchilov and Hutter, Decoupled Weight Decay Regularization (arXiv:1711.05101)](https://arxiv.org/abs/1711.05101): why L2 and weight decay differ under Adam.
- [Shazeer and Stern, Adafactor (arXiv:1804.04235)](https://arxiv.org/abs/1804.04235): factored second moments and update clipping.
- [Dettmers et al., 8-bit Optimizers via Block-wise Quantization (arXiv:2110.02861)](https://arxiv.org/abs/2110.02861): how to quantize optimizer state safely.
- [Muon: An optimizer for hidden layers, Keller Jordan](https://kellerjordan.github.io/posts/muon/): the design and Newton-Schulz coefficients, from the author.
- [Liu et al., Muon is Scalable for LLM Training (arXiv:2502.16982)](https://arxiv.org/abs/2502.16982): weight decay, RMS matching, Moonlight.
- [Defazio et al., The Road Less Scheduled (arXiv:2405.15682)](https://arxiv.org/abs/2405.15682): schedule-free SGD and AdamW.
- [Wen et al., Fantastic Pretraining Optimizers and Where to Find Them (arXiv:2509.02046)](https://arxiv.org/abs/2509.02046): a careful, tuned benchmark of modern optimizers.
- Also useful: [Lion (arXiv:2302.06675)](https://arxiv.org/abs/2302.06675), [Sophia (arXiv:2305.14342)](https://arxiv.org/abs/2305.14342), [Shampoo (arXiv:1802.09568)](https://arxiv.org/abs/1802.09568), [SOAP (arXiv:2409.11321)](https://arxiv.org/abs/2409.11321), [Kimi K2 (arXiv:2507.20534)](https://arxiv.org/abs/2507.20534) for MuonClip, [MiniCPM (arXiv:2404.06395)](https://arxiv.org/abs/2404.06395) for WSD, [AlgoPerf (arXiv:2306.07179)](https://arxiv.org/abs/2306.07179).

**Watch**

- [Gradient Descent With Momentum, Andrew Ng (DeepLearningAI)](https://www.youtube.com/watch?v=k8fTYJPd3_I): short, clear EMA view of momentum.
- [Adam Optimization Algorithm, Andrew Ng (DeepLearningAI)](https://www.youtube.com/watch?v=JXQT_vxqwIs): Adam as momentum plus RMSProp, with bias correction.
- [Jeremy Bernstein: Depths of First Order Optimization (Cohere)](https://www.youtube.com/watch?v=4OAiakkmKQs): the norm-based view of optimization behind Muon.

### Project

**An optimizer bake-off on your GPT-2.**

1. Use the GPT-2 small (or a ~20-50M parameter version) from Fundamentals and a fixed dataset and token budget (for example 500M tokens).
2. Implement or import: AdamW (your `MyAdamW`), Lion, Adafactor, and Muon (Muon on hidden 2D matrices, AdamW on the rest).
3. For each optimizer, sweep learning rate over at least 5 values on a log grid with a WSD schedule. Use the same warmup and decay fraction for all. Record final validation loss, tokens/sec and peak memory.
4. Plot validation loss vs learning rate for each optimizer (the "LR sensitivity curve") and loss vs wall-clock time at each optimizer's best LR.
5. Re-run the best two at a second model size (2-4x larger). Does the gap grow or shrink?
6. Stretch: from a single WSD run, branch decays at 25%, 50% and 100% of the budget and compare against three separate cosine runs of those lengths.

Summarize with a table and a one-paragraph recommendation, stating clearly what you tuned and what you did not.

---

previous: [distillation](/posts/learn-ai-04-distillation/) · next: [mixture of experts](/posts/learn-ai-07-mixture-of-experts/)
