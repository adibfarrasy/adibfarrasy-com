---
title: "learn ai fundamentals, 8: linear attention"
date: 2026-10-18T10:00:00+07:00
description: "linear attention, state space models, and hybrids"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 8 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/05-linear-attention.html).

In the fundamentals course you made softmax attention fast with FlashAttention and cheap at inference with GQA and a KV cache. None of that changed the underlying asymptotics: every new token still looks at every previous token, so compute grows quadratically with sequence length and the KV cache grows linearly. This chapter is about the family of architectures that change the asymptotics themselves, by replacing the ever-growing cache with a fixed-size state. They go by many names (linear attention, linear RNNs, state space models, RetNet, RWKV, Mamba, DeltaNet), and one of the most useful things you can take away is that they are mostly the same idea with different choices of "how to write into the state" and "how to forget".

## What you will learn

- Exactly where the $O(n^2)$ cost of softmax attention comes from, and why removing the softmax lets you reorder the computation.
- The "Transformers are RNNs" view: causal linear attention as a recurrence with a constant-size matrix state.
- Feature maps (elu+1, Performer's random features) and why they matter.
- How decay and gating evolved: RetNet, gated linear attention, Mamba, Mamba-2 and its state space duality, RWKV.
- The delta rule (DeltaNet, Gated DeltaNet) and why "overwrite" beats "accumulate".
- The recall problem that fixed-size states cannot escape, and why production models are hybrids.
- How to implement causal linear attention in parallel, recurrent and chunked form and check they agree.

## Prerequisites

From LLM Fundamentals, review:

- **Attention, GQA, KV cache and FlashAttention**: the $QK^\top$ score matrix, causal masking, and KV cache memory arithmetic.
- **Transformer blocks**: where attention sits relative to the MLP, residual stream and normalization. Linear attention variants usually swap only the token-mixing layer.
- **RoPE**: positional information in softmax attention; many linear models drop it because decay already encodes recency.
- **Evaluation**: needle-in-a-haystack and long-context benchmarks, which are where these models' weaknesses show up.
- Basic linear algebra: outer products, matrix associativity, and what "rank-1 update" means.

## 1. Intuition: where the quadratic comes from and how to remove it

Write causal softmax attention for a single head. For position $i$,

$$
o_i = \sum_{j=1}^{i} \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{l=1}^{i} \exp(q_i^\top k_l / \sqrt{d})}\, v_j
$$

where $q_i, k_j \in \mathbb{R}^{d_k}$ are query and key vectors, $v_j \in \mathbb{R}^{d_v}$ is a value vector, $d = d_k$, and $o_i \in \mathbb{R}^{d_v}$ is the output. In matrix form with $Q, K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$ and sequence length $n$:

$$
O = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
$$

where $M$ is the causal mask ($0$ on and below the diagonal, $-\infty$ above). The $n \times n$ matrix $QK^\top$ is the quadratic term: $O(n^2 d)$ FLOPs to build, and at inference each new token must read all $n$ cached keys and values. FlashAttention avoids *materializing* that matrix in HBM, but still does all $O(n^2 d)$ of the arithmetic.

Why can we not just compute $K^\top V$ first? Matrix multiplication is associative: $(QK^\top)V = Q(K^\top V)$, and $K^\top V$ is only $d_k \times d_v$, independent of $n$. The softmax is what blocks this. It applies a nonlinear function to each entry of $QK^\top$ and then normalizes each row, so the product no longer factors.

{{< fig "linear-attention/01" >}}

So the idea of **linear attention** is: replace $\exp(q^\top k)$ with something that factors. If the similarity is a dot product of feature maps, $\mathrm{sim}(q, k) = \phi(q)^\top \phi(k)$, then

$$
o_i = \frac{\sum_{j \le i} \phi(q_i)^\top \phi(k_j)\, v_j^\top}{\sum_{j \le i} \phi(q_i)^\top \phi(k_j)}
= \frac{\phi(q_i)^\top \overbrace{\left(\sum_{j \le i} \phi(k_j) v_j^\top\right)}^{S_i \,\in\, \mathbb{R}^{d_k \times d_v}}}{\phi(q_i)^\top \underbrace{\left(\sum_{j \le i} \phi(k_j)\right)}_{z_i \,\in\, \mathbb{R}^{d_k}}}
$$

(here $o_i$ is written as a row vector). The query has been pulled outside the sum. The quantity inside, $S_i$, is a sum of outer products that does not depend on $q_i$ at all, and it can be updated incrementally. That is the kernel trick, and it is the whole engine of this chapter.

A useful mental model: $S$ is a tiny **associative memory**. Writing $\phi(k)v^\top$ stores the value $v$ "at address" $\phi(k)$. Reading with $\phi(q)^\top S$ retrieves a mix of stored values weighted by how well $q$ matches each stored key. Softmax attention keeps every key and value around and does exact lookups; linear attention compresses them all into a $d_k \times d_v$ matrix. Everything good and bad about these models follows from that compression.

{{< fig "linear-attention/02" >}}

## 2. The math: Transformers are RNNs

### Recurrent form

Katharopoulos et al. (2020) made the RNN connection explicit. Define the state recursively:

$$
S_i = S_{i-1} + \phi(k_i)\, v_i^\top, \qquad z_i = z_{i-1} + \phi(k_i), \qquad
o_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}
$$

with $S_0 = 0 \in \mathbb{R}^{d_k \times d_v}$ and $z_0 = 0 \in \mathbb{R}^{d_k}$. Each step costs $O(d_k d_v)$ and the memory is constant in $n$. For a model with $H$ heads per layer, the entire "KV cache" of a layer is $H$ matrices of size $d_k \times d_v$, no matter whether the context is 1k or 1M tokens. Generating token $n+1$ costs the same as generating token 2.

{{< fig "linear-attention/03" >}}

### Parallel form

For training you want to process the whole sequence at once on a GPU. The parallel form is

$$
O = \frac{\left(\phi(Q)\phi(K)^\top \odot M_{\text{causal}}\right) V}{\left(\phi(Q)\phi(K)^\top \odot M_{\text{causal}}\right)\mathbf{1}}
$$

where $\phi(Q) \in \mathbb{R}^{n \times d_k}$ applies $\phi$ row-wise, $M_{\text{causal}}$ is a lower-triangular matrix of ones (multiplicative, not additive, since there is no softmax to absorb $-\infty$), $\odot$ is elementwise product, $\mathbf{1}$ is a ones vector, and the division is row-wise. This is quadratic again, but it is exactly the same function as the recurrence, and it uses large matmuls that GPUs love.

### Chunkwise form: the one actually used for training

Real implementations (the `flash-linear-attention` library, Mamba-2's SSD kernels) use a hybrid. Split the sequence into chunks of size $C$ (typically 64 to 256). Within a chunk, use the parallel form; across chunks, carry the state $S$. For chunk $c$ with queries $Q_c$ and so on:

$$
O_c = \underbrace{\phi(Q_c)\, S_{c-1}}_{\text{all earlier chunks}} + \underbrace{\left(\phi(Q_c)\phi(K_c)^\top \odot M\right) V_c}_{\text{within chunk}}, \qquad
S_c = S_{c-1} + \phi(K_c)^\top V_c
$$

(normalizer omitted for clarity). Cost is $O(n C d + n d^2)$, linear in $n$, with matmul-friendly blocks. Keep these three forms in mind: **parallel** for understanding, **recurrent** for inference, **chunkwise** for training.

{{< fig "linear-attention/04" >}}

### Feature maps

The choice of $\phi$ decides what similarity you are computing.

- **elu+1** (Katharopoulos): $\phi(x) = \mathrm{elu}(x) + 1$, applied elementwise. Cheap, positive (so the normalizer cannot vanish or flip sign), but a poor approximation of softmax's sharpness. Linear attention with this map is noticeably worse at language modeling than softmax.
- **Performer / FAVOR+** (Choromanski et al., 2020): use random features so that $\mathbb{E}[\phi(q)^\top \phi(k)] = \exp(q^\top k)$. With $w_1, \dots, w_m \sim \mathcal{N}(0, I)$ (orthogonalized in practice),
  $$\phi(x) = \frac{\exp(-\|x\|^2/2)}{\sqrt{m}}\left[\exp(w_1^\top x), \dots, \exp(w_m^\top x)\right] \in \mathbb{R}^m.$$
  This gives an unbiased, always-positive estimate of the softmax kernel. In practice the variance is high unless $m$ is large, and approximating softmax turned out to be the wrong goal.
- **Identity or simple maps with normalization elsewhere**: most modern models (RetNet, GLA, Mamba-2, DeltaNet) drop the denominator $\phi(q)^\top z$ entirely, use $\phi$ = identity or a SiLU/L2-normalized projection, and rely on a normalization layer on the output instead. The denominator was numerically fragile and turned out to be unnecessary.

The lesson of 2020-2022 was that trying to *approximate* softmax with a kernel underperforms. The progress since 2023 came from making the recurrence itself smarter.

{{< fig "linear-attention/05" >}}

## 3. Code: causal linear attention three ways

The following script implements the parallel, recurrent and chunkwise forms and checks that they produce the same output. Shapes follow the usual multi-head convention: `(batch, heads, seq_len, head_dim)`.

```python
import torch
import torch.nn.functional as F


def feature_map(x):
    # elu+1 keeps features positive so the normalizer never hits zero or flips sign
    return F.elu(x) + 1


def linear_attn_parallel(q, k, v, eps=1e-6):
    # q, k: (B, H, N, d_k); v: (B, H, N, d_v)
    q, k = feature_map(q), feature_map(k)
    N = q.shape[-2]
    scores = q @ k.transpose(-1, -2)                       # (B, H, N, N), quadratic like softmax attention
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask, 0.0)                # zero, not -inf: there is no softmax to absorb it
    return (scores @ v) / (scores.sum(-1, keepdim=True) + eps)


def linear_attn_recurrent(q, k, v, eps=1e-6):
    q, k = feature_map(q), feature_map(k)
    B, H, N, d_k = q.shape
    d_v = v.shape[-1]
    S = q.new_zeros(B, H, d_k, d_v)   # the whole "KV cache": size independent of N
    z = q.new_zeros(B, H, d_k)        # running sum of keys, for the normalizer
    out = []
    for t in range(N):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        S = S + kt.unsqueeze(-1) * vt.unsqueeze(-2)        # rank-1 outer-product write
        z = z + kt
        num = (qt.unsqueeze(-2) @ S).squeeze(-2)            # (B, H, d_v)
        den = (qt * z).sum(-1, keepdim=True)
        out.append(num / (den + eps))
    return torch.stack(out, dim=2)


def linear_attn_chunked(q, k, v, chunk=16, eps=1e-6):
    # Training kernels use this form: matmuls inside a chunk, recurrence across chunks
    q, k = feature_map(q), feature_map(k)
    B, H, N, d_k = q.shape
    d_v = v.shape[-1]
    S = q.new_zeros(B, H, d_k, d_v)
    z = q.new_zeros(B, H, d_k)
    mask = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    out = []
    for s in range(0, N, chunk):
        qc, kc, vc = q[:, :, s:s+chunk], k[:, :, s:s+chunk], v[:, :, s:s+chunk]
        C = qc.shape[-2]
        intra = (qc @ kc.transpose(-1, -2)).masked_fill(~mask[:C, :C], 0.0)
        num = qc @ S + intra @ vc                            # past chunks via state, current chunk via masked scores
        den = (qc @ z.unsqueeze(-1)).squeeze(-1) + intra.sum(-1)
        out.append(num / (den.unsqueeze(-1) + eps))
        S = S + kc.transpose(-1, -2) @ vc
        z = z + kc.sum(-2)
    return torch.cat(out, dim=2)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, d_k, d_v = 2, 3, 50, 8, 16
    q, k, v = (torch.randn(B, H, N, d) for d in (d_k, d_k, d_v))
    a = linear_attn_parallel(q, k, v)
    b = linear_attn_recurrent(q, k, v)
    c = linear_attn_chunked(q, k, v)
    print("parallel vs recurrent max diff:", (a - b).abs().max().item())
    print("parallel vs chunked   max diff:", (a - c).abs().max().item())
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)
```

Note `N = 50` with `chunk = 16`: the last chunk is partial on purpose, so the slicing logic is exercised. Expect differences around `1e-7`, which is float32 rounding from summing in a different order.

{{< fig "linear-attention/06" >}}

The second script adds a **per-token, data-dependent scalar decay** $a_t \in (0, 1)$, which is the core of RetNet (fixed decay), Mamba-2 and Gated DeltaNet (learned decay). The recurrence becomes $S_t = a_t S_{t-1} + k_t v_t^\top$. Unrolling it, the contribution of token $j$ to output $i$ is scaled by $\prod_{m=j+1}^{i} a_m$, so the parallel form multiplies the score matrix by a decay mask $L$ with $L_{ij} = \prod_{m=j+1}^{i} a_m$ for $j \le i$ and $0$ otherwise.

{{< fig "linear-attention/07" >}}

```python
import torch


def decay_parallel(q, k, v, log_a):
    # log_a: (B, H, N), log of a per-token decay in (0, 1); working in log space avoids underflow of long products
    cum = log_a.cumsum(-1)                                  # (B, H, N)
    N = q.shape[-2]
    L = cum.unsqueeze(-1) - cum.unsqueeze(-2)               # L[i, j] = sum_{j<m<=i} log a_m
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    L = L.masked_fill(~mask, float("-inf")).exp()           # 1-semiseparable decay mask
    return ((q @ k.transpose(-1, -2)) * L) @ v


def decay_recurrent(q, k, v, log_a):
    B, H, N, d_k = q.shape
    S = q.new_zeros(B, H, d_k, v.shape[-1])
    out = []
    for t in range(N):
        a = log_a[:, :, t].exp()[..., None, None]
        S = a * S + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)  # forget a little, then write
        out.append((q[:, :, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(out, dim=2)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, d_k, d_v = 2, 2, 40, 8, 8
    q, k, v = (torch.randn(B, H, N, d) for d in (d_k, d_k, d_v))
    log_a = torch.nn.functional.logsigmoid(torch.randn(B, H, N) + 2)  # data-dependent gate, mostly near 1
    a, b = decay_parallel(q, k, v, log_a), decay_recurrent(q, k, v, log_a)
    print("max diff:", (a - b).abs().max().item())
    assert torch.allclose(a, b, atol=1e-4)
```

There is no normalizer here; models with decay use output normalization instead. Keep this second script in mind: in section 4 you will see that it is, almost line for line, Mamba-2.

## 4. The modern family: decay, selectivity and the delta rule

Plain linear attention has a fatal flaw for language modeling: $S_t = S_{t-1} + \phi(k_t)v_t^\top$ only ever adds. After a few thousand tokens the state is a blurry sum of everything, and nothing can be removed. Every successful successor adds a way to **forget**, and the best ones add a way to **overwrite**. A helpful way to organize the zoo is by the state update rule, writing $S_t \in \mathbb{R}^{d_k \times d_v}$ and reading out with $o_t = S_t^\top q_t$:

{{< fig "linear-attention/08" >}}

| Model | State update | Forgetting |
|---|---|---|
| Linear attention (2020) | $S_t = S_{t-1} + k_t v_t^\top$ | none |
| RetNet (2023) | $S_t = \gamma S_{t-1} + k_t v_t^\top$ | fixed scalar per head |
| GLA (2023) | $S_t = \mathrm{Diag}(\alpha_t) S_{t-1} + k_t v_t^\top$ | data-dependent, per key dimension |
| Mamba-2 (2024) | $S_t = a_t S_{t-1} + k_t v_t^\top$ | data-dependent scalar per head |
| DeltaNet (2024) | $S_t = (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ | targeted overwrite |
| Gated DeltaNet (2024) | $S_t = a_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$ | both |

### RetNet: fixed exponential decay

RetNet (Sun et al., 2023) multiplies the state by a constant $\gamma \in (0,1)$ each step, with a different $\gamma$ per head so some heads remember long and others short. Its parallel form is $\left(QK^\top \odot D\right)V$ with $D_{ij} = \gamma^{i-j}$ for $i \ge j$, which is exactly the decay script above with constant $a_t$. It also uses an xPos-like rotation on $q$ and $k$, and GroupNorm on outputs. RetNet's contribution was presenting the three forms (parallel, recurrent, chunkwise) as a deliberate design goal: train like a transformer, infer like an RNN. Its weakness is that the decay ignores content: an important token decays as fast as filler.

### Gated linear attention (GLA)

GLA (Yang et al., 2023) makes the decay **data-dependent** and **per dimension**: $\alpha_t = \sigma(W_\alpha x_t)^{1/\tau} \in (0,1)^{d_k}$ is computed from the current input $x_t$, so the model can decide, token by token, which parts of memory to keep. The paper's other major contribution was a hardware-efficient chunkwise algorithm (FlashLinearAttention) that made these models competitive in wall-clock training speed, not just asymptotically.

### State space models: S4 to Mamba

State space models arrive at the same place from control theory. A continuous-time linear SSM maps an input signal $x(t) \in \mathbb{R}$ to an output $y(t)$ through a hidden state $h(t) \in \mathbb{R}^{N}$:

{{< fig "linear-attention/09" >}}

$$
h'(t) = A\, h(t) + B\, x(t), \qquad y(t) = C\, h(t)
$$

with $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$. Discretizing with step size $\Delta$ (zero-order hold) gives a linear recurrence

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \qquad y_t = C h_t, \qquad \bar{A} = \exp(\Delta A), \quad \bar{B} \approx \Delta B.
$$

**S4** (Gu et al., 2021) intuition: because $\bar A, \bar B, C$ are fixed across time, unrolling the recurrence gives $y = \bar{K} * x$, a long convolution with kernel $\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \dots)$. You train with FFT convolutions (parallel, $O(n \log n)$) and infer with the recurrence (constant memory). Two ingredients made it work: the **HiPPO** initialization of $A$, which is derived so that $h$ optimally compresses the history of $x$ onto a polynomial basis, and a structured parameterization of $A$ that makes computing $\bar{K}$ cheap. S4 dominated long-range benchmarks but lagged transformers on language.

**Mamba** (Gu and Dao, 2023) diagnosed why: S4 is **linear time-invariant**. Its dynamics are the same for every token, so it cannot decide to ignore filler words or to latch onto a key fact. Mamba makes $\Delta$, $B$ and $C$ functions of the input:

$$
\Delta_t = \mathrm{softplus}(W_\Delta x_t), \quad B_t = W_B x_t, \quad C_t = W_C x_t, \quad h_t = \exp(\Delta_t A)\, h_{t-1} + \Delta_t B_t x_t, \quad y_t = C_t h_t.
$$

This is the **selection mechanism**. A large $\Delta_t$ resets the state toward the current input (focus on this token); a small $\Delta_t$ keeps the state nearly unchanged (ignore this token). Losing time-invariance means losing the convolution trick, so Mamba computes the recurrence with a hardware-aware **parallel scan** that keeps the expanded state in SRAM, the same "do not round-trip to HBM" lesson as FlashAttention. Mamba uses diagonal $A$ and runs an independent SSM per channel, with state size $N = 16$ per channel. Mamba-1 was the first attention-free model to match transformers of the same size on language modeling perplexity at small scales.

{{< fig "linear-attention/10" >}}

Now compare $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t,\ y_t = C_t h_t$ with gated linear attention $S_t = a_t S_{t-1} + k_t v_t^\top,\ o_t = S_t^\top q_t$. Map $B_t \leftrightarrow k_t$, $x_t \leftrightarrow v_t$, $C_t \leftrightarrow q_t$. They are the same shape of computation.

### Mamba-2 and state space duality

**Mamba-2** (Dao and Gu, 2024) made that correspondence exact. Restrict $A$ to a **scalar times identity** per head, so $\bar{A}_t = a_t I$. Then unrolling the recurrence gives

$$
y = \left(L \odot C B^\top\right) x, \qquad L_{ij} = \begin{cases} \prod_{m=j+1}^{i} a_m & i \ge j \\ 0 & i \lt  j \end{cases}
$$

where $C, B \in \mathbb{R}^{n \times N}$ stack the per-token $C_t, B_t$ and $x \in \mathbb{R}^{n \times P}$ is the head's input with head dimension $P$. This is masked linear attention with $Q = C$, $K = B$, $V = x$, and a decay mask $L$. A matrix of the form $L$ is called **1-semiseparable**, and the paper's "state space duality" (SSD) is the statement that the recurrent SSM and this masked-attention form compute the same thing, so you can pick whichever is efficient. It is precisely `decay_parallel` versus `decay_recurrent` from section 3.

Giving up per-dimension decay in exchange for a scalar lets SSD use a chunkwise algorithm built on matmuls (tensor cores) instead of a scan, making training 2 to 8 times faster than Mamba-1 and allowing much larger states ($N = 64$ to $256$). Mamba-2 also adopts multi-head structure, analogous to multi-head attention.

### RWKV

**RWKV** (Peng et al., 2023) evolved independently from the "attention-free transformer" line, with its own vocabulary: "time mixing" (the token mixer) and "channel mixing" (the MLP). RWKV-4's time mixing computes, per channel,

$$
wkv_t = \frac{\sum_{i\lt t} e^{-(t-1-i)w + k_i}\, v_i + e^{u + k_t}\, v_t}{\sum_{i\lt t} e^{-(t-1-i)w + k_i} + e^{u + k_t}}
$$

where $w \ge 0$ is a learned per-channel decay rate, $u$ is a learned "bonus" for the current token, and $k_i, v_i$ are per-channel scalars. It is a linear-attention-like weighted average with exponential decay, computable as an RNN. Later versions moved steadily toward the table above: RWKV-5/6 ("Eagle" and "Finch") introduced matrix-valued states and data-dependent decay, and RWKV-7 adopts a generalized delta rule. RWKV is notable as a community-trained open model family that scaled to 14B parameters early.

### DeltaNet and Gated DeltaNet: overwrite, do not accumulate

Return to the associative-memory view. Suppose the state already stores value $v_{\text{old}}$ under key $k$ and a new token wants to store $v_{\text{new}}$ under the same key. Additive linear attention writes $S + k v_{\text{new}}^\top$, so reading with $k$ now returns roughly $v_{\text{old}} + v_{\text{new}}$, a corrupted mix. Decay helps only by blurring everything uniformly.

{{< fig "linear-attention/11" >}}

The **delta rule** (Schlag et al., 2021, made trainable at scale by Yang et al., 2024) first reads what is currently stored under $k_t$, then writes the *difference*:

$$
v_t^{\text{old}} = S_{t-1}^\top k_t, \qquad
S_t = S_{t-1} + \beta_t\, k_t \left(v_t - v_t^{\text{old}}\right)^\top = \left(I - \beta_t k_t k_t^\top\right) S_{t-1} + \beta_t k_t v_t^\top
$$

with $k_t$ L2-normalized and $\beta_t = \sigma(w_\beta^\top x_t) \in (0,1)$ a learned write strength. With $\beta_t = 1$ the old value under $k_t$ is fully replaced. Equivalently, $S_t$ is one step of online gradient descent on the regression loss $\tfrac{1}{2}\|S^\top k_t - v_t\|^2$ with learning rate $\beta_t$, which is why this line of work connects to "test-time training" and Titans: the state is a tiny model being trained on the context as it streams by.

The update is no longer a simple sum, so parallelizing it is harder. Songlin Yang's DeltaNet work shows that products of $(I - \beta k k^\top)$ matrices can be represented compactly (the WY representation from numerical linear algebra), giving a chunkwise algorithm with matmuls.

**Gated DeltaNet** (Yang, Kautz, Hatamizadeh, 2024) combines both mechanisms: $S_t = a_t (I - \beta_t k_t k_t^\top) S_{t-1} + \beta_t k_t v_t^\top$. The gate $a_t$ can clear memory quickly (useful at document boundaries or topic shifts); the delta rule edits specific associations precisely. It outperforms both Mamba-2 and DeltaNet on language modeling and retrieval benchmarks at equal size, and it is the linear layer in Qwen3-Next.

## 5. Tradeoffs and hybrids

### The recall problem

A fixed-size state is a lossy compression of the context. For a head with $d_k = d_v = 128$, the state holds 16,384 numbers. A softmax attention head at 32k tokens holds $2 \times 32{,}768 \times 128 \approx 8.4$ million. No update rule can make the small state store arbitrary amounts of exact information.

{{< fig "linear-attention/12" >}}

This shows up concretely in **associative recall** tasks: present many key-value pairs, then query a key. The Zoology paper (Arora et al., 2023) showed that gaps between attention-free models and transformers in perplexity are dominated by tokens that require this kind of recall, and that the gap grows as the number of pairs exceeds what the state can hold. "Repeat After Me" (Jelassi et al., 2024) proved and showed empirically that transformers can copy long strings that fixed-state models cannot. In practice:

- **Strong** for linear/SSM models: perplexity on ordinary text, long-range smooth dependencies, throughput, memory, very long streaming inputs.
- **Weak**: needle-in-a-haystack retrieval, copying, in-context learning with many examples, multi-hop lookups over long documents, and phone-book style "find this exact string" tasks.
- **Partially mitigated** by larger states (Mamba-2's bigger $N$), better update rules (delta rule), and more heads, but never eliminated.

{{< fig "linear-attention/13" >}}

### Hybrids: a few attention layers go a long way

If a model only needs exact lookup occasionally, it does not need softmax attention in every layer. Hybrids interleave a small number of full (or sliding-window) attention layers with many linear layers. The attention layers handle precise retrieval; the linear layers carry most of the compute and keep the KV cache small.

{{< fig "linear-attention/14" >}}

- **Jamba** (AI21, 2024): Mamba layers and attention layers in a 1:7 attention-to-Mamba ratio, plus mixture-of-experts MLPs. 52B total parameters (12B active), supporting 256k context, with about 140k tokens of context fitting on a single 80GB GPU because only one layer in eight keeps a KV cache.
- **MiniMax-01** (2025): "lightning attention" (a linear attention variant) with a softmax attention layer after every seven linear layers, scaled to 456B total parameters and multi-million-token contexts.
- **Qwen3-Next** (2025): Gated DeltaNet layers with gated full attention in a 3:1 ratio.
- **NVIDIA Nemotron-H** (2025): mostly Mamba-2 layers with a small fraction of attention layers.

The pattern across these papers: hybrids match or beat pure transformers on standard benchmarks and recover most of the retrieval ability, while cutting KV-cache memory by roughly the fraction of layers that are linear. Pure linear models remain mostly a research and edge-device story.

{{< fig "linear-attention/15" >}}

### Practical tradeoffs summary

| Aspect | Softmax attention | Linear / SSM | Hybrid |
|---|---|---|---|
| Training FLOPs | $O(n^2 d)$ | $O(n d^2)$ | between |
| Inference memory per layer | grows with $n$ (KV cache) | constant state | small cache on few layers |
| Decode cost per token | grows with $n$ | constant | mostly constant |
| Exact recall / copying | excellent | limited by state size | good |
| Ecosystem maturity | FlashAttention, paged KV, everything | `flash-linear-attention`, `mamba-ssm`, growing vLLM support | improving |

One subtlety: at short contexts (a few thousand tokens), $n \lt  d$ for typical models and FlashAttention is extremely optimized, so linear attention may be *slower*. The advantage appears at long contexts and, above all, at decode time with large batches, where KV-cache memory is the bottleneck.

## Common confusions

- **"Linear attention approximates softmax attention."** Performer tried to; modern linear models do not. RetNet, Mamba-2 and DeltaNet are different sequence mixers with different inductive biases, not approximations.
- **"Linear means the model is linear."** Only the recurrence is linear in the state. Projections, gates (which depend nonlinearly on the input), MLPs and normalization keep the network nonlinear.
- **"The parallel form of linear attention is $O(n)$."** It is $O(n^2)$, same as softmax. Linear cost comes from the recurrent or chunkwise forms.
- **"Mamba is not attention."** Mamba-2's SSD paper shows it is exactly masked linear attention with a structured decay mask. Mamba-1 differs mainly in using per-channel rather than per-head decay.
- **"Constant memory means infinite context."** The state does not grow, but it also cannot hold more. Information is overwritten or decays; you can process infinite input, but you cannot recall arbitrary details from it.
- **"Linear models need RoPE."** Most do not use it: the decay mask already encodes relative position and recency. Hybrids often use RoPE (or no positional encoding) in their attention layers only.

## Check yourself

1. Explain, using associativity, why $\phi(Q)(\phi(K)^\top V)$ is cheaper than $(\phi(Q)\phi(K)^\top)V$ for long sequences, and why the causal mask prevents you from using the first form directly in parallel.
2. A model has 32 layers, 32 heads per layer, $d_k = d_v = 128$, bf16. Compare the memory of a pure linear-attention state with a softmax KV cache (no GQA) at 128k tokens.
3. Why did making $\Delta$, $B$ and $C$ input-dependent in Mamba force the authors to abandon S4's convolution-based training?
4. Write the DeltaNet update for $\beta_t = 1$ and a unit-norm $k_t$. What does reading the new state with $k_t$ return, and why can additive linear attention not do this?
5. In `decay_parallel`, why is the mask built from `cumsum` of `log_a` rather than by multiplying the decays directly?
6. Your hybrid model has one attention layer per eight. A user complains it misses facts buried in a 200k-token document that a pure transformer finds. Name two architecture-level changes you could try.

<details>
<summary>Answers</summary>

1. $\phi(K)^\top V$ is $d_k \times d_v$ and costs $O(n d_k d_v)$; multiplying by $\phi(Q)$ costs another $O(n d_k d_v)$, so total $O(n d^2)$. The other order builds an $n \times n$ matrix, $O(n^2 d)$. With a causal mask, each row $i$ needs a *different* prefix sum $\sum_{j \le i} \phi(k_j) v_j^\top$, so a single $\phi(K)^\top V$ is not enough. You need either the recurrence (a running sum) or the chunkwise mix.
2. Linear: $32 \times 32 \times 128 \times 128 \times 2$ bytes $= 32$ MiB total, regardless of length. Softmax KV cache: $2 \text{ (K and V)} \times 32 \times 32 \times 128 \times 131{,}072 \times 2$ bytes $= 64$ GiB. That is a factor of 2048.
3. The convolution trick relies on time invariance: the same $\bar{A}, \bar{B}, C$ at every step makes the output a convolution with one fixed kernel. With input-dependent parameters, each step has different dynamics and there is no single kernel, so Mamba uses a parallel scan of the recurrence instead.
4. $S_t = (I - k_t k_t^\top) S_{t-1} + k_t v_t^\top$. Reading: $S_t^\top k_t = S_{t-1}^\top (I - k_t k_t^\top) k_t + v_t (k_t^\top k_t) = S_{t-1}^\top(k_t - k_t) + v_t = v_t$. The old association under $k_t$ is erased and replaced exactly. Additive attention would return $S_{t-1}^\top k_t + v_t$, the old value plus the new one.
5. The mask entries are products of up to $n$ numbers in $(0,1)$, which underflow to zero in floating point and are expensive to compute pairwise. Prefix sums of logs turn every product into one subtraction, $\text{cum}_i - \text{cum}_j$, and the final `exp` is stable because masked entries are set to $-\infty$ before exponentiation.
6. Increase the fraction of attention layers (for example 1:3 instead of 1:7); increase the state size of the linear layers (more heads or larger $d_k$, or Mamba-2's larger $N$); switch the linear layers to a delta-rule variant for better recall; or make sure the attention layers are full rather than sliding-window. Also check that training included long-context retrieval data.

</details>

## Resources

### Read

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236) **(start here)**: short, clear derivation of the kernel trick and the recurrent form.
- [Performer (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794), [RetNet (Sun et al., 2023)](https://arxiv.org/abs/2307.08621) and [Gated Linear Attention (Yang et al., 2023)](https://arxiv.org/abs/2312.06635): the lineage from approximating softmax, to fixed decay with the parallel/recurrent/chunkwise trio, to data-dependent gating with hardware-efficient kernels.
- [The Annotated S4 (Sasha Rush and Sidd Karamcheti)](https://srush.github.io/annotated-s4/): S4 implemented step by step in JAX with the math alongside.
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) and [Mamba-2 / Transformers are SSMs](https://arxiv.org/abs/2405.21060): selection, then state space duality.
- [State Space Duality (Mamba-2) blog series (Tri Dao)](https://tridao.me/blog/2024/mamba2-part1-model/): the authors' own walkthrough of SSD, more readable than the paper.
- [A Visual Guide to Mamba and State Space Models (Maarten Grootendorst)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state): diagram-heavy intuition for SSMs and selectivity.
- [DeltaNet Explained (Songlin Yang)](https://sustcsonglin.github.io/blog/2024/deltanet-1/): the delta rule as online regression and how to parallelize it. Papers: [DeltaNet](https://arxiv.org/abs/2406.06484), [Gated DeltaNet](https://arxiv.org/abs/2412.06464).
- [RWKV (Peng et al., 2023)](https://arxiv.org/abs/2305.13048), [Jamba (Lieber et al., 2024)](https://arxiv.org/abs/2403.19887) and [Zoology (Arora et al., 2023)](https://arxiv.org/abs/2312.04927): the RNN-style lineage, the first large hybrid, and the recall analysis.

### Watch

- [Mamba and S4 Explained (Umar Jamil)](https://www.youtube.com/watch?v=8Q_tqwpTpVU): long-form derivation covering discretization, convolution and parallel scan.
- [Efficiently Modeling Long Sequences with Structured State Spaces (Albert Gu, Stanford MLSys)](https://www.youtube.com/watch?v=EvQ3ncuriCM): the S4 author on HiPPO and why SSMs work.

## Project

**Build an associative-recall benchmark and watch fixed-size states fail.**

1. Generate a synthetic multi-query associative recall (MQAR) task: sequences of $P$ random key-value token pairs followed by queries of some of the keys, where the target is the matching value. Vary $P \in \{8, 32, 128, 512\}$.
2. Implement four tiny 2-layer models that differ only in the token mixer: softmax attention, linear attention with elu+1 (from section 3), decayed linear attention (the `decay_parallel` script with a learned gate), and DeltaNet (write a recurrent version first, then optionally a chunked one).
3. Train each at two head dimensions, $d = 32$ and $d = 128$, and plot recall accuracy against $P$.
4. Explain the curves with state-size arithmetic: at what $P$ should each model start failing, and does the delta rule beat additive updates at equal state size?
5. Bonus: add a hybrid (one softmax layer plus one DeltaNet layer) and check whether it recovers softmax-level recall. Compare wall-clock decode speed for 8k-token sequences between softmax attention with a KV cache and your recurrent DeltaNet.

---

previous: [mixture of experts](/posts/learn-ai-07-mixture-of-experts/) · next: [reinforcement learning](/posts/learn-ai-09-reinforcement-learning/)
