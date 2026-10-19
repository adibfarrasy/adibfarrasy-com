---
title: "learn ai fundamentals, 7: mixture of experts"
date: 2026-10-18T11:00:00+07:00
description: "sparse models: more parameters without more compute per token"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 7 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/07-mixture-of-experts.html).

Most frontier open-weight models released since 2024 (Mixtral, DeepSeek-V3, Qwen3-MoE, Kimi K2, Llama 4, OLMoE) are not dense transformers. They replace the feed-forward block with a *mixture of experts*: many parallel FFNs, of which each token only uses a few. This chapter explains why that trade is worth making, how the router is trained without collapsing, and what it costs you at training and inference time.

## What you will learn

- The difference between parameter count and compute per token, and why sparse models decouple them.
- How a top-k softmax router works, and the lineage from Shazeer et al. (2017) through GShard and Switch Transformer.
- Capacity factor, token dropping, and dropless routing.
- The load-balancing auxiliary loss (derived), the router z-loss, and DeepSeek-V3's auxiliary-loss-free balancing.
- Expert parallelism and why all-to-all communication dominates MoE systems work.
- Fine-grained and shared experts (DeepSeekMoE), Mixtral 8x7B, OLMoE.
- Inference implications: memory scales with total parameters, FLOPs with active parameters.
- How to implement a top-k MoE layer with a load-balancing loss in PyTorch.

## Prerequisites

From LLM Fundamentals, review:

- **Transformer blocks**: the FFN / SwiGLU sublayer is exactly what MoE replaces.
- **GPT-2 from scratch**: you should be comfortable writing a block and a training loop.
- **Scaling laws**: MoE is a way to buy a better loss for the same training FLOPs.
- **Distributed training (ZeRO/FSDP, tensor/pipeline parallelism)**: expert parallelism is a fourth axis that sits beside these.
- **Llama 3**: as the reference dense architecture we compare against.

---

## Mixture of Experts (MoE)

### 1. Intuition: dense vs sparse compute

In a dense transformer, every parameter participates in every token's forward pass. If you have $N$ parameters, a forward pass costs roughly $2N$ FLOPs per token and a training step roughly $6N$. Scaling laws tell you that loss improves with both parameters and data, so to get a better model you pay more FLOPs per token, forever, including at inference.

A sparse model breaks that coupling. Instead of one FFN of width $d_{ff}$, a layer holds $E$ FFNs ("experts"). A small learned *router* looks at each token and sends it to only $k$ of them (typically $k = 1$, $2$ or $8$). The layer now *stores* roughly $E$ times as many FFN parameters but each token only *computes* through $k$ of them. Two numbers now describe the model:

- **Total parameters**: everything you must store (and load into GPU memory).
- **Active parameters**: what a single token actually touches. This sets FLOPs per token.

{{< fig "mixture-of-experts/01" >}}

Mixtral 8x7B has about 46.7B total parameters but only about 12.9B active per token, so it runs at roughly the speed of a 13B dense model while scoring like a much larger one. DeepSeek-V3 has 671B total and 37B active.

{{< fig "mixture-of-experts/02" >}}

Why should this work? The intuition is that a large model's capacity is mostly "knowledge storage" (the FFNs are often described as key-value memories), and different tokens need different slices of that knowledge. A Python token and a French token rarely need the same memories. Conditional computation lets you grow the memory without growing the per-token arithmetic.

Two things go wrong if you naively do this, and most of the MoE literature is about fixing them:

1. **Routing collapse.** Early in training a few experts get slightly better, the router sends them more tokens, they get even better, and the rest starve. You end up with a dense model wearing a costume.
2. **Systems imbalance.** Experts live on different GPUs. If one expert receives 5x its share of tokens, its GPU becomes the straggler and everyone waits.

{{< fig "mixture-of-experts/03" >}}

### 2. The router: top-k softmax gating

Let $x \in \mathbb{R}^{d}$ be a token's hidden state entering the MoE layer, and $E$ the number of experts. The router is a single linear map $W_r \in \mathbb{R}^{E \times d}$ (no bias, usually):

$$
h = W_r x \in \mathbb{R}^{E}, \qquad p = \mathrm{softmax}(h), \qquad p_i = \frac{e^{h_i}}{\sum_{j=1}^{E} e^{h_j}}
$$

- $h_i$ is the router logit for expert $i$.
- $p_i$ is the router's probability that expert $i$ should handle this token.

Select the top-$k$ experts $\mathcal{T} = \mathrm{TopK}(p, k)$ and combine their outputs:

$$
y = \sum_{i \in \mathcal{T}} g_i \, \mathrm{FFN}_i(x)
$$

{{< fig "mixture-of-experts/04" >}}

- $\mathrm{FFN}_i$ is expert $i$, usually a SwiGLU MLP with the same shape as (or smaller than) a dense FFN.
- $g_i$ is the gate weight. Two conventions exist: use $g_i = p_i$ directly (Switch, DeepSeek), or renormalize over the selected set, $g_i = p_i / \sum_{j \in \mathcal{T}} p_j$ (Mixtral, which equivalently does a softmax over only the top-$k$ logits).

{{< fig "mixture-of-experts/05" >}}

The output $y$ goes through the usual residual connection. Attention, norms and embeddings stay dense; only the FFN is sparse.

**How does the router learn?** TopK is not differentiable with respect to *which* experts were chosen, but $y$ is differentiable with respect to $g_i$, which depends on $h$, which depends on $W_r$. So the router gets gradient only through the gate values of the experts it picked. If expert 3 was selected and its output reduced the loss, the gradient increases $g_3$ and therefore $h_3$. Experts not selected get no direct signal. This "only learn about what you tried" structure is exactly why collapse happens and why we need explicit balancing.

### 3. A short history

**Shazeer et al. 2017, "Outrageously Large Neural Networks".** Introduced the sparsely-gated MoE layer between stacked LSTM layers, with up to thousands of experts and 137B parameters. Key ideas that survived: top-$k$ gating, *noisy* gating (add tunable Gaussian noise to $h$ before TopK so that the router explores), and auxiliary losses to balance "importance" (sum of gate values per expert) and "load" (number of tokens per expert).

**GShard (Lepikhin et al. 2020).** Scaled MoE transformers to 600B parameters for multilingual translation on TPUs. Every other FFN became a top-2 MoE. Introduced *expert capacity* (a hard limit on tokens per expert), the auxiliary loss in the $\sum_i f_i P_i$ form we derive below, local dispatching within groups, and random routing of the second expert. Its other big contribution is the systems one: SPMD sharding annotations that make expert parallelism expressible.

**Switch Transformer (Fedus, Zoph, Shazeer 2021).** Argued that top-1 routing is enough. With $k=1$ the router compute, communication and expert computation are all halved compared with top-2, and quality held up. Switch scaled to 1.6T parameters and reported up to 7x pretraining speedups over T5 at equal FLOPs. It also popularized the training tricks that make MoE stable: computing the router in float32 ("selective precision"), smaller initialization, and expert dropout during fine-tuning.

**ST-MoE (Zoph et al. 2022).** "Designing Stable and Transferable Sparse Expert Models." Introduced the router z-loss, studied fine-tuning instability, and provides the most practical recipe-level advice of the early papers.

{{< fig "mixture-of-experts/06" >}}

### 4. Capacity factor and token dropping

Hardware wants static shapes. If you allocate a buffer per expert, how big should it be? With $T$ tokens in a batch, $E$ experts and top-$k$ routing, perfectly balanced routing gives each expert $Tk/E$ tokens. The **expert capacity** is

$$
C = \left\lceil \mathrm{CF} \cdot \frac{T \cdot k}{E} \right\rceil
$$

- $T$: tokens in the routing group (batch times sequence length, or a sub-group).
- $k$: experts per token.
- $E$: number of experts.
- $\mathrm{CF}$: capacity factor, a slack multiplier, typically 1.0 to 1.25 for training and higher for evaluation.

If more than $C$ tokens pick an expert, the overflow tokens are **dropped**: they skip the expert and pass through on the residual connection only. That is not catastrophic (the residual stream still carries the token) but it is lost computation and adds noise. A larger CF drops fewer tokens but wastes memory and FLOPs on padding.

{{< fig "mixture-of-experts/07" >}}

Modern implementations (MegaBlocks, used by OLMoE and others) are often **dropless**: they use block-sparse or grouped GEMM kernels that handle variable tokens per expert, so no capacity limit is needed. DeepSeek-V3 also reports dropping no tokens in training or inference. Dropless routing makes load balancing a speed concern rather than a quality concern, but it is still a concern: an overloaded expert is a slow GPU.

### 5. The load-balancing auxiliary loss, derived

We want every expert to receive about $1/E$ of the tokens. Define, over a batch $\mathcal{B}$ of $T$ tokens:

$$
f_i = \frac{1}{T k} \sum_{x \in \mathcal{B}} \mathbb{1}\{ i \in \mathrm{TopK}(p(x), k) \}, \qquad
P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} p_i(x)
$$

- $f_i$: the fraction of routing slots that went to expert $i$ (the actual load). $\sum_i f_i = 1$.
- $P_i$: the average router probability assigned to expert $i$ (the router's "intent"). $\sum_i P_i = 1$.
- $\mathbb{1}\{\cdot\}$: indicator function, 1 if the condition is true.

The obvious objective is to penalize the variance of $f$, but $f$ comes from an argmax and has zero gradient almost everywhere. $P$ is differentiable but is not the thing we care about. The Switch / GShard trick is to multiply them:

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \, P_i
$$

- $\alpha$: coefficient, typically $10^{-2}$ (Switch) down to $10^{-3}$ or less.
- $E$: number of experts; the factor keeps the loss scale independent of $E$.

{{< fig "mixture-of-experts/08" >}}

Why this works:

1. **Its gradient points the right way.** Treat $f$ as a constant (it is, for the gradient). Then $\partial \mathcal{L}_{\text{aux}} / \partial P_i = \alpha E f_i$. Experts that are overloaded (large $f_i$) have their probability pushed down hardest. Because $p$ comes from a softmax, pushing one probability down pushes others up.
2. **Its minimum is at uniform routing.** If $f$ and $P$ agree (the router sends tokens where its probability mass is), then $\sum_i f_i P_i \approx \sum_i f_i^2$, which on the simplex is minimized at $f_i = 1/E$, giving $\sum_i f_i^2 = 1/E$. Multiplying by $E$ gives a minimum value of 1, which is a handy sanity check: a well-balanced model logs an aux loss close to $\alpha$.
3. **It is cheap.** Two reductions over the batch.

Without $\alpha E$, the loss for a uniform router would be $1/E$ and you would have to retune $\alpha$ whenever you changed $E$.

The aux loss has a cost: it is a second objective competing with language modeling. Too large an $\alpha$ forces the router toward uniform, which can mean routing tokens to experts that are worse for them. This tension motivated auxiliary-loss-free balancing (section 8).

### 6. Router z-loss

The router logits $h$ can drift to large magnitudes. Large logits make the softmax saturate, and in bf16 they lose precision: exponentiation of numbers around 100 has very few significant bits left. ST-MoE found this was a major source of training instability, and added

$$
\mathcal{L}_z = \beta \cdot \frac{1}{T} \sum_{x \in \mathcal{B}} \left( \log \sum_{j=1}^{E} e^{h_j(x)} \right)^2
$$

- $\log \sum_j e^{h_j}$: the log-partition function (logsumexp) of the router logits for token $x$.
- $\beta$: coefficient, $10^{-3}$ in ST-MoE.

Squaring the logsumexp penalizes logits that are large in magnitude without caring about their relative order, so routing decisions are untouched while numerics stay healthy. OLMoE uses both the load-balancing loss and z-loss. The same trick appears on the output softmax of some dense models (PaLM used a z-loss on the LM head).

{{< fig "mixture-of-experts/09" >}}

### 7. Expert parallelism and all-to-all

Recall from Fundamentals that data parallelism replicates the model, tensor parallelism splits matrices, and pipeline parallelism splits layers. **Expert parallelism (EP)** splits the experts: with $E$ experts on $D$ devices, each device holds $E/D$ experts. Attention and other dense parts are still handled with data/tensor parallelism.

A forward pass through one MoE layer then looks like:

1. Each device computes router scores for its local tokens.
2. **All-to-all dispatch**: each device sends each token to the device that owns its chosen expert(s). Every device sends to and receives from every other device.
3. Each device runs its local experts on the tokens it received (a batched or grouped GEMM).
4. **All-to-all combine**: results are sent back to the tokens' home devices and weighted by $g_i$.

That is two all-to-alls per MoE layer in the forward pass and two more in the backward. Unlike all-reduce, all-to-all does not benefit from ring tricks; its cost is bounded by bisection bandwidth, and it gets painful across nodes where InfiniBand is several times slower than intra-node NVLink. This is why:

{{< fig "mixture-of-experts/10" >}}

- Load balance matters for speed: the all-to-all and expert compute finish when the busiest device finishes.
- DeepSeek-V3 uses **node-limited routing**: each token may only be sent to experts on at most 4 nodes, and the team wrote custom all-to-all kernels that overlap communication with computation (their "DualPipe" schedule).
- Top-$k$ choice is a systems decision as much as a modeling one. Doubling $k$ doubles the bytes moved.

{{< fig "mixture-of-experts/11" >}}

Memory note: with EP, optimizer states for experts are naturally sharded, so EP composes with ZeRO-style sharding of the dense parameters.

### 8. Modern refinements

#### Fine-grained and shared experts (DeepSeekMoE)

DeepSeekMoE (Dai et al. 2024) made two changes that most recent MoEs adopt.

**Fine-grained expert segmentation.** Split each expert into $m$ smaller experts (each with $1/m$ the hidden width) and activate $m$ times as many. Active compute is unchanged, but the number of possible expert combinations explodes. With 16 experts choosing 2 there are 120 combinations; split each into 4 (64 experts choosing 8) and there are about 4.4 billion. More combinations means experts can be more specialized and recombined flexibly.

{{< fig "mixture-of-experts/12" >}}

**Shared experts.** Reserve $K_s$ experts that every token always uses, in addition to the routed ones:

$$
y = \sum_{i=1}^{K_s} \mathrm{FFN}^{\text{shared}}_i(x) + \sum_{i \in \mathcal{T}} g_i \, \mathrm{FFN}^{\text{routed}}_i(x)
$$

The idea is that some knowledge (common syntax, frequent tokens) is needed by everyone. Without shared experts, many routed experts each relearn it, which is redundant. Shared experts absorb it so routed experts can specialize. DeepSeek-V3 uses 1 shared expert and 256 routed experts with 8 active per token.

{{< fig "mixture-of-experts/13" >}}

#### Auxiliary-loss-free balancing (DeepSeek-V3)

DeepSeek-V3 (building on Wang et al. 2024, "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts") replaces the main balancing loss with a per-expert bias that only affects *selection*, not the gate value:

$$
s_{i}(x) = \sigma(u_i^\top x), \qquad
\mathcal{T} = \mathrm{TopK}\big(\{ s_i(x) + b_i \}_{i=1}^{E}, k\big), \qquad
g_i = \frac{s_i(x)}{\sum_{j \in \mathcal{T}} s_j(x)}
$$

- $u_i$: the centroid vector for expert $i$ (a row of the router weight).
- $\sigma$: sigmoid. DeepSeek-V3 uses sigmoid affinities instead of softmax.
- $b_i$: a bias for expert $i$, *not* trained by gradient descent.

After each training step, measure each expert's load. If expert $i$ is overloaded, $b_i \leftarrow b_i - \gamma$; if underloaded, $b_i \leftarrow b_i + \gamma$, with update speed $\gamma$ (0.001 for most of training). The bias nudges routing toward balance like a controller, while the gate weights $g_i$ that multiply expert outputs remain purely what the model learned. No gradient from a balancing objective interferes with the language modeling loss. DeepSeek-V3 keeps a tiny sequence-level balance loss ($\alpha = 10^{-4}$) as a safety net against extreme imbalance within a single sequence.

{{< fig "mixture-of-experts/14" >}}

### 9. Landmark models

**Mixtral 8x7B (Mistral, 2024).** Same architecture as Mistral 7B but each FFN is replaced by 8 SwiGLU experts with top-2 routing at every layer. 46.7B total, 12.9B active. It matched or beat Llama 2 70B on most benchmarks at a fraction of the inference FLOPs. The name is misleading: it is not 8 copies of a 7B model (that would be 56B), because attention and embeddings are shared and only the FFNs are replicated.

**DeepSeek-V3 (2024).** 671B total, 37B active, MLA attention, 1 shared + 256 routed experts with 8 active, aux-loss-free balancing, node-limited routing, FP8 training. Trained on 14.8T tokens for a reported 2.788M H800 GPU hours, which reset expectations about what frontier training costs.

**OLMoE (AI2, 2024).** OLMoE-1B-7B: 6.9B total, 1.3B active, 64 experts with 8 active per layer, trained on 5T tokens. Its value for learners is that everything is open: data, code, checkpoints and logs. The paper also runs many ablations (fine-grained vs coarse experts, shared experts, token-choice vs expert-choice routing, load-balancing and z-loss) at a scale you can reason about.

{{< fig "mixture-of-experts/15" >}}

### 10. Do experts specialize?

It is tempting to imagine a "math expert" and a "French expert". The evidence is more nuanced:

- **ST-MoE** found encoder experts specialize by token type: punctuation, proper nouns, verbs, numbers. Decoder experts showed much less specialization, and experts did not specialize by language in multilingual training.
- **Mixtral** analyzed routing across domains (ArXiv, biology, philosophy, GitHub, math) and found no clear domain-level specialization. Routing correlated more with syntax: tokens like `self` in Python, indentation, and consecutive tokens often went to the same expert.
- **OLMoE** found clearer *domain* specialization (for example, arXiv and GitHub tokens concentrated on particular experts) and low co-activation between experts, which the authors attribute partly to fine-grained experts and to training from scratch rather than upcycling a dense model.

The safe summary: experts specialize, but along axes the model finds useful (often lexical or syntactic), which do not map cleanly onto human topic labels. Specialization tends to increase with finer-grained experts.

{{< fig "mixture-of-experts/16" >}}

### 11. Code: a top-k MoE layer with load-balancing loss

This is a readable reference implementation, not a fast one. It loops over experts and uses index gathers. Production systems use grouped GEMMs (MegaBlocks, or `torch._grouped_mm` style kernels) and expert parallelism.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TopKMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, k=2, n_shared=0,
                 aux_coef=1e-2, z_coef=1e-3):
        super().__init__()
        self.n_experts, self.k = n_experts, k
        self.aux_coef, self.z_coef = aux_coef, z_coef
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUExpert(d_model, d_ff) for _ in range(n_experts))
        self.shared = nn.ModuleList(SwiGLUExpert(d_model, d_ff) for _ in range(n_shared))

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        T = x_flat.shape[0]

        # Router math in fp32: bf16 logits lose precision in exp() and destabilize training.
        logits = self.router(x_flat).float()
        probs = logits.softmax(dim=-1)
        topk_p, topk_idx = probs.topk(self.k, dim=-1)
        # Renormalize over chosen experts (Mixtral convention) so outputs keep a stable scale.
        gates = (topk_p / topk_p.sum(dim=-1, keepdim=True)).to(x.dtype)

        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            token_ids, slot = (topk_idx == e).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            y = self.experts[e](x_flat[token_ids])
            out.index_add_(0, token_ids, y * gates[token_ids, slot, None])

        for s in self.shared:
            out = out + s(x_flat)

        # f is an argmax count and carries no gradient; P carries the gradient back to the router.
        counts = F.one_hot(topk_idx, self.n_experts).sum(dim=(0, 1)).float()
        f = counts / (T * self.k)
        P = probs.mean(dim=0)
        aux_loss = self.aux_coef * self.n_experts * (f * P).sum()
        z_loss = self.z_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()

        self.last_load = f.detach()
        return out.reshape(B, S, D), aux_loss + z_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    moe = TopKMoE(d_model=64, d_ff=128, n_experts=8, k=2, n_shared=1)
    x = torch.randn(4, 32, 64)
    y, aux = moe(x)
    assert y.shape == x.shape

    # Overfit a random regression target and watch the load stay spread across experts.
    target = torch.randn_like(x)
    opt = torch.optim.AdamW(moe.parameters(), lr=3e-3)
    for step in range(300):
        y, aux = moe(x)
        loss = F.mse_loss(y, target) + aux
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(step, f"mse={F.mse_loss(y, target).item():.3f}",
                  "load=", [round(v, 3) for v in moe.last_load.tolist()])
    # Uniform load would be 1/8 = 0.125 per expert.
    assert moe.last_load.max() < 0.5, "router collapsed onto one expert"
```

{{< fig "mixture-of-experts/17" >}}

Things to try with this code:

- Set `aux_coef=0` and watch whether `load` drifts toward a few experts.
- Replace softmax-then-topk with topk-then-softmax over the selected logits; confirm the gates are identical (they are, mathematically).
- Implement a capacity limit: for each expert, keep only the first $C$ tokens and count how many are dropped as you vary CF.

### 12. Inference implications

MoE changes the economics of serving in a way that surprises people coming from dense models.

{{< fig "mixture-of-experts/18" >}}

- **Memory scales with total parameters.** Every expert must be resident (or paged in), because the next token might route anywhere. Mixtral 8x7B needs about 94 GB in bf16 even though it computes like a 13B model. DeepSeek-V3 needs multiple nodes or aggressive quantization.
- **FLOPs scale with active parameters.** Prefill (compute-bound) benefits fully from sparsity.
- **Decode is memory-bandwidth bound.** At batch size 1, each token reads the weights of only its $k$ experts per layer, so you read about the active parameters and MoE is fast. At large batch, different tokens hit different experts, so each step reads nearly *all* experts, and the bandwidth advantage shrinks. The sweet spot is large batches with enough tokens per expert to make each expert's GEMM efficient, which is why MoE serving uses expert parallelism across many GPUs.
- **Quantization and offloading.** Because most experts are idle for any given token, CPU offloading of experts (and expert caching) is more practical than offloading layers of a dense model, though latency suffers.
- **Fine-tuning.** MoEs historically overfit and were unstable to fine-tune (ST-MoE). Common practice: lower learning rate, keep or increase aux loss, sometimes freeze the router.

{{< fig "mixture-of-experts/19" >}}

The headline: an MoE is cheaper per token than a dense model of the same quality, but needs more memory and more engineering to serve. That trade favors providers serving many users at high batch, which is why it dominates frontier deployment.

### Common confusions

- **"Mixtral 8x7B has 56B parameters."** No. Only the FFNs are replicated; attention, norms and embeddings are shared. It has about 46.7B total and 12.9B active.
- **"An MoE is an ensemble of models."** No. Routing happens per token, per layer. A single sequence passes through many different experts, and consecutive layers route independently.
- **"Experts are topic specialists."** Mostly not in coarse MoEs; specialization is often syntactic. Finer-grained experts show more domain specialization.
- **"Active parameters determine memory."** Active parameters determine FLOPs per token. Memory is set by total parameters.
- **"The aux loss makes routing uniform per token."** It balances the *aggregate* load over a batch. Individual tokens still route sharply.
- **"Dropless means balancing doesn't matter."** It removes quality loss from dropping, but imbalance still makes one device the straggler.
- **"Aux-loss-free means no balancing objective at all."** The bias update is a balancing mechanism; it just does not send gradients into the model. DeepSeek-V3 even keeps a tiny sequence-level aux loss.

### Check yourself

1. A model has 32 experts per layer with top-4 routing, and each expert has the same shape as the dense FFN of a 2B model. Roughly how do its FLOPs per token compare to the 2B dense model, and how does its memory footprint compare?
2. Why can't you directly minimize the variance of $f_i$ (the fraction of tokens per expert) with gradient descent, and how does the $f_i P_i$ product get around this?
3. With $T = 4096$ tokens, $E = 64$, $k = 8$ and capacity factor 1.25, what is the expert capacity? What happens to the 600th token assigned to an expert?
4. What problem does the router z-loss solve that the load-balancing loss does not?
5. In DeepSeek-V3's aux-loss-free scheme, why is the bias $b_i$ added only for the TopK selection and not to the gate value $g_i$?
6. Explain why MoE decoding is relatively faster at batch size 1 than at batch size 256, compared with a dense model with the same active parameters.

<details>
<summary>Answers</summary>

1. Each token uses 4 experts, so the FFN compute per token is about 4x the dense FFN (attention unchanged), so somewhat under 4x the dense model's FLOPs overall. Memory holds 32 FFNs, so FFN parameters are 32x; total memory is roughly an order of magnitude larger than the dense 2B model, dominated by the experts.
2. $f_i$ is computed from an argmax/TopK, which is piecewise constant, so its gradient is zero almost everywhere. Multiplying by the differentiable mean probability $P_i$ gives a gradient $\alpha E f_i$ on $P_i$, so overloaded experts get their router probability reduced most. At equilibrium, where $f \approx P$, the loss behaves like $E \sum f_i^2$, minimized at uniform load.
3. $C = \lceil 1.25 \cdot 4096 \cdot 8 / 64 \rceil = 640$. The 600th token fits. A token beyond the 640th would be dropped: it skips that expert and only its residual (plus any other selected experts) carries forward.
4. Logit magnitude. The balancing loss cares about which experts are chosen; z-loss keeps the router logits small so the softmax does not saturate and bf16 exponentiation does not lose precision, which ST-MoE found to be a cause of instability.
5. The bias is a control signal for load, not a statement about how useful an expert is for this token. Adding it to the gate would change the model's output and therefore its loss, reintroducing the interference with language modeling that the method is designed to avoid.
6. At batch 1, each token only reads its $k$ experts per layer, so bytes read per step are roughly the active parameters. At batch 256, the union of selected experts covers nearly all experts, so each step reads close to the total parameters while doing only the active compute per token, and memory bandwidth, not FLOPs, is the bottleneck.

</details>

### Resources

**Read**

- [Mixture of Experts Explained (Hugging Face blog)](https://huggingface.co/blog/moe) (start here): a thorough tour from Shazeer 2017 to Mixtral, with capacity, balancing and serving.
- [A Visual Guide to Mixture of Experts, Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts): diagram-heavy build-up of routing, balancing and capacity.
- [Shazeer et al. 2017, Outrageously Large Neural Networks (arXiv:1701.06538)](https://arxiv.org/abs/1701.06538): the original sparsely-gated MoE layer with noisy top-k gating.
- [Fedus et al. 2021, Switch Transformers (arXiv:2101.03961)](https://arxiv.org/abs/2101.03961): top-1 routing, capacity factor, the canonical aux loss.
- [Zoph et al. 2022, ST-MoE (arXiv:2202.08906)](https://arxiv.org/abs/2202.08906): router z-loss and stability/fine-tuning practice.
- [Dai et al. 2024, DeepSeekMoE (arXiv:2401.06066)](https://arxiv.org/abs/2401.06066): fine-grained and shared experts.
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437): aux-loss-free balancing, node-limited routing, large-scale EP.
- [Muennighoff et al. 2024, OLMoE (arXiv:2409.02060)](https://arxiv.org/abs/2409.02060): fully open MoE with extensive ablations and specialization analysis.
- Also useful: [GShard (arXiv:2006.16668)](https://arxiv.org/abs/2006.16668), [Mixtral of Experts (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088), [Auxiliary-Loss-Free Load Balancing (arXiv:2408.15664)](https://arxiv.org/abs/2408.15664).

**Watch**

- [A Visual Guide to Mixture of Experts (MoE) in LLMs, Maarten Grootendorst](https://www.youtube.com/watch?v=sOPDGQjFcuM): the video companion to the visual guide.
- [Switch Transformers paper explained, Yannic Kilcher](https://www.youtube.com/watch?v=iAR8LkkMMIM): walkthrough of top-1 routing and the Switch results.
- [Sparse Expert Models with the authors, Yannic Kilcher](https://www.youtube.com/watch?v=ccBMRryxGog): interview with Fedus and Zoph on Switch, GLaM and practical lessons.

### Project

**Turn your GPT-2 into an MoE and measure the trade.**

1. Take the GPT-2 (124M-style) model you built in Fundamentals. Replace the FFN in every other block with `TopKMoE` (8 experts, top-2, each expert the same size as the original FFN, or half-size for a fine-grained variant).
2. Train both the dense and MoE models on the same data (for example, a few hundred million tokens of FineWeb-Edu or TinyStories) for the same number of steps. Log loss vs tokens and loss vs wall-clock time.
3. Report: total params, active params, tokens/sec, peak memory, final validation loss.
4. Ablations: (a) `aux_coef = 0`, (b) `z_coef = 0`, (c) top-1 vs top-2, (d) add one shared expert. For each, plot the per-expert load over training (an `E x steps` heatmap of `last_load`).
5. Stretch: implement the DeepSeek-V3 bias update instead of the aux loss and compare load balance and validation loss.
6. Stretch: for a trained model, record which expert each token goes to in layer $L/2$ and check whether code, numbers, punctuation or a particular language cluster on specific experts.

Write a one-page summary of what you found about the parameter/compute/quality trade.

---

previous: [optimizers](/posts/learn-ai-08-optimizers/) · next: [linear attention](/posts/learn-ai-05-linear-attention/)
