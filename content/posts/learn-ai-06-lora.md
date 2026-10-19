---
title: "learn ai fundamentals, 4: lora"
date: 2026-10-18T14:00:00+07:00
description: "fine-tuning a large model by training small low-rank adapters"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 4 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/06-lora.html).


{{< fig "lora/01" >}}
In the fundamentals course you fine-tuned models with SFT and DPO by updating every weight. That is the most expressive option, and for a frontier lab with a cluster it is often the right one. For almost everyone else it is wasteful: the optimizer state alone for a 70B model is hundreds of gigabytes, every fine-tuned variant is a full copy of the model, and serving ten customer-specific variants means ten copies in GPU memory. Low-Rank Adaptation (LoRA) fixes all three problems with one simple idea: freeze the pretrained weights and learn only a small, low-rank correction. This chapter covers why that works, the math and its sharp edges, the variants that patch those edges (QLoRA, DoRA, LoRA+, rsLoRA), how to serve hundreds of adapters from one base model, and what recent evidence says about when LoRA matches full fine-tuning and when it does not.

## What you will learn

- Where the memory of full fine-tuning goes, and how much LoRA saves.
- Why low-rank updates are plausible: intrinsic dimensionality of fine-tuning.
- The LoRA parameterization $W_0 + \frac{\alpha}{r}BA$, its initialization, and the role of $\alpha$ and $r$.
- Which modules to adapt, how merging works, and when you should not merge.
- QLoRA's three tricks: NF4, double quantization and paged optimizers.
- DoRA, LoRA+ and rsLoRA: what each one fixes.
- Multi-adapter serving (S-LoRA, vLLM), and the "learns less, forgets less" tradeoff.
- How to write a LoRA layer from scratch and how to use HuggingFace PEFT in practice.

## Prerequisites

From LLM Fundamentals, review:

- **AdamW**: first and second moment buffers and why they cost 8 bytes per parameter in fp32.
- **Distributed training (ZeRO/FSDP)**: the 16-bytes-per-parameter accounting for mixed-precision training, and what sharding does and does not save.
- **Transformer blocks and attention/GQA**: the names and shapes of `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **SFT and DPO**: the training loops you will now run with adapters instead of full updates. DPO's reference model becomes free with LoRA: it is the base model with the adapter disabled.
- **Llama 3**: you will use a Llama-family model in the PEFT examples.

## 1. Intuition: why a small update can be enough

### The cost of full fine-tuning

Mixed-precision training with AdamW stores, per parameter: a bf16 weight (2 bytes), a bf16 gradient (2 bytes), an fp32 master weight (4 bytes) and two fp32 Adam moments (8 bytes). That is 16 bytes per parameter before activations. For an 8B model, roughly 128 GB; for 70B, roughly 1.1 TB. ZeRO-3/FSDP spreads this across GPUs but does not reduce the total. Then there is storage and serving: every fine-tuned variant is another full checkpoint (16 GB for 8B in bf16), and serving several variants concurrently means holding several full copies in GPU memory.

With LoRA, the base weights are frozen and stored once in bf16 (2 bytes per parameter, no gradients, no optimizer state). Only the adapter, typically 0.1% to 1% of the parameter count, needs gradients and Adam moments. For an 8B model that turns ~128 GB of training state into ~16 GB of frozen weights plus well under 1 GB of adapter state, plus activations. A LoRA checkpoint is tens to hundreds of megabytes.

{{< fig "lora/02" >}}

### Intrinsic dimensionality

Why would a tiny update be enough? Li et al. (2018) introduced the **intrinsic dimension** of a task: train in a random low-dimensional subspace of parameter space, $\theta = \theta_0 + P\,\omega$ with $P \in \mathbb{R}^{D \times d}$ a fixed random projection, $\omega \in \mathbb{R}^d$ the only trainable vector and $d \ll D$, and find the smallest $d$ that reaches 90% of full performance. Aghajanyan et al. (2020) applied this to fine-tuning pretrained language models and found strikingly small numbers: on MRPC, tuning about 200 parameters projected into RoBERTa's full space recovered 90% of full fine-tuning performance. Moreover, larger and better-pretrained models had *lower* intrinsic dimension.

{{< fig "lora/03" >}}

The interpretation: pretraining has already learned the features; fine-tuning mostly re-weights and steers them. The change needed is a small move in a well-chosen direction, not a rewrite. LoRA's authors (Hu et al., 2021) made the specific bet that this small move, for each weight matrix, is **low-rank**.

## 2. The math of LoRA

### Parameterization

Take one linear layer with pretrained weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and input $x \in \mathbb{R}^{d_{\text{in}}}$. Full fine-tuning learns $W_0 + \Delta W$ with $\Delta W$ unrestricted. LoRA constrains the update to rank at most $r$:

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A x, \qquad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}
$$

where:

- $r \ll \min(d_{\text{in}}, d_{\text{out}})$ is the **rank**, typically 8 to 64 (sometimes 1 to 256).
- $A$ is the "down projection": it maps the input into an $r$-dimensional bottleneck.
- $B$ is the "up projection": it maps the bottleneck back to the output dimension.
- $\alpha$ is a constant scaling hyperparameter, and $\frac{\alpha}{r}$ is the effective multiplier on the update.
- $W_0$ is frozen; only $A$ and $B$ receive gradients.

The adapter has $r(d_{\text{in}} + d_{\text{out}})$ parameters instead of $d_{\text{in}} d_{\text{out}}$. For a $4096 \times 4096$ projection at $r = 16$, that is 131,072 instead of 16,777,216: 0.78%.

{{< fig "lora/04" >}}

In batched form, with $X \in \mathbb{R}^{N \times d_{\text{in}}}$ holding $N$ tokens as rows, $H = X W_0^\top + \frac{\alpha}{r} (X A^\top) B^\top$. Computing $XA^\top$ first keeps the intermediate at shape $N \times r$, so the extra FLOPs are small and the full $\Delta W$ is never materialized during training.

### Initialization: why $B = 0$

LoRA initializes $A$ randomly (Kaiming-uniform in PEFT, Gaussian in the paper) and $B = 0$. Two consequences:

1. **The adapted model starts exactly at the pretrained model.** $BA = 0$ at step zero, so the first forward pass is identical to the base. Fine-tuning begins from known-good behavior instead of from a random perturbation.
2. **Gradients still flow.** With $B = 0$, the gradient with respect to $A$ is zero at first ($\partial L / \partial A = \frac{\alpha}{r} B^\top (\partial L / \partial h) x^\top$), but the gradient with respect to $B$ is $\frac{\alpha}{r} (\partial L/\partial h) (Ax)^\top$, which is nonzero because $A$ is random. After one step $B \ne 0$ and both matrices train. If you initialized *both* to zero, both gradients would be zero forever.

{{< fig "lora/05" >}}

### The scaling factor $\alpha / r$

Why have $\alpha$ at all when $B$ can learn any scale? Because of the optimizer. With Adam, the size of each parameter update is roughly the learning rate, regardless of gradient magnitude. If you change $r$, the number of terms summed in $BA$ changes, and the effective size of $\Delta W$ changes with it. Dividing by $r$ was meant to keep the update magnitude roughly constant across ranks so you can change $r$ without re-tuning the learning rate. In practice, people set $\alpha$ once (commonly $\alpha = r$ or $\alpha = 2r$, or a fixed 16 or 32) and treat it as part of the learning rate. Note that changing $\alpha$ with Adam is nearly equivalent to changing the learning rate of $B$ and $A$.

**rsLoRA** (Kalajdzievski, 2023) showed that $\frac{\alpha}{r}$ over-corrects: as $r$ grows, the update shrinks too much, gradients collapse, and higher ranks stop helping. A stability analysis gives the "rank-stabilized" factor $\frac{\alpha}{\sqrt{r}}$, under which higher ranks do keep improving. If you plan to use ranks above ~64, use rsLoRA (`use_rslora=True` in PEFT).

{{< fig "lora/06" >}}

### Which modules to target

The original paper adapted only $W_q$ and $W_v$ in attention, because it was memory-constrained and those gave the best result per parameter on GPT-3. That advice is now outdated:

- **QLoRA** found that adapting **all linear layers** (attention projections and MLP projections) was necessary to match full fine-tuning; rank mattered much less than coverage.
- **Thinking Machines' "LoRA Without Regret"** found MLP layers matter most: MLP-only LoRA beat attention-only LoRA, even at equal parameter count. MLPs hold most of the parameters and most of the "knowledge".
- Embeddings and the LM head are usually left frozen. Adapt them (or train them fully, via `modules_to_save`) only when you add new tokens, for example new special tokens for a chat template.

The modern default is `target_modules="all-linear"`.

{{< fig "lora/07" >}}

### Merging

Since $\Delta W = \frac{\alpha}{r}BA$ has the same shape as $W_0$, you can fold it in after training: $W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$. The merged model has exactly the original architecture and **zero extra inference latency**, which was one of LoRA's key selling points over earlier adapter methods that added sequential layers. You can also subtract it back out to unmerge.

{{< fig "lora/08" >}}

When **not** to merge:

- When you serve many adapters on one base (section 4); merged weights would require a full model copy per adapter.
- When the base is quantized (QLoRA). Merging into 4-bit weights requires dequantizing, adding, and requantizing; the requantization error changes the model's behavior. The standard practice is to load the base in bf16, merge there, and quantize the merged model afterwards if needed.

{{< fig "lora/09" >}}

## 3. Code: LoRA from scratch and with PEFT

### A LoRA layer from scratch

{{< fig "lora/10" >}}

```python
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.0,
                 rslora: bool = False):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)   # the pretrained weight is never updated
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.r = r
        # rsLoRA divides by sqrt(r) so the update does not shrink as rank grows
        self.scaling = alpha / math.sqrt(r) if rslora else alpha / r
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))   # B = 0 makes the adapted model start identical to the base
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # A must be random or B's gradient would be zero forever
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.merged:
            return out
        # x @ A^T @ B^T keeps the intermediate at width r instead of materializing a d_out x d_in matrix
        return out + (self.dropout(x) @ self.A.T @ self.B.T) * self.scaling

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.B @ self.A) * self.scaling
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight -= (self.B @ self.A) * self.scaling
            self.merged = False


if __name__ == "__main__":
    torch.manual_seed(0)
    d_in, d_out, r = 64, 32, 4
    base = nn.Linear(d_in, d_out)
    layer = LoRALinear(base, r=r, alpha=8)
    x = torch.randn(256, d_in)

    # B = 0 at init, so the wrapped layer is exactly the base layer
    assert torch.allclose(layer(x), base(x))

    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in layer.parameters())
    print(f"trainable {trainable} / {total} ({100 * trainable / total:.1f}%)")

    # Target: the base layer plus a hidden rank-2 change; LoRA with r=4 should fit it
    with torch.no_grad():
        delta = torch.randn(d_out, 2) @ torch.randn(2, d_in) * 0.1
        y = x @ (base.weight + delta).T + base.bias
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
    for step in range(2000):
        loss = ((layer(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("final mse:", loss.item())
    assert loss.item() < 1e-4

    before = layer(x)
    layer.merge()
    assert torch.allclose(layer(x), before, atol=1e-5)  # merging must not change outputs
    layer.unmerge()
    assert torch.allclose(layer(x), before, atol=1e-5)
    print("merge/unmerge ok")
```

The toy is tiny so the percentage is large; at real sizes it drops below 1%. To apply this to a whole model, walk `model.named_modules()`, find `nn.Linear` modules whose names end in your target list, and replace them with `LoRALinear(module)` via `setattr` on the parent. That is essentially what PEFT does, plus a lot of bookkeeping.

A good experiment: change the hidden delta to rank 8 while keeping `r=4`. The loss will plateau well above zero. That is the capacity limit you will meet again in section 5.

### HuggingFace PEFT: LoRA and QLoRA

{{< fig "lora/11" >}}

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "meta-llama/Llama-3.2-1B"  # any causal LM works; this one is small enough for a single consumer GPU

# QLoRA: frozen base stored in 4-bit NF4, matmuls computed in bf16
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
# Casts norms to fp32 and enables gradient checkpointing, which 4-bit training needs for stability and memory
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",   # attention and MLP projections; MLPs matter most
    task_type="CAUSAL_LM",
    # use_rslora=True,             # switch on when going to high ranks
    # use_dora=True,               # magnitude/direction decomposition, slower per step
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# ... train with transformers.Trainer or TRL's SFTTrainer / DPOTrainer as in the fundamentals course ...

model.save_pretrained("my-adapter")   # writes only the adapter, tens of MB

# Merging: reload the base in bf16, not 4-bit, so the merged weights are not requantized
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, "my-adapter").merge_and_unload()
merged.save_pretrained("my-merged-model")
```

TRL's `SFTTrainer` and `DPOTrainer` accept a `peft_config` argument directly, and with LoRA, `DPOTrainer` uses the base model with the adapter disabled as the reference policy, so you do not hold a second model in memory. For LoRA+, PEFT provides `peft.optimizers.create_loraplus_optimizer`.

## 4. Variants: QLoRA, DoRA, LoRA+ and multi-adapter serving

### QLoRA

QLoRA (Dettmers et al., 2023) asked: if the base weights are frozen, why store them in 16 bits? It quantizes the base to 4 bits and trains LoRA adapters in bf16 on top, backpropagating *through* the dequantized weights. It fine-tuned a 65B model on a single 48 GB GPU with no measurable quality loss versus 16-bit LoRA on their benchmarks. Three ingredients:

{{< fig "lora/12" >}}

**NF4 (4-bit NormalFloat).** Pretrained weights are approximately zero-mean normally distributed. A uniform 4-bit grid wastes levels in the tails where few weights live. NF4 places its 16 levels at quantiles of a standard normal distribution (adjusted to include an exact zero), so each level is used by roughly the same number of weights, which is information-theoretically optimal for normal data. Weights are quantized in blocks of 64: each block is divided by its absolute maximum $c$ (stored in higher precision), mapped to the nearest NF4 level, and stored as a 4-bit index. During the forward pass, each weight is dequantized as $w \approx c \cdot \text{NF4}[\text{index}]$ into bf16 for the matmul.

{{< fig "lora/13" >}}

**Double quantization.** Those per-block constants $c$ add overhead: one 32-bit float per 64 weights is $32/64 = 0.5$ bits per parameter. QLoRA quantizes the constants themselves to 8 bits in blocks of 256, with one fp32 constant per block of constants. Overhead falls to about $8/64 + 32/(64 \cdot 256) \approx 0.127$ bits per parameter, saving roughly 0.37 bits per parameter, about 3 GB on a 65B model.

{{< fig "lora/14" >}}

**Paged optimizers.** Long sequences produce memory spikes during gradient checkpointing that can OOM an otherwise-fitting job. QLoRA allocates optimizer states in NVIDIA unified memory, so they page to CPU RAM automatically under pressure and back when needed, like virtual memory.

{{< fig "lora/15" >}}

The tradeoffs: QLoRA is slower per step than bf16 LoRA (dequantization on every forward and backward), and the adapter is trained against a quantized base, so it is best deployed against that same quantized base or merged carefully as described above.

### DoRA

DoRA (Liu et al., 2024) starts from an analysis: decompose each weight matrix into a **magnitude** (column norms) and a **direction** (normalized columns). Comparing full fine-tuning with LoRA, they found full fine-tuning often changes magnitude and direction in opposite proportions (big directional change with small magnitude change, or vice versa), whereas LoRA's updates tend to change both together. DoRA reparameterizes:

$$
W' = m \odot \frac{W_0 + BA}{\lVert W_0 + BA \rVert_c}
$$

where $\lVert \cdot \rVert_c$ is the L2 norm of each weight vector belonging to one output channel (the paper writes weights transposed relative to PyTorch and calls this "column-wise"), giving one norm per output channel; $m$ is a trainable vector with one magnitude per output channel, initialized to $\lVert W_0 \rVert_c$; and the division and $\odot$ broadcast per channel. LoRA's $BA$ updates only the direction; $m$ updates only the magnitude. DoRA typically improves on LoRA at the same rank, especially at low ranks, at the cost of extra compute per step (the norm must be recomputed). It merges into a plain weight matrix after training, so inference cost is unchanged.

{{< fig "lora/16" >}}

### LoRA+

LoRA+ (Hayou et al., 2024) points out that $A$ and $B$ play asymmetric roles and, in the large-width limit, should not share a learning rate. Using a single learning rate makes feature learning inefficient. The fix is one line: set $\eta_B = \lambda\, \eta_A$ with $\lambda$ around 16. They report up to about 2x faster convergence and small accuracy gains, at no extra cost.

{{< fig "lora/17" >}}

### Serving many adapters

Because adapters are small and the base is shared, one GPU can serve many fine-tuned "models":

- **Unmerged batched inference.** Keep $W_0$ shared and compute each request's $x A_i^\top B_i^\top$ with its own adapter $i$. Requests for different adapters can be batched together: the expensive $W_0$ matmul is shared across the batch, and the small per-adapter matmuls are handled by gather-style kernels (Punica's SGMV is the well-known one).
- **S-LoRA** (Sheng et al., 2023) scaled this to thousands of concurrent adapters. Adapters live in CPU memory and are paged into GPU memory on demand; a **unified paging** pool manages both adapter weights (of varying ranks) and the KV cache to avoid fragmentation; custom kernels handle heterogeneous ranks in one batch; and tensor parallelism is extended to adapters. It reported up to 4x throughput over vLLM with naive adapter support and orders of magnitude more adapters than PEFT-based serving.
- In practice, **vLLM** and **SGLang** support multi-LoRA serving out of the box (in vLLM, start with `enable_lora=True` and pass a `LoRARequest` per request). This is how "one fine-tuned model per customer" products are built economically.

{{< fig "lora/18" >}}

{{< fig "lora/19" >}}

{{< fig "lora/20" >}}

## 5. Modern practice: how well does LoRA actually work?

{{< fig "lora/21" >}}

### LoRA learns less and forgets less

Biderman et al. (2024) ran careful comparisons of LoRA versus full fine-tuning on Llama-2 7B in two regimes, continued pretraining (billions of tokens) and instruction fine-tuning, in code and math. Findings:

- **LoRA learns less.** In most settings, especially continued pretraining on code, LoRA substantially underperformed full fine-tuning on the target domain. The gap was smaller for instruction tuning than for continued pretraining.
- **LoRA forgets less.** LoRA better preserved the base model's performance on tasks outside the target domain. It acts as a regularizer, stronger than weight decay or dropout.
- **Full fine-tuning's updates are not low-rank.** The spectrum of $\Delta W$ from full fine-tuning has rank 10 to 100 times higher than typical LoRA configurations. The low-rank hypothesis holds less well for absorbing large amounts of new knowledge.

{{< fig "lora/22" >}}

### LoRA Without Regret

Thinking Machines' 2025 study refined this picture with careful sweeps:

- In the regime most people care about (post-training datasets up to the size of typical SFT and preference sets), LoRA with good settings **matches full fine-tuning** in sample efficiency and final loss. It underperforms once the dataset contains more information than the adapter can hold, which is the regime Biderman et al. probed.
- **Apply LoRA to all layers, especially MLPs.** Attention-only LoRA underperforms even at matched parameter count.
- **The optimal LoRA learning rate is about 10x the full fine-tuning learning rate**, and with the $\frac{1}{r}$ scaling it is roughly independent of rank.
- LoRA is **less tolerant of large batch sizes** than full fine-tuning.
- For **reinforcement learning**, even rank-1 LoRA matches full fine-tuning, consistent with RL providing very few bits of information per episode.

{{< fig "lora/23" >}}

Put together: LoRA is an excellent default for instruction tuning, preference optimization, RL and style or format adaptation. For injecting large volumes of new knowledge (a new language, a large codebase, continued pretraining) prefer full fine-tuning, or at least much larger ranks with rsLoRA, and measure.

### Practical hyperparameters

| Setting | Starting point | Notes |
|---|---|---|
| `target_modules` | `"all-linear"` | MLP coverage matters more than rank. |
| `r` | 16 to 64 | Go higher (with rsLoRA) for larger datasets; 1 to 8 is often enough for RL or small style tasks. |
| `lora_alpha` | 16 to 32, or $2r$ | Interacts with learning rate; fix it and tune LR. |
| Learning rate | 1e-4 to 2e-4 for 7B-class | About 10x full fine-tuning LR. Sweep it before anything else. |
| `lora_dropout` | 0 to 0.05 | Mostly matters for small datasets. |
| Batch size | Modest (e.g. 32 to 128 sequences) | LoRA degrades faster than full FT at large batch. |
| Epochs | 1 to 3 | Multiple epochs on small static datasets often overfit. |
| Scheduler | Cosine or linear with short warmup | As in the fundamentals SFT recipe. |

{{< fig "lora/24" >}}

Sebastian Raschka's experiments also found that QLoRA saves about a third of memory versus LoRA at a moderate cost in runtime, that $\alpha = 2r$ is a reasonable heuristic, and that results are quite consistent across random seeds, so a single careful run is informative.

## Common confusions

- **"LoRA makes inference faster."** No. A merged LoRA model is exactly as fast as the base. Unmerged adapters add a small cost. LoRA saves training memory, storage and multi-tenant serving memory.
- **"LoRA reduces activation memory."** Barely. Activations depend on the base model's width and sequence length, and backprop still flows through every frozen layer. Use gradient checkpointing for activations.
- **"Higher rank is always better."** With standard $\alpha/r$ scaling, very high ranks can train worse because the update is scaled down. Use rsLoRA for high ranks, and remember that coverage of modules usually matters more than rank.
- **"Initializing $B = 0$ means nothing trains."** $B$'s gradient depends on $A$, which is random, so $B$ moves on the first step; after that, $A$ gets gradients too. Only initializing both to zero is broken.
- **"QLoRA trains a 4-bit model."** The base is stored in 4-bit and frozen; the adapters and all computation are in bf16. The trainable parameters are never quantized.
- **"Merging a QLoRA adapter is free."** Merging into 4-bit weights requires requantization and changes outputs. Merge into a bf16 copy of the base.
- **"LoRA is just a cheaper approximation of full fine-tuning."** It is also a different inductive bias: it forgets less, which is sometimes exactly what you want.

## Check yourself

1. For a $4096 \times 14336$ MLP projection, how many trainable parameters does LoRA add at $r = 16$, and what fraction of the full matrix is that?
2. Why does LoRA initialize $B$ to zero and $A$ randomly, rather than both randomly or both zero?
3. You raise $r$ from 16 to 256 with standard scaling and keep $\alpha = 16$. Training gets worse. Explain why, and give two fixes.
4. In QLoRA, what does NF4 optimize for, and what problem does double quantization solve? Do the arithmetic for the overhead with and without it.
5. A team wants to serve 500 customer-specific fine-tunes of an 8B model. Compare memory for merged full checkpoints against a shared base with unmerged rank-16 adapters on all linear layers (rough numbers are fine).
6. Your LoRA fine-tune on 20B tokens of a new programming language underperforms full fine-tuning, but your LoRA chat fine-tune on 50k examples matches it. Explain using the two studies discussed in section 5.

<details>
<summary>Answers</summary>

1. $r(d_{\text{in}} + d_{\text{out}}) = 16 \times (4096 + 14336) = 294{,}912$. The full matrix has $58{,}720{,}256$ entries, so about 0.5%.
2. $B = 0$ makes $\Delta W = 0$ at the start, so training begins exactly at the pretrained model. $A$ must be random so that $B$'s gradient, which is proportional to $Ax$, is nonzero; if both were zero all gradients would vanish. If both were random, the model would start at a randomly perturbed, worse point.
3. The update is scaled by $\alpha/r = 16/256 = 0.0625$ instead of $1$, so its effective step size shrinks and, per rsLoRA's analysis, learning at high rank slows or collapses. Fixes: use rsLoRA scaling $\alpha/\sqrt{r}$; raise $\alpha$ proportionally (for example $\alpha = 2r$); or re-tune the learning rate upward.
4. NF4 places its 16 levels at quantiles of a normal distribution, so for normally distributed weights each level is used equally often, minimizing information loss for that distribution. Double quantization cuts the overhead of per-block scale constants. Without it: one fp32 constant per 64 weights is $32/64 = 0.5$ bits per parameter. With it: 8-bit constants plus one fp32 constant per 256 blocks gives $8/64 + 32/16384 \approx 0.127$ bits per parameter, saving about 0.37 bits per parameter.
5. Merged: $500 \times 16$ GB $= 8$ TB of bf16 weights, impossible to keep resident. Shared base: 16 GB once. A Llama-3-8B-style model has 32 layers with seven linear projections; at $r=16$ the adapter is roughly 40M to 45M parameters, about 85 MB in bf16. 500 of them is about 42 GB, and with S-LoRA-style paging only active adapters need to be on GPU.
6. Biderman et al. found LoRA learns less in high-information regimes like continued pretraining, because full fine-tuning's updates there are high-rank. Thinking Machines found that when the dataset's information content fits within the adapter's capacity, as with typical instruction tuning, a well-tuned LoRA matches full fine-tuning. 20B tokens of a new language exceeds that capacity; 50k chat examples does not.

</details>

## Resources

### Read

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685): the original paper; sections 4 and 7 cover the method and the rank analysis. Pair it with [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning (Aghajanyan et al., 2020)](https://arxiv.org/abs/2012.13255) for the empirical motivation.
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314): NF4, double quantization, paged optimizers, and the all-linear-layers finding.
- [DoRA (Liu et al., 2024)](https://arxiv.org/abs/2402.09353), [LoRA+ (Hayou et al., 2024)](https://arxiv.org/abs/2402.12354) and [rsLoRA (Kalajdzievski, 2023)](https://arxiv.org/abs/2312.03732): the three most useful refinements, each a small change.
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters (Sheng et al., 2023)](https://arxiv.org/abs/2311.03285): unified paging and batched heterogeneous adapters.
- [LoRA Learns Less and Forgets Less (Biderman et al., 2024)](https://arxiv.org/abs/2405.09673): the careful counterpoint on knowledge-heavy fine-tuning.
- [LoRA Without Regret (Thinking Machines)](https://thinkingmachines.ai/blog/lora/) **(start here)**: the most practical modern guide to when LoRA matches full fine-tuning and how to set it up.
- [Practical Tips for Finetuning LLMs Using LoRA (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) and [Improving LoRA: Implementing DoRA from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch): hands-on experiments and from-scratch code.
- [HuggingFace PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora): every `LoraConfig` option, initialization variants, merging and multi-adapter APIs.

### Watch

- [What is Low-Rank Adaptation (LoRA), explained by the inventor (Edward Hu)](https://www.youtube.com/watch?v=DhRoTONcyZE): short, first-hand motivation for the design.
- [LoRA explained visually + PyTorch code from scratch (Umar Jamil)](https://www.youtube.com/watch?v=PXWYUTMt-AU): derivation plus an implementation you can compare with the one above.

## Project

**Find LoRA's capacity limit on a real model.**

1. Take a small instruct model (Llama-3.2-1B-Instruct or Qwen2.5-0.5B-Instruct) and two datasets of different information content: (a) a small style or format task, such as 2k examples of answering in a strict JSON schema, and (b) a knowledge-heavy task, such as 50M+ tokens of a domain corpus (a programming language's documentation, legal texts or a non-English Wikipedia subset).
2. Fine-tune with: full fine-tuning; your from-scratch `LoRALinear` injected into all linear layers at $r \in \{4, 32, 256\}$; and PEFT LoRA at $r = 256$ with `use_rslora=True`. Sweep the learning rate for each (at least three values) and keep the best.
3. Measure: target-domain loss or accuracy, and forgetting on a held-out general benchmark (for example a few hundred MMLU or HellaSwag questions).
4. Plot target performance against forgetting for every run. Identify where LoRA matches full fine-tuning and where it falls short, and relate it to the "learns less, forgets less" and "LoRA Without Regret" findings.
5. Bonus: repeat the best LoRA configuration with QLoRA and DoRA, and report memory, step time and quality. Then serve two of your adapters simultaneously from one base with vLLM's multi-LoRA support and confirm each request gets its own behavior.

---

previous: [post-training](/posts/learn-ai-03-post-training/) · next: [distillation](/posts/learn-ai-04-distillation/)
