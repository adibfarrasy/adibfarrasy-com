---
title: "learn ai fundamentals, 2: pre-training"
date: 2026-10-18T16:00:00+07:00
description: "objectives, scaling laws, data, systems, and a llama 3 case study"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 2 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/02-pretraining.html).

Chapter 1 built the machine: tokenizer, embeddings, RoPE, attention, the transformer block, and a working GPT-2. This chapter is about what happens when you feed that machine a large slice of the internet and a large pile of GPUs for weeks. Pretraining is where almost all of a language model's knowledge and most of its compute bill come from. Everything after it (instruction tuning, RLHF, tool use) is a comparatively thin layer on top.

{{< fig "pretraining/01" >}}

## What you will learn

- What a pretraining run produces, what it costs, and how a base model differs from a chat model.
- Why the next-token cross-entropy loss is the objective, how it relates to perplexity, and how it compares with BERT-style and T5-style objectives.
- The small architectural tweaks (pre-norm, no biases, QK-norm, z-loss, careful init) that keep large runs stable.
- Scaling laws: how to split a compute budget between parameters and tokens, and why modern models deliberately ignore "compute-optimal".
- Optimizer and schedule details used in practice: AdamW, warmup, cosine and WSD schedules, batch size, muP.
- How a raw web crawl becomes a training corpus: extraction, filtering, deduplication, mixtures.
- How a model too large for one GPU is trained across thousands: memory budgets, mixed precision, the parallelism zoo, FSDP, MFU, fault tolerance.
- Objectives beyond plain next-token prediction: FIM, multi-token prediction, UL2, long-context and mid-training stages.
- How to evaluate a model while it is still training, and where those evaluations mislead.
- How all of this came together in one real frontier run: Llama 3.

## Prerequisites

- Chapter 1 (Architecture). You should be able to write a GPT-2 forward pass and explain what attention, GQA, RoPE, RMSNorm and SwiGLU do. This chapter will not re-explain them.
- Python, and basic comfort with logarithms and exponents. Linear algebra only shows up as "matrix multiply costs about 2 x rows x inner x cols FLOPs".
- Optional but helpful: having run a small PyTorch training loop once.

All code snippets use only the Python standard library so they run anywhere with `python3`.

---

## 01. Overview

### What pretraining produces

Pretraining takes a randomly initialized transformer and a very large text corpus and runs one task over and over: given the tokens so far, predict the next one. After trillions of such predictions the weights encode grammar, facts, coding idioms, arithmetic patterns, and a surprising amount of reasoning structure, because all of these help predict the next token. The output is a **base model** (also called a pretrained or foundation model): a single checkpoint file of weights that, given any prefix, returns a probability distribution over the vocabulary for the next token.

A base model is a document completer, not an assistant. Prompt it with "What is the capital of France?" and it may answer "Paris", or it may continue with "What is the capital of Germany? What is the capital of Spain?", because a list of quiz questions is a perfectly plausible document. A **chat model** is a base model that has gone through post-training (supervised fine-tuning on conversations, then preference optimization such as RLHF or DPO) so that it reliably follows instructions, uses a chat template, refuses some requests, and stops when done. Post-training changes behavior; pretraining sets capability. A rough but useful rule: if the base model cannot do something with a good few-shot prompt, post-training rarely makes it appear.

### The three knobs: parameters, data, compute

Three quantities describe a run:

- **N**, the number of parameters (GPT-2 small: 124M; Llama 3: 8B, 70B, 405B).
- **D**, the number of training tokens (GPT-2: roughly 10B seen; GPT-3: 300B; Llama 2: 2T; Llama 3: about 15T).
- **C**, the training compute in floating point operations (FLOPs). For a dense transformer, a good approximation is

$$
C \approx 6ND
$$

where 6 comes from 2 FLOPs per parameter per token for the forward pass (one multiply and one add per weight) plus about 4 for the backward pass (gradients with respect to both activations and weights). Section 03 derives this properly. Plug in Llama 3 405B: $6 \times 4.05\times10^{11} \times 1.56\times10^{13} \approx 3.8\times10^{25}$ FLOPs. An H100 delivers roughly $10^{15}$ dense bf16 FLOP/s on paper and around 40% of that in practice, so the run needs on the order of $10^{11}$ GPU-seconds, about 26 million GPU-hours, which is why it ran on 16,384 GPUs for months.

{{< fig "pretraining/02" >}}

{{< fig "pretraining/03" >}}

{{< fig "pretraining/04" >}}

### The pipeline from raw web to checkpoint

A pretraining project is mostly data and systems engineering with a modeling core:

1. **Collect** raw text: web crawls (Common Crawl), code (GitHub), books, papers (arXiv), Wikipedia, math sites, multilingual sources.
2. **Process**: extract text from HTML, identify language, filter for quality, remove duplicates, scrub PII and toxic content, decontaminate against benchmarks (Section 04).
3. **Tokenize** everything once and store as packed arrays of token IDs, sharded across files.
4. **Choose a mixture**: what fraction of each batch comes from web, code, math, multilingual, and so on.
5. **Pick a model size and token budget** with scaling laws fitted on small proxy runs (Section 03).
6. **Train** across thousands of accelerators with parallelism, mixed precision and checkpointing (Section 05), watching loss and evals (Section 07).
7. **Late stages**: anneal on high-quality data, extend context length (Section 06).
8. **Release** a base checkpoint, which then goes to post-training.

Notice that model architecture, which consumed all of Chapter 1, is only one step. Teams that train frontier models spend most of their people-time on data and infrastructure.

### Why it works at all

Next-token prediction is a deceptively rich task. To predict the next token of a Python function you need to track variable names and types. To predict the next token of a physics derivation you need the algebra. To predict the next line of a dialogue you need a model of the speakers. The loss keeps falling as long as the model can find any structure left in the data, so scale (more parameters to store structure, more tokens to find it) keeps paying off in a smooth, predictable way. That predictability is what lets labs commit hundreds of millions of dollars to a run: Section 03 shows you can forecast the final loss from runs a thousand times smaller.

```python
# Back-of-envelope cost of a pretraining run.
# Why: before any code is written, this arithmetic decides whether a run is feasible.

def train_cost(params, tokens, gpus, peak_flops=989e12, mfu=0.40, usd_per_gpu_hour=2.5):
    flops = 6 * params * tokens
    seconds = flops / (gpus * peak_flops * mfu)
    gpu_hours = gpus * seconds / 3600
    return flops, seconds / 86400, gpu_hours, gpu_hours * usd_per_gpu_hour

for name, n, d, g in [
    ("GPT-2 124M, 10B tokens", 124e6, 10e9, 8),
    ("Llama 3 8B, 15T tokens", 8e9, 15e12, 4096),
    ("Llama 3 405B, 15.6T tokens", 405e9, 15.6e12, 16384),
]:
    f, days, gh, usd = train_cost(n, d, g)
    print(f"{name:28s} C={f:.2e} FLOPs  {days:7.1f} days on {g} H100s  ~{gh/1e6:.2f}M GPU-h  ~${usd/1e6:.1f}M")
```

### Common confusions

- "Pretraining teaches the model to chat." It does not. A base model continues documents. Chat behavior is post-training.
- "More parameters is always better." Only for a fixed amount of data and compute; Section 03 shows a too-big model trained on too few tokens loses to a smaller one trained longer.
- "6ND is exact." It ignores attention's quadratic term and embedding lookups. It is accurate to within roughly 10% for typical sizes and context lengths, which is all you need for planning.
- "Training tokens = dataset size." Tokens seen can exceed unique tokens when data is repeated for multiple epochs (common for high-quality sources), or be smaller when only part of a corpus is used.

### Check yourself

1. A base model is prompted with "Q: Who wrote Hamlet?\nA:". Why does this format work better than the bare question?
2. Estimate the training FLOPs for a 7B model on 2T tokens.
3. Which pipeline steps would you expect to take the most engineering time, and why?
4. Why can't post-training fix a base model that has never seen any code?

<details>
<summary>Answers</summary>

1. The Q/A scaffold makes "an answer follows" the most likely continuation. Bare questions might be continued as part of a list of questions.
2. $6 \times 7\times10^9 \times 2\times10^{12} = 8.4\times10^{22}$ FLOPs.
3. Data processing and distributed training. Data quality drives model quality per FLOP, and keeping thousands of GPUs busy and recovering from failures is a large systems problem. The model code itself is small.
4. Post-training uses a tiny amount of data relative to pretraining (millions vs trillions of tokens). It reshapes how existing knowledge is expressed; it cannot install a large body of new knowledge or skill.

</details>

### Resources

**Read**
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) (start here). Course site with lecture notes and assignments covering this entire chapter end to end.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). The most detailed public account of a frontier pretraining pipeline; skim the table of contents now, Section 08 walks through it.

**Watch**
- [CS336 lecture playlist (YouTube)](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). Lecture 1 frames the full pipeline and its costs.
- [Karpathy, "Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU). Four hours of a real, small pretraining run, from code to loss curves to evals.

---

## 02. Training Objectives and Architectural Details

### The causal language modeling loss

A causal LM factorizes the probability of a sequence $x_1, \dots, x_T$ with the chain rule of probability:

$$
p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{\lt t})
$$

- $\theta$: all model parameters.
- $x_t$: the token at position $t$, an integer in $\{0, \dots, V-1\}$ with $V$ the vocabulary size.
- $x_{\lt t}$: all tokens before position $t$.

This is not an approximation: any distribution over sequences can be written this way. The modeling choice is to let a transformer with a causal mask (Chapter 1) compute every conditional in parallel.

{{< fig "pretraining/05" >}}

Training maximizes the likelihood of the corpus, which is the same as minimizing the negative log-likelihood. Taking logs turns the product into a sum, and averaging over positions gives the per-token loss:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{\lt t})
$$

How does the model output $p_\theta(\cdot \mid x_{\lt t})$? The final hidden state at position $t-1$ is multiplied by the unembedding matrix to give a vector of **logits** $z \in \mathbb{R}^V$, one score per vocabulary entry. Softmax turns logits into probabilities:

$$
p_\theta(x_t = i \mid x_{\lt t}) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}} = \frac{e^{z_i}}{Z}
$$

so the loss at one position is

$$
\ell_t = -\log p_\theta(x_t \mid x_{\lt t}) = \log Z - z_{x_t} = \log \sum_j e^{z_j} - z_{x_t}
$$

Read this as: "push up the logit of the correct token, push down the log-sum-exp of all logits." This is exactly the **cross-entropy** between the one-hot target distribution and the model distribution, hence the name. Its gradient with respect to the logits is beautifully simple: $\partial \ell_t / \partial z_i = p_i - \mathbb{1}[i = x_t]$, the predicted probability minus the target. Every wrong token gets pushed down in proportion to how much probability it stole.

Two reference points to calibrate intuition. A uniformly random model over $V$ tokens has loss $\log V$; for GPT-2's $V = 50{,}257$ that is about 10.8 nats, which is what you see at step 0. A well-trained GPT-2 124M reaches around 3.3 nats on web text; Llama-class models are around 1.5 to 2 nats depending on tokenizer and data. Losses are only comparable across models with the same tokenizer and evaluation data, because a bigger vocabulary packs more text into each token.

### Perplexity

**Perplexity** is the exponential of the average loss:

$$
\mathrm{PPL} = \exp\left(\mathcal{L}\right) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{\lt t})\right)
$$

Interpretation: a perplexity of 20 means the model is, on average, as uncertain as if it were choosing uniformly among 20 tokens. Random guessing gives PPL $= V$, perfect prediction gives 1. Perplexity is convenient for humans; loss is what you plot because its differences are additive.

{{< fig "pretraining/06" >}}

### Teacher forcing

During training the model always conditions on the **true** previous tokens, never on its own predictions. This is called teacher forcing, and it is what allows all $T$ positions to be trained in a single parallel forward pass: the input is `tokens[:-1]`, the target is `tokens[1:]`, and the causal mask stops each position from peeking ahead. The downside is **exposure bias**: at inference the model conditions on its own outputs, which it never practiced on, so early mistakes can compound. In practice, at scale, this matters less than early papers feared, and nobody pretrains without teacher forcing because it would destroy parallelism.

### Packing and document boundaries

Corpora are tokenized, documents are joined with an end-of-text token, and the stream is cut into fixed-length sequences (for example 8,192 tokens). One training sequence therefore often contains the end of one document and the start of another. Most older models let attention cross the boundary; Llama 3 applies a mask so tokens only attend within their own document, which mattered for long-context training.

### Alternative objectives: masked LM and span corruption

**Masked language modeling (BERT, 2018)** uses an encoder with bidirectional attention. Randomly choose 15% of tokens, replace most of them with a `[MASK]` token, and predict the originals from both left and right context. This gives excellent representations for classification and retrieval, but only 15% of positions produce a loss (less signal per FLOP), and the model cannot naturally generate text left to right.

**Span corruption (T5, 2019)** uses an encoder-decoder. Contiguous spans of the input (mean length 3, 15% of tokens) are replaced with sentinel tokens `<X>`, `<Y>`; the decoder must output `<X> span1 <Y> span2`. This is efficient and good for tasks with a clear input and output, such as translation and summarization.

Why did decoder-only causal LMs win for general-purpose models? Every token is a training target, generation is native, the same model does both understanding and generation through prompting, and the architecture is simple to scale and serve (one stack, one KV cache). Experiments such as T5's and UL2's found encoder-decoders competitive at equal compute on some benchmarks, but in-context learning and open-ended generation favored decoder-only, and the ecosystem converged there. Section 06 covers hybrids such as prefix LM and UL2.

{{< fig "pretraining/07" >}}

### Architectural choices for stability

Large runs fail in ways small runs do not: the loss suddenly spikes or diverges after weeks of training. Modern architectures carry a set of small changes, mostly about keeping activation and logit magnitudes bounded.

- **Pre-norm.** The original transformer normalized after the residual add (post-norm). Pre-norm puts the normalization inside the residual branch, $h \leftarrow h + f(\mathrm{Norm}(h))$, so the residual stream is an untouched sum and gradients flow straight through it. Pre-norm trains stably without long warmup at depth; post-norm needs careful warmup. Essentially every modern LLM is pre-norm, plus one final norm before the unembedding.
- **RMSNorm** (Chapter 1) instead of LayerNorm: drops the mean subtraction and bias, slightly faster, equally stable.
- **No bias terms** in linear layers (LLaMA, PaLM). Biases add little capacity, and removing them improved stability in PaLM and simplifies sharding.
- **QK-norm.** Apply RMSNorm (or LayerNorm) to queries and keys per head before the dot product. Without it, attention logits $q \cdot k$ can grow without bound during training, softmax saturates to one-hot, and training diverges (observed in ViT-22B and in the "small-scale proxies" study). Used in OLMo 2, Gemma 2/3, Qwen 3 and others.
- **z-loss.** An auxiliary penalty on the log of the softmax normalizer of the output logits:

$$
\mathcal{L}_z = \lambda_z \cdot (\log Z)^2, \qquad Z = \sum_{j=1}^{V} e^{z_j}
$$

with $\lambda_z \approx 10^{-4}$ (PaLM). It keeps $\log Z$ near 0, which stops output logits from drifting to huge values where bf16 round-off becomes significant. Cheap insurance against late-training divergence.

- **Initialization.** Weights are drawn from a normal distribution with small standard deviation (0.02 in GPT-2, or scaled like $1/\sqrt{d_{\text{model}}}$). The output projections of each residual branch (attention out-projection and MLP down-projection) are further scaled by $1/\sqrt{2L}$ for $L$ layers, because the residual stream sums $2L$ branch outputs and you want its variance to stay roughly constant with depth. OLMo 2 and others also report that truncated normal init around 0.02 is more stable than fancier schemes at scale.
- **Weight decay choices.** AdamW decay (typically 0.1) applies to matrices in linear layers but usually **not** to norm gains or biases, because shrinking a norm gain toward 0 just fights the normalization. Whether to decay embeddings varies; OLMo 2 found removing weight decay on embeddings improved stability.
- **Tied vs untied embeddings.** GPT-2 ties the input embedding and output unembedding. Large models usually untie them; small models tie them to save parameters (a 128K vocab at $d=2048$ is 262M parameters per matrix).

{{< fig "pretraining/08" >}}

```python
# Cross-entropy, perplexity and z-loss on a toy vocabulary, stdlib only.
# Why log-sum-exp with max subtraction: raw exp() overflows for logits above ~700.
import math

def log_softmax(logits):
    m = max(logits)
    lse = m + math.log(sum(math.exp(z - m) for z in logits))
    return [z - lse for z in logits], lse

def lm_loss(logit_rows, targets, z_coef=1e-4):
    nll, zl = 0.0, 0.0
    for logits, y in zip(logit_rows, targets):
        logp, log_z = log_softmax(logits)
        nll += -logp[y]
        zl += z_coef * log_z ** 2
    T = len(targets)
    return nll / T, zl / T

V = 8
uniform = [[0.0] * V for _ in range(4)]
confident = [[5.0 if i == y else 0.0 for i in range(V)] for y in [1, 2, 3, 4]]
targets = [1, 2, 3, 4]

for name, rows in [("uniform", uniform), ("confident", confident)]:
    ce, z = lm_loss(rows, targets)
    print(f"{name:9s} loss={ce:.3f} nats  ppl={math.exp(ce):.2f}  z-loss={z:.5f}")
print(f"log(V) = {math.log(V):.3f}  (uniform loss should equal this, ppl should equal V)")
```

### Common confusions

- "Loss 2.0 is better than loss 2.3, full stop." Only with the same tokenizer and eval set. A tokenizer with longer tokens has higher per-token loss for the same text quality; compare bits per byte if tokenizers differ.
- "Perplexity and accuracy move together." Mostly, but not exactly. A model can lower perplexity by being better calibrated on easy tokens while not improving on the hard tokens that benchmarks test.
- "BERT is an older version of GPT." They are different objectives and architectures (bidirectional encoder vs causal decoder). BERT-style encoders remain the right tool for embeddings and classification.
- "z-loss is a regularizer for generalization." Its purpose is numerical stability, not better held-out loss.
- "Pre-norm and RMSNorm are the same idea." Pre-norm is about where the norm sits; RMSNorm is which norm. Independent choices.

### Check yourself

1. Derive the gradient of $\ell_t = \log\sum_j e^{z_j} - z_{x_t}$ with respect to $z_i$.
2. A model has validation loss 2.3 nats. What is its perplexity, and what does that number mean?
3. Why does BERT get less learning signal per forward pass than a causal LM?
4. What failure does QK-norm prevent, and why does it show up only at scale or late in training?
5. Why do we not apply weight decay to RMSNorm gains?

<details>
<summary>Answers</summary>

1. $\partial/\partial z_i \log\sum_j e^{z_j} = e^{z_i}/Z = p_i$, and $\partial z_{x_t}/\partial z_i = \mathbb{1}[i = x_t]$. So the gradient is $p_i - \mathbb{1}[i=x_t]$.
2. $e^{2.3} \approx 10$. On average the model is as uncertain as a uniform choice among about 10 tokens.
3. Only the masked 15% of positions contribute a loss, versus 100% of positions in a causal LM.
4. Unbounded growth of query/key norms makes attention logits huge, softmax saturates, gradients vanish or explode and loss diverges. Weight norms grow slowly with training, so it appears after many steps and is worse with larger learning rates and widths.
5. Decay pulls the gain toward 0, which shrinks the normalized activations the norm exists to keep at unit scale; the optimizer then fights the decay for no benefit.

</details>

### Resources

**Read**
- [Small-scale proxies for large-scale Transformer training instabilities (Wortsman et al., 2023)](https://arxiv.org/abs/2309.14322) (start here). Reproduces attention-logit growth and output-logit divergence in small models and shows QK-norm and z-loss fix them.
- [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745). Why pre-norm trains without warmup and post-norm does not.
- [BERT (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805) and [T5 (Raffel et al., 2019)](https://arxiv.org/abs/1910.10683). The masked LM and span-corruption objectives; T5 also has a large objective comparison study.
- [2 OLMo 2 Furious (OLMo team, 2024)](https://arxiv.org/abs/2501.00656). A fully open model report with a detailed section on the stability changes listed above.

**Watch**
- [CS336 lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). The architectures and hyperparameters lecture surveys exactly these choices across recent models.

---

## 03. Scaling Laws and Optimization

### The empirical fact

Train many models of different sizes on different amounts of data and plot the final loss on log-log axes. You get straight lines over many orders of magnitude. Loss falls as a **power law** in parameters, data and compute, until it approaches an irreducible floor (the entropy of natural text itself). This regularity is the reason modern LLM development looks like engineering rather than alchemy: fit the curve on runs costing thousands of dollars, then predict a run costing tens of millions.

{{< fig "pretraining/09" >}}

### Kaplan et al. (2020)

OpenAI's "Scaling Laws for Neural Language Models" fitted separate power laws:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_N \approx 0.076,\ \alpha_D \approx 0.095
$$

- $L$: test loss in nats per token.
- $N$: non-embedding parameter count. $D$: training tokens.
- $N_c, D_c$: fitted constants. $\alpha$: fitted exponents.

Their headline conclusion on allocation was that, as compute grows, most of it should go into parameters: optimal $N \propto C^{0.73}$, with data growing much more slowly. This shaped GPT-3 (175B params, only 300B tokens) and Gopher (280B, 300B tokens). A key flaw, identified later, was that they used a fixed learning-rate schedule length regardless of how many tokens a run actually trained for, which penalized short runs of small models and biased the result toward big models.

### Chinchilla (Hoffmann et al., 2022)

DeepMind redid the study with schedules matched to each run's length, using three methods (fixing model sizes and varying tokens, IsoFLOP profiles, and fitting a parametric loss). The parametric form is the one to remember:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

- $E$: irreducible loss, the entropy of text (what an infinite model on infinite data would reach).
- $A/N^\alpha$: the penalty for a finite model.
- $B/D^\beta$: the penalty for finite data.
- $A, B, \alpha, \beta$: fitted constants.

Minimize $L$ subject to $C = 6ND$ and you get $N_{\text{opt}} \propto C^{a}$, $D_{\text{opt}} \propto C^{b}$ with $a = \beta/(\alpha+\beta)$ and $b = \alpha/(\alpha+\beta)$. Chinchilla found $a \approx b \approx 0.5$: **scale parameters and tokens equally**. In ratio form, compute-optimal training uses roughly **20 tokens per parameter**. Their 70B Chinchilla trained on 1.4T tokens beat the 280B Gopher (300B tokens) at the same compute, which reset the field.

{{< fig "pretraining/10" >}}

A cautionary footnote that is also a good lesson about reading papers: Epoch AI tried to replicate the parametric fit from the paper's own plotted data and found the published constants inconsistent with the other two methods; their corrected fit ($E \approx 1.82$, $A \approx 482$, $B \approx 2085$, $\alpha \approx 0.348$, $\beta \approx 0.366$) agrees with the ~20 tokens/param rule. The snippet below uses the corrected values.

### Where $6ND$ comes from

For one token, each weight matrix of shape $m \times n$ does $mn$ multiply-adds in the forward pass, which is $2mn$ FLOPs. Summed over the model that is $2N$ FLOPs per token. The backward pass needs two matrix multiplies per weight matrix: one for the gradient with respect to inputs (to keep backpropagating) and one for the gradient with respect to weights, so $4N$. Total $6N$ per token, $6ND$ per run. Attention scores add roughly $6 \cdot n_{\text{layer}} \cdot T \cdot d_{\text{model}}$ per token for context length $T$, small relative to $6N$ unless the context is very long relative to model width.

### Over-training for inference

Chinchilla optimizes **training** compute only. But a model that will serve billions of queries pays inference cost proportional to $N$ forever. A smaller model trained far past 20 tokens/param reaches a given loss with more training compute but much cheaper serving. The loss curve keeps improving slowly with more data, so labs now deliberately over-train:

| Model | Params | Tokens | Tokens/param |
|---|---|---|---|
| Chinchilla | 70B | 1.4T | 20 |
| Llama 2 7B | 7B | 2T | ~290 |
| Llama 3 8B | 8B | ~15T | ~1,900 |
| Llama 3 405B | 405B | 15.6T | ~39 |

"Beyond Chinchilla-Optimal" (Sardana et al.) formalizes this by adding expected inference tokens to the objective. The practical takeaway: the 20:1 rule tells you the cheapest way to hit a loss with training compute alone. It is a starting point, not a target.

{{< fig "pretraining/11" >}}

### AdamW as used in pretraining

Almost every LLM is trained with **AdamW**. For each parameter $\theta$ with gradient $g_t$ at step $t$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \eta_t\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\right)
$$

- $m_t$: running mean of gradients (momentum). $v_t$: running mean of squared gradients (per-parameter scale).
- $\beta_1, \beta_2$: decay rates; LLMs typically use $\beta_1 = 0.9$, $\beta_2 = 0.95$ (lower than the 0.999 default, so $v$ adapts faster and avoids instability after rare large gradients).
- $\hat m, \hat v$: bias-corrected estimates (they start at 0).
- $\eta_t$: learning rate at step $t$ (from the schedule). $\epsilon \approx 10^{-8}$: avoids division by zero.
- $\lambda$: weight decay, typically 0.1, **decoupled** from the gradient (the "W" in AdamW), so decay is not distorted by the adaptive scaling.

Standard companions: gradient clipping to global norm 1.0, and peak learning rates around $3\times10^{-4}$ for ~1B models, falling to around $10^{-4}$ or lower for 100B+ models. The memory cost matters for Section 05: Adam keeps two extra fp32 tensors ($m$ and $v$) the size of the model.

{{< fig "pretraining/12" >}}

### Learning-rate schedules

- **Warmup**: increase $\eta$ linearly from ~0 to peak over the first few thousand steps (Llama 3 405B: 8,000 steps). Early on, Adam's $v$ estimates are noisy and the weights are random, so large steps can blow up.
- **Cosine decay**: after warmup, $\eta_t = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})(1+\cos(\pi\,\tfrac{t - t_w}{T - t_w}))$, ending at $\eta_{\min} \approx 0.1\,\eta_{\max}$ or lower. The catch: you must know the total steps $T$ in advance, and a checkpoint halfway through is not a "finished" model.
- **WSD (warmup-stable-decay)**: warm up, hold $\eta$ constant for most of training, then decay quickly (often over the final 10-20%). Popularized by MiniCPM and analyzed by Hägele et al. It matches cosine at the end, and you can branch a decay from any stable-phase checkpoint, so one long run yields models at many token budgets, which makes scaling-law studies far cheaper. The loss drops sharply during the decay, which looks dramatic on a plot and is expected.

{{< fig "pretraining/13" >}}

### Batch size and the critical batch size

Batch size (in tokens per optimizer step) trades steps for parallelism. Up to a point, doubling the batch halves the number of steps needed with no extra total compute: the gradient estimate is just less noisy. Beyond the **critical batch size**, extra samples mostly average away noise that was already small, so you burn compute for little gain. McCandlish et al. estimate it with the gradient noise scale $B_{\text{noise}} = \mathrm{tr}(\Sigma)/|G|^2$ (per-example gradient variance divided by squared true gradient norm). The critical batch grows as loss falls, which is why runs often ramp batch size up: Llama 3 405B went from 4M to 8M to 16M tokens per batch. Typical large-model batches are 1M to 16M tokens.

### muP at a high level

With standard parameterization, the best learning rate depends on model width: tune on a 100M model, and the same LR will be too large for a 10B model. **Maximal Update Parameterization (muP)** rescales initialization variances and per-layer learning rates with width so that each layer's update size stays constant as you widen the model. The payoff: hyperparameters tuned on a small proxy model transfer to the large one ("muTransfer"). Cerebras-GPT, MiniCPM and several labs use it; others simply fit LR-vs-size scaling laws instead. You do not need to implement it to understand pretraining; know that it exists and what problem it solves.

### Loss spikes

A loss spike is a sudden jump in training loss, sometimes recovering in a few hundred steps, sometimes diverging. Common causes: attention or output logits growing too large, a bad data batch (long runs of repeated tokens, binary junk), too-high LR, or Adam's $v$ being stale for parameters that rarely receive gradient (for example rare token embeddings) and then getting a large one. Practical responses, roughly in the order teams reach for them: rewind to an earlier checkpoint and skip the offending batches (PaLM did this), lower the LR, add QK-norm or z-loss, tighten gradient clipping, lower $\beta_2$ or raise $\epsilon$. Monitoring the gradient norm and per-layer activation norms gives early warning.

```python
# LR schedules plus a compute-optimal split, stdlib only.
import math

def cosine_lr(step, total, warmup, peak, floor_ratio=0.1):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = peak * floor_ratio
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))

def wsd_lr(step, total, warmup, peak, decay_frac=0.2):
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return peak * step / warmup
    if step < decay_start:
        return peak
    # Linear decay to 0; why: WSD's gains come from this short final cooldown.
    return peak * (total - step) / (total - decay_start)

total, warmup, peak = 10_000, 500, 3e-4
for s in [0, 250, 500, 2_500, 5_000, 8_000, 9_000, 10_000]:
    print(f"step {s:6d}  cosine {cosine_lr(s, total, warmup, peak):.2e}  wsd {wsd_lr(s, total, warmup, peak):.2e}")

# Chinchilla parametric fit, Epoch AI's corrected constants (see text).
E, A, B, alpha, beta = 1.82, 482.01, 2085.43, 0.3478, 0.3658

def compute_optimal(C):
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, C / (6 * N)

for C in [1e21, 5.76e23, 3.8e25]:
    N, D = compute_optimal(C)
    L = E + A / N ** alpha + B / D ** beta
    print(f"C={C:.1e}: N={N:.2e} params, D={D:.2e} tokens, D/N={D/N:.0f}, predicted loss={L:.3f}")
```

### Common confusions

- "Chinchilla says always train at 20 tokens/param." It says that is the cheapest way to reach a given loss if training compute is all you pay for. With inference costs counted, smaller and longer is better.
- "Scaling laws predict benchmark scores." They predict loss. Mapping loss to task accuracy needs a second fit (Llama 3 did this, Section 08) and is noisier.
- "Kaplan was wrong." Its power-law form held up; its compute allocation was skewed by the fixed-schedule methodology.
- "A bigger batch is always faster." Faster per step, but past the critical batch size you pay more total compute per unit of loss reduction.
- "WSD is a different optimizer." It is only a learning-rate schedule; the optimizer is still AdamW.

### Check yourself

1. You have $10^{23}$ FLOPs. Roughly what model size and token count does the 20:1 rule suggest?
2. Why would a lab train an 8B model on 15T tokens when Chinchilla suggests about 160B tokens?
3. Why does LLM pretraining use $\beta_2 = 0.95$ instead of Adam's default 0.999?
4. What practical advantage does WSD give over cosine for scaling-law studies?
5. A loss spike appears at step 120k and the loss does not recover. List three things you could try.

<details>
<summary>Answers</summary>

1. $C = 6ND$ with $D = 20N$ gives $C = 120N^2$, so $N = \sqrt{10^{23}/120} \approx 2.9\times10^{10}$ (about 29B params) and $D \approx 580$B tokens.
2. Inference cost scales with $N$ and dominates over the model's lifetime; the 8B model keeps improving with more tokens, so extra training compute buys a cheaper-to-serve model at a given quality.
3. A shorter memory makes the second-moment estimate react quickly when gradient scales change, reducing the chance that a stale, too-small $v$ produces a huge update.
4. Decay can be branched off any stable-phase checkpoint, so one run produces fully-annealed models at many token budgets instead of one run per budget.
5. Rewind to a checkpoint before the spike and skip the data batches around it; lower the peak LR; add QK-norm or z-loss; tighten gradient clipping. Also inspect the batch contents.

</details>

### Resources

**Read**
- [Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) (start here). The Chinchilla paper; Figure 1 and Section 3 are the core.
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361). The original power-law study and the source of $6ND$.
- [Chinchilla Scaling: A replication attempt (Epoch AI, 2024)](https://arxiv.org/abs/2404.10102). Short, readable, and a lesson in checking fitted constants.
- [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (Hägele et al., 2024)](https://arxiv.org/abs/2405.18392). WSD schedules and why they make scaling studies cheaper.
- [Tensor Programs V: muTransfer (Yang et al., 2022)](https://arxiv.org/abs/2203.03466). The muP paper; read the intro and the hyperparameter transfer figures.

**Watch**
- [CS336 lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). The two scaling-laws lectures cover Kaplan, Chinchilla, muP and WSD with derivations.

---

## 04. Training Data Engineering

### Why data dominates

At a fixed architecture and compute budget, the single largest lever on model quality is the data. FineWeb-Edu, a filtered subset of the web, lets a 1.8B model match on knowledge benchmarks what unfiltered web data needs roughly 10x more tokens to reach. DCLM showed that the choice of a quality classifier alone moved MMLU by double-digit points at 7B scale. This section walks the pipeline that turns petabytes of crawl into a few trillion good tokens.

{{< fig "pretraining/14" >}}

### Common Crawl and text extraction

**Common Crawl** is a nonprofit that has crawled the web since 2008 and publishes snapshots every month or two, each a few billion pages. It comes in two forms: WARC (raw HTTP responses, full HTML) and WET (pre-extracted plain text). The WET extraction is crude and keeps menus, footers and boilerplate, so serious pipelines re-extract from WARC with a tool like **trafilatura** or resiliparse, which use heuristics on the DOM to keep the main content. FineWeb's ablations showed re-extraction from WARC clearly beats using WET files. Extraction is also where you lose or keep math (LaTeX, MathML) and code formatting, so specialized pipelines for those domains exist.

### Language identification

A fastText classifier (trained on 176 languages) labels each document with a language and a confidence. English-focused pipelines keep documents with English score above roughly 0.65. Multilingual pipelines route each language to its own filtering thresholds, because heuristics tuned on English (stopword ratios, punctuation) misbehave on other scripts.

### Heuristic quality filtering

Cheap rules remove obvious junk. The canonical sets come from Gopher ("MassiveText") and C4:

- Document length between roughly 50 and 100,000 words.
- Mean word length between 3 and 10 characters.
- Symbol-to-word ratio below 0.1 (too many `#` or `...` signals spam or code fragments).
- At least 80% of words contain an alphabetic character.
- Must contain some common stopwords ("the", "and", "of"), which indicates natural prose.
- Few lines ending in an ellipsis, few lines starting with bullets.
- Repetition filters: fraction of duplicated lines, paragraphs, and top n-grams below thresholds.
- C4 rules: drop lines without terminal punctuation, drop pages with "lorem ipsum" or curly braces, drop "javascript must be enabled" lines.
- URL blocklists for adult and spam domains.

FineWeb added its own filters chosen by ablation (for example, removing documents where most lines are very short or where a large fraction of characters sit in duplicated lines).

### Model-based quality filtering

Heuristics only catch junk. To find **good** documents, train a classifier:

- **Reference-based**: positives are text resembling a trusted source (Wikipedia, books, pages linked from Wikipedia), negatives are random crawl. GPT-3 and Llama used variants of this.
- **LLM-annotated**: ask a strong LLM to grade a sample of documents (FineWeb-Edu prompted Llama 3 70B for 0 to 5 "educational value" on 450K documents), train a small classifier on those labels, and run it on everything. FineWeb-Edu keeps score 3 and above, yielding 1.3T tokens from 15T.
- **Instruction-like positives**: DCLM's best filter was a fastText classifier with positives from OpenHermes 2.5 instruction data and r/ExplainLikeImFive answers. It beat more elaborate approaches.

Model filters are powerful and dangerous. They encode the biases of their positives: a filter trained on Wikipedia-like text may delete dialects, informal writing, or whole topics. Keep track of what you remove.

### Deduplication

The web is massively redundant: boilerplate, mirrors, scraped copies, templated pages. Duplicates waste compute, increase memorization (and thus privacy and copyright risk), and can hurt quality. Three granularities:

- **Exact dedup**: hash each document (or each line, or paragraph) and keep one copy per hash. Line-level dedup across a corpus removes repeated navigation menus and cookie banners.
- **Exact substring dedup**: suffix arrays to find long repeated spans (for example 50+ tokens) across documents. Used for C4 cleanup; expensive.
- **Fuzzy dedup with MinHash + LSH**: find near-duplicates (the same article with a different footer). Represent each document by its set of word n-grams ("shingles"). The **Jaccard similarity** of two sets is $J(A,B) = |A\cap B| / |A\cup B|$. MinHash approximates it: apply $k$ random hash functions to every shingle and keep the minimum per function. The probability that two documents share the same minimum for one hash function equals exactly $J(A,B)$, so the fraction of matching minima across $k$ functions estimates $J$.

{{< fig "pretraining/15" >}}

Comparing every pair is $O(n^2)$, impossible for billions of documents. **Locality-sensitive hashing** splits the $k$ minima into $b$ bands of $r$ rows; two documents become candidates if any band matches exactly. The probability of becoming candidates is $1 - (1 - J^r)^b$, an S-curve whose threshold sits near $(1/b)^{1/r}$. FineWeb used 112 hashes as 14 bands of 8, targeting about 75% similarity.

{{< fig "pretraining/16" >}}

A FineWeb surprise: deduplicating across all 96 crawl snapshots globally made results **worse** than deduplicating within each snapshot. Global dedup disproportionately kept low-quality documents that happened to be unique, while removing good content that had been legitimately recrawled. Dedup is not monotonically good.

### PII and toxicity

Pipelines mask emails, phone numbers and IP addresses with regexes (Dolma, FineWeb) and sometimes run named-entity models. Toxicity filtering uses classifiers or word lists (the "List of Dirty, Naughty, Obscene and Otherwise Bad Words" in C4). Aggressive toxicity filtering also removes useful content about health, identity and history, and can make the model worse at recognizing toxicity, so modern pipelines usually filter conservatively and leave behavior to post-training.

### Mixtures and domain weighting

After filtering you have pools: web, code, math, academic papers, books, Wikipedia, multilingual. The **mixture** is the sampling probability for each pool. Two key ideas:

- **Upsampling**: small high-quality sources (Wikipedia, textbooks) are repeated several epochs; huge low-quality ones are subsampled. Repeating data up to about 4 epochs costs little compared with fresh data (Muennighoff et al.); beyond that returns drop fast.
- **Choosing weights empirically**: train small proxy models on different mixtures and measure downstream loss, or fit mixture scaling laws. DoReMi and RegMix are published methods; most labs run grid ablations. Llama 3 ended at roughly 50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual.

{{< fig "pretraining/17" >}}

**Code and math** are disproportionately valuable: code improves reasoning and structured output even for non-coding tasks, and math data (OpenWebMath, FineMath, proof corpora) is needed for quantitative ability. Both need their own extraction (keeping indentation, LaTeX) and their own quality filters.

### Curriculum and annealing

Order of data mostly does not matter during the main phase (it is shuffled). The big exception is the **end** of training: upweighting the highest-quality data while the LR decays ("annealing" or "mid-training", Section 06) gives outsized gains on benchmarks. It also serves as a cheap test of a new dataset: anneal a partly trained model on a mix containing it and see if scores rise.

### Open datasets to know

| Dataset | Size | Notes |
|---|---|---|
| **RedPajama** v1 / v2 | 1.2T / ~30T dedup tokens | v1 reproduced the LLaMA 1 mix; v2 is raw CC with quality signals precomputed for you to filter. |
| **Dolma** (AI2) | 3T tokens | Web, code, papers, books, Wikipedia, social; fully documented toolkit, trains OLMo. |
| **FineWeb** (HF) | 15T tokens | 96 CC snapshots, extensive ablations behind every filter. |
| **FineWeb-Edu** | 1.3T (score 3+), 5.4T (score 2+) | Educational-quality classifier subset; big gains on MMLU and ARC. |
| **DCLM-Baseline** | ~4T tokens | Built by the DataComp-LM benchmark; 7B model on 2.6T tokens reached 64% MMLU. |

{{< fig "pretraining/18" >}}

### Data contamination

If benchmark questions (MMLU, GSM8K) appear in training data, scores measure memorization. Benchmarks leak via forums, study sites and GitHub. Decontamination checks n-gram overlap (GPT-3 used 13-grams; Llama 3 used 8-gram overlap) between training documents and test sets, then removes matches. Contamination is hard to rule out entirely; Section 07 revisits it from the evaluation side.

```python
# MinHash near-duplicate detection sketch, stdlib only.
# Why salted SHA1 instead of 128 independent hash families: same statistical behavior, no deps.
import hashlib, random, re

def shingles(text, n=5):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}

def minhash(sh, num_perm=128):
    sig = []
    for seed in range(num_perm):
        salt = seed.to_bytes(4, "little")
        sig.append(min(int.from_bytes(hashlib.sha1(salt + s.encode()).digest()[:8], "little") for s in sh))
    return sig

def lsh_buckets(sig, bands=16, rows=8):
    return {(b, tuple(sig[b * rows:(b + 1) * rows])) for b in range(bands)}

random.seed(0)
base = " ".join(random.choice(["data", "model", "token", "train", "scale", "loss", "web", "text"]) + str(i % 37) for i in range(400))
near_dup = base.replace("loss3 ", "gain3 ", 2) + " subscribe to our newsletter"
unrelated = " ".join(random.choice(["cat", "dog", "tree", "river"]) + str(i) for i in range(400))

docs = {"base": base, "near_dup": near_dup, "unrelated": unrelated}
sets = {k: shingles(v) for k, v in docs.items()}
sigs = {k: minhash(s) for k, s in sets.items()}

for a, b in [("base", "near_dup"), ("base", "unrelated")]:
    true_j = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
    est_j = sum(x == y for x, y in zip(sigs[a], sigs[b])) / len(sigs[a])
    candidate = bool(lsh_buckets(sigs[a]) & lsh_buckets(sigs[b]))
    print(f"{a} vs {b}: true J={true_j:.3f}  MinHash est={est_j:.3f}  LSH candidate={candidate}")
```

### Common confusions

- "More data is always better." More **good** data is. Adding low-quality tokens at a fixed budget displaces better ones.
- "Dedup everything as hard as possible." FineWeb found global cross-snapshot dedup hurt. Some repetition of good data is fine.
- "MinHash computes Jaccard similarity." It estimates it; the error shrinks as $1/\sqrt{k}$ in the number of hash functions.
- "A quality classifier measures quality." It measures similarity to its positive examples. Choose positives carefully.
- "Decontamination guarantees clean evals." Paraphrases, translations and solutions posted in different formats slip past n-gram checks.

### Check yourself

1. Why re-extract text from WARC instead of using Common Crawl's WET files?
2. With 20 bands of 5 rows, roughly where is the LSH similarity threshold?
3. Give one argument for and one against aggressive toxicity filtering in pretraining.
4. Why might deduplicating across all crawl snapshots hurt quality?
5. How can annealing be used to evaluate a new dataset cheaply?

<details>
<summary>Answers</summary>

1. WET extraction keeps navigation, footers and boilerplate and mangles structure; tools like trafilatura extract main content much more cleanly, and ablations show better downstream results.
2. $(1/20)^{1/5} \approx 0.55$ Jaccard.
3. For: fewer harmful generations and less offensive text memorized. Against: removes legitimate content about health, identity and history, can hurt the model's ability to recognize and handle toxic content, and is often biased against dialects.
4. Content that is recrawled across snapshots tends to be popular and good; global dedup keeps one copy of it but keeps every unique low-quality page, so the remaining distribution skews toward junk.
5. Take a partially trained checkpoint, run a short LR decay on a mix that includes the new data (versus a control mix), and compare benchmark scores. Much cheaper than a full pretraining ablation.

</details>

### Resources

**Read**
- [FineWeb blog: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) (start here). Every pipeline step with an ablation showing whether it helped; the most practical data document published.
- [The FineWeb Datasets (Penedo et al., 2024)](https://arxiv.org/abs/2406.17557). Paper version, including FineWeb-Edu.
- [Dolma (Soldaini et al., 2024)](https://arxiv.org/abs/2402.00159). An open corpus with every design decision documented, including PII and toxicity handling.
- [DataComp-LM (Li et al., 2024)](https://arxiv.org/abs/2406.11794). Benchmark for data curation; the fastText-on-instruction-data filter result.
- [RedPajama (Weber et al., 2024)](https://arxiv.org/abs/2411.12372). Reproducing the LLaMA mix and shipping quality signals rather than a fixed filter.

**Watch**
- [CS336 lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). The two data lectures cover sources, filtering and dedup with worked examples.

---

## 05. Training Infrastructure and Systems

### The memory budget

Why can't you just train a 70B model on one 80GB GPU? Count bytes. Standard mixed-precision training with Adam stores, per parameter:

| Item | Precision | Bytes/param |
|---|---|---|
| Model weights (working copy) | bf16 | 2 |
| Gradients | bf16 | 2 |
| Master weights | fp32 | 4 |
| Adam first moment $m$ | fp32 | 4 |
| Adam second moment $v$ | fp32 | 4 |
| **Total** | | **16** |

{{< fig "pretraining/19" >}}

So a 7B model needs 112GB before any activations, and 70B needs 1.12TB. Then come **activations**: the intermediate tensors saved during the forward pass for use in backward. For a standard transformer layer, Korthikanti et al. estimate

$$
\text{activation bytes per layer} \approx s\,b\,h\left(34 + 5\frac{a\,s}{h}\right)
$$

- $s$: sequence length, $b$: micro-batch size, $h$: hidden dimension, $a$: number of attention heads.
- The $34\,sbh$ term covers linear layer inputs, norms, dropout masks. The $5as^2b$ term is the attention score matrix, quadratic in sequence length.

With FlashAttention (which never materializes the score matrix), the second term essentially disappears, leaving about $34\,sbh$ bytes per layer (slightly less without dropout). At $s=8192$, $b=1$, $h=8192$, that is 2.3GB per layer, or 180GB for 80 layers. Activations, not parameters, often dominate.

### Mixed precision

Training in pure fp32 is slow; tensor cores are 10 to 20x faster in 16-bit formats. **bf16** (bfloat16) has fp32's 8-bit exponent with only 7 mantissa bits, so it represents the same range (no overflow, no loss scaling needed, unlike fp16) at lower precision. The recipe: matmuls in bf16, accumulate in fp32 inside the tensor core, keep master weights and optimizer states in fp32 because tiny updates (LR $\times$ gradient $\approx 10^{-7}$ relative) would round to zero in bf16. Softmax, norms and the loss are computed in fp32.

**FP8** (E4M3 for forward, E5M2 with more range for gradients) doubles matmul throughput again on H100s. It needs per-tensor or finer-grained (per-block) scaling factors to keep values in range. DeepSeek-V3 trained mostly in FP8 with block-wise scaling; Llama 3 trained in bf16. FP8 is now mainstream for frontier runs but still demands careful engineering.

### The parallelism zoo

When one GPU cannot hold the model or would take years, split work across GPUs. Each strategy splits a different axis.

- **Data parallelism (DP).** Every GPU holds a full model replica and processes a different slice of the batch; gradients are averaged with an all-reduce after backward. Simple and communication-light per step, but requires the whole 16 bytes/param on every GPU.
- **ZeRO / FSDP (sharded data parallelism).** Notice that in DP every GPU stores identical optimizer states, gradients and weights. ZeRO shards them across the $N_d$ DP ranks: stage 1 shards optimizer states, stage 2 also gradients, stage 3 also parameters. Per-GPU memory for model state becomes roughly $16\Psi/N_d$ for $\Psi$ parameters at stage 3. The cost is communication: each layer's parameters are all-gathered just before use and freed after; gradients are reduce-scattered. PyTorch's **FSDP** is the standard implementation of ZeRO-3. Communication volume is about 1.5x plain DP, often overlapped with compute.
- **Tensor parallelism (TP, Megatron-LM).** Split individual weight matrices across GPUs. In the MLP, the first matrix is split by columns and the second by rows, so each GPU computes a partial result and one all-reduce combines them; attention splits naturally by heads. Requires two all-reduces per layer in forward, so it needs very fast interconnect and is kept **within a node** (8 GPUs on NVLink).
- **Sequence parallelism (SP).** A companion to TP: the parts of the layer TP does not split (norms, dropout) are split along the sequence dimension, so their activations are not replicated across the TP group.
- **Pipeline parallelism (PP).** Split layers into stages on different GPUs (GPUs 0-7 hold layers 1-16, the next group 17-32, and so on). Micro-batches flow through the pipeline. Naive pipelining leaves GPUs idle while waiting (the **bubble**, fraction about $(p-1)/m$ for $p$ stages and $m$ micro-batches); schedules like 1F1B and interleaved stages shrink it. Communication is only point-to-point activations between neighbors, so PP works across nodes.
- **Context parallelism (CP).** For very long sequences (128K tokens), split the sequence itself across GPUs, including attention: each GPU holds a chunk of queries and the keys/values are passed around (ring attention) or all-gathered. Needed because activations grow with sequence length even with FlashAttention.
- **Expert parallelism** applies to mixture-of-experts models (placing different experts on different GPUs); covered later in the course.

Large runs combine them: **3D or 4D parallelism**. A typical layout is TP within a node, PP across a few nodes, CP for long sequences, and FSDP/DP across everything else. The global batch is micro-batch $\times$ gradient accumulation steps $\times$ DP degree.

{{< fig "pretraining/20" >}}

{{< fig "pretraining/21" >}}

{{< fig "pretraining/22" >}}

{{< fig "pretraining/23" >}}

### Activation checkpointing

Trade compute for memory: during forward, keep only each layer's input; during backward, recompute the layer's forward to regenerate the activations you need. Full recomputation costs one extra forward (~33% more compute, since forward is about a third of the $6N$). **Selective** recomputation (Korthikanti et al.) only recomputes cheap but memory-hungry parts like attention softmax, recovering most of the memory for a few percent overhead.

{{< fig "pretraining/24" >}}

### MFU: are the GPUs actually busy?

**Model FLOPs Utilization**:

$$
\mathrm{MFU} = \frac{\text{tokens per second} \times 6N}{\text{number of GPUs} \times \text{peak FLOP/s per GPU}}
$$

It counts only the FLOPs the model mathematically needs, not recomputation. Good large-scale runs reach 35 to 55% MFU on H100s (peak dense bf16 is about 989 TFLOP/s on H100 SXM). The missing half goes to communication not overlapped with compute, pipeline bubbles, memory-bound kernels (norms, elementwise ops) and stragglers. **HFU** (hardware FLOPs utilization) also counts recomputation, so it is always higher; quote MFU when comparing.

{{< fig "pretraining/25" >}}

### Checkpointing and fault tolerance

At 16,000 GPUs, something breaks every few hours: GPUs fail, HBM throws errors, NICs flap, nodes hang. Llama 3 saw 419 unexpected interruptions in 54 days. Strategies:

- Save **checkpoints** (weights, optimizer state, data loader position, RNG state) frequently. A 405B checkpoint with optimizer state is about 6TB, so saving must be sharded and asynchronous, written in parallel by all ranks to a distributed filesystem.
- Fast failure detection: watchdog timers on collectives, NCCL error tracing, health checks that remove bad nodes before restart.
- Hot spare nodes so a restart does not wait for repair.
- Deterministic data loading so a restart resumes exactly where it stopped, without repeating or skipping data.

The metric is **effective training time** (fraction of wall-clock time spent making progress); Llama 3 reported over 90%.

### Networking

Inside a node, GPUs talk over NVLink/NVSwitch at around 900GB/s per GPU (H100). Between nodes, InfiniBand or RoCE (RDMA over Ethernet) at 400Gb/s per GPU (50GB/s), so roughly 18x slower. This hierarchy dictates placement: bandwidth-hungry TP stays inside the node, PP and DP go across nodes. Collective algorithms (ring, tree) in NCCL, network topology (rail-optimized fat trees), and congestion control matter at scale. Even data center power is a systems issue: thousands of GPUs pausing and resuming together cause megawatt-scale swings in power draw.

```python
# Per-GPU memory and MFU calculator, stdlib only.
# Why ignore temporary buffers and fragmentation: this is a planning estimate, real usage runs ~10-20% higher.

def model_state_gb(params, zero_stage=0, dp=1):
    weights, grads, optim = 2 * params, 2 * params, 12 * params  # bf16 w, bf16 g, fp32 master + m + v
    if zero_stage >= 1: optim /= dp
    if zero_stage >= 2: grads /= dp
    if zero_stage >= 3: weights /= dp
    return (weights + grads + optim) / 1e9

def activation_gb(layers, seq, micro_batch, hidden, tp=1, checkpointing=False):
    per_layer = 34 * seq * micro_batch * hidden / tp  # FlashAttention: no s^2 term
    if checkpointing:
        per_layer = 2 * seq * micro_batch * hidden / tp  # only bf16 layer input kept, sharded by sequence parallelism
    return layers * per_layer / 1e9

def mfu(tokens_per_sec, params, gpus, peak=989e12):
    return tokens_per_sec * 6 * params / (gpus * peak)

P = 70e9
for stage in [0, 1, 2, 3]:
    print(f"70B ZeRO-{stage}, DP=64: model state {model_state_gb(P, stage, 64):7.1f} GB/GPU")
print(f"70B activations, 80 layers, s=8192, b=1, h=8192, TP=8: {activation_gb(80, 8192, 1, 8192, tp=8):.1f} GB/GPU")
print(f"  ...with full activation checkpointing: {activation_gb(80, 8192, 1, 8192, tp=8, checkpointing=True):.1f} GB/GPU")
print(f"MFU if 2048 H100s process 1.6M tokens/s on 70B: {mfu(1.6e6, P, 2048):.1%}")
```

### Common confusions

- "Mixed precision halves memory." The weights in bf16 are half, but master weights and Adam states stay fp32, so model state is 16 bytes/param versus 16 bytes/param in pure fp32 Adam (4+4+4+4). The win is speed, plus smaller activations.
- "FSDP is model parallelism." It shards storage, but each GPU still computes full layers on its own data. It is data parallelism with sharded state.
- "Tensor and pipeline parallelism are interchangeable." TP splits within a layer and needs NVLink-class bandwidth; PP splits across layers and tolerates slower links but introduces bubbles.
- "Activation checkpointing makes training 2x slower." Full recomputation adds about one forward pass, roughly a third of total compute; selective recomputation adds a few percent.
- "MFU 100% is the goal." Unreachable; 40-50% at scale is excellent.

### Check yourself

1. How many bytes per parameter does mixed-precision Adam need for model state, and what are they?
2. You have 64 GPUs with 80GB each and a 30B model. Which ZeRO stage makes model state fit, ignoring activations?
3. Why is tensor parallelism kept inside a node?
4. A run reports 55% HFU with full activation checkpointing. Roughly what is its MFU?
5. Why must a checkpoint include the data loader state?

<details>
<summary>Answers</summary>

1. 16: bf16 weights (2), bf16 grads (2), fp32 master weights (4), fp32 Adam $m$ (4) and $v$ (4).
2. Total model state is 480GB. ZeRO-1: $2+2$ bytes $\times$ 30B $= 120$GB plus $12 \times 30\text{B}/64 \approx 5.6$GB, does not fit. ZeRO-2: 60 + 0.9 + 5.6 is about 67GB, fits but tightly. ZeRO-3: 7.5GB, comfortable.
3. It does two all-reduces of activation-sized tensors per layer per pass, which only NVLink-class bandwidth can sustain without stalling compute.
4. Recompute adds about one forward ($2N$) to the $6N$, so hardware does $8N$ per token: MFU $\approx 55\% \times 6/8 \approx 41\%$.
5. On restart you must resume at the exact same position in the data stream, otherwise you repeat or skip data, which changes the effective mixture and can break reproducibility.

</details>

### Resources

**Read**
- [The Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (start here). Interactive, thorough walkthrough of memory, DP, ZeRO, TP, SP, CP, PP and how to combine them, with real benchmarks.
- [ZeRO (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054). The sharding stages and their memory math.
- [Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053) and [Efficient Large-Scale LM Training with Megatron-LM (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473). Tensor parallelism, then its combination with pipeline and data parallelism.
- [Reducing Activation Recomputation in Large Transformer Models (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198). Source of the activation memory formula, sequence parallelism and selective recomputation.
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135). Why attention memory need not be quadratic.

**Watch**
- [Karpathy, "Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU). The second hour covers bf16, TF32, torch.compile, FlashAttention and DDP on 8 GPUs, measured step by step.

---

## 06. Advanced Pretraining Objectives

Plain next-token prediction on shuffled documents gets you most of the way. Modern runs add objective tweaks and training stages to buy specific capabilities.

### Fill-in-the-middle (FIM)

A left-to-right model cannot naturally fill a gap given both what comes before and after it, yet that is exactly what code completion inside an editor needs: the cursor sits between existing code. Bavarian et al. (2022) showed a simple data transform teaches it with almost no cost. Take a document, split it at two random points into prefix, middle, suffix, and rearrange:

```
PSM:  <PRE> prefix <SUF> suffix <MID> middle <EOT>
SPM:  <PRE> <SUF> suffix <MID> prefix middle <EOT>
```

The model is still trained with ordinary next-token loss on this rearranged sequence; it just learns that after `<MID>` comes the text that connects prefix to suffix. At inference you supply prefix and suffix and let it generate after `<MID>`. Applying FIM to about 50% of documents (rate 0.5) gave full infilling ability while leaving left-to-right performance unchanged, which the authors called "FIM-for-free". Splitting at the character level and at random positions inside the document works better than restricting to line boundaries. Code models (StarCoder, Code Llama, DeepSeek-Coder, Qwen-Coder) all use it.

{{< fig "pretraining/26" >}}

### Multi-token prediction (MTP)

Standard training predicts only token $t+1$ from position $t$. Gloeckle et al. (2024) attach $n$ output heads to a shared trunk, head $k$ predicting token $t+k$:

$$
\mathcal{L}_{\text{MTP}} = -\sum_{t}\sum_{k=1}^{n} \log p_\theta^{(k)}(x_{t+k} \mid x_{\le t})
$$

- $n$: number of future tokens predicted (they found 4 works well for code).
- $p_\theta^{(k)}$: the distribution from head $k$, sharing the transformer trunk.

Why would this help? Forcing the trunk's hidden state to support several future tokens encourages planning-like representations and gives more signal per position. Their 13B model solved 12% more HumanEval problems and 17% more MBPP problems than next-token training, with benefits growing with model size (it hurt small models). A bonus: the extra heads can draft tokens for **self-speculative decoding**, giving up to 3x faster inference. DeepSeek-V3 uses a sequential MTP variant (one extra token, predicted by a small module that also sees the next token's embedding) as an auxiliary training loss and for speculative decoding. The memory trick: compute each head's loss and backward sequentially so you never hold $n$ vocab-sized logit tensors at once.

{{< fig "pretraining/27" >}}

### Prefix LM

A prefix LM uses a decoder-only transformer but makes attention **bidirectional over a prefix** (say, the prompt) and causal over the rest. Loss is computed only on the causal part. It combines encoder-style understanding of the input with decoder-style generation in a single stack. It was used in T5 comparisons, UL2 and some multimodal models (PaliGemma). It is rarely used for pure-text frontier models because it complicates training (two masking regimes) and gives no clear gain at scale.

### UL2: mixture of denoisers

Tay et al. (2022) argued that different objectives teach different skills and proposed training on a mixture, flagged by a mode token so the model knows which task it is doing:

- **R-denoiser** (`[R]`, regular): T5-style span corruption, short spans (~3 tokens), ~15% corruption. Teaches local knowledge.
- **X-denoiser** (`[X]`, extreme): long spans (32+ tokens) or high corruption (~50%). Forces generating long text from little context.
- **S-denoiser** (`[S]`, sequential): prefix LM, predict the continuation of a prefix. Closest to standard LM.

UL2 20B beat comparable T5 and GPT-style models across a broad suite, and "UL2R" (continuing to train PaLM briefly with the UL2 mixture) improved PaLM's few-shot scores at small extra cost. Today the main legacy is conceptual: objectives are composable, and most frontier models stayed with causal LM plus FIM.

{{< fig "pretraining/28" >}}

### Long-context continued pretraining

Training at 128K context from the start would be wasteful: attention cost grows with length, and most documents are short. So models pretrain at 4K to 8K tokens and then run a **context extension** stage:

1. Increase RoPE's base frequency $\theta$ (for example from 10,000 to 500,000 or more) so rotations at long distances do not alias, or use interpolation schemes (Position Interpolation, NTK-aware scaling, YaRN) that rescale positions.
2. Continue pretraining on a mix that upweights genuinely long documents (books, long code repositories, concatenated related documents), gradually increasing sequence length.
3. Keep short-context data in the mix so short-task performance does not regress.

Typical cost is a few percent of total pretraining tokens (Llama 3: about 800B tokens across six length stages from 8K to 128K). Evaluation uses needle-in-a-haystack retrieval plus harder tests like RULER, since perplexity on long text can look fine even when the model ignores distant context.

### Mid-training and annealing

The final phase of pretraining, while the learning rate decays to near zero, is when the model is most sensitive to what it sees. **Annealing** (also called mid-training, or the decay phase in WSD) exploits this: switch to a mix heavy in the best data you have (curated math, code, textbook-like and instruction-like text, sometimes synthetic data) for the last few percent of tokens. OLMo 2 dedicated a mid-training stage ("Dolmino" mix) to this; Llama 3 annealed on the final 40M tokens. The gains on benchmarks like GSM8K can be large for small models and shrink for very large ones. Two warnings: annealing on benchmark-like data blurs the line with contamination, and the improvements partly reflect formatting familiarity rather than new knowledge.

{{< fig "pretraining/29" >}}

```python
# Fill-in-the-middle transform at character level, stdlib only.
# Why character-level split points: the paper found random character splits beat line-level ones for infilling quality.
import random

PRE, SUF, MID, EOT = "<PRE>", "<SUF>", "<MID>", "<EOT>"

def fim_transform(doc, rng, fim_rate=0.5, spm_rate=0.5):
    if rng.random() >= fim_rate:
        return doc + EOT  # Why keep plain docs: preserves normal left-to-right ability ("FIM-for-free").
    a, b = sorted(rng.sample(range(len(doc) + 1), 2))
    prefix, middle, suffix = doc[:a], doc[a:b], doc[b:]
    if rng.random() < spm_rate:
        return f"{PRE}{SUF}{suffix}{MID}{prefix}{middle}{EOT}"
    return f"{PRE}{prefix}{SUF}{suffix}{MID}{middle}{EOT}"

rng = random.Random(3)
doc = "def area(r):\n    return 3.14159 * r * r\n"
for _ in range(4):
    print(repr(fim_transform(doc, rng)))
```

### Common confusions

- "FIM needs a new architecture or loss." No. It is a data transform plus three special tokens; the loss is still next-token cross-entropy.
- "Multi-token prediction means generating several tokens at once at inference." Training uses multiple heads; inference can still be one token at a time, with the extra heads optionally used for speculative drafts that are then verified.
- "Long context is just a bigger max_position." Without changing RoPE scaling and training on long data, quality degrades sharply past the trained length.
- "Annealing is fine-tuning." It is still pretraining-style next-token loss on documents, just with a shifted mix and decaying LR. Post-training uses chat formats and different objectives.
- "UL2 replaced causal LM." It influenced thinking and some models, but most frontier LLMs use causal LM with FIM.

### Check yourself

1. Write the PSM form of the document "abcdef" split into prefix "ab", middle "cd", suffix "ef".
2. Why does FIM at rate 0.5 not hurt left-to-right performance much?
3. What extra memory problem does multi-token prediction create, and how is it avoided?
4. Why not just pretrain at 128K context from the start?
5. What is a risk of putting benchmark-style math data in the annealing mix?

<details>
<summary>Answers</summary>

1. `<PRE>ab<SUF>ef<MID>cd<EOT>`.
2. Half the documents are untouched, and even in transformed documents the model still does left-to-right prediction on all tokens; the paper measured no significant loss in standard perplexity or benchmarks.
3. $n$ heads produce $n$ vocab-sized logit tensors per position. Compute and backpropagate each head's loss in sequence, freeing logits before the next head.
4. Attention compute and activation memory grow with length, most training documents are short, and the long-range skill can be learned cheaply at the end with a few hundred billion tokens.
5. It can inflate benchmark scores through format familiarity or near-duplicates of test items, which looks like capability but is closer to contamination.

</details>

### Resources

**Read**
- [Efficient Training of Language Models to Fill in the Middle (Bavarian et al., 2022)](https://arxiv.org/abs/2207.14255) (start here). Short, clear, and full of practical ablations on FIM rate and split strategy.
- [Better & Faster Large Language Models via Multi-token Prediction (Gloeckle et al., 2024)](https://arxiv.org/abs/2404.19737). The MTP architecture, memory trick and code results.
- [UL2: Unifying Language Learning Paradigms (Tay et al., 2022)](https://arxiv.org/abs/2205.05131). Mixture of denoisers and mode switching.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). Section on MTP as used in a frontier model, plus FP8 training.
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656). A documented mid-training stage with ablations on what data to anneal on.

**Watch**
- [CS336 lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). Later lectures touch on long-context training and mid-training in recent open models.

---

## 07. Evaluation During Pretraining

### Why evaluate during training at all

A run takes weeks or months. You need to know early whether it is healthy, whether a data change helped, and which checkpoint to keep. Evaluation during pretraining has different goals than evaluating a finished chat model: you want **cheap, low-noise signals that move smoothly** as the model improves, so you can compare runs and spot regressions.

### Validation loss and perplexity

The first signal is loss on held-out data, ideally several held-out sets (web, code, math, Wikipedia, each language) so you can see which capabilities are moving. Validation loss is smooth, cheap and directly tied to the training objective. Its weaknesses: it is tokenizer-dependent, it says little about whether the model can follow a task format, and improvement on held-out web text can coincide with stagnation on reasoning. Use **bits per byte** ($\text{loss in nats} \times \text{tokens} / (\text{bytes} \times \ln 2)$) to compare across tokenizers.

### Downstream benchmarks

The standard pretraining suite, with what each measures:

- **MMLU**: 57 subjects, 14K four-way multiple-choice questions from high school to professional level. Knowledge breadth.
- **HellaSwag**: choose the most plausible continuation of a short scenario among four; adversarially filtered. Commonsense; saturates near 90%+ for strong models but very useful early.
- **ARC** (Easy and Challenge): grade-school science multiple choice.
- **GSM8K**: 8.5K grade-school math word problems requiring multi-step arithmetic; scored by exact match on the final number, usually with chain-of-thought.
- **HumanEval**: 164 Python function-writing problems scored by running unit tests; reported as pass@1 (or pass@k).

Others you will see: PIQA, WinoGrande, BoolQ, MATH, MBPP, BBH, DROP, TriviaQA, and newer harder sets like MMLU-Pro and GPQA.

{{< fig "pretraining/30" >}}

### How a base model is scored

Base models are evaluated **few-shot**: the prompt contains $k$ solved examples (MMLU: 5-shot; GSM8K: 8-shot) followed by the test question. Two scoring styles:

{{< fig "pretraining/31" >}}

- **Log-likelihood (ranked classification).** For each answer option, compute the log-probability the model assigns to the option's text given the prompt. Pick the highest. Variants normalize by length (per-token or per-byte) because long options accumulate more negative log-probability. No generation needed, very cheap and stable.
  - **Cloze formulation (CF)**: score the answer **text** ("Paris"). Works for small models.
  - **Multiple-choice formulation (MCF)**: show options labeled A/B/C/D and score the **letter**. Small models score at chance in MCF because they have not learned the format; large models do better in MCF. Switching formulations mid-project creates fake jumps.
- **Generation.** Sample (usually greedily) and parse an answer: GSM8K final number, HumanEval code executed against tests. More realistic, noisier, more expensive.

{{< fig "pretraining/32" >}}

The prompt format, number of shots, normalization and answer parsing all change scores by several points. This is why comparing numbers across papers is risky unless they used the same harness and settings.

### lm-evaluation-harness

EleutherAI's **lm-evaluation-harness** is the de facto open tool: hundreds of tasks defined in YAML, consistent few-shot construction and scoring, support for HuggingFace models and inference servers. It powered the original Open LLM Leaderboard. Running it looks like:

```
lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag,arc_easy --num_fewshot 0 --batch_size 16
```

Other frameworks include HuggingFace's lighteval (used for FineWeb ablations) and OLMES from AI2, which standardizes formats to reduce ambiguity.

### Noise and early-signal benchmarks

Benchmarks have sampling noise: with $n$ questions and accuracy $p$, the standard error is $\sqrt{p(1-p)/n}$. For ARC-Challenge (~1,170 questions) at 50%, that is about 1.5 points, so a 1-point difference between two runs is noise. Checkpoint-to-checkpoint variation in the same run can be similar. Practices:

{{< fig "pretraining/33" >}}

- For small-scale ablations, choose benchmarks that rise above chance early and monotonically. FineWeb picked tasks (CommonsenseQA, HellaSwag, OpenBookQA, PIQA, SIQA, WinoGrande, ARC, MMLU in cloze form) by checking that they show low variance across seeds and a clear signal at 1-2B scale.
- Average across several benchmarks.
- Run multiple seeds for important comparisons.
- Prefer smooth metrics (log-likelihood of the correct answer) over hard accuracy when the model is small.

### Contamination from the evaluation side

If test items were in the training data, scores are inflated. Detection approaches: n-gram overlap between test items and the corpus, comparing performance on items flagged as contaminated vs clean (Llama 3 reports this per benchmark), checking whether the model can complete a test question verbatim from its first half, and evaluating on fresh or private variants (GSM1K was built to mirror GSM8K and exposed some overfit models). Treat suspiciously large gains on one benchmark as a contamination question first.

### The emergent abilities debate

Wei et al. (2022) documented "emergent abilities": tasks where performance sits at chance for small models and then jumps sharply above some scale (multi-digit arithmetic, some BIG-Bench tasks). This suggested capabilities cannot be predicted from small runs. Schaeffer et al. (2023) countered that many emergences are **artifacts of the metric**: exact-match accuracy on a multi-token answer is all-or-nothing, so steadily improving per-token probabilities show up as a sudden jump. With continuous metrics (token edit distance, log-likelihood of the correct answer), the same capabilities improve smoothly. The current consensus is in between: per-token loss is smooth and predictable, task-level metrics can still show sharp transitions, and predicting them requires modeling the loss-to-accuracy mapping, which is exactly the two-step approach Llama 3 used.

{{< fig "pretraining/34" >}}

```python
# Log-likelihood multiple-choice scoring, the core of lm-evaluation-harness, stdlib only.
# Why a toy character bigram model: it shows the scoring logic without downloading weights; swap in a real LM's logprob.
import math
from collections import Counter

corpus = "the cat sat on the mat. the dog sat on the log. paris is the capital of france. " * 50
pairs = Counter(zip(corpus, corpus[1:]))
firsts = Counter(corpus[:-1])
vocab = sorted(set(corpus))

def logprob(context, continuation):
    # Why add-one smoothing: unseen pairs must not give log(0).
    lp, prev = 0.0, context[-1]
    for ch in continuation:
        lp += math.log((pairs[(prev, ch)] + 1) / (firsts[prev] + len(vocab)))
        prev = ch
    return lp

# Why report both: normalization can flip the pick, which is why harnesses report acc and acc_norm separately.
def score(question, options):
    raw = [logprob(question, o) for o in options]
    per_char = [lp / len(o) for lp, o in zip(raw, options)]
    return raw, per_char

q = "the capital of france is"
opts = [" paris.", " the dog sat on the log.", " xqz."]
raw, norm = score(q, opts)
for o, r, n in zip(opts, raw, norm):
    print(f"{o!r:30s} sum logprob={r:8.2f}  per-char={n:6.2f}")
print("pick (raw):", opts[max(range(3), key=raw.__getitem__)], "| pick (normalized):", opts[max(range(3), key=norm.__getitem__)])

# Why standard error matters: differences below ~2 SE are not evidence of improvement.
for n, p in [(1172, 0.5), (14042, 0.6), (164, 0.3)]:
    print(f"n={n:5d} acc={p:.2f}  standard error = {100 * math.sqrt(p * (1 - p) / n):.1f} points")
```

### Common confusions

- "Lower validation loss means better benchmark scores." Usually correlated, but loss can improve on web text while reasoning benchmarks stall, or a mix change can trade one domain for another.
- "A 1-point gain on a benchmark is a real improvement." Often within noise; check the standard error and seeds.
- "Few-shot scores reflect what the chat model will do." Base model few-shot scores measure latent capability; chat models are usually evaluated zero-shot with different prompts.
- "Emergence was debunked." The metric artifact explains many cases, but sharp task-level transitions still exist and matter for planning.
- "MMLU at 25% means the model knows nothing." For small models in MCF format it may mean the model does not understand the A/B/C/D format; try cloze.

### Check yourself

1. Why do evaluation harnesses often normalize log-likelihood by answer length?
2. A 1B model scores 25.3% on MMLU in MCF format. What would you try before concluding it lacks knowledge?
3. Two runs differ by 1.2 points on ARC-Challenge. Is that meaningful?
4. Explain the Schaeffer et al. argument in one or two sentences.
5. Name two ways to detect benchmark contamination.

<details>
<summary>Answers</summary>

1. Each additional token adds a negative log-probability, so raw sums favor short answers regardless of correctness.
2. Evaluate in cloze format (score answer text, not letters), and look at the log-probability of the correct answer as a continuous metric.
3. Probably not: standard error is about 1.5 points at 50% accuracy with ~1,170 questions; you would need multiple seeds or a larger benchmark set.
4. Discontinuous metrics like exact match turn smooth improvements in per-token probability into apparent sudden jumps; with continuous metrics the improvements look smooth and predictable.
5. N-gram overlap between test items and training data; comparing scores on flagged vs clean subsets; verbatim completion tests; evaluating on fresh mirrored benchmarks.

</details>

### Resources

**Read**
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (start here). Install it and run HellaSwag on GPT-2; read a task YAML to see exactly how prompts and scoring are defined.
- [Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer et al., 2023)](https://arxiv.org/abs/2304.15004) and [Emergent Abilities of Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2206.07682). Both sides of the debate.
- [MMLU (Hendrycks et al., 2020)](https://arxiv.org/abs/2009.03300), [GSM8K (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168), [HumanEval (Chen et al., 2021)](https://arxiv.org/abs/2107.03374). Read how each benchmark is built and scored.
- [FineWeb blog](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). Its ablation setup section explains how they chose early-signal benchmarks.

**Watch**
- [Karpathy, "Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU). The final hour adds HellaSwag evaluation to the training loop and compares against GPT-2 and GPT-3.

---

## 08. Case Study: LLaMA 3

"The Llama 3 Herd of Models" (Meta, July 2024) is a 92-page report covering the 8B, 70B and 405B models. It is the most detailed public description of a frontier pretraining run and ties together every section of this chapter. Read it alongside this walkthrough; section numbers below refer to the paper.

{{< fig "pretraining/35" >}}

### Architecture: deliberately boring

Llama 3 is a standard dense decoder-only transformer, very close to Llama 2. The report says explicitly that they avoided mixture-of-experts to maximize training stability, and spent their effort on data, scale and infrastructure instead. Choices you already know from Chapter 1:

- Pre-norm with RMSNorm, SwiGLU MLP, RoPE, no biases.
- **GQA with 8 key-value heads** for all sizes, to shrink the KV cache at inference.
- **RoPE base frequency 500,000** (up from 10,000), which made later extension to 128K context easier.
- **128K vocabulary** (128,256 tokens): 100K from tiktoken plus 28K extra tokens for non-English languages. Compression on English improved from 3.17 to 3.94 characters per token versus Llama 2, so the same compute sees more text.
- An attention mask that **blocks attention across document boundaries** within a packed sequence. Little effect at 8K, important for long-context training.

| | 8B | 70B | 405B |
|---|---|---|---|
| Layers | 32 | 80 | 126 |
| Model dim | 4,096 | 8,192 | 16,384 |
| FFN dim | 14,336 | 28,672 | 53,248 |
| Attention heads | 32 | 64 | 128 |
| KV heads | 8 | 8 | 8 |
| Peak LR | 3e-4 | 1.5e-4 | 8e-5 |

### Data: about 15 trillion tokens

The pretraining corpus has a knowledge cutoff at the end of 2023 and about 15T multilingual tokens (versus 1.8T for Llama 2). The pipeline (Section 3.1) matches Section 04 of this chapter:

- Custom HTML parser that keeps math and code structure; they found markdown markers harmful and removed them.
- Domain and PII filters, adult-content filtering.
- **Three levels of dedup**: URL-level, document-level MinHash, and line-level (lines appearing more than 6 times in each bucket of 30M documents are removed, which cleans boilerplate but also removes some good repeated text; they accepted the tradeoff after ablations).
- Heuristics: duplicated n-gram coverage ratio for repetitive content, dirty-word counts, and token-distribution KL divergence against the corpus to catch outliers.
- **Model-based quality**: fastText classifiers trained to recognize text that resembles Wikipedia references, plus RoBERTa-based classifiers trained on Llama 2's quality judgments.
- Separate pipelines for code and math-heavy reasoning pages, and fastText language ID for 176 languages with per-language dedup and filtering.

**Choosing the mix**: they classified web data by topic to downsample overrepresented categories (arts and entertainment), then ran scaling-law experiments on small models with candidate mixes to predict large-model performance. Final mix: about **50% general knowledge, 25% math and reasoning, 17% code, 8% multilingual**.

**Annealing as data evaluation**: to judge a small, specialized dataset, they took a 50%-trained 8B model, annealed the LR to zero over 40B tokens with 30% weight on the new data, and measured the benchmark change. Much cheaper than a full ablation.

### Using scaling laws to pick model size and predict scores

Two problems with classic scaling laws (Section 3.2.1): next-token loss is not what you care about, and fits on small compute budgets are noisy. Meta's approach:

1. Run IsoFLOP experiments from $6\times10^{18}$ to $10^{22}$ FLOPs (models from 40M to 16B params), fit the compute-optimal frontier. Extrapolated to their $3.8\times10^{25}$ FLOP budget, it predicted about **402B parameters on 16.55T tokens**. They chose 405B on about 15.6T, essentially compute-optimal for the flagship. (The 8B and 70B models, in contrast, are heavily over-trained for inference efficiency.)
2. **Two-step prediction of benchmark accuracy**: first relate compute to the normalized negative log-likelihood of the correct answer on a benchmark (using small models), then relate that NLL to accuracy with a sigmoid fit (using small models plus older Llama 2 models). This predicted the 405B model's ARC-Challenge score closely, before training it. This is the pragmatic answer to the emergence debate in Section 07.

{{< fig "pretraining/36" >}}

### Training recipe

- AdamW, peak LR $8\times10^{-5}$, **linear warmup over 8,000 steps**, **cosine decay to $8\times10^{-7}$** over 1.2M steps.
- **Batch size ramp**: start at 4M tokens with 4,096-token sequences, move to 8M tokens with 8,192-token sequences after 252M tokens, then 16M tokens after 2.87T tokens. They report few loss spikes and no interventions needed for divergence.
- Mix adjustments during training: more non-English and math data, more recent web data in later stages, downsampling subsets judged lower quality.

### 405B infrastructure and 4D parallelism

Hardware (Section 3.3): up to **16,384 H100 GPUs** (80GB HBM3, 700W), 8 per server on NVLink, connected with a RoCE fabric at 400Gb/s per GPU (they also had an InfiniBand cluster). Storage: a distributed filesystem with 240PB capacity and 2 to 7 TB/s sustained throughput, needed because checkpoints are huge and saved often.

**4D parallelism**, from innermost to outermost: tensor (TP), context (CP), pipeline (PP), data (DP, implemented as FSDP that shards optimizer state and gradients, keeping parameters unsharded after the forward to avoid an extra all-gather). The ordering follows bandwidth: TP needs NVLink so it stays inside a server; DP tolerates the most latency so it is outermost.

{{< fig "pretraining/37" >}}

| GPUs | TP | CP | PP | DP | Seq len | Batch/DP | Tokens/batch | TFLOPs/GPU | MFU |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38% |

Notable engineering details: a modified pipeline schedule that allows a flexible number of micro-batches and balances the first and last stages (which also hold the embedding and output layers, so they got one fewer transformer layer); all-gather-based context parallelism rather than ring attention, because it works with the document mask and the KV is small thanks to GQA; and their own NCCLX collective library tuned for their network.

### Failures and interruptions

Section 3.3.4 is required reading for anyone who thinks training is just `model.fit()`. During a **54-day snapshot** of pretraining there were **466 job interruptions**: 47 planned (firmware upgrades, config changes) and **419 unexpected**. About 78% of the unexpected ones were attributed to confirmed or suspected hardware issues. The largest category was **faulty GPUs (30.1%)**, followed by **GPU HBM3 memory (17.2%)**; software bugs, network switches and cables, and host maintenance made up much of the rest. Despite roughly one unexpected failure every three hours, they achieved **over 90% effective training time**, and only three interruptions needed significant manual intervention. Tools: PyTorch NCCL flight recorder for fast diagnosis of hangs, automatic straggler detection, reduced job startup and checkpoint times.

{{< fig "pretraining/38" >}}

{{< fig "pretraining/39" >}}

Two more operational surprises: throughput varied **1 to 2% with time of day** because of datacenter temperature affecting GPU clocks, and synchronized pauses (checkpointing, collective waits) across tens of thousands of GPUs caused **power swings of tens of megawatts**, a real stress on the power grid.

### Long-context stage

The model pretrains at 8K. Then (Section 3.4.2) context is extended **in six stages from 8K to 128K**, using about **800B tokens** in total. Each increase happens only after checking that short-context benchmark performance has fully recovered and that needle-in-a-haystack retrieval succeeds at the new length. The 128K configuration uses CP=16 (third row of the table above).

{{< fig "pretraining/40" >}}

### Annealing

In the final stage (Section 3.4.3), during the last **40M tokens**, the learning rate is linearly annealed to 0 while keeping 128K context, and high-quality sources are upsampled. The final model is a **Polyak average** of checkpoints from the annealing phase. Annealing on curated math and code data improved the 8B model's GSM8K by 24.0 points and MATH by 6.4 points, but gains on the 405B model were negligible, suggesting the big model already had strong in-context abilities without needing the data nudged into it. To stay honest, they excluded training sets of commonly used benchmarks from the annealing data.

{{< fig "pretraining/41" >}}

### What to take away

- The best-documented frontier run uses no exotic architecture. Gains came from 8x more and better data, compute-optimal sizing from scaling laws, and engineering for reliability.
- Scaling laws were used for three distinct decisions: model size, data mix, and predicting benchmark scores.
- Every stage of this chapter shows up: objective and stability (Section 02), scaling and schedules (03), data (04), systems (05), long-context and annealing (06), evaluation and contamination analysis (07).

```python
# Sanity-check Llama 3 405B's reported numbers with this chapter's formulas, stdlib only.
N, D = 405e9, 15.6e12
C = 6 * N * D
print(f"6ND = {C:.2e} FLOPs (paper: 3.8e25)")

peak, gpus, mfu = 989e12, 16384, 0.40
days = C / (gpus * peak * mfu) / 86400
print(f"At {mfu:.0%} MFU on {gpus} H100s: {days:.0f} days of pure compute, {gpus * days * 24 / 1e6:.1f}M GPU-hours")

print(f"Model + optimizer state: {16 * N / 1e12:.1f} TB, which is why a single checkpoint needs a parallel filesystem")

interruptions, days_snapshot = 419, 54
print(f"Unexpected interruptions: one every {days_snapshot * 24 / interruptions:.1f} hours")
print(f"Tokens per parameter: {D / N:.0f} (Chinchilla ~20)")
```

### Common confusions

- "Llama 3 405B is over-trained like the 8B." The 405B at about 39 tokens/param is close to compute-optimal per their own scaling law; the 8B (about 1,900 tokens/param) is the over-trained one.
- "Hardware failures mean bad engineering." At 16K GPUs, failures every few hours are statistical certainty; the engineering is in recovery speed.
- "The 128K context came from training at 128K all along." Pretraining was at 8K; 128K came from a staged extension using about 5% of total tokens.
- "Annealing always helps big models." Their data shows big gains at 8B and negligible ones at 405B.
- "4D parallelism means four times as many GPUs." It means four independent axes along which the work is split; their product equals the GPU count.

### Check yourself

1. Why did Meta choose a dense model over mixture-of-experts for Llama 3?
2. Verify that TP=8, CP=1, PP=16, DP=128 uses 16,384 GPUs. Why is TP the innermost dimension?
3. How did Meta predict benchmark accuracy before training the 405B model?
4. Compute the average time between unexpected interruptions in the 54-day snapshot.
5. What two conditions were checked before each context-length increase?

<details>
<summary>Answers</summary>

1. Training stability and simplicity; they preferred to put risk budget into scale and data rather than architecture.
2. $8 \times 1 \times 16 \times 128 = 16{,}384$. TP needs the highest bandwidth (all-reduces every layer), which only NVLink inside a server provides, so it is placed on the 8 GPUs of one server.
3. Two steps: fit compute to normalized NLL of the correct answer on the benchmark, then fit NLL to accuracy with a sigmoid using small models and older models.
4. $54 \times 24 / 419 \approx 3.1$ hours.
5. Short-context benchmark performance had fully recovered, and needle-in-a-haystack retrieval worked at the new length.

</details>

### Resources

**Read**
- [The Llama 3 Herd of Models (Meta, 2024)](https://arxiv.org/abs/2407.21783) (start here). Read Sections 3.1 to 3.4 for pretraining; Section 3.3.4 for reliability.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook). Use it as a companion when the paper's parallelism section gets dense.
- [Beyond Chinchilla-Optimal (Sardana et al., 2023)](https://arxiv.org/abs/2401.00448). Why the 8B and 70B are trained far past 20 tokens/param.

**Watch**
- [CS336 lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). The architecture, scaling and parallelism lectures repeatedly use Llama 3 as a reference point.

---

## Chapter project: pretrain a small GPT end to end

Goal: experience every stage of this chapter at a scale that fits a single GPU (or a few dollars of cloud time).

1. **Data (Section 04).** Download a 1B-token sample of FineWeb-Edu (`sample-10BT` subset, take 10%) from the Hugging Face Hub. Write a filter script with three Gopher-style heuristics and your MinHash dedup from Section 04 (run it on a 100K-document subset). Report how many documents each step removed and eyeball 10 removed documents per step.
2. **Tokenize** with the GPT-2 tokenizer (`tiktoken`) and write packed `uint16` shards, like Karpathy's `build-nanogpt` or `llm.c` data scripts.
3. **Model and objective (Section 02).** Start from your Chapter 1 GPT-2 (124M). Add: QK-norm, z-loss ($10^{-4}$), no weight decay on norm gains and biases. Log loss, z-loss and gradient norm every step.
4. **Optimization (Section 03).** Train with AdamW ($\beta = (0.9, 0.95)$, wd 0.1, clip 1.0) using a WSD schedule. From one run, branch two decays (at 50% and 100% of the stable phase) to get models at two token budgets.
5. **Systems (Section 05).** Use bf16 autocast, FlashAttention (`F.scaled_dot_product_attention`), and `torch.compile`. Compute and log MFU. If you have multiple GPUs, use DDP or FSDP. Record tokens/s before and after each optimization.
6. **Evaluation (Section 07).** Every N steps compute validation loss and HellaSwag accuracy (via your own log-likelihood scorer or lm-evaluation-harness). Plot both against tokens.
7. **Mini scaling law (Section 03).** Train 4 smaller models (for example 10M, 25M, 50M, 85M non-embedding params) at roughly 20 tokens/param, fit $L(C) = a\,C^{-b} + E$, and predict your 124M model's loss before you train it. How close were you?
8. **Write-up.** A one-page report: data removed and why, loss curves, MFU, HellaSwag versus GPT-2's published score, scaling-law prediction vs actual, and one thing that surprised you.

Stretch goals: apply FIM at rate 0.5 to a code subset and test infilling; add a 4-head multi-token prediction loss and compare.

## Consolidated resources

**Courses and guides**
- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) and its [lecture playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_). The closest thing to a textbook for this chapter.
- [The Ultra-Scale Playbook (Hugging Face)](https://huggingface.co/spaces/nanotron/ultrascale-playbook). Distributed training, from one GPU to thousands.
- [FineWeb blog](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). Data pipeline design backed by ablations.
- [Karpathy, "Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) and [llm.c](https://github.com/karpathy/llm.c). A complete small-scale pretraining run.
- [Epoch AI](https://epoch.ai/). Data and analysis on training compute, scaling trends and hardware.

**Objectives and architecture**
- [BERT](https://arxiv.org/abs/1810.04805), [T5](https://arxiv.org/abs/1910.10683), [UL2](https://arxiv.org/abs/2205.05131)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322)
- [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

**Scaling and optimization**
- [Scaling Laws for Neural Language Models (Kaplan et al.)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Chinchilla Scaling: A replication attempt (Epoch AI)](https://arxiv.org/abs/2404.10102)
- [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)
- [Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations (WSD)](https://arxiv.org/abs/2405.18392)
- [MiniCPM](https://arxiv.org/abs/2404.06395)
- [Tensor Programs V (muP)](https://arxiv.org/abs/2203.03466)

**Data**
- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- [Dolma](https://arxiv.org/abs/2402.00159)
- [DataComp-LM (DCLM)](https://arxiv.org/abs/2406.11794)
- [RedPajama](https://arxiv.org/abs/2411.12372)

**Systems**
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP](https://arxiv.org/abs/2304.11277)
- [Megatron-LM](https://arxiv.org/abs/1909.08053) and [Efficient Large-Scale LM Training with Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

**Evaluation**
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MMLU](https://arxiv.org/abs/2009.03300), [GSM8K](https://arxiv.org/abs/2110.14168), [HumanEval](https://arxiv.org/abs/2107.03374)
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) and [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004)

**Case studies**
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

---

previous: [architecture](/posts/learn-ai-01-architecture/) · next: [post-training](/posts/learn-ai-03-post-training/)
