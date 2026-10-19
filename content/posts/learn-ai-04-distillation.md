---
title: "learn ai fundamentals, 5: distillation"
date: 2026-10-18T13:00:00+07:00
description: "training a small student model to copy a large teacher"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 5 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/04-distillation.html).

Most small open models you can download today were not trained the way GPT-2 was in the fundamentals course, by predicting one-hot next tokens from raw web text and nothing else. They were taught by a bigger model. Gemma 2 2B, Llama 3.2 1B and 3B, the Minitron family and the DeepSeek-R1 "distill" checkpoints all owe much of their quality to a teacher. This chapter explains what that teaching signal is, why it carries more information than ordinary labels, and how the recipe changed when the student became an autoregressive language model.

## What you will learn

- Why a teacher's full output distribution ("soft targets") is a richer training signal than a one-hot label, and what temperature does to it.
- The classic Hinton distillation loss, including the often-misunderstood $T^2$ factor.
- White-box versus black-box distillation, and forward versus reverse KL for generative models.
- Sequence-level KD, on-policy distillation (GKD, MiniLLM) and why they fix the train/inference mismatch of naive token-level KD.
- How modern labs use distillation: DistilBERT, synthetic-data SFT from frontier models (Tulu 3, Llama, DeepSeek-R1 distills), logit distillation in Gemma 2 and Llama 3.2, and pruning plus distillation in Minitron.
- How to implement a correct token-level KD loss in PyTorch.

## Prerequisites

From LLM Fundamentals, review:

- **Pretraining objectives**: cross-entropy over the vocabulary, teacher forcing, and how logits become probabilities through softmax.
- **GPT-2 from scratch**: the shape of the logits tensor `(batch, seq_len, vocab)` and how the loss is shifted by one position.
- **SFT and DPO**: chat templates, loss masking on prompt tokens, and the idea of a reference policy with a KL penalty.
- **Tulu 3 and Llama 3**: their post-training pipelines, since both lean heavily on synthetic data.
- **Scaling laws**: why a small model trained on "better" tokens can beat the compute-optimal frontier for its size.

## 1. Intuition: dark knowledge and soft targets

Imagine a digit classifier shown an image of a "2". The label is a one-hot vector: 1.0 on class "2", zero elsewhere. A well-trained large classifier, however, outputs something like 0.9 on "2", 0.08 on "7", 0.01 on "3" and tiny values on the rest. Those small numbers are not noise. They say "this particular 2 looks a bit like a 7", which is a statement about the geometry of the data that the one-hot label throws away. Hinton, Vinyals and Dean called this information **dark knowledge**.

{{< fig "distillation/01" >}}


If you train a small student to match the teacher's whole distribution instead of only the label, each example transfers far more bits. A one-hot label over $V$ classes carries at most $\log_2 V$ bits and in practice much less, since the student mostly already knows the answer. A full distribution tells the student how the teacher ranks every alternative. Empirically, students trained this way generalize better and need fewer examples than students trained on hard labels alone.

Language modeling makes the argument even stronger. The "label" for a next-token prediction is whatever token happened to appear in the corpus, which is one sample from a highly uncertain distribution. After "The capital of France is", the corpus might say " Paris", but after "I went to the store and bought some" it might say " milk" when " eggs", " bread" and " apples" were all equally valid. Cross-entropy against that single sample is a very noisy estimate of the true distribution. A strong teacher's next-token distribution is a far lower-variance target: it puts mass on all the plausible continuations at once. This is why distilled small LMs beat same-size models trained on the same number of raw tokens.

{{< fig "distillation/02" >}}


There is one practical problem. A confident teacher puts almost all of its mass on the top token, so the interesting small probabilities are numerically tiny and contribute almost nothing to a cross-entropy or KL gradient. **Temperature** fixes this. Dividing the logits by $T \gt  1$ before the softmax flattens the distribution, lifting the small probabilities so the student can actually learn the ranking among the "wrong" answers.

{{< fig "distillation/03" >}}


## 2. The math of classic knowledge distillation

### Softmax with temperature

Let $z \in \mathbb{R}^V$ be the teacher's logit vector for one prediction (one image, or one token position), where $V$ is the number of classes or the vocabulary size. Let $v \in \mathbb{R}^V$ be the student's logits for the same input. The tempered distributions are

$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}, \qquad
q_i^{(T)} = \frac{\exp(v_i / T)}{\sum_{j=1}^{V} \exp(v_j / T)}
$$

where $p^{(T)} \in \Delta^{V-1}$ is the teacher's softened distribution (a point on the probability simplex), $q^{(T)}$ is the student's, $i$ indexes a class or token, and $T \gt  0$ is the temperature. $T = 1$ is the ordinary softmax. As $T \to \infty$ both distributions approach uniform, and as $T \to 0$ they approach one-hot on the argmax.

### The combined loss

The distillation loss for one prediction with ground-truth label $y$ is

$$
\mathcal{L} = \alpha \, T^2 \, \mathrm{KL}\!\left(p^{(T)} \,\|\, q^{(T)}\right) + (1 - \alpha)\, \mathrm{CE}\!\left(y, q^{(1)}\right)
$$

where $\alpha \in [0, 1]$ weights the soft term against the hard term, $\mathrm{CE}(y, q^{(1)}) = -\log q_y^{(1)}$ is the usual cross-entropy at temperature 1, and

$$
\mathrm{KL}(p \,\|\, q) = \sum_{i=1}^{V} p_i \log \frac{p_i}{q_i}.
$$

Because $\sum_i p_i \log p_i$ does not depend on the student, minimizing this KL is identical to minimizing the cross-entropy $-\sum_i p_i \log q_i$ of the student against soft labels. Both are common in code; KL is preferred because it is zero at the optimum, which makes logs easier to read.

At inference time the student is always run at $T = 1$. The temperature is only a training device.

### Why multiply by $T^2$

Take the gradient of the soft cross-entropy term with respect to a student logit $v_i$:

$$
\frac{\partial}{\partial v_i}\left(-\sum_j p_j^{(T)} \log q_j^{(T)}\right) = \frac{1}{T}\left(q_i^{(T)} - p_i^{(T)}\right).
$$

The $1/T$ comes from the chain rule through $v_i / T$. Now, for large $T$, the softmax is close to uniform and you can Taylor-expand $\exp(x/T) \approx 1 + x/T$. If the logits are zero-mean per example, this gives $q_i^{(T)} - p_i^{(T)} \approx \frac{1}{VT}(v_i - z_i)$, so

$$
\frac{\partial \mathcal{L}_{\text{soft}}}{\partial v_i} \approx \frac{1}{V T^2}\,(v_i - z_i).
$$

Two things fall out of this. First, in the high-temperature limit, distillation is just matching logits with a squared error, which is what earlier "model compression" work (Bucila et al., Ba and Caruana) did directly. Second, the gradient magnitude shrinks as $1/T^2$. If you change $T$ from 2 to 8 without compensating, the soft term silently becomes 16 times weaker relative to the hard-label term, and your carefully tuned $\alpha$ no longer means what you think. Multiplying by $T^2$ keeps the soft term's gradient scale roughly independent of $T$, so $\alpha$ and $T$ can be tuned separately.

{{< fig "distillation/04" >}}


### White-box versus black-box

The loss above needs the teacher's logits, or at least its full probability vector. That is **white-box** (or "logit") distillation. It requires running the teacher yourself, and for language models it requires the teacher and student to share a tokenizer, because $p$ and $q$ must be distributions over the same vocabulary. Llama 3.1 8B can teach Llama 3.2 1B this way; GPT-4 cannot teach a Llama model this way.

**Black-box** distillation uses only the teacher's sampled outputs: text. The student is fine-tuned with ordinary next-token cross-entropy on teacher-generated responses. It works with closed API models and across tokenizers, but transfers fewer bits per token, since every position collapses to a single sampled token. Most "distillation from GPT-4" or "distillation from R1" you read about is black-box: it is SFT on synthetic data.

Some APIs expose top-$k$ log-probabilities. That is a middle ground: you can build a truncated soft target over the top $k$ tokens, renormalize, and treat the remainder as one bucket.

{{< fig "distillation/05" >}}


## 3. Distilling generative models: which KL, and on whose samples

Classification has one prediction per example. A language model defines a distribution over **sequences**, $p(y \mid x) = \prod_{t=1}^{L} p(y_t \mid y_{\lt t}, x)$, where $x$ is the prompt, $y = (y_1, \dots, y_L)$ is the response, and $L$ is its length. That raises two questions classic KD never had to ask.

### Forward versus reverse KL

Let $p$ be the teacher and $q_\theta$ the student with parameters $\theta$.

$$
\text{Forward KL: } \mathrm{KL}(p \,\|\, q_\theta) = \mathbb{E}_{y \sim p}\left[\log \frac{p(y)}{q_\theta(y)}\right]
\qquad
\text{Reverse KL: } \mathrm{KL}(q_\theta \,\|\, p) = \mathbb{E}_{y \sim q_\theta}\left[\log \frac{q_\theta(y)}{p(y)}\right]
$$

**Forward KL is mode-covering.** The expectation is over the teacher's samples, and the penalty $\log(p/q_\theta)$ explodes wherever the teacher has mass but the student has almost none. So the student is forced to spread mass over everything the teacher might say. When the student is much smaller and cannot represent all of the teacher's modes, it compromises by putting mass in the space between modes, which in text means generating blends that the teacher would never produce. Hinton-style KD and standard SFT on teacher outputs both minimize forward KL.

**Reverse KL is mode-seeking.** The expectation is over the student's own samples, and the penalty explodes wherever the student puts mass that the teacher considers unlikely. The student is allowed to ignore some teacher modes entirely as long as whatever it does say, the teacher finds plausible. For a capacity-limited generator this is usually what you want: fewer, higher-quality behaviors rather than a smeared average. The cost is reduced diversity.

{{< fig "distillation/06" >}}


**MiniLLM** (Gu et al., 2023) argues exactly this and optimizes sequence-level reverse KL. Because the expectation is over student samples, you cannot just backpropagate through a fixed dataset. MiniLLM derives a policy-gradient estimator (with variance-reduction tricks such as single-step decomposition and teacher-mixed sampling), so the training loop looks like RL in which the "reward" is the teacher's log-likelihood.

There are also interpolations. The **generalized Jensen-Shannon divergence** with mixing weight $\beta$,

$$
\mathrm{JSD}_\beta(p \,\|\, q) = \beta\, \mathrm{KL}\!\left(p \,\|\, m\right) + (1 - \beta)\, \mathrm{KL}\!\left(q \,\|\, m\right), \quad m = \beta p + (1-\beta) q,
$$

behaves like forward KL as $\beta \to 0$ and like reverse KL as $\beta \to 1$ (up to scaling), and is bounded, which makes it numerically friendlier.

### Sequence-level knowledge distillation

**Sequence-level KD** (Kim and Rush, 2016) was the first adaptation to sequence models. Instead of matching per-token distributions, run beam search on the teacher for each training input, then train the student with plain cross-entropy on those teacher outputs. The intuition: the teacher's beam output approximates the mode of its sequence distribution, and a small student has an easier time fitting one clean mode than the noisy, multi-modal human references. Notice that this is precisely "SFT on synthetic data". The modern black-box recipe is sequence-level KD with a sampler instead of beam search.

### The exposure-bias problem and on-policy distillation

Both token-level KD on a fixed dataset and sequence-level KD train the student on prefixes $y_{\lt t}$ that came from the data or the teacher. At inference time the student conditions on its **own** previous tokens. Once it makes a mistake that the teacher would never have made, it is in a region of prefix space it was never trained on, and errors compound. This is the same exposure bias that teacher forcing causes in ordinary pretraining, and it hurts small students most because they make the most mistakes.

{{< fig "distillation/07" >}}


**On-policy distillation** fixes the distribution mismatch by training on student-generated prefixes. **GKD** (Generalized Knowledge Distillation, Agarwal et al., 2023, titled "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes") formalizes the loop:

1. Sample a response $\hat{y} \sim q_\theta(\cdot \mid x)$ from the current student.
2. Run the teacher forward on $(x, \hat{y})$ to get its next-token distribution $p(\cdot \mid \hat{y}_{\lt t}, x)$ at **every** position.
3. Minimize a per-token divergence $D$ (forward KL, reverse KL, or $\mathrm{JSD}_\beta$) between teacher and student distributions at each position of $\hat{y}$:

$$
\mathcal{L}_{\text{GKD}}(\theta) = \mathbb{E}_{x}\, \mathbb{E}_{\hat{y} \sim q_\theta(\cdot \mid x)} \left[ \frac{1}{|\hat{y}|} \sum_{t=1}^{|\hat{y}|} D\!\left( p(\cdot \mid \hat{y}_{\lt t}, x) \,\|\, q_\theta(\cdot \mid \hat{y}_{\lt t}, x) \right) \right]
$$

GKD does not backpropagate through the sampling step; it treats the sampled tokens as a fixed dataset that gets refreshed each step. It also mixes on-policy and off-policy data with a ratio $\lambda$, and shows that on-policy data with reverse KL or JSD often works best. Because the teacher scores every token, the signal is dense, which is the key contrast with RL from a sparse sequence-level reward. Thinking Machines' "On-Policy Distillation" post makes this point vividly: you get the on-policy benefits of RL at a fraction of the sample cost, since each episode yields a gradient signal per token instead of one bit at the end.

{{< fig "distillation/08" >}}


The cost is that you need the teacher online during training and it must share the student's tokenizer. That is why on-policy distillation is mostly used inside a model family.

## 4. Code: a token-level KD loss in PyTorch

The function below computes the per-token distillation loss for causal LMs. It handles the three details that most buggy implementations get wrong: the $T^2$ factor, masking padding and prompt tokens, and averaging over real tokens rather than over the batch dimension.

```python
import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,   # (B, L, V)
    teacher_logits: torch.Tensor,   # (B, L, V)
    labels: torch.Tensor,           # (B, L), -100 marks positions to ignore
    temperature: float = 2.0,
    alpha: float = 0.5,
    reverse: bool = False,
) -> torch.Tensor:
    mask = labels != -100                             # (B, L)
    s = student_logits.float() / temperature          # fp32: log_softmax over a 128k vocab underflows in bf16
    t = teacher_logits.float().detach() / temperature  # teacher is a fixed target, never trained

    log_q = F.log_softmax(s, dim=-1)
    log_p = F.log_softmax(t, dim=-1)

    if reverse:
        # KL(q || p): penalizes student mass where the teacher has little (mode-seeking)
        kl = (log_q.exp() * (log_q - log_p)).sum(-1)
    else:
        # KL(p || q): penalizes missing teacher mass (mode-covering), Hinton's original choice
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)

    # Mean over real tokens only, so long padded batches do not dilute the signal
    soft = (kl * mask).sum() / mask.sum().clamp(min=1)
    # T^2 keeps the soft-term gradient scale independent of the chosen temperature
    soft = soft * temperature**2

    if alpha >= 1.0:
        return soft
    hard = F.cross_entropy(
        student_logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100
    )
    return alpha * soft + (1 - alpha) * hard


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, V = 2, 5, 11
    teacher = torch.randn(B, L, V) * 3
    labels = torch.randint(0, V, (B, L))
    labels[:, :2] = -100  # e.g. prompt tokens that should not be distilled

    # Identical logits must give zero soft loss
    assert kd_loss(teacher, teacher, labels, alpha=1.0).abs() < 1e-5

    # Verify the T^2 argument: gradients should stay roughly the same size as T grows
    for T in (1.0, 4.0, 16.0):
        student = torch.randn(B, L, V, requires_grad=True)
        kd_loss(student, teacher, labels, temperature=T, alpha=1.0).backward()
        print(f"T={T:5.1f}  grad norm={student.grad.norm():.4f}")

    # A student trained only on the KD loss converges toward the teacher
    student = torch.zeros(B, L, V, requires_grad=True)
    opt = torch.optim.Adam([student], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        loss = kd_loss(student, teacher, labels, temperature=1.0, alpha=1.0)
        loss.backward()
        opt.step()
    print("final KD loss:", loss.item())
    assert loss.item() < 1e-3
```

Run it and look at the gradient norms. Without the `temperature**2` line, they fall by roughly 16x from $T=4$ to $T=16$; with it they stay in the same ballpark (they are not identical, because the high-$T$ approximation only holds for large $T$).

A few practical notes for using this with real models:

- **Memory.** At $V = 128{,}256$ (Llama 3), a single `(B, L, V)` tensor in fp32 for `B=8, L=4096` is about 16 GB, and you need several of them. Production code computes the loss in chunks along the sequence dimension, or fuses it into a kernel (Liger Kernel ships a fused distillation loss for this reason).
- **Caching teacher logits** is tempting but storage-heavy: the full distribution for one token is 128k floats. Offline pipelines usually store only the top-$k$ probabilities per position and renormalize; a few dozen to a few hundred entries capture nearly all of the mass.
- **Tokenizer match** is non-negotiable for this loss. Check `teacher_tok.get_vocab() == student_tok.get_vocab()` before you start, not after a day of training.

{{< fig "distillation/09" >}}

The on-policy version reuses the same function. The only change is where the token sequences come from:

```python
@torch.no_grad()
def sample_on_policy(student, prompt_ids, max_new_tokens=128):
    # The student must see its own mistakes, so prefixes come from student samples
    return student.generate(prompt_ids, do_sample=True, max_new_tokens=max_new_tokens)


def on_policy_step(student, teacher, prompt_ids, optimizer, temperature=1.0):
    seq = sample_on_policy(student, prompt_ids)
    labels = seq.clone()
    labels[:, : prompt_ids.shape[1]] = -100  # only distill the generated part
    with torch.no_grad():
        t_logits = teacher(seq).logits
    s_logits = student(seq).logits
    # alpha=1: no hard labels exist on-policy, the teacher distribution is the only target
    loss = kd_loss(s_logits[:, :-1], t_logits[:, :-1], labels[:, 1:],
                   temperature=temperature, alpha=1.0, reverse=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

With HuggingFace `transformers`, a pair like `gpt2-large` (teacher) and `distilgpt2` (student) shares a tokenizer and is small enough to try this on a laptop GPU. The TRL library also ships a `GKDTrainer` that implements the full GKD loop with $\lambda$ and $\beta$ knobs.

## 5. Modern practice: where distillation shows up

### DistilBERT: the template

DistilBERT (Sanh et al., 2019) took BERT-base, removed every other layer (6 instead of 12), **initialized the student from the teacher's layers**, and trained on the same corpus with a triple loss: soft-target KD on the masked-LM distribution, the ordinary MLM loss, and a cosine loss aligning student and teacher hidden states. The result kept about 97% of BERT's GLUE score with 40% fewer parameters and 60% faster inference. Two ideas from DistilBERT recur in every later recipe: initialize the student from a cut-down teacher rather than from scratch, and combine a soft target with the normal objective.

{{< fig "distillation/10" >}}

{{< fig "distillation/11" >}}


### Synthetic-data distillation from frontier models

The most widespread form of distillation in 2023-2025 is black-box sequence-level KD: generate responses with a strong model, then SFT a smaller one on them.

- **Tulu 3** builds much of its SFT mixture from prompts answered by frontier models (GPT-4o and others), plus persona-driven synthetic prompts for math, code and instruction-following. You saw this pipeline in the fundamentals course; now you can name it: it is distillation with a forward-KL objective on sampled text.
- **Llama 3** post-training uses rejection sampling from its own larger models to produce SFT data for later rounds, and the Llama 3.1 405B model was explicitly used to improve the post-training data of the 8B and 70B. That is self-distillation within a family.
- **DeepSeek-R1** released six "R1-Distill" models (Qwen and Llama bases from 1.5B to 70B) trained with plain SFT on about 800k samples curated from R1. The paper's striking finding is that distilling R1's reasoning traces into a small model beat running large-scale RL directly on that small model. Reasoning behavior, it turns out, transfers well through text.

The caveats: black-box distillation inherits the teacher's errors and stylistic tics, is bounded by the teacher's capability on the prompt distribution you chose, and is restricted by many providers' terms of service. It also transfers the *outputs* of reasoning without the dense per-token signal, which is why on-policy distillation is attractive when you own the teacher.

{{< fig "distillation/12" >}}


### Logit distillation in pretraining: Gemma 2 and Llama 3.2

Distillation is no longer only a post-training trick.

- **Gemma 2** trains its 2B and 9B models with knowledge distillation during **pretraining**: instead of one-hot next tokens, the student minimizes cross-entropy against the next-token distribution of a larger teacher, over many more tokens than would be compute-optimal for its size (the 2B saw 2T tokens). The report frames it as replacing the noisy one-hot target with a richer one so a small model can keep improving past the point where raw-token training saturates. The 27B model was trained from scratch.
- **Llama 3.2 1B and 3B** were built by pruning Llama 3.1 8B, then using logits from Llama 3.1 8B and 70B as token-level targets during pretraining to recover performance after pruning.

This is the most information-dense form of distillation available, and it is only possible for labs that own a large teacher with the same tokenizer.

### Pruning plus distillation: Minitron

NVIDIA's **Minitron** paper ("Compact Language Models via Pruning and Knowledge Distillation") asked: instead of training each size from scratch, can you train the largest model once, then produce smaller ones by pruning and short distillation runs? Their recipe:

1. **Estimate importance** of attention heads, MLP neurons, embedding channels and whole layers from activations on a small calibration set (about 1k samples).
2. **Prune** along width (heads, neurons, hidden size) and/or depth (layers). They found width pruning generally beats depth pruning at the same parameter count.
3. **Retrain with distillation** from the unpruned teacher, using logit KL as the main loss, on a tiny fraction of the original token budget.

From Nemotron-4 15B they derived 8B and 4B models using up to 40x fewer training tokens than training from scratch, with quality comparable to or better than similar-size open models. Llama-3.1-Minitron 4B later applied the same recipe to Llama 3.1 8B, adding a short "teacher correction" fine-tune so the teacher matched the distillation corpus. This "train one big model, prune and distill the family" approach is now standard for small model releases.

{{< fig "distillation/13" >}}


### Choosing a recipe

| Situation | Recipe |
|---|---|
| Teacher is a closed API | Black-box SFT on sampled outputs (sequence-level KD). Filter aggressively. |
| You own the teacher, same tokenizer, pretraining scale | Logit KD with top-$k$ cached targets (Gemma 2 style). |
| You own the teacher and want a smaller sibling cheaply | Prune, then logit KD (Minitron style). |
| Post-training a small model on a narrow skill | On-policy distillation with reverse KL or JSD (GKD). |

{{< fig "distillation/14" >}}


## Common confusions

- **"Temperature is applied at inference too."** No. $T \gt  1$ is only used to soften both distributions inside the training loss. The deployed student runs at $T = 1$ (or whatever sampling temperature you choose for generation, which is unrelated).
- **"The $T^2$ factor is a magic constant."** It exactly cancels the $1/T^2$ gradient scaling derived above. Drop it and your $\alpha$ silently depends on $T$.
- **"KL and cross-entropy against soft labels give different gradients."** They differ by the teacher's entropy, which is constant with respect to the student. Gradients are identical.
- **"Fine-tuning on GPT-4 outputs is not distillation."** It is: black-box, sequence-level, forward-KL distillation. It just transfers fewer bits per token than logit distillation.
- **"Reverse KL is always better for LMs."** It is better at avoiding low-quality blends when the student is much smaller, but it reduces diversity. GKD found the best divergence is task-dependent.
- **"On-policy distillation is RL."** It shares RL's sampling loop, but the signal is a dense per-token divergence from a teacher, not a scalar reward, and GKD does not need a policy-gradient estimator because it differentiates the per-token divergence directly on fixed samples.
- **"Any teacher can teach any student via logits."** Only if they share the vocabulary. Cross-tokenizer logit distillation is an open research area.

## Check yourself

1. A teacher outputs logits `[8, 2, 1]`. Compute (roughly) the teacher distribution at $T=1$ and $T=4$. Which one teaches the student more about class 2 versus class 3?
2. You double the distillation temperature from 2 to 4 but forget the $T^2$ factor. What happens to the balance between soft and hard losses, and how would you notice it in training curves?
3. A 1B student is distilled from a 70B teacher with forward KL on teacher samples. Its outputs are fluent but sometimes mix two unrelated plausible answers. Explain why, and name two changes that would help.
4. Why does GKD sample from the student rather than the teacher, and what extra infrastructure does that require compared to SFT on synthetic data?
5. Why could Llama 3.2 1B use logit distillation from Llama 3.1 70B, while most open models distilled from GPT-4 could not?
6. In Minitron, why is the pruned model distilled rather than simply fine-tuned with ordinary next-token cross-entropy on the same small token budget?

<details>
<summary>Answers</summary>

1. At $T=1$, $e^8 \approx 2981$, $e^2 \approx 7.4$, $e^1 \approx 2.7$, so roughly `[0.997, 0.0025, 0.0009]`. At $T=4$ the logits become `[2, 0.5, 0.25]`, giving about `[0.71, 0.16, 0.13]`. The $T=4$ distribution makes the 2-versus-3 ranking large enough to produce a meaningful gradient; at $T=1$ it is buried in the third decimal place.
2. The soft-term gradient shrinks by about $4$x, so the hard-label term dominates and training behaves more like ordinary SFT. You would see the KD loss plateau higher and the student's agreement with the teacher's top-$k$ drop, while the hard cross-entropy trains normally.
3. Forward KL is mode-covering: the student is penalized heavily for assigning low probability to anything the teacher might say, so a capacity-limited student spreads mass between modes and produces blends. Switch to reverse KL or $\mathrm{JSD}_\beta$ with large $\beta$, and/or train on-policy so the teacher can penalize the student's actual blended outputs. Sequence-level KD with low-temperature teacher samples also helps by giving the student one mode per prompt.
4. Training on teacher or dataset prefixes leaves a train/inference mismatch: at inference the student conditions on its own (possibly wrong) prefixes. Sampling from the student puts training on the states it will actually visit. It requires running the student's generation loop inside training and keeping the teacher online to score every generated sequence, typically with a separate inference server or engine.
5. Logit distillation needs both models to output distributions over the same vocabulary. Llama 3.2 and Llama 3.1 share a tokenizer and Meta owns the 70B weights. GPT-4 is a closed API with a different tokenizer and at most exposes top-$k$ logprobs, so only black-box text distillation is possible.
6. After pruning, the model is damaged but still close to the teacher; the teacher's full distribution tells it exactly how to recover the original function, giving far more signal per token than one-hot labels. That is why Minitron recovers quality with a tiny fraction of the original token budget.

</details>

## Resources

### Read

- [Distilling the Knowledge in a Neural Network (Hinton, Vinyals, Dean, 2015)](https://arxiv.org/abs/1503.02531) **(start here)**: the original paper; short, and the $T^2$ derivation is in section 2.
- [Sequence-Level Knowledge Distillation (Kim and Rush, 2016)](https://arxiv.org/abs/1606.07947): the bridge from classification KD to training on teacher-generated sequences.
- [DistilBERT (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108): layer-skipping initialization plus a triple loss; the template for later recipes.
- [MiniLLM: Knowledge Distillation of Large Language Models (Gu et al., 2023)](https://arxiv.org/abs/2306.08543): the case for reverse KL and a policy-gradient way to optimize it.
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (Agarwal et al., 2023)](https://arxiv.org/abs/2306.13649): GKD, the cleanest formulation of on-policy distillation.
- [On-Policy Distillation (Thinking Machines)](https://thinkingmachines.ai/blog/on-policy-distillation/): a practitioner's view of why dense per-token teacher signal beats sparse RL rewards for post-training.
- [Compact Language Models via Pruning and Knowledge Distillation (Minitron)](https://arxiv.org/abs/2407.14679): the prune-then-distill recipe with careful ablations.
- [Gemma 2 technical report](https://arxiv.org/abs/2408.00118): section on using distillation as the pretraining objective for small models.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948): see the "Distillation" section for why SFT on R1 traces beat RL on small models.

### Watch

- [Knowledge Distillation in Large Language Models (John Olafenwa)](https://www.youtube.com/watch?v=1RvdM-q6kDQ): an overview walkthrough of KD applied to LLMs, useful as a refresher after reading Hinton.

## Project

**Distill a small GPT-2 two ways and compare.**

1. Pick a teacher and student that share a tokenizer: `gpt2-large` (774M) and `distilgpt2` (82M), or `Qwen2.5-1.5B-Instruct` and `Qwen2.5-0.5B` if you have a bigger GPU.
2. Choose a narrow task with an automatic metric, such as GSM8K-style arithmetic word problems or short story continuation scored by the teacher's own perplexity.
3. Train three students from the same initialization with the same token budget:
   - **(a) SFT** on teacher-sampled responses (black-box, sequence-level KD).
   - **(b) Off-policy logit KD** using `kd_loss` with forward KL on the same teacher-sampled sequences.
   - **(c) On-policy KD** using `on_policy_step` with reverse KL.
4. Report: task metric, the teacher's average per-token log-likelihood of each student's samples, and a diversity measure (distinct-2 n-grams across 100 samples per prompt).
5. Write a paragraph explaining the results in terms of forward versus reverse KL and exposure bias. Bonus: sweep $T \in \{1, 2, 4\}$ for (b) with and without the $T^2$ factor and plot the training curves.

---

previous: [lora](/posts/learn-ai-06-lora/) · next: [optimizers](/posts/learn-ai-08-optimizers/)
