---
title: "learn ai fundamentals, 1: architecture"
date: 2026-10-18T17:00:00+07:00
description: "how a transformer turns text into a next-token guess, ending with a gpt-2 you write yourself"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 1 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/01-architecture.html).

This chapter takes you from "what is a language model" to a working GPT-2 that you wrote yourself and that loads OpenAI's real weights. Each section builds on the previous one, so read them in order.

## What you will learn

- What a large language model actually computes, and why "predict the next token" is enough to produce useful behavior.
- How raw text becomes integer token IDs (BPE and its relatives), and why tokenization explains many odd LLM failures.
- How token IDs become vectors (embeddings), and how the model knows where each token sits in the sequence (positional encodings, including RoPE).
- Exactly what attention computes, with shapes, plus the inference-time tricks that make it practical (causal masking, multi-head, GQA, KV cache, FlashAttention).
- How a full transformer block is assembled (residual stream, normalization, MLP), and what interpretability research says the layers are doing.
- How to write GPT-2 in about 150 lines of PyTorch, load the official weights, sample text with temperature, top-k and top-p, and train a tiny model yourself.

## Prerequisites

- Python and basic PyTorch: tensors, `nn.Module`, autograd, writing a training loop.
- Matrix multiplication and dot products, and comfort reading tensor shapes like `(B, T, C)`. If this is rusty, watch the first four chapters of 3Blue1Brown's *Essence of Linear Algebra* before starting.
- Softmax, cross-entropy loss, and the idea of gradient descent. Karpathy's *Neural Networks: Zero to Hero* videos 1 to 5 cover exactly this.
- A machine with Python 3.10+ and `pip install torch tiktoken transformers`. A GPU is nice but not required for anything in this chapter.

Notation used throughout: $B$ is batch size, $T$ is sequence length (number of tokens), $C$ or $d_{\text{model}}$ is the model width (embedding dimension), $V$ is the vocabulary size, $h$ is the number of attention heads, and $d_k = d_{\text{model}} / h$ is the per-head dimension.

---

## 01. Introduction

### What an LLM is

A large language model is a function that takes a sequence of tokens (think of them as word pieces for now; section 02 makes this precise) and returns a probability distribution over what the next token should be. That is the whole contract. Given "The capital of France is", a well-trained model puts most of its probability mass on " Paris".

Everything an LLM appears to do, such as answering questions, writing code, or translating, is produced by running this one function in a loop:

1. Feed in the tokens so far.
2. Get a probability for every token in the vocabulary.
3. Pick one (the most likely one, or a random draw weighted by the probabilities).
4. Append it to the sequence and go back to step 1.

This loop is called **autoregressive generation**: each output becomes part of the next input. The model has no separate "answer mode". A chat reply is just a continuation of a document that happens to look like a conversation.

{{< fig "architecture/02" >}}

### Why next-token prediction is so powerful

It seems too simple to explain the capabilities. The intuition: to predict the next token of arbitrary internet text *well*, you have to model whatever process produced that text. Predicting the next line of a Python file rewards understanding Python. Predicting the end of a physics derivation rewards some grasp of the physics. Predicting what a character in a novel says next rewards modeling that character's beliefs. The objective is simple, but the data is so diverse that minimizing the loss forces the model to learn a large amount of structure about language and the world.

Formally, a language model defines the probability of a whole sequence $x_1, \dots, x_T$ using the chain rule of probability:

$$
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1})
$$

Here $x_t$ is the token at position $t$ (an integer ID), and $p(x_t \mid x_{\lt t})$ is what the network outputs at position $t-1$. Training minimizes the average **negative log-likelihood**, also called cross-entropy loss:

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{\lt t})
$$

where $\theta$ is the set of all model weights. A key engineering fact: a transformer computes the prediction for *every* position in one forward pass, so a single training sequence of 1024 tokens gives 1024 training examples at once. This parallelism is a large part of why transformers replaced earlier architectures.

{{< fig "architecture/03" >}}

### The decoder-only transformer, end to end

Modern LLMs (GPT, Llama, Claude, Gemini, Mistral, Qwen) are almost all **decoder-only transformers**. Here is the path from text to next token. Every box gets its own section later.

```
"The cat sat"                                      text
   | tokenizer (section 02)
[464, 3797, 3332]                                  token IDs, shape (T,)
   | embedding lookup (section 03)
x  : (T, d_model)                                  one vector per token
   | + position information (section 04)
   | N x transformer block (sections 05, 06):
   |     x = x + Attention(Norm(x))   tokens exchange information
   |     x = x + MLP(Norm(x))         each token processed on its own
   | final Norm
   | unembedding / LM head (section 03)
logits : (T, V)                                    a score per vocab entry, per position
   | softmax over the last position
p(next token) : (V,)                               probabilities summing to 1
   | sampling (section 07)
" on"
```

{{< fig "architecture/01" >}}

Two ideas carry the design:

- **Attention** is the only place where information moves *between* token positions. The token "sat" can look back at "cat" to figure out who sat.
- **The MLP** (a small two-layer network) processes each position independently. It is where much of the model's stored knowledge appears to live.

"Decoder-only" is a historical name. The original transformer had an encoder (reads the input, like a French sentence) and a decoder (writes the output, like the English translation). GPT kept only the decoder half, with its rule that each position may look only at earlier positions. That restriction is exactly what next-token prediction needs.

### A short history

**RNNs (1980s to 2010s).** A recurrent neural network reads tokens one at a time and keeps a hidden state vector, updating it at each step: $h_t = f(h_{t-1}, x_t)$. The hidden state is a running summary of everything seen so far. Two problems: training must step through the sequence one token at a time (slow on GPUs, which want big parallel matrix multiplies), and gradients flowing back through many steps tend to shrink toward zero or blow up, so long-range dependencies are hard to learn.

**LSTMs (1997).** Hochreiter and Schmidhuber added gates (learned switches that decide what to keep, forget and output) and a separate cell state that gradients can flow through more easily. LSTMs dominated language modeling and machine translation through about 2017. The 2014 sequence-to-sequence models used one LSTM to encode a sentence into a vector and another to decode a translation from it.

**Attention (2014).** Squeezing a whole sentence into one fixed vector was a bottleneck. Bahdanau, Cho and Bengio let the decoder look back at *all* encoder hidden states and take a weighted average, with weights computed from how relevant each one is. This "attention" was an add-on to RNNs.

**"Attention Is All You Need" (2017).** Vaswani et al. at Google asked: what if we drop recurrence entirely and use only attention? The resulting **Transformer** processes all positions in parallel during training, connects any two positions in one step instead of many, and scaled far better on GPUs. It beat LSTMs on translation at a fraction of the training cost.

{{< fig "architecture/04" >}}

**The GPT lineage.**

- **GPT-1 (2018, 117M parameters).** OpenAI pre-trained a 12-layer decoder-only transformer on books with next-token prediction, then fine-tuned it on specific tasks. The lesson: generic pre-training transfers.
- **BERT (2018)**, from Google, went the other way: encoder-only, trained to fill in masked words. Great for classification and search, not for generation. It is the main branch this course does not follow.
- **GPT-2 (2019, up to 1.5B parameters).** Same recipe, ten times bigger, trained on WebText (about 40GB of web pages). It could do tasks like summarization or translation "zero-shot", just by being prompted. We build it in section 07.
- **GPT-3 (2020, 175B parameters).** Showed **in-context learning**: put a few examples in the prompt and the model picks up the task with no weight updates. It also made **scaling** the central story: performance improved predictably with more parameters, data and compute (covered in Chapter 2).
- **InstructGPT and ChatGPT (2022)** added post-training (supervised fine-tuning plus reinforcement learning from human feedback) to turn a document-continuer into an assistant. That is Chapter 3.

{{< fig "architecture/05" >}}

Architecturally, a 2025 frontier model is still recognisably GPT-2 with a set of refinements: RoPE instead of learned positions, RMSNorm instead of LayerNorm, SwiGLU instead of GELU, grouped-query attention, and often mixture-of-experts MLPs. You will meet all but the last in this chapter.

{{< fig "architecture/06" >}}

### Snippet: next-token prediction with counting

Before any neural network, here is the whole idea with a lookup table. We "train" by counting which character follows which, then generate autoregressively. A transformer replaces the count table with a learned function that can look at the entire history, not just one previous character.

```python
import random
from collections import Counter, defaultdict

text = "the cat sat on the mat. the cat ate the rat. the rat sat on the cat."

# The "model": for each character, how often each next character follows it.
counts = defaultdict(Counter)
for a, b in zip(text, text[1:]):
    counts[a][b] += 1

def next_char_distribution(prev):
    c = counts[prev]
    total = sum(c.values())
    return {ch: n / total for ch, n in c.items()}

print(next_char_distribution("t"))  # p(next | 't')

random.seed(0)
out = "t"
for _ in range(40):
    dist = next_char_distribution(out[-1])
    # Sampling, not argmax: argmax would loop on the single most common pair.
    out += random.choices(list(dist), weights=list(dist.values()))[0]
print(out)
```

The output is gibberish with the right "texture". That gap, between one character of context and thousands of tokens of context, is what the rest of the chapter closes.

### Common confusions

- **"The model outputs a word."** It outputs a probability distribution over the whole vocabulary. Choosing a token from it is a separate step (sampling) with its own knobs.
- **"The model remembers the conversation."** The model is stateless between calls. "Memory" is the transcript being fed back in as input each time. Anything outside the context window does not exist for the model.
- **"Decoder-only means it can't understand input."** The prompt is processed by the same stack; "decoder-only" only means every position can see earlier positions and not later ones.
- **"Transformers replaced RNNs because they are smarter."** Mostly because they parallelize during training. Training cost per token is much lower on GPUs, which let people train far larger models on far more data.
- **"Generation is parallel too."** Training is parallel across positions; generation is inherently sequential, one token per forward pass. Section 05 covers the KV cache that makes this bearable.

### Check yourself

1. Write the chain-rule factorization of $p(\text{"a b c"})$ into three conditional terms.
2. Why can a transformer get $T$ training signals from one sequence of length $T$ in a single forward pass?
3. Name the two sub-layers of a transformer block and say which one moves information between positions.
4. What was the main practical advantage of the 2017 Transformer over LSTMs?
5. What did GPT-3 demonstrate that GPT-1 did not?

<details>
<summary>Answers</summary>

1. $p(a)\,p(b \mid a)\,p(c \mid a, b)$.
2. The causal mask lets every position compute its own prediction using only earlier positions, so all $T$ predictions are computed simultaneously and each one is compared to the true next token.
3. Attention and the MLP (feed-forward network). Attention moves information between positions; the MLP works on each position independently.
4. Parallelism during training (no step-by-step recurrence) plus direct connections between any two positions, which made it much cheaper to train at scale and better at long-range dependencies.
5. In-context learning (doing new tasks from a few examples in the prompt, without fine-tuning) and that capabilities improve smoothly with scale.

</details>

### Resources

**Read**
- [Vaswani et al., "Attention Is All You Need" (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762): the original Transformer paper. Read sections 1 to 3 now; the rest will make more sense after section 05.
- [Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2 paper, PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): short and readable; section 2 explains the "language modeling as multitask learning" framing.
- [Brown et al., "Language Models are Few-Shot Learners" (GPT-3, arXiv:2005.14165)](https://arxiv.org/abs/2005.14165): read the abstract and figures 1.1 to 1.3 for in-context learning and scaling.
- [Christopher Olah, "Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/): the clearest picture of what came before transformers and why gates helped.

**Watch**
- [3Blue1Brown, "But what is a GPT? Visual intro to Transformers" (Deep Learning ch. 5)](https://www.3blue1brown.com/lessons/gpt) (start here): a 27-minute visual tour of the whole pipeline in this section.
- [Andrej Karpathy, "Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html): the "makemore" videos build exactly the counting model above, then replace it with neural networks.

---

## 02. Tokenization

### Why not just use characters or words?

A neural network needs numbers, so text must be chopped into units and each unit given an integer ID. The choice of unit is a tradeoff:

- **Characters.** Tiny vocabulary (about 100 for English, but over 150,000 Unicode code points in general), never an unknown symbol. But sequences become very long: a 1000-word essay is about 5000 characters. Since attention cost grows with the square of sequence length (section 05), and the model must spend capacity just learning to spell, this is expensive.
- **Words.** Short sequences, and each unit carries meaning. But the vocabulary explodes ("run", "runs", "running", "rerun", every name, every typo, every language), and any word not seen in training becomes an unknown `<UNK>` token. Word-level models cannot handle "ChatGPTification".
- **Subwords.** The compromise every modern LLM uses: frequent words get a single token (" the", " because"), rare words are split into reusable pieces (" Chat", "G", "PT", "ification"). Vocabulary stays fixed (typically 32k to 256k), nothing is ever unknown, and common text is compressed to roughly 4 characters per token in English.

{{< fig "architecture/07" >}}

### Byte Pair Encoding (BPE), step by step

BPE started as a 1994 data compression trick and was adapted to neural machine translation by Sennrich et al. in 2015. GPT-2 onward uses a byte-level variant. The training algorithm is greedy and simple:

1. Start with a base vocabulary of single symbols and represent the training corpus as sequences of those symbols.
2. Count every adjacent pair of symbols in the corpus.
3. Take the most frequent pair, say `("t", "h")`, create a new symbol `"th"`, and replace every occurrence of the pair with it. Record the **merge rule** `t + h -> th`.
4. Repeat steps 2 and 3 until the vocabulary reaches the target size.

Worked example on the corpus `"hug pug pun bun hugs"` (ignoring spaces for clarity; each word is a separate sequence):

| Step | Most frequent pair | New token | Corpus after merge |
|---|---|---|---|
| start | | | `h u g`, `p u g`, `p u n`, `b u n`, `h u g s` |
| 1 | `u g` (3 times) | `ug` | `h ug`, `p ug`, `p u n`, `b u n`, `h ug s` |
| 2 | `u n` and `h ug` tie at 2; we pick `u n` | `un` | `h ug`, `p ug`, `p un`, `b un`, `h ug s` |
| 3 | `h ug` (2) | `hug` | `hug`, `p ug`, `p un`, `b un`, `hug s` |

The output of training is the ordered list of merges. **Encoding** new text replays those merges in the order they were learned: split into base symbols, then repeatedly apply the earliest-learned merge that is present until none apply. **Decoding** is trivial: look up each token's string and concatenate.

{{< fig "architecture/08" >}}

### Byte-level BPE

If the base symbols are Unicode characters, you either need a huge base vocabulary or you hit unknown characters (emoji, rare scripts). GPT-2's fix: run BPE over the **UTF-8 bytes** of the text. The base vocabulary is exactly 256 byte values, so *any* string can be encoded, and merges build up multi-byte characters and then words. GPT-2's vocabulary is 50,257 = 256 bytes + 50,000 learned merges + 1 special `<|endoftext|>` token.

GPT-2 also adds **pre-tokenization**: before BPE, a regex splits text into chunks (letters, numbers, punctuation, whitespace, contractions like `'s`), and merges never cross chunk boundaries. This stops the tokenizer from learning silly tokens like `"dog."` and `"dog!"` separately. Note the convention that a leading space attaches to the following word: `" dog"` (with a space) is one token, different from `"dog"`.

{{< fig "architecture/09" >}}

### WordPiece, Unigram and SentencePiece

- **WordPiece** (used by BERT) is BPE with a different merge criterion: instead of the most frequent pair, merge the pair that most increases the likelihood of the training data, roughly $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$. Continuation pieces are marked with `##` (`"hug", "##s"`).
- **Unigram LM** (Kudo, 2018) works top-down. Start from a large candidate vocabulary, fit a model where each token has a probability and a word's segmentation probability is the product of its tokens' probabilities, then repeatedly remove the tokens whose removal hurts the likelihood least. A single string can have several segmentations, which allows "subword regularization" (sampling different segmentations during training as data augmentation).
- **SentencePiece** (Kudo and Richardson, 2018) is a *library*, not an algorithm. It implements both BPE and Unigram, treats input as a raw stream (spaces become the visible symbol `▁`), and needs no language-specific pre-tokenizer. That makes it popular for multilingual models. Llama 1 and 2 used SentencePiece BPE with a 32k vocabulary and "byte fallback" for unknown characters.

{{< fig "architecture/10" >}}

### tiktoken

`tiktoken` is OpenAI's fast byte-level BPE implementation (written in Rust). It ships the encodings used by OpenAI models: `gpt2` (50k vocab), `cl100k_base` (about 100k, GPT-3.5 and GPT-4) and `o200k_base` (about 200k, GPT-4o). It only *encodes and decodes*; it does not train. Llama 3 switched to a tiktoken-based tokenizer with a 128k vocabulary.

### Vocabulary size tradeoffs

- **Bigger vocab:** shorter sequences (more text fits in the context window, fewer forward passes to generate a reply, cheaper attention), and better coverage of non-English languages and code.
- **Costs of a bigger vocab:** the embedding matrix and output layer have $V \times d_{\text{model}}$ parameters each (section 03), and the final softmax over $V$ entries costs compute at every position. Rare tokens get few training examples, so their embeddings are poorly learned.

The trend has been upward: 32k (Llama 2) to 128k (Llama 3) to 200k+ (GPT-4o, Gemma), because for large models the embedding cost is a small fraction of the total and the compression win is significant.

{{< fig "architecture/11" >}}

### Tokenization pitfalls

Many "LLMs are dumb" moments are tokenization artifacts:

- **Spelling and counting letters.** The model sees `" strawberry"` as one or a few token IDs, not ten letters. It must have memorized the spelling of each token to count its r's.
- **Arithmetic.** Numbers split inconsistently (`"1234"` might be `"123"` + `"4"`, while `"1235"` is `"12"` + `"35"`), making digit-level algorithms awkward. Many newer tokenizers split numbers into individual digits or groups of at most three.
- **Leading spaces.** `"Hello"` and `" Hello"` are different tokens. A prompt that ends with a trailing space can make the model act strangely, because in training the space almost always belonged to the *next* word's token.
- **Non-English text** often costs 2 to 10 times more tokens for the same meaning, which means higher price, slower generation and less effective context.
- **Glitch tokens.** Tokens that were frequent in the tokenizer's training data but almost absent from the model's training data (the famous `" SolidGoldMagikarp"`, a Reddit username in GPT-2's vocabulary) have nearly untrained embeddings and can trigger bizarre outputs.
- **Code whitespace.** GPT-2's tokenizer wasted many tokens on Python indentation (each space a token); later tokenizers added tokens for runs of spaces.

{{< fig "architecture/12" >}}

### Snippet: train a tiny BPE, then compare with tiktoken

```python
from collections import Counter

def get_pair_counts(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

text = "low lower lowest newer newest wider widest " * 20
ids = list(text.encode("utf-8"))  # byte-level: base vocab is 0..255
merges = {}
for step in range(10):
    pair = get_pair_counts(ids).most_common(1)[0][0]
    new_id = 256 + step
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id

vocab = {i: bytes([i]) for i in range(256)}
for (a, b), i in merges.items():  # dicts keep insertion order, so parents exist first
    vocab[i] = vocab[a] + vocab[b]
print([vocab[i].decode() for i in merges.values()])
print(f"compression: {len(text.encode())} bytes -> {len(ids)} tokens")

import tiktoken
enc = tiktoken.get_encoding("gpt2")
for s in ["hello world", " hello world", "strawberry", "1234567", "Привет мир"]:
    toks = enc.encode(s)
    print(repr(s), toks, [enc.decode([t]) for t in toks])
```

Run it and notice: the learned merges are pieces like `"we"`, `"st"`, `"lowe"`, `"newe"`; the leading space changes the tokens; Russian text uses far more tokens per character than English. (Decoding a single token on its own can show a broken character when a token holds only part of a multi-byte UTF-8 character; that is expected.)

### Common confusions

- **"Tokens are words."** They are subword pieces; on average about 0.75 English words per token, much less for other languages and for code.
- **"The tokenizer is learned with the model."** It is trained separately, beforehand, usually on a different sample of data, and then frozen. The model only ever sees integer IDs.
- **"BPE encoding picks the longest matching token."** It replays the learned merges in order of priority. The result is usually but not always the same as greedy longest-match.
- **"SentencePiece is an alternative to BPE."** SentencePiece is a library that can run BPE or Unigram.
- **"Token ID numbers mean something."** IDs are arbitrary labels (in BPE, roughly the order merges were learned). Meaning lives in the embedding vectors, next section.

### Check yourself

1. Why does byte-level BPE never produce an unknown token?
2. In BPE, what exactly is saved after training, and how is it used to encode new text?
3. Give two reasons a model might be bad at reversing the letters of a word.
4. You double the vocabulary size of a model with $d_{\text{model}} = 4096$. How many parameters does that add to the input embedding alone if $V$ goes from 32k to 64k?
5. Why does GPT-2 pre-tokenize with a regex before running BPE?

<details>
<summary>Answers</summary>

1. Every string is a sequence of UTF-8 bytes, and all 256 byte values are in the base vocabulary, so any input can be represented, in the worst case as one token per byte.
2. The ordered list of merge rules (plus the resulting vocabulary). Encoding splits text into bytes and repeatedly applies the highest-priority (earliest-learned) merge present until none apply.
3. The model sees token IDs, not letters, so it must know each token's spelling from memory; and the reversed string tokenizes into completely different, often rare, pieces.
4. $32{,}000 \times 4096 \approx 131$ million extra parameters (and the same again for the output layer if it is not tied).
5. To stop merges from crossing category boundaries (letters with punctuation, words with following spaces), which would waste vocabulary on many near-duplicate tokens like `"dog."`, `"dog!"`, `"dog?"`.

</details>

### Resources

**Read**
- [Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with Subword Units" (arXiv:1508.07909)](https://arxiv.org/abs/1508.07909): the paper that brought BPE to NLP; section 3.2 includes the full algorithm in a few lines of Python.
- [Hugging Face LLM Course, chapter 6.5: Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5): worked example plus code; neighboring pages 6.6 and 6.7 cover WordPiece and Unigram the same way.
- [Kudo and Richardson, "SentencePiece" (arXiv:1808.06226)](https://arxiv.org/abs/1808.06226): short paper explaining the language-independent design.
- [karpathy/minbpe](https://github.com/karpathy/minbpe): minimal, readable BPE (including the GPT-4 regex) to read after the video; do its `exercise.md`.
- [openai/tiktoken](https://github.com/openai/tiktoken): the production tokenizer; the README shows how to inspect encodings.

**Watch**
- [Andrej Karpathy, "Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE) (start here): two hours building byte-level BPE from scratch; the final 30 minutes on tokenization pitfalls is especially worth it.

---

## 03. The Embedding Layer

### From integer IDs to vectors

After tokenization, the input is a list of integers like `[464, 3797, 3332]`. These numbers are labels, not quantities: token 3797 is not "more" than token 464. A neural network multiplies and adds its inputs, so feeding raw IDs in would impose a fake ordering. We need a representation where every token is equally distinct to begin with, but which can then *learn* to place related tokens near each other.

**One-hot vectors** are the first step. Represent token $i$ as a vector of length $V$ that is all zeros except a 1 at position $i$. Every pair of distinct tokens is now equally far apart. But one-hot vectors are huge ($V$ = 50,257 for GPT-2) and carry no notion of similarity.

So we multiply by a learned matrix. Let $W_E \in \mathbb{R}^{V \times d_{\text{model}}}$ be the **embedding matrix**, with one row per vocabulary entry. For a one-hot row vector $e_i \in \mathbb{R}^{1 \times V}$:

$$
e_i \, W_E = W_E[i, :] \in \mathbb{R}^{1 \times d_{\text{model}}}
$$

Multiplying a one-hot vector by a matrix just selects row $i$. So in practice nobody builds the one-hot vector: the **embedding layer is a lookup table**, `W_E[token_ids]`, which is mathematically identical and far cheaper. The gradient is identical too: during backpropagation only the rows for tokens that appeared in the batch receive updates.

{{< fig "architecture/13" >}}

Shapes for a batch:

- Input token IDs: `(B, T)`, integers in $[0, V)$.
- Embedding matrix $W_E$: `(V, d_model)`. For GPT-2 small, `(50257, 768)`, about 38.6M parameters, roughly 31% of the 124M total.
- Output: `(B, T, d_model)`, one vector per token. This tensor is what flows through the rest of the network. From here on we call it the **residual stream** (section 06 explains the name).

In PyTorch this is `nn.Embedding(V, d_model)`. Weights start as small random numbers (GPT-2 uses a normal distribution with standard deviation 0.02) and are learned by gradient descent along with everything else.

{{< fig "architecture/14" >}}

### The output side: unembedding

At the top of the network, each position's final vector $x \in \mathbb{R}^{d_{\text{model}}}$ has to become a score for every vocabulary entry. That is another matrix, the **unembedding** or **LM head**, $W_U \in \mathbb{R}^{d_{\text{model}} \times V}$:

$$
\text{logits} = x \, W_U \in \mathbb{R}^{V}, \qquad p(\text{next} = j) = \frac{\exp(\text{logits}_j)}{\sum_{k=1}^{V} \exp(\text{logits}_k)}
$$

The logit for token $j$ is the dot product of $x$ with column $j$ of $W_U$. The dot product is large when two vectors point in a similar direction, so the model predicts token $j$ by producing a final vector that points toward token $j$'s column. Softmax then turns the $V$ scores into probabilities that are positive and sum to 1.

### Weight tying

Notice the symmetry: $W_E$ maps tokens into vector space, and $W_U$ scores vectors against tokens. Press and Wolf (2016) and Inan et al. (2016) proposed using the same matrix for both: $W_U = W_E^\top$. This is **weight tying**. The reasoning: if two tokens are interchangeable as inputs ("big" and "large"), they are probably interchangeable as predictions too, so they should share one representation. Benefits:

- Saves $V \times d_{\text{model}}$ parameters (38.6M for GPT-2, which ties).
- Every row of $W_E$ now gets a gradient on every step through the output layer, not just when that token appears in the input, so rare tokens learn faster.

Tying is common in small and medium models (GPT-2, Gemma, many small Llama variants such as Llama 3.2 1B and 3B). Larger models often untie them, because the parameter cost is relatively small and separate matrices give the model more freedom: the input side has to say "what is this token" while the output side has to say "what would come next", and those are not exactly the same job.

{{< fig "architecture/15" >}}

### What embeddings learn

Since every row starts random and is shaped only by the prediction loss, the geometry the embeddings settle into reflects how tokens are *used*. Tokens that appear in similar contexts get similar vectors (this is the "distributional hypothesis": you know a word by the company it keeps). Concretely, in trained models:

- Nearest neighbours of a token's embedding are synonyms, morphological variants (" run", " running", " ran"), the same word with and without capitals or a leading space, and members of the same category (days of the week, programming keywords).
- Some *directions* in embedding space carry meaning. The classic word2vec (Mikolov et al., 2013) result is that $\text{vec(king)} - \text{vec(man)} + \text{vec(woman)}$ lands near $\text{vec(queen)}$. In LLM input embeddings these linear relationships are messier, because the input embedding only has to be a good *starting point*; later layers build the context-dependent meaning.
- An input embedding cannot know which sense of a word is meant (" bank" of a river vs a financial bank). Context is added by attention in later layers. The vector at position $t$ after many layers is very different from the raw embedding.

{{< fig "architecture/16" >}}

{{< fig "architecture/17" >}}

A useful mental model: the embedding table is a dictionary of initial "guesses" about each token, and the transformer blocks then edit those guesses using context.

### Snippet: lookup equals one-hot matmul, and tying

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

V, d = 10, 4
emb = nn.Embedding(V, d)
ids = torch.tensor([[3, 7, 3]])                  # (B=1, T=3)

x_lookup = emb(ids)                               # (1, 3, 4)
one_hot = F.one_hot(ids, num_classes=V).float()   # (1, 3, 10)
x_matmul = one_hot @ emb.weight                   # (1, 3, 10) @ (10, 4) -> (1, 3, 4)
print(torch.allclose(x_lookup, x_matmul))         # True: same operation

lm_head = nn.Linear(d, V, bias=False)             # weight shape (V, d), computes x @ W.T
lm_head.weight = emb.weight                       # tie: one tensor, two roles
logits = lm_head(x_lookup)                        # (1, 3, 10)
print(logits.shape, lm_head.weight is emb.weight)

# Same token in, same vector out: the embedding has no idea about context yet.
print(torch.equal(x_lookup[0, 0], x_lookup[0, 2]))
```

Note that `nn.Linear(d, V)` stores its weight as `(V, d)` and computes `x @ W.T`, which is exactly why the embedding matrix `(V, d)` can be assigned to it directly without a transpose.

### Common confusions

- **"The embedding layer is a separate pretrained model (like word2vec)."** In LLMs it is just the first weight matrix of the network, trained from scratch together with everything else.
- **"Embedding vectors are the model's understanding of a word."** They are a context-free starting point. The same token in different contexts has the same input embedding; contextual meaning is built later.
- **"An embedding lookup and a linear layer are different things."** A lookup is a linear layer applied to a one-hot vector, implemented efficiently.
- **"Weight tying means the input and output vectors of a token are identical at every layer."** It only means $W_U = W_E^\top$. The final hidden state that gets scored against $W_U$ has gone through the whole network.
- **"Bigger $d_{\text{model}}$ is only about embeddings."** $d_{\text{model}}$ is the width of the whole residual stream, so it sets the size of nearly every weight matrix in the model.

### Check yourself

1. What is the shape of GPT-2 small's embedding matrix, and how many parameters is that?
2. Why is multiplying a one-hot vector by $W_E$ never done explicitly?
3. With weight tying, how does the logit for token $j$ relate to token $j$'s input embedding?
4. A token appears only once in the training data. Why might weight tying help its embedding?
5. Why can't an input embedding distinguish two meanings of " bank"?

<details>
<summary>Answers</summary>

1. `(50257, 768)`, which is $50{,}257 \times 768 = 38{,}597{,}376 \approx 38.6$M parameters.
2. It selects one row, so indexing `W_E[id]` gives the same result with no wasted multiply-by-zero work and no $V$-sized vector in memory.
3. The logit is the dot product of the final hidden vector with token $j$'s embedding row: $\text{logit}_j = x \cdot W_E[j]$.
4. With tying, the row gets gradient from the output softmax at every position (pushing it down when it is not the answer), not just on the one occasion it appears as input.
5. The lookup depends only on the token ID, which is the same in both cases; only attention to surrounding tokens can disambiguate.

</details>

### Resources

**Read**
- [Jay Alammar, "The Illustrated Word2vec"](https://jalammar.github.io/illustrated-word2vec/) (start here): builds intuition for what embedding vectors and their directions represent.
- [Press and Wolf, "Using the Output Embedding to Improve Language Models" (arXiv:1608.05859)](https://arxiv.org/abs/1608.05859): the weight-tying paper; sections 1 to 3 are short.
- [Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (arXiv:1301.3781)](https://arxiv.org/abs/1301.3781): the original word2vec paper with the vector arithmetic results.

**Watch**
- [3Blue1Brown, "But what is a GPT?"](https://www.3blue1brown.com/lessons/gpt): revisit the second half, which visualizes embeddings, directions, and the unembedding matrix.
- [Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY): the first 30 minutes builds a bigram model that is literally just an embedding table used as a logits table.

---

## 04. Positional Encoding

### Attention does not know about order

Section 05 explains attention fully, but one property matters now. Attention computes, for each token, a weighted average of other tokens' vectors, with weights based on vector similarity. Nothing in that computation refers to *where* a token is. If you shuffle the input tokens, each token's output is the same as before, just shuffled along with it. Formally, attention is **permutation equivariant**: for any permutation matrix $P$, $\text{Attn}(PX) = P\,\text{Attn}(X)$. (People often say "permutation invariant"; the precise term is equivariant, but the consequence is the same.)

So without help, "dog bites man" and "man bites dog" look like the same bag of tokens. (The causal mask leaks a little order information, since position $t$ sees exactly $t$ tokens, and some models do learn to exploit it, but it is far too weak to rely on.) We need to inject position information. There are two broad families:

{{< fig "architecture/18" >}}

- **Absolute:** give each position $0, 1, 2, \dots$ its own signal and add it to the token embedding.
- **Relative:** make the attention score between two tokens depend on their *distance* $m - n$, not on their absolute positions.

Relative schemes matter because language is mostly translation invariant: the relationship between an adjective and the noun right after it is the same at position 5 and at position 5000.

### Sinusoidal encoding (original Transformer)

Vaswani et al. added a fixed vector $PE_{pos} \in \mathbb{R}^{d_{\text{model}}}$ to the embedding at each position:

$$
PE_{pos, 2i} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
PE_{pos, 2i+1} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- $pos$ is the token's position (0-indexed).
- $i$ runs from $0$ to $d_{\text{model}}/2 - 1$ and indexes pairs of dimensions.
- Each pair oscillates at its own frequency $\omega_i = 10000^{-2i/d_{\text{model}}}$, from 1 radian per position (fast, $i=0$) down to $1/10000$ (very slow).

Think of it like a clock with many hands: the fast hands distinguish neighbouring positions, the slow hands distinguish far-apart ones, and together they give every position a unique fingerprint. A nice property: $PE_{pos+k}$ is a fixed linear transformation (a rotation of each pair) of $PE_{pos}$, which in principle makes relative offsets easy to compute. No parameters are learned.

{{< fig "architecture/19" >}}

### Learned absolute positions (GPT-1, GPT-2, BERT)

Simpler still: a second embedding table $W_P \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}$, one learned row per position, added to the token embedding: $x_t = W_E[\text{token}_t] + W_P[t]$. GPT-2 uses $T_{\max} = 1024$, so `wpe` has shape `(1024, 768)`. It works well, but the model has never seen a row for position 1025, so it **cannot run past its training length at all**, and positions near the end get fewer training updates.

### RoPE: Rotary Position Embedding

RoPE (Su et al., 2021, popularized by EleutherAI's GPT-NeoX and then Llama) is the dominant choice today. It is absolute in how it is applied and relative in effect.

**The idea.** Do not add anything to the embeddings. Instead, inside attention, *rotate* the query and key vectors by an angle proportional to their position. Since the attention score is a dot product of a query and a key, and the dot product of two rotated vectors depends only on the *difference* of their rotation angles, the score automatically depends on relative position.

**In 2D first.** Treat a 2D vector as a complex number $q$. Rotating it by angle $m\theta$ is multiplication by $e^{i m \theta}$. For a query at position $m$ and a key at position $n$:

$$
\langle q e^{i m\theta},\; k e^{i n\theta} \rangle = \text{Re}\!\left[ q \, \overline{k} \, e^{i (m-n)\theta} \right]
$$

where $\overline{k}$ is the complex conjugate and $\text{Re}$ takes the real part. The absolute positions $m$ and $n$ appear only as $m - n$. That is the whole trick.

{{< fig "architecture/20" >}}

**In $d$ dimensions.** Split the $d_k$-dimensional query or key vector into $d_k/2$ pairs. Rotate pair $i$ by angle $m\theta_i$, using the same frequency ladder as sinusoidal encoding:

$$
\theta_i = b^{-2i/d_k}, \quad i = 0, \dots, d_k/2 - 1, \qquad
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
$$

- $b$ is the base, 10,000 in the original paper (Llama 3 uses 500,000).
- $m$ is the position of the token whose query (or key) is being rotated.
- The same rotation is applied to keys with their own position $n$. Values are **not** rotated.

Then $q'_m \cdot k'_n$ depends on $q$, $k$ and $m - n$ only. Properties worth knowing:

- No extra parameters, and rotation preserves vector length, so it does not distort magnitudes.
- Low-index pairs rotate fast and encode fine-grained local position; high-index pairs rotate slowly and act almost like position-free channels that attention can use for content matching over long distances.
- It is applied per head, to $q$ and $k$ after their projections, at every layer.

Implementation note: there are two pairing conventions. The paper pairs adjacent dimensions $(0,1), (2,3), \dots$. GPT-NeoX and Llama pair the first half with the second half, $(0, d/2), (1, d/2+1), \dots$, via a "rotate_half" helper. They are equivalent up to a fixed permutation of dimensions, but weights trained with one convention give garbage with the other, a classic bug when porting checkpoints.

### ALiBi

ALiBi (Press, Smith and Lewis, 2021, "Train Short, Test Long") uses no position embeddings at all. It adds a fixed penalty to each attention score proportional to the distance between query and key:

$$
\text{score}(m, n) = \frac{q_m \cdot k_n}{\sqrt{d_k}} - s_h \,(m - n), \qquad n \le m
$$

where $s_h$ is a fixed slope for head $h$, taken from a geometric sequence (for 8 heads: $1/2, 1/4, \dots, 1/256$). Distant tokens get lower scores, more strongly in heads with steep slopes, so some heads focus locally and others look far. Its selling point was extrapolating to sequences longer than those seen in training without any fine-tuning. It was used in BLOOM and MPT, but RoPE plus context-extension methods (below) won out in practice.

{{< fig "architecture/21" >}}

### Extending the context window

Models are pre-trained on a fixed length (say 4k or 8k tokens) because attention is expensive. Users want 128k or more. With RoPE, positions beyond the training length produce rotation angles the model has never seen in the slow-rotating pairs, and quality collapses. Fixes, roughly in historical order:

- **Position Interpolation (PI)** (Chen et al., 2023). To go from training length $L$ to $L' = sL$, use position $m/s$ instead of $m$. All angles now stay inside the trained range, at the cost of squeezing nearby tokens closer together. A short fine-tune (about 1000 steps) recovers quality.
- **NTK-aware scaling** (community, 2023). Instead of squeezing all frequencies equally, increase the base $b$. This stretches slow (low-frequency) dimensions a lot and leaves fast (high-frequency) ones almost unchanged, preserving local resolution.
- **YaRN** (Peng et al., 2023). Refines this per frequency: dimensions whose wavelength is short relative to the context are not interpolated at all, dimensions with wavelength longer than the context are fully interpolated, and a ramp blends the ones in between. It also multiplies attention logits by a small temperature factor, because interpolation makes the softmax distribution flatter. YaRN extends context with a small fraction of the original pre-training compute and is used by many open models (Qwen, DeepSeek).
- **Base frequency adjustment** in pre-training: Llama 3 uses $b = 500{,}000$ from the start and extends context in a final training stage; Code Llama raised the base to 1,000,000 for 100k-token contexts.

{{< fig "architecture/22" >}}

{{< fig "architecture/23" >}}

### Snippet: sinusoidal encoding and RoPE's relative property

```python
import torch

def sinusoidal(T, d):
    pos = torch.arange(T).unsqueeze(1)                       # (T, 1)
    freq = 10000 ** (-torch.arange(0, d, 2) / d)             # (d/2,)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe                                                # (T, d)

def rope(x, pos, base=10000.0):
    # x: (..., d) with d even; pairs are (0, d/2), (1, d/2+1), ... as in Llama
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)   # (d/2,)
    ang = pos * theta                                                   # (d/2,)
    cos, sin = torch.cos(ang).repeat(2), torch.sin(ang).repeat(2)       # (d,)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated_half = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated_half * sin

print(sinusoidal(6, 8).round(decimals=2))

torch.manual_seed(0)
q, k = torch.randn(64), torch.randn(64)
# Same offset (3) at very different absolute positions gives the same score.
s1 = rope(q, 10) @ rope(k, 7)
s2 = rope(q, 1010) @ rope(k, 1007)
print(s1.item(), s2.item())                 # equal up to float error
print((rope(q, 10).norm() - q.norm()).item())  # ~0: rotation keeps length
```

### Common confusions

- **"RoPE is added to the embeddings like sinusoidal encoding."** RoPE is applied inside every attention layer, to queries and keys only. Nothing is added to the residual stream.
- **"Relative position encoding means the model can't tell absolute position."** It mostly can't directly, but the causal mask and things like the start-of-sequence token give it anchors; in practice this has not been a problem.
- **"Context extension is free."** PI and YaRN need at least some fine-tuning on long sequences to work well, and a model claiming 128k context may still use the far end of it poorly.
- **"Longer context only needs a position fix."** Attention compute and KV cache memory also grow with length (section 05); position encoding is just one of the obstacles.
- **"Sinusoidal encoding extrapolates to any length."** It produces valid vectors at any position, but models trained with it still degrade beyond their training length because they have never learned to use those patterns.

### Check yourself

1. Why do transformers need positional information while RNNs do not?
2. Why can't GPT-2 process a sequence of 2000 tokens without modification?
3. In RoPE, which of $Q$, $K$, $V$ are rotated, and why does the attention score end up depending only on $m-n$?
4. Position Interpolation extends a 4k model to 16k. What position value is used for the token at index 12,000?
5. What problem with Position Interpolation do NTK-aware scaling and YaRN address?

<details>
<summary>Answers</summary>

1. An RNN processes tokens in order, so order is built into its computation. Attention treats inputs as an unordered set; shuffling inputs only shuffles outputs.
2. It uses a learned absolute position table with exactly 1024 rows; there is no learned vector for positions 1024 and beyond.
3. $Q$ and $K$. Rotating $q$ by $m\theta$ and $k$ by $n\theta$ and taking a dot product gives the same result as rotating by the difference $(m-n)\theta$, because rotations compose and the dot product only sees the relative angle.
4. $12{,}000 / 4 = 3{,}000$ (scale factor $s = 16\text{k}/4\text{k} = 4$).
5. PI compresses all frequencies equally, which blurs the high-frequency dimensions that encode fine local order. NTK-aware and YaRN interpolate mostly the low-frequency dimensions and leave the high-frequency ones nearly intact.

</details>

### Resources

**Read**
- [EleutherAI, "Rotary Embeddings: A Relative Revolution"](https://blog.eleuther.ai/rotary-embeddings/) (start here): derivation, intuition and comparisons, with code.
- [Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864): the RoPE paper; section 3 builds RoPE from the 2D case used above.
- [Press, Smith, Lewis, "Train Short, Test Long" (ALiBi, arXiv:2108.12409)](https://arxiv.org/abs/2108.12409): figure 3 explains ALiBi in one picture.
- [Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (arXiv:2306.15595)](https://arxiv.org/abs/2306.15595): the PI paper; its opening figures contrast extrapolation with interpolation.
- [Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (arXiv:2309.00071)](https://arxiv.org/abs/2309.00071): the methods section walks from PI through NTK-aware scaling to YaRN.

**Watch**
- [Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY): watch where learned position embeddings are added to the bigram model, to see the "position as a second lookup table" approach in code.

---

## 05. Attention

### The intuition: a soft dictionary lookup

At this point every token is a vector that knows its own identity (embedding) and, with RoPE or added position vectors, its position. But it knows nothing about its neighbours. In "The animal didn't cross the street because **it** was too tired", the vector for " it" needs information from " animal". Attention is the mechanism that moves information between positions.

Think of a Python dictionary, but fuzzy. Each token produces three vectors from its current state:

- A **query** $q$: "what am I looking for?" (for " it": something like "a noun this pronoun could refer to").
- A **key** $k$: "what do I contain, for the purpose of being found?" (for " animal": "I am an animal-type noun").
- A **value** $v$: "what information do I hand over if someone attends to me?"

Each token compares its query against every key it is allowed to see. A high dot product means a good match. The scores are turned into weights with softmax, and the token receives the weighted average of the values. Unlike a real dictionary that returns exactly one entry, attention returns a blend, which is what makes it differentiable and trainable.

{{< fig "architecture/24" >}}

{{< fig "architecture/25" >}}

### Scaled dot-product attention

Let $X \in \mathbb{R}^{T \times d_{\text{model}}}$ be the input (one row per token). Three learned matrices project it:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V, \qquad W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$

so $Q, K, V \in \mathbb{R}^{T \times d_k}$. Then:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
$$

Reading it piece by piece, with shapes:

- $Q K^\top$: `(T, d_k) @ (d_k, T) -> (T, T)`. Entry $(i, j)$ is $q_i \cdot k_j$, how much token $i$ wants to attend to token $j$.
- $\sqrt{d_k}$: scaling, explained below.
- $M \in \mathbb{R}^{T \times T}$: the mask, explained below.
- softmax is applied to each row separately, so row $i$ becomes a set of non-negative weights $a_{i1}, \dots, a_{iT}$ that sum to 1. This $(T, T)$ matrix is the **attention pattern**.
- Multiply by $V$: `(T, T) @ (T, d_k) -> (T, d_k)`. Output row $i$ is $\sum_j a_{ij} v_j$.

{{< fig "architecture/26" >}}

### Why divide by $\sqrt{d_k}$?

Suppose the entries of $q$ and $k$ are independent with mean 0 and variance 1 (roughly true at initialization with normalized inputs). Then $q \cdot k = \sum_{l=1}^{d_k} q_l k_l$ is a sum of $d_k$ terms each with variance 1, so its variance is $d_k$ and its standard deviation is $\sqrt{d_k}$. With $d_k = 64$, typical scores are around $\pm 8$, and with $d_k = 128$, around $\pm 11$.

Softmax of large-magnitude scores is nearly one-hot: one weight close to 1, the rest close to 0. In that regime the gradient of softmax is almost zero, so learning stalls. Dividing by $\sqrt{d_k}$ brings the scores back to unit variance regardless of head size, keeping softmax in a range where gradients flow.

{{< fig "architecture/28" >}}

### The causal mask

For next-token prediction, position $i$ must not see positions $j \gt  i$; otherwise during training it could just read the answer. The mask is

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j \gt  i \end{cases}
$$

Adding $-\infty$ before softmax makes $e^{-\infty} = 0$, so future positions get exactly zero weight. The allowed region is the lower triangle of the $(T, T)$ matrix. This is what lets one forward pass produce $T$ valid predictions at once: row $i$ only uses tokens $0..i$, exactly as it would during generation.

{{< fig "architecture/27" >}}

### Multi-head attention

One attention pattern can only express one kind of relationship at a time. A token might simultaneously need "the previous token", "the subject of this sentence" and "the matching open bracket". **Multi-head attention** runs $h$ attention operations in parallel, each with its own $W_Q, W_K, W_V$ of width $d_k = d_{\text{model}} / h$, then concatenates their outputs and mixes them with an output projection:

{{< fig "architecture/29" >}}

$$
\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}), \qquad
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

with $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. Because each head has width $d_{\text{model}}/h$, the total cost is about the same as one big head. In code, you do one big projection to `(B, T, d_model)`, reshape to `(B, h, T, d_k)`, run attention over the last two dims for all heads at once, and reshape back. GPT-2 small has $h = 12$ heads of size 64.

### MQA and GQA: shrinking the keys and values

During generation (next subsection), the keys and values of every past token must be kept in memory. With $h$ heads, that is $h$ keys and $h$ values per token per layer.

- **Multi-Query Attention (MQA)** (Shazeer, 2019): all query heads share a *single* key head and value head. KV memory shrinks by a factor of $h$. Quality drops somewhat.
- **Grouped-Query Attention (GQA)** (Ainslie et al., 2023): a middle ground. Split the $h$ query heads into $g$ groups; each group shares one K/V head. $g = h$ is ordinary multi-head attention, $g = 1$ is MQA. Llama 2 70B and Llama 3 use 8 KV heads; for Llama 3 8B that is 32 query heads sharing 8 KV heads, a 4x memory saving with negligible quality loss. GQA is the default in most current open models.

{{< fig "architecture/30" >}}

### The KV cache

Generation is sequential. Naively, to produce token $t+1$ you would run the model on all $t$ tokens again, recomputing the same keys and values for the prefix every step. But because of the causal mask, the keys and values of past tokens never change when new tokens are appended. So we **cache** them:

1. **Prefill:** run the whole prompt once, storing $K$ and $V$ for every layer.
2. **Decode:** for each new token, compute only its own $q, k, v$; append $k, v$ to the cache; attend from the single new query to all cached keys and values.

Each decode step then costs $O(t)$ attention work instead of recomputing everything. The price is memory. Per sequence:

{{< fig "architecture/31" >}}

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_k \times T \times \text{bytes per number}
$$

(the 2 is for K and V). Llama 2 7B (32 layers, 32 KV heads, $d_k$ = 128, fp16 = 2 bytes) at 4096 tokens: $2 \times 32 \times 32 \times 128 \times 4096 \times 2 \approx 2.1$ GB, for a single sequence. This is why GQA exists, and why long-context serving is memory-bound.

{{< fig "architecture/32" >}}

### Complexity: the $O(n^2)$ problem

For sequence length $n$ (we have been writing $T$), $Q K^\top$ is an $n \times n$ matrix. Compute is $O(n^2 \, d)$ per layer and the attention matrix needs $O(n^2)$ memory per head. Double the context and attention costs four times as much. By contrast, the MLP and projections cost $O(n \, d^2)$, linear in $n$. For short sequences the MLP dominates; past a few thousand tokens (depending on $d$), attention does. This quadratic cost is the reason for context limits and for the research into sparse and linear attention covered in the advanced course.

{{< fig "architecture/33" >}}

### FlashAttention, at a high level

On a GPU, arithmetic is cheap and moving data between the large, slow main memory (HBM) and the small, fast on-chip memory (SRAM) is expensive. Standard attention writes the full $n \times n$ score matrix to HBM, reads it back for softmax, writes it again, and reads it again to multiply by $V$. For long sequences, that memory traffic, not the math, is the bottleneck.

**FlashAttention** (Dao et al., 2022) computes exactly the same result without ever materializing the full matrix. It splits $Q$, $K$, $V$ into tiles that fit in SRAM, and for each query tile it loops over key/value tiles, using an **online softmax** trick: keep a running maximum and running sum of exponentials, and rescale the partial output whenever a new maximum appears. The result is exact (not an approximation), memory use is $O(n)$ instead of $O(n^2)$, and it is 2 to 4 times faster. FLOPs are still $O(n^2)$: FlashAttention makes quadratic attention practical, it does not make it linear. In PyTorch you get it via `F.scaled_dot_product_attention`, which dispatches to a fused kernel when it can.

{{< fig "architecture/34" >}}

### Snippet: attention from scratch, checked against PyTorch

```python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, h, T, dk = 2, 4, 5, 16
q, k, v = (torch.randn(B, h, T, dk) for _ in range(3))

scores = q @ k.transpose(-2, -1) / math.sqrt(dk)         # (B, h, T, T)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~causal, float("-inf"))
weights = scores.softmax(dim=-1)                         # each row sums to 1
out = weights @ v                                        # (B, h, T, dk)

ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(out, ref, atol=1e-6))               # True
print(weights[0, 0].round(decimals=2))                   # lower-triangular pattern

# Why sqrt(dk): unscaled scores have std ~ sqrt(dk)
a, b = torch.randn(10000, 128), torch.randn(10000, 128)
print((a * b).sum(-1).std().item(), math.sqrt(128))

# GQA: 8 query heads share 2 KV heads by repeating each KV head 4 times
qg = torch.randn(1, 8, T, dk)
kg, vg = torch.randn(1, 2, T, dk), torch.randn(1, 2, T, dk)
kg, vg = kg.repeat_interleave(4, dim=1), vg.repeat_interleave(4, dim=1)
print(F.scaled_dot_product_attention(qg, kg, vg, is_causal=True).shape)  # (1, 8, 5, 16)
```

### Common confusions

- **"Q, K and V are different inputs."** In self-attention they are three different linear projections of the *same* input $X$.
- **"Attention weights tell you what the model thinks is important."** They show where information was read from in one head of one layer. Interpreting them requires care (section 06).
- **"The causal mask is only needed in training."** It is needed in any forward pass over multiple tokens, including prefill. With a KV cache during decode, the single new query only sees past keys anyway, so masking happens implicitly.
- **"FlashAttention is an approximation."** It is exact; it reorders the computation to reduce memory traffic.
- **"Multi-head attention costs $h$ times more."** Each head has width $d_{\text{model}}/h$, so total cost is about the same as one full-width head.
- **"The KV cache stores queries too."** Past queries are never needed again, only past keys and values.

### Check yourself

1. Write the shapes of $Q$, $K^\top$, $QK^\top$ and the output for a single head with $T = 10$, $d_k = 64$.
2. If $d_k = 256$ and you forget the $\sqrt{d_k}$ scaling, what happens to the softmax and to training?
3. Why can the KV cache reuse keys and values from previous steps without recomputing them?
4. A model has 40 layers, 8 KV heads, $d_k$ = 128 and uses fp16. How much KV cache does one 32k-token sequence need?
5. Does FlashAttention change the asymptotic FLOP count of attention? What does it change?

<details>
<summary>Answers</summary>

1. $Q$: `(10, 64)`, $K^\top$: `(64, 10)`, $QK^\top$: `(10, 10)`, output: `(10, 64)`.
2. Scores have standard deviation around 16, so softmax becomes nearly one-hot, its gradients vanish, and training is slow or unstable.
3. With causal masking, a past token's representation at every layer depends only on tokens at or before it, so appending new tokens never changes it; its $k$ and $v$ are fixed.
4. $2 \times 40 \times 8 \times 128 \times 32{,}768 \times 2$ bytes $= 5{,}368{,}709{,}120$ bytes, about 5.4 GB (5 GiB).
5. No, still $O(n^2 d)$. It reduces memory from $O(n^2)$ to $O(n)$ and cuts slow HBM reads and writes, making it much faster in wall-clock time.

</details>

### Resources

**Read**
- [Jay Alammar, "The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/): step-by-step pictures of Q, K, V and multi-head attention (the encoder-decoder parts can be skimmed).
- [Vaswani et al., "Attention Is All You Need" (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762): now read section 3.2 in full; footnote 4 is the $\sqrt{d_k}$ variance argument.
- [Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245): figure 2 compares MHA, GQA and MQA at a glance.
- [Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135): read section 1 and figure 1 for the memory-hierarchy argument.
- [Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (arXiv:1911.02150)](https://arxiv.org/abs/1911.02150): the MQA paper and a clear statement of why decoding is memory-bound.

**Watch**
- [3Blue1Brown, "Attention in transformers, step-by-step" (Deep Learning ch. 6)](https://www.3blue1brown.com/lessons/attention) (start here): the best visual explanation of queries, keys, values and masking.
- [Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY): the "self-attention" segment builds the masked weighted average with the tril trick, then multi-head attention, in code.

---

## 06. Layers of Understanding

### The transformer block

You now have every ingredient. A transformer block combines them:

$$
\begin{aligned}
x' &= x + \text{Attn}(\text{Norm}_1(x)) \\
x'' &= x' + \text{MLP}(\text{Norm}_2(x'))
\end{aligned}
$$

where $x \in \mathbb{R}^{T \times d_{\text{model}}}$ and every intermediate has the same shape. A model is a stack of $N$ identical blocks (each with its own weights), followed by a final normalization and the unembedding. GPT-2 small has $N = 12$; Llama 3 70B has 80.

{{< fig "architecture/35" >}}

### The residual stream

The `x + ...` structure is a **residual connection** (from ResNets, 2015). The most useful way to picture it, popularised by Anthropic's interpretability work, is as a **residual stream**: a $d_{\text{model}}$-wide channel running from the embedding at the bottom to the unembedding at the top, one per token position. Each attention layer and each MLP *reads* from the stream (through a normalization) and *adds* its output back in. No layer overwrites the stream; they all write small updates into a shared workspace.

{{< fig "architecture/36" >}}

Why this matters:

- **Gradients:** the derivative of $x + f(x)$ with respect to $x$ includes an identity term, so the gradient from the loss reaches early layers directly, without being multiplied through dozens of layers. This is what makes 100-layer networks trainable.
- **Communication:** a layer can write information into some subspace of the stream that a much later layer reads. Attention moves information *between* positions' streams; the MLP transforms information *within* one position's stream.
- **Default behaviour:** a freshly initialised block adds roughly nothing, so the network starts close to "embedding goes straight to unembedding" (a bigram model) and learns to add useful refinements.

### LayerNorm vs RMSNorm

Values in the residual stream can drift in scale as layers add to it. Normalization rescales each token's vector before a sub-layer reads it, which keeps training stable.

**LayerNorm** (Ba et al., 2016), for one token's vector $x \in \mathbb{R}^d$:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2, \quad
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

$\gamma, \beta \in \mathbb{R}^d$ are learned scale and shift; $\epsilon$ (about $10^{-5}$) avoids division by zero; $\odot$ is element-wise multiplication. Note it normalizes across the *features of one token*, not across the batch (unlike BatchNorm).

**RMSNorm** (Zhang and Sennrich, 2019) drops the mean subtraction and the shift:

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}
$$

It is cheaper (one reduction instead of two) and works just as well in practice. Llama, Mistral, Qwen, Gemma and most current models use RMSNorm. GPT-2 uses LayerNorm.

{{< fig "architecture/38" >}}

### Pre-norm vs post-norm

The original 2017 Transformer used **post-norm**: $x' = \text{Norm}(x + \text{Attn}(x))$. The normalization sits *on* the residual path, so the stream gets renormalized after every sub-layer. This trains poorly in deep models without a careful learning-rate warmup, because gradients near the output are much larger than near the input.

**Pre-norm** (GPT-2 onward, analysed by Xiong et al., 2020) normalizes only the *input to each sub-layer*: $x' = x + \text{Attn}(\text{Norm}(x))$. The residual path itself stays an untouched sum, so the identity gradient path is clean. Pre-norm needs one extra normalization at the very end (`ln_f` in GPT-2) since the stream itself is never normalized. Nearly every modern LLM is pre-norm. (Some recent models add extra norms, e.g. on the sub-layer output or on queries and keys, for stability at scale; the basic pre-norm structure stays.)

{{< fig "architecture/37" >}}

### The MLP (feed-forward network)

After attention has gathered context into each position, the MLP processes each position independently:

$$
\text{MLP}(x) = \phi(x W_1 + b_1)\, W_2 + b_2, \qquad W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}, \; W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}
$$

$\phi$ is a non-linear activation, and the hidden width is usually $d_{\text{ff}} = 4\,d_{\text{model}}$ (3072 for GPT-2 small). The MLP holds about two thirds of a dense transformer's parameters. A productive way to think about it (Geva et al., 2021): each of the $d_{\text{ff}}$ hidden neurons has an input pattern (a column of $W_1$, a "key") and an output pattern (a row of $W_2$, a "value"). When the residual stream matches the key, the neuron fires and adds its value to the stream. This makes MLPs look like a huge learned key-value memory, and experiments that locate and edit factual associations ("The Eiffel Tower is in Paris") find them largely in mid-layer MLPs.

{{< fig "architecture/39" >}}

### GELU and SwiGLU

**ReLU** is $\max(0, x)$. **GELU** (Hendrycks and Gimpel, 2016), used by GPT-2 and BERT, is $\text{GELU}(x) = x\,\Phi(x)$, where $\Phi$ is the standard normal cumulative distribution function: a smooth version of ReLU that lets small negative values through slightly. GPT-2 uses a tanh-based approximation, which matters when loading its weights.

**SwiGLU** (Shazeer, 2020) is a *gated* MLP:

$$
\text{SwiGLU}(x) = \big(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big)\, W_{\text{down}}, \qquad \text{SiLU}(z) = z \cdot \sigma(z)
$$

$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and $\sigma$ is the logistic sigmoid. One projection decides *how much* of each hidden unit to let through (the gate), the other provides *what* to let through. It has three matrices instead of two, so to keep the parameter count equal $d_{\text{ff}}$ is usually set to about $\frac{8}{3} d_{\text{model}}$ instead of $4\,d_{\text{model}}$. Shazeer reports consistently better quality for the same compute, and Llama, Mistral, Qwen and PaLM all use it. (His paper's conclusion famously attributes the gain to "divine benevolence": nobody has a crisp theory for why gating helps.)

### Stacking layers, and what they learn

More layers means more sequential steps of computation: a later layer can build on features computed by earlier ones. Probing studies (training small classifiers on each layer's activations to predict linguistic properties) and interpretability research paint a consistent rough picture:

- **Early layers** handle local, surface-level things: combining multi-token words into a single concept (" Michael", " Jordan" into "the basketball player"), part of speech, simple syntax, and "previous token" patterns.
- **Middle layers** carry most of the semantic and factual content: entity attributes, relations, and much of what gets retrieved when you ask a factual question.
- **Late layers** focus on the specific output: turning an internal representation into a distribution over the exact next token, including formatting and surface form.

These are tendencies, not hard boundaries, and the picture differs by model.

### The logit lens

Because of the residual stream, every layer's output lives in the same $d_{\text{model}}$-dimensional space that the final layer reads from. So you can take the stream *after layer $\ell$*, apply the final norm and unembedding as if the model stopped there, and see what it would predict. This is the **logit lens** (nostalgebraist, 2020). In GPT-2 you typically see the correct answer emerge in the middle layers and sharpen toward the end: the model forms a guess early and refines it. The trick works poorly on some other model families, which motivated the **tuned lens** (Belrose et al., 2023): train a small linear map per layer to translate the intermediate stream into the final layer's "language" first.

### Mechanistic interpretability, pointers

Mechanistic interpretability tries to reverse-engineer the algorithms inside the network. Key ideas to know by name:

- **Circuits:** specific heads and neurons that implement an identifiable algorithm. Anthropic's "A Mathematical Framework for Transformer Circuits" (2021) gives the residual-stream view used in this section.
- **Induction heads:** a two-head circuit that implements "if `A B` appeared earlier and now I see `A`, predict `B`". It appears suddenly during training and is a major mechanism behind in-context learning.
- **Superposition:** models represent more features than they have dimensions by using nearly orthogonal directions, which is why single neurons are often "polysemantic" (fire for unrelated things).
- **Sparse autoencoders (SAEs):** a tool to decompose activations into many interpretable features, scaled to production models in Anthropic's "Scaling Monosemanticity" (2024).

{{< fig "architecture/40" >}}

### Snippet: RMSNorm, a SwiGLU block, and a logit lens on GPT-2

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class LlamaStyleBlock(nn.Module):
    def __init__(self, d=256, n_head=4):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, bias=False, batch_first=True)
        self.mlp = SwiGLU(d, int(8 * d / 3))
    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # True = blocked
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))

print(LlamaStyleBlock()(torch.randn(2, 10, 256)).shape)   # (2, 10, 256)

# Logit lens on pretrained GPT-2 (downloads ~500MB the first time)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
ids = tok("The Eiffel Tower is located in the city of", return_tensors="pt").input_ids
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
for layer, h in enumerate(out.hidden_states):         # embeddings + 12 blocks
    # HF already applied ln_f to the last entry; applying it twice would distort it.
    h_last = h[0, -1] if layer == len(out.hidden_states) - 1 else model.transformer.ln_f(h[0, -1])
    top = model.lm_head(h_last).topk(3).indices
    print(layer, [tok.decode(t) for t in top])
```

The block above uses PyTorch's built-in `nn.MultiheadAttention` and no RoPE to stay short; section 07 writes attention by hand. Watch the logit-lens output: early layers predict generic continuations, and " Paris" should appear only in the later layers.

### Common confusions

- **"Each layer replaces the representation."** Each layer adds an update to the residual stream; the original embedding is still in there, summed with everything else.
- **"LayerNorm normalizes over the batch."** It normalizes each token's feature vector on its own. That is why it works the same at batch size 1 during inference.
- **"Pre-norm and post-norm are cosmetic."** Where the norm sits decides whether the residual path is a clean identity; it strongly affects trainability of deep models.
- **"The MLP is the less important part."** It holds about two thirds of the parameters and much of the factual knowledge.
- **"Neuron 1234 means 'cat'."** Because of superposition, most individual neurons respond to several unrelated features; meaningful units are usually directions, not single neurons.
- **"The logit lens shows what the model is thinking."** It shows what the final unembedding would read from an intermediate state that was never trained to be read that way. It is a useful but biased probe.

### Check yourself

1. Write the two equations of a pre-norm transformer block.
2. What does RMSNorm drop compared with LayerNorm, and why is that acceptable?
3. Why do pre-norm models need a final normalization before the unembedding?
4. A SwiGLU MLP with $d_{\text{model}}$ = 4096 uses $d_{\text{ff}}$ = 11008 (as in Llama 2 7B). How many parameters does it have (no biases)? Compare with a standard GELU MLP using $d_{\text{ff}} = 4 \times 4096$.
5. What does the logit lens rely on to make sense at all?

<details>
<summary>Answers</summary>

1. $x' = x + \text{Attn}(\text{Norm}_1(x))$ and $x'' = x' + \text{MLP}(\text{Norm}_2(x'))$.
2. It drops the mean subtraction and the learned bias $\beta$, normalizing only by root-mean-square. Empirically, re-centering contributes little; rescaling is what stabilizes training, and the simpler version is faster.
3. The residual stream itself is never normalized in pre-norm (only copies fed to sub-layers are), so its scale can grow through the layers; `ln_f` rescales it before scoring against the vocabulary.
4. SwiGLU: $3 \times 4096 \times 11008 = 135{,}266{,}304 \approx 135$M. GELU MLP: $2 \times 4096 \times 16384 = 134{,}217{,}728 \approx 134$M. Nearly identical, by design.
5. The residual stream: every layer reads from and writes to the same vector space, so intermediate states are roughly in the "format" the unembedding expects.

</details>

### Resources

**Read**
- [Elhage et al. (Anthropic), "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) (start here): the residual-stream view; read the "Transformer Overview" and "Residual stream" parts first.
- [nostalgebraist, "interpreting GPT: the logit lens"](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens): the original logit-lens post with layer-by-layer plots on GPT-2.
- [Shazeer, "GLU Variants Improve Transformer" (arXiv:2002.05202)](https://arxiv.org/abs/2002.05202): two pages; the source of SwiGLU.
- [Xiong et al., "On Layer Normalization in the Transformer Architecture" (arXiv:2002.04745)](https://arxiv.org/abs/2002.04745): why pre-norm trains without warmup; figure 1 shows both layouts.
- [Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (arXiv:2012.14913)](https://arxiv.org/abs/2012.14913): the key-value memory view of MLPs.
- [Olsson et al. (Anthropic), "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): the induction-head story; read the introduction and key concepts.

**Watch**
- [3Blue1Brown, "How might LLMs store facts" (Deep Learning ch. 7)](https://www.3blue1brown.com/lessons/mlp): a visual walk through the MLP block and superposition.

---

## 07. GPT-2 from Scratch

### The plan

You will now write GPT-2 yourself, using only things from sections 01 to 06, and prove it is correct by loading OpenAI's released weights into it and checking that it produces the same logits as the Hugging Face implementation. Then you will add sampling and train a small version from scratch. The work splits into five steps:

1. A config object holding the hyperparameters.
2. The model: causal self-attention, MLP, block, and the full GPT module (about 150 lines).
3. Loading the pretrained weights from Hugging Face.
4. Sampling: temperature, top-k, top-p.
5. A training loop on a small text.

Save the model code as `gpt2.py`; later snippets import from it.

### Step 1: the configuration

GPT-2 came in four sizes. All share the same code; only these numbers change:

| Model | `n_layer` | `n_head` | `n_embd` | Parameters |
|---|---|---|---|---|
| `gpt2` (small) | 12 | 12 | 768 | 124M |
| `gpt2-medium` | 24 | 16 | 1024 | 350M |
| `gpt2-large` | 36 | 20 | 1280 | 774M |
| `gpt2-xl` | 48 | 25 | 1600 | 1558M |

All use `vocab_size = 50257` and `block_size = 1024` (the maximum context). In terms of earlier sections, GPT-2 is: byte-level BPE (02), tied embeddings (03), learned absolute positions (04), multi-head causal attention with $d_k = 64$ (05), and pre-norm blocks with LayerNorm and a GELU MLP of width $4 \times$ `n_embd` (06).

{{< fig "architecture/41" >}}

Where the 124M parameters go (you will verify this number when the model loads):

- Token embedding `wte`: $50257 \times 768 = 38.6$M (shared with the LM head, so counted once).
- Position embedding `wpe`: $1024 \times 768 = 0.8$M.
- Each block: attention $768 \times 2304 + 768 \times 768$ plus biases $\approx 2.36$M; MLP $2 \times 768 \times 3072$ plus biases $\approx 4.72$M; two LayerNorms $\approx 3$k. Total $\approx 7.09$M per block, $\times 12 = 85.1$M.
- Final LayerNorm: 1.5k. Sum: 124,439,808.

{{< fig "architecture/42" >}}

### Step 2: the model

Design decisions worth noticing as you read:

- **Attribute names match Hugging Face's** (`transformer.wte`, `h.0.attn.c_attn`, `ln_f`, ...) so that the pretrained `state_dict` can be copied key by key.
- **`c_attn` computes Q, K and V in one matrix multiply** of shape `(768, 2304)`, then splits the result. Three separate projections would be equivalent but slower.
- **Attention is written out by hand** so you can see section 05's formula. For speed, the four lines from `att = ...` to `y = att @ v` can be replaced with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, which uses FlashAttention where available.
- **Initialization** follows GPT-2: normal with standard deviation 0.02, with the output projections of each residual branch scaled down by $1/\sqrt{2N}$ (there are $2N$ additions into the residual stream).
- **`forward` returns logits for every position** and, if targets are given, the cross-entropy loss from section 01 averaged over all $B \times T$ positions.

```python
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024   # max context length (number of learned positions)
    vocab_size: int = 50257  # GPT-2 BPE vocabulary
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # One matmul produces Q, K and V together; it is faster than three and
        # matches the layout of OpenAI's checkpoint.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.SCALE_INIT = True
        self.n_head = cfg.n_head
        self.drop = nn.Dropout(cfg.dropout)
        # Not a parameter and not saved: HF checkpoints do not contain it.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, hd) so every head attends independently.
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)           # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = att @ v                                               # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)          # re-merge heads
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.SCALE_INIT = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # GPT-2 was trained with the tanh approximation; exact GELU would
        # slightly change the outputs of the pretrained weights.
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Pre-norm: the residual stream x itself is never normalized, which
        # keeps a clean gradient path from the loss to the embeddings.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Names mirror Hugging Face's GPT2LMHeadModel so weights load by key.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "SCALE_INIT", False):
                # Each block adds two outputs into the residual stream; scaling
                # keeps the stream's variance from growing with depth.
                std *= (2 * self.cfg.n_layer) ** -0.5
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)  # (B, T, C)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))            # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, name="gpt2"):
        from transformers import GPT2LMHeadModel
        sizes = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        model = cls(GPTConfig(**sizes[name]))
        sd = model.state_dict()
        sd_hf = GPT2LMHeadModel.from_pretrained(name).state_dict()
        # OpenAI stored these as Conv1D, i.e. (in, out); nn.Linear wants (out, in).
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        for k, v in sd_hf.items():
            if k.endswith((".attn.bias", ".attn.masked_bias")):
                continue  # HF's own causal-mask buffers, not weights
            src = v.t() if k.endswith(transposed) else v
            assert sd[k].shape == src.shape, k
            with torch.no_grad():
                sd[k].copy_(src)
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth] = float("-inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                # Drop a token if the tokens ranked above it already cover top_p,
                # so the most likely token always survives.
                drop = probs.cumsum(dim=-1) - probs > top_p
                sorted_logits[drop] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
```

Trace the shapes through `forward` once by hand with `B = 2`, `T = 5`, `n_embd = 768`, `n_head = 12`: `idx (2, 5)` becomes `x (2, 5, 768)`, inside attention `q (2, 12, 5, 64)` and `att (2, 12, 5, 5)`, each block preserves `(2, 5, 768)`, and `logits` are `(2, 5, 50257)`.

{{< fig "architecture/43" >}}

### Step 3: load the pretrained weights and check them

OpenAI trained GPT-2 in TensorFlow and its linear layers were stored as "Conv1D" modules whose weight matrices have shape `(in, out)`. PyTorch's `nn.Linear` stores `(out, in)`. That is why `from_pretrained` transposes exactly four kinds of weights, and why a shape assertion guards each copy. The Hugging Face checkpoint also contains causal-mask buffers named `attn.bias` (not to be confused with the `c_attn.bias` weights), which we skip because our model builds its own mask.

{{< fig "architecture/44" >}}

```python
import torch
import tiktoken
from transformers import GPT2LMHeadModel
from gpt2 import GPT

model = GPT.from_pretrained("gpt2").eval()
print(sum(p.numel() for p in model.parameters()))      # 124439808

# Correctness check: our logits should match Hugging Face's to float precision.
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
enc = tiktoken.get_encoding("gpt2")
idx = torch.tensor([enc.encode("The capital of France is")])
with torch.no_grad():
    ours, _ = model(idx)
    theirs = hf(idx).logits
print((ours - theirs).abs().max().item())               # around 1e-4 or smaller

# Greedy prediction for the next token
print(repr(enc.decode([ours[0, -1].argmax().item()])))
```

If the maximum difference is large, the usual culprits are: a missing transpose, exact GELU instead of `approximate="tanh"`, a forgotten final `ln_f`, or the mask not being applied.

### Step 4: sampling

The model gives logits $z \in \mathbb{R}^V$ for the next token. Choosing a token from them is a separate decision with large effects on output quality.

**Greedy decoding** takes $\arg\max_j z_j$. It is deterministic and tends to fall into repetitive loops, because once a phrase is likely it stays likely.

**Temperature** $\tau \gt  0$ divides the logits before softmax:

$$
p_j = \frac{\exp(z_j / \tau)}{\sum_{k=1}^{V} \exp(z_k / \tau)}
$$

$\tau = 1$ is the model's own distribution. $\tau \lt  1$ sharpens it (approaching greedy as $\tau \to 0$); $\tau \gt  1$ flattens it (approaching uniform random tokens as $\tau \to \infty$). Typical values are 0.7 to 1.0.

{{< fig "architecture/45" >}}

**Top-k** keeps only the $k$ highest logits, sets the rest to $-\infty$, and renormalizes. This removes the long tail of individually unlikely but collectively likely junk tokens. Its weakness is that $k$ is fixed: when the model is confident, 50 candidates is too many, and when many continuations are plausible, it may be too few.

**Top-p (nucleus) sampling** (Holtzman et al., 2019) adapts the candidate set: sort tokens by probability and keep the smallest set whose cumulative probability reaches $p$ (for example 0.9). When the model is confident, that set may be one or two tokens; when it is uncertain, it may be hundreds. The implementation keeps a token if the probability mass ranked *above* it is still below $p$, which guarantees the top token always survives.

{{< fig "architecture/46" >}}

In `generate`, the order is: crop the context to `block_size`, take the last position's logits, apply temperature, then top-k, then top-p, then sample with `torch.multinomial`. Note that `generate` recomputes the whole context every step: it has no KV cache. Adding one is part of the chapter project.

```python
torch.manual_seed(42)
prompt = torch.tensor([enc.encode("In a shocking finding, scientists discovered")])
for settings in [dict(temperature=1e-6),                 # effectively greedy
                 dict(temperature=0.8, top_k=50),
                 dict(temperature=1.0, top_p=0.9),
                 dict(temperature=1.5)]:                  # too hot: watch it fall apart
    out = model.generate(prompt, max_new_tokens=30, **settings)
    print(settings, "\n", enc.decode(out[0].tolist()), "\n")
```

### Step 5: train a tiny GPT

Finally, train the same architecture from random initialization. To make it fast enough for a laptop CPU, use a 0.8M-parameter configuration and character-level tokens on Tiny Shakespeare (about 1MB of text). Every training example is a window of `block_size` characters, with targets equal to the same window shifted one character to the right, so a single window gives `block_size` next-token predictions (section 01).

{{< fig "architecture/47" >}}

```python
import os
import urllib.request

import torch

from gpt2 import GPT, GPTConfig

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(url, "input.txt")
text = open("input.txt").read()

# Character-level keeps the vocab at ~65, so a tiny model learns fast on a CPU.
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)

data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
cfg = GPTConfig(block_size=128, vocab_size=len(chars), n_layer=4, n_head=4, n_embd=128, dropout=0.1)
model = GPT(cfg).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {device}")

def get_batch(split, B=32):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (B,))
    x = torch.stack([d[i : i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + cfg.block_size] for i in ix])  # shifted by one: next-token targets
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(iters=20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = sum(model(*get_batch(split))[1].item() for _ in range(iters)) / iters
    model.train()
    return out

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
max_steps = int(os.environ.get("STEPS", 2000))
for step in range(max_steps + 1):
    if step % 250 == 0:
        losses = estimate_loss()
        print(f"step {step}: train {losses['train']:.3f}, val {losses['val']:.3f}")
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    # Clipping guards against the occasional huge gradient early in training.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

model.eval()
start = torch.tensor([encode("ROMEO:")], device=device)
print(decode(model.generate(start, 300, temperature=0.8, top_k=20)[0].tolist()))
```

On a laptop CPU this takes a few minutes. Expect the loss to start near $\ln 65 \approx 4.17$ (a uniform guess over 65 characters), fall below 2.0 within about 1000 steps, and keep improving slowly. The samples go from random characters to word-like strings with character names, line breaks and punctuation in the right places. That progression is the same thing the 124M model did, at a much larger scale, on 40GB of web text.

Things to try once it runs: switch to `tiktoken`'s GPT-2 encoding (vocab 50257, so the embedding dominates the parameter count), change `n_layer` and `n_embd` and compare validation loss, and set `dropout=0.0` to see overfitting appear as the gap between train and validation loss.

### Common confusions

- **"Temperature changes what the model knows."** It only reshapes the output distribution at sampling time; the logits are unchanged.
- **"Top-k and top-p are alternatives you must choose between."** They compose; many APIs apply both, and temperature is applied before either.
- **"Matching Hugging Face's logits exactly to 0 is required."** Different operation orders give tiny float differences (around $10^{-5}$ to $10^{-4}$). Differences of 0.1 or more mean a real bug.
- **"Training loss going down means the model is getting better."** Watch validation loss. If train loss keeps falling while validation loss rises, the model is memorizing.
- **"The trained character model is a small GPT-2."** It is the same architecture with a different tokenizer and far fewer parameters; it cannot use GPT-2's weights because its vocabulary and sizes differ.

### Check yourself

1. Why does `from_pretrained` transpose some weights but not others (for example, not `wte` or the LayerNorm weights)?
2. What is the expected initial loss of the character model, and why?
3. With $\tau = 0.5$, how does the probability ratio between two tokens with logits 2.0 and 1.0 change compared with $\tau = 1$?
4. Why does the top-p implementation subtract each token's own probability before comparing the cumulative sum with $p$?
5. Name two changes that would make `generate` much faster for long outputs.

<details>
<summary>Answers</summary>

1. Only the four linear layers stored as Conv1D in the original checkpoint have `(in, out)` layout; embeddings and LayerNorm parameters are stored the same way in both implementations.
2. About $\ln(65) \approx 4.17$: at random initialization the model's output is close to uniform over the 65 characters, and cross-entropy of a uniform guess is $\ln V$.
3. At $\tau = 1$ the ratio is $e^{2-1} = e \approx 2.7$. At $\tau = 0.5$ it is $e^{(2-1)/0.5} = e^2 \approx 7.4$: the more likely token becomes relatively more likely.
4. So that the token that crosses the threshold is kept, and in particular the top token always survives even if it alone has probability above $p$.
5. A KV cache (compute only the new token's $q, k, v$ each step instead of rerunning the whole context), and using `F.scaled_dot_product_attention` (fused FlashAttention kernels). Running in half precision on a GPU and batching several sequences also help.

</details>

### Resources

**Read**
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT): the reference implementation this section follows closely; read `model.py` after writing your own.
- [Jay Alammar, "The Illustrated GPT-2"](https://jalammar.github.io/illustrated-gpt2/): pictures of GPT-2's exact data flow, including how the weight matrices are laid out.
- [Holtzman et al., "The Curious Case of Neural Text Degeneration" (arXiv:1904.09751)](https://arxiv.org/abs/1904.09751): the nucleus (top-p) sampling paper; figure 1 shows why greedy and beam search degenerate.
- [Sebastian Raschka, *Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) and its [code repository](https://github.com/rasbt/LLMs-from-scratch): chapters 4 and 5 build and pretrain the same GPT-2 architecture and load OpenAI's weights, with more explanation than nanoGPT.

**Watch**
- [Andrej Karpathy, "Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) (start here): four hours going from an empty file to a trained GPT-2; the first hour is exactly steps 1 to 4 of this section.
- [Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY): the same build on Tiny Shakespeare, from a bigram model upward; ideal companion for step 5.

---

## Chapter project

**Build "picoGPT-plus": upgrade your GPT-2 to a modern architecture and measure every change.**

Starting from your `gpt2.py` and the Tiny Shakespeare training loop:

1. **Baseline.** Train the 0.8M character model for 3000 steps with a fixed seed. Record validation loss and a sample. Then switch to `tiktoken`'s GPT-2 tokenizer and a model with `n_embd=192`, and record the same. Note how many tokens the dataset becomes in each case (section 02).
2. **Modernize, one change at a time** (re-train and record validation loss after each, so you can attribute any gain or loss):
   - LayerNorm to RMSNorm (section 06).
   - GELU MLP to SwiGLU with $d_{\text{ff}} \approx \frac{8}{3} d_{\text{model}}$ (section 06).
   - Learned position embeddings to RoPE applied to $q$ and $k$ (section 04). Delete `wpe`.
   - Multi-head attention to grouped-query attention with 2 KV heads (section 05).
3. **KV cache.** Add a cache to `generate` (section 05). Verify that cached and uncached generation produce identical tokens with greedy decoding, and time both for 500 new tokens.
4. **Length extrapolation.** Train with `block_size=128`, then generate with a 256-token context. Compare learned positions (impossible without changes), plain RoPE, and RoPE with position interpolation (scale positions by 0.5).
5. **Logit lens.** Apply the logit lens to your trained model and to pretrained GPT-2 on the same prompt (section 06). At which layer does the right answer first appear in each?

Write up a short table: change, parameter count, validation loss, tokens per second. The goal is not the best number but being able to explain every row.

---

## Consolidated resources

**Papers**
- [Vaswani et al., "Attention Is All You Need" (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
- [Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Brown et al., "Language Models are Few-Shot Learners" (GPT-3, arXiv:2005.14165)](https://arxiv.org/abs/2005.14165)
- [Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (arXiv:1508.07909)](https://arxiv.org/abs/1508.07909)
- [Kudo and Richardson, "SentencePiece" (arXiv:1808.06226)](https://arxiv.org/abs/1808.06226)
- [Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (arXiv:1301.3781)](https://arxiv.org/abs/1301.3781)
- [Press and Wolf, "Using the Output Embedding to Improve Language Models" (arXiv:1608.05859)](https://arxiv.org/abs/1608.05859)
- [Su et al., "RoFormer" (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864)
- [Press et al., "Train Short, Test Long" (ALiBi, arXiv:2108.12409)](https://arxiv.org/abs/2108.12409)
- [Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (arXiv:2306.15595)](https://arxiv.org/abs/2306.15595)
- [Peng et al., "YaRN" (arXiv:2309.00071)](https://arxiv.org/abs/2309.00071)
- [Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA, arXiv:1911.02150)](https://arxiv.org/abs/1911.02150)
- [Ainslie et al., "GQA" (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
- [Dao et al., "FlashAttention" (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)
- [Xiong et al., "On Layer Normalization in the Transformer Architecture" (arXiv:2002.04745)](https://arxiv.org/abs/2002.04745)
- [Shazeer, "GLU Variants Improve Transformer" (arXiv:2002.05202)](https://arxiv.org/abs/2002.05202)
- [Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (arXiv:2012.14913)](https://arxiv.org/abs/2012.14913)
- [Holtzman et al., "The Curious Case of Neural Text Degeneration" (arXiv:1904.09751)](https://arxiv.org/abs/1904.09751)

**Blogs, docs and code**
- [Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)
- [Christopher Olah, "Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [EleutherAI, "Rotary Embeddings: A Relative Revolution"](https://blog.eleuther.ai/rotary-embeddings/)
- [Lilian Weng, "The Transformer Family Version 2.0"](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/): a dense survey of attention and architecture variants, good for review after this chapter
- [Anthropic, "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) and ["In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [nostalgebraist, "interpreting GPT: the logit lens"](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- [Hugging Face LLM Course, chapter 6 (tokenizers)](https://huggingface.co/learn/llm-course/chapter6/5)
- [karpathy/minbpe](https://github.com/karpathy/minbpe), [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT), [openai/tiktoken](https://github.com/openai/tiktoken)
- [Sebastian Raschka, *Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) ([code](https://github.com/rasbt/LLMs-from-scratch))

**Videos**
- [Andrej Karpathy, Neural Networks: Zero to Hero (series)](https://karpathy.ai/zero-to-hero.html)
- [Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Andrej Karpathy, "Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE)
- [Andrej Karpathy, "Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU)
- 3Blue1Brown Deep Learning chapters 5 to 7: ["But what is a GPT?"](https://www.3blue1brown.com/lessons/gpt), ["Attention in transformers, step-by-step"](https://www.3blue1brown.com/lessons/attention), ["How might LLMs store facts"](https://www.3blue1brown.com/lessons/mlp)

---

next: [pre-training](/posts/learn-ai-02-pretraining/)
