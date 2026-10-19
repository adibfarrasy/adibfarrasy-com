---
title: "learn ai fundamentals, 9: reinforcement learning"
date: 2026-10-18T09:00:00+07:00
description: "rl fundamentals, rlhf, and reasoning-model training with grpo"
tags: ["llm", "learn-ai"]
math: true
figures: true
draft: false
---

part 9 of 9 in [learn ai fundamentals](/learn-ai/). [visual version with diagrams](/learn-ai/09-reinforcement-learning.html).

In Fundamentals you met RLHF and PPO only as intuition, and saw DPO as a way to skip RL altogether. Then reasoning models (o1, DeepSeek-R1) arrived and RL came back to the center of post-training. This chapter gives you the RL you need to read those papers properly: first the general theory, derived from scratch and mapped onto language models, then the RLHF and reasoning-RL algorithms built on it.

## What you will learn

- The MDP formalism and how a language model generating text fits it.
- Value functions, Q functions and the Bellman equations.
- The policy gradient theorem derived, REINFORCE, baselines, advantages, actor-critic and GAE.
- Why trust regions matter (TRPO) and how PPO's clipped objective approximates one.
- The full InstructGPT RLHF pipeline, the KL penalty, and the four models PPO keeps in memory.
- Reward hacking and the scaling laws of reward model overoptimization.
- RLAIF, RLVR and the reasoning-RL turn: GRPO derived and compared with PPO, plus RLOO, Dr. GRPO and DAPO.
- Process vs outcome reward models, test-time compute, and the main open-source frameworks.
- How to implement REINFORCE on a toy problem and GRPO's advantage and loss in PyTorch.

## Prerequisites

From LLM Fundamentals, review:

- **Pretraining objective**: log-probabilities of tokens under the model are the central quantity in this chapter.
- **SFT** and **reward models**: RLHF starts from both.
- **DPO and preference optimization**: we will contrast online RL with offline preference methods.
- **Tulu 3 incl. RLVR**: the high-level picture we now make precise.
- **Attention/KV cache**: generation (rollouts) dominates the cost of RL for LLMs.

Basic probability (expectations, the log-derivative $\nabla p = p \nabla \log p$) is assumed.

---

## 01 RL Fundamentals

### 1.1 Intuition: learning from consequences

Supervised learning tells the model the right answer for each input. Reinforcement learning only tells it *how good* its choices turned out to be, often much later. An agent acts in an environment, receives rewards, and must figure out which of its actions deserved credit.

For a language model, this is a natural fit for things we can judge but cannot easily demonstrate: "was this answer helpful?", "is this proof correct?", "did this code pass the tests?". We cannot write the perfect response for every prompt, but we can score one. RL turns scores into gradients.

{{< fig "reinforcement-learning/01" >}}

Three properties distinguish RL from supervised learning, and all three show up in LLM training:

1. **Data depends on the policy.** The model generates its own training data (rollouts). Change the model, and the data distribution changes.
2. **Delayed credit.** The reward for a 500-token answer arrives at the end. Which tokens deserve it?
3. **Exploration.** The model only learns about outputs it actually samples. If it never tries a better strategy, it never gets rewarded for it.

{{< fig "reinforcement-learning/02" >}}

### 1.2 Markov decision processes

An MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: set of states $s$.
- $\mathcal{A}$: set of actions $a$.
- $P(s' \mid s, a)$: transition probability of landing in state $s'$ after taking action $a$ in state $s$.
- $R(s, a)$: reward received for taking $a$ in $s$ (written $r_t$ at time $t$).
- $\gamma \in [0, 1]$: discount factor, how much future reward is worth relative to immediate reward.

"Markov" means the next state depends only on the current state and action, not on the history before it. An agent interacts for an **episode**: $s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T$. The sequence $\tau = (s_0, a_0, s_1, a_1, \dots)$ is a **trajectory**.

The **return** from time $t$ is the discounted sum of future rewards:

$$
G_t = \sum_{k=0}^{T - t - 1} \gamma^k \, r_{t+k}
$$

- $G_t$: return from step $t$ onward.
- $r_{t+k}$: reward at step $t + k$.
- $\gamma^k$: discount applied to a reward $k$ steps in the future.

Discounting does two jobs: it keeps returns finite in infinite-horizon problems and it expresses a preference for sooner reward. In finite episodes with a single terminal reward, $\gamma = 1$ is common.

**Mapping onto LLMs.** This mapping is the key to everything that follows:

| RL concept | LLM generation |
| --- | --- |
| State $s_t$ | The prompt plus all tokens generated so far, $(x, y_{\lt t})$ |
| Action $a_t$ | The next token $y_t$, chosen from the vocabulary (~100k-200k actions) |
| Transition $P$ | Deterministic: append $y_t$ to the context |
| Policy $\pi_\theta(a \mid s)$ | The LM's next-token distribution $\pi_\theta(y_t \mid x, y_{\lt t})$ |
| Episode | One full response, ending at EOS or a length limit |
| Reward | Usually a single scalar at the end (reward model score, or "answer correct"), zero elsewhere |

Two features of this MDP are unusual. Transitions are deterministic and known (no world model needed), and rewards are **sparse and terminal**: one number per sequence. The state is the whole history, so the Markov property holds trivially. Many LLM RL algorithms treat the whole response as a single action (a contextual bandit), which we will see is a reasonable simplification.

### 1.3 Policies, value functions, Q functions

A **policy** $\pi(a \mid s)$ is a distribution over actions given a state. A parameterized policy $\pi_\theta$ is a neural network; for us, the LLM itself.

The **state-value function** is the expected return from a state when following $\pi$:

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi}\left[ G_t \mid s_t = s \right]
$$

The **action-value function** (Q function) conditions also on the first action:

$$
Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi}\left[ G_t \mid s_t = s, a_t = a \right]
$$

The **advantage** measures how much better action $a$ is than the policy's average behavior in $s$:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

- $\mathbb{E}_{\tau \sim \pi}$: expectation over trajectories generated by running $\pi$ in the environment.
- $V^\pi(s)$: "how good is it to be here?"
- $Q^\pi(s, a)$: "how good is it to do $a$ here, then follow $\pi$?"
- $A^\pi(s, a)$: positive if $a$ is better than average, negative if worse. Note $\mathbb{E}_{a \sim \pi}[A^\pi(s, a)] = 0$.

In LLM terms: $V(s)$ for a partial response is "how likely is this partial solution to end up correct?"; $A(s, a)$ for a token is "did this token make success more or less likely than what the model would typically write here?".

{{< fig "reinforcement-learning/03" >}}

### 1.4 Bellman equations

Value functions satisfy recursive consistency conditions, obtained by splitting the return into the first reward plus the rest:

$$
V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s), \, s' \sim P(\cdot \mid s, a)}\left[ R(s, a) + \gamma V^\pi(s') \right]
$$

$$
Q^\pi(s, a) = R(s, a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot \mid s, a), \, a' \sim \pi(\cdot \mid s')}\left[ Q^\pi(s', a') \right]
$$

- $s'$: next state; $a'$: next action.

The **optimal** value functions $V^*, Q^*$ satisfy the Bellman optimality equations, replacing the expectation over $\pi$ with a max:

$$
Q^*(s, a) = R(s, a) + \gamma \, \mathbb{E}_{s'}\left[ \max_{a'} Q^*(s', a') \right]
$$

These equations underpin value-based methods (Q-learning, DQN) and give us the **temporal-difference (TD) error**, which we will reuse in GAE:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

If $V$ were exact, $\mathbb{E}[\delta_t] = 0$. A nonzero $\delta_t$ says "this step went better (or worse) than $V$ predicted", which is a one-step estimate of the advantage.

Value-based methods are rarely used directly for LLMs: a max over 100k+ actions at every step, combined with a model that is already a good policy, makes policy-based methods the natural choice.

### 1.5 Exploration

An agent that always takes the action it currently thinks is best never discovers anything better. Classic remedies: $\epsilon$-greedy (random action with probability $\epsilon$), optimism, entropy bonuses. For LLMs, exploration comes from **sampling temperature** and from sampling multiple responses per prompt. A pretrained model is an excellent prior: it already proposes plausible actions, so exploration is about covering diverse reasoning paths, not random tokens.

The flip side is that RL for LLMs can only reinforce behaviors the model sometimes produces. If a prompt is so hard that all samples fail, the reward is constant and there is no learning signal. This "exploration bottleneck" is why curriculum and difficulty filtering matter in reasoning RL.

### 1.6 The policy gradient theorem, derived

We want to maximize expected return:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ R(\tau) \right] = \sum_\tau p_\theta(\tau) \, R(\tau)
$$

- $R(\tau)$: total (discounted) return of trajectory $\tau$.
- $p_\theta(\tau)$: probability of trajectory $\tau$ under policy $\pi_\theta$.

The trajectory probability factorizes:

$$
p_\theta(\tau) = \rho(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t)
$$

where $\rho(s_0)$ is the initial-state distribution (for LLMs, the prompt distribution).

**Step 1: the log-derivative trick.** Since $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$,

$$
\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta p_\theta(\tau) R(\tau) = \sum_\tau p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \nabla_\theta \log p_\theta(\tau) \, R(\tau) \right]
$$

This turned the gradient of an expectation into an expectation of something we can sample. We never differentiate through the reward, so the reward can be a non-differentiable black box: a unit test, a human click, a regex.

**Step 2: the dynamics drop out.** Take the log of the factorization:

$$
\log p_\theta(\tau) = \log \rho(s_0) + \sum_t \log \pi_\theta(a_t \mid s_t) + \sum_t \log P(s_{t+1} \mid s_t, a_t)
$$

Only the middle term depends on $\theta$, so

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

We do not need to know the environment's dynamics. That is why this is called a *model-free* method.

**Step 3: causality (reward-to-go).** An action at time $t$ cannot affect rewards received before $t$. Those terms contribute zero in expectation, so we can replace $R(\tau)$ with the return from $t$ onward:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t \right]
$$

This is the **policy gradient theorem** in the form used by **REINFORCE** (Williams, 1992). The algorithm:

1. Sample $N$ trajectories with the current policy.
2. For each step compute $G_t$.
3. Ascend $\frac{1}{N}\sum_{i}\sum_t \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i) \, G_t^i$.

{{< fig "reinforcement-learning/04" >}}

In code, you implement this as minimizing the "loss" $-\sum_t \log \pi_\theta(a_t \mid s_t) \, G_t$ with $G_t$ treated as a constant. It looks like supervised learning on the model's own samples, weighted by how well they did. With a terminal reward $r$ and $\gamma = 1$, every token's $G_t = r$: REINFORCE on an LLM is "SFT on your own samples, weighted by reward".

### 1.7 Baselines and variance reduction

REINFORCE is unbiased but has very high variance. Suppose all rewards are positive (say between 5 and 10). Then every sampled action gets pushed up, and only the *relative* sizes tell good from bad. The signal is buried in noise.

{{< fig "reinforcement-learning/05" >}}

**Baselines.** Subtract any function $b(s_t)$ that does not depend on the action:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, \big(G_t - b(s_t)\big) \right]
$$

This is still unbiased because

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \, b(s) \right] = b(s) \sum_a \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) = b(s) \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s) \nabla_\theta 1 = 0
$$

A good baseline is close to the expected return, so $G_t - b(s_t)$ is positive for better-than-expected actions and negative for worse ones. The natural choice is $b(s) = V^\pi(s)$, which makes the weight an estimate of the **advantage**:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^\pi(s_t, a_t) \right]
$$

Every algorithm in the rest of this chapter is this formula with a different way of estimating $A$. Keep that in mind: PPO, GRPO, RLOO, Dr. GRPO and DAPO differ mainly in (1) how they estimate the advantage and (2) how they constrain the update size.

### 1.8 Actor-critic and GAE

**Actor-critic** methods learn the baseline: an *actor* (the policy $\pi_\theta$) and a *critic* $V_\phi(s)$ trained by regression toward observed returns. The critic lets us estimate advantages without waiting for full returns, using TD errors.

There is a bias-variance trade-off in how far ahead we look:

- One-step: $\hat A_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. Low variance (one random reward), but biased if $V$ is wrong.
- Monte Carlo: $\hat A_t = G_t - V(s_t)$. Unbiased, but high variance (sums all future randomness).

**Generalized Advantage Estimation** (Schulman et al. 2015) interpolates with an exponentially weighted sum of TD errors:

$$
\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \, \delta_{t+l}
$$

- $\lambda \in [0, 1]$: the trade-off knob. $\lambda = 0$ gives the one-step TD advantage; $\lambda = 1$ gives the Monte Carlo advantage $G_t - V(s_t)$ (the $V$ terms telescope).
- $\delta_{t+l}$: TD error at step $t + l$.

It is computed with a backward recursion $\hat A_t = \delta_t + \gamma\lambda \hat A_{t+1}$. Common choices are $\gamma = 0.99$, $\lambda = 0.95$ in classic control, and $\gamma = 1$, $\lambda = 0.95$ or $1.0$ for LLMs.

{{< fig "reinforcement-learning/06" >}}

**For LLMs**, the critic is usually another transformer of similar size to the policy, with a scalar head predicting $V$ at every token. With a sparse terminal reward, $r_t = 0$ until the last token, so per-token advantages come entirely from the critic's changing predictions: if $V$ jumps after a token, that token gets credit. This is the main promise of a critic (token-level credit assignment) and also its main cost (another large model to train, which is hard to train well on sparse rewards).

### 1.9 Trust regions: TRPO intuition

Policy gradient gives a direction, not a step size. Too large a step can collapse the policy (for example, put all mass on one token), and since the policy generates its own data, a collapsed policy produces garbage data and cannot recover. In supervised learning a bad step is corrected by the next batch; in RL a bad step can poison all future batches.

We would also like to reuse a batch of rollouts for several gradient steps, since generation is expensive. After the first step, the data comes from the old policy $\pi_{\theta_{\text{old}}}$, not the current one. Importance sampling corrects for this:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}}\left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \, \hat A(s, a) \right]
$$

- $\pi_{\theta_{\text{old}}}$: the policy that generated the data.
- The ratio $\pi_\theta / \pi_{\theta_{\text{old}}}$ reweights old samples to estimate the objective for the new policy. Its gradient at $\theta = \theta_{\text{old}}$ equals the policy gradient.

This surrogate is only trustworthy near $\theta_{\text{old}}$: the advantages were estimated for the old policy and the state distribution has shifted. **TRPO** (Schulman et al. 2015) maximizes $L^{\text{CPI}}$ subject to a constraint on the average KL divergence, $\mathbb{E}_s[\mathrm{KL}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s))] \le \delta$, and solves it with conjugate gradients and a line search. It works, but it is complex and awkward with large networks and shared parameters.

### 1.10 PPO: the clipped objective, derived

PPO (Schulman et al. 2017) gets most of TRPO's benefit with a first-order objective. Write the ratio $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$. The clipped objective is

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\Big( \rho_t(\theta) \hat A_t, \;\; \mathrm{clip}\big(\rho_t(\theta), 1 - \varepsilon, 1 + \varepsilon\big) \hat A_t \Big) \right]
$$

- $\rho_t(\theta)$: probability ratio; equals 1 at the start of each update phase.
- $\hat A_t$: advantage estimate (usually GAE), treated as a constant.
- $\varepsilon$: clip range, typically 0.2.
- $\mathrm{clip}(\rho, 1 - \varepsilon, 1 + \varepsilon)$: limits $\rho$ to the interval $[1 - \varepsilon, 1 + \varepsilon]$.

How to derive the behavior, case by case:

- **$\hat A_t \gt  0$ (good action).** We want to increase $\rho_t$. The objective is $\min(\rho \hat A, \mathrm{clip}(\rho)\hat A)$. For $\rho \lt  1 + \varepsilon$ the two terms are equal (or the unclipped is smaller), so the gradient flows. Once $\rho \gt  1 + \varepsilon$, the clipped term is the min and it is constant in $\theta$, so the gradient is zero. We stop pushing the action up once it is already $1 + \varepsilon$ times more likely than before.
- **$\hat A_t \lt  0$ (bad action).** We want to decrease $\rho_t$. With negative $\hat A$, the min selects the clipped term when $\rho \lt  1 - \varepsilon$, giving zero gradient. We stop pushing down once the action is already $1 - \varepsilon$ times as likely.
- **The min makes it pessimistic.** If an update has gone the *wrong* way (for example $\hat A \gt  0$ but $\rho$ dropped below $1 - \varepsilon$), the unclipped term is smaller, so it is selected and the gradient still flows to fix the mistake. Clipping only removes the incentive to move further in the beneficial direction; it never hides a loss.

{{< fig "reinforcement-learning/07" >}}

The result is a lower bound on the unclipped surrogate that removes any incentive to move $\rho$ outside $[1 - \varepsilon, 1 + \varepsilon]$. This lets PPO run several epochs of minibatch SGD on each batch of rollouts safely. The full PPO loss adds a value-function regression term and often an entropy bonus:

$$
L^{\text{PPO}} = -L^{\text{CLIP}}(\theta) + c_1 \, \mathbb{E}_t\left[ (V_\phi(s_t) - \hat G_t)^2 \right] - c_2 \, \mathbb{E}_t\left[ \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \right]
$$

- $\hat G_t$: return target for the critic (often $\hat A_t + V_{\text{old}}(s_t)$).
- $\mathcal{H}$: entropy of the policy at $s_t$, encouraging exploration.
- $c_1, c_2$: weighting coefficients.

### 1.11 Code: REINFORCE on a toy sequence task

A toy that has the shape of LLM RL: the "policy" emits a sequence of $L$ tokens from a vocabulary of $V$, one token per step, conditioned on the previous token. The episode ends after $L$ tokens and gets a single terminal reward: the fraction of positions that match a hidden target. Sequence = episode, token = action, sparse terminal reward. Pure PyTorch, runs in seconds on a CPU.

{{< fig "reinforcement-learning/08" >}}

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
V, L, BATCH = 8, 6, 64
TARGET = torch.tensor([3, 1, 4, 1, 5, 2])
# All-positive rewards (like a reward model whose scores never go below zero) are where a missing baseline hurts most.
REWARD_SHIFT = 5.0
BOS = V  # an extra input id that marks the start, like a BOS token


class TinyPolicy(nn.Module):
    """Next-token distribution given (position, previous token): a minimal autoregressive LM."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, 32)
        self.pos = nn.Embedding(L, 32)
        self.head = nn.Linear(32, V)

    def forward(self, prev_tok, t):
        return self.head(torch.tanh(self.emb(prev_tok) + self.pos.weight[t]))


def rollout(policy):
    prev = torch.full((BATCH,), BOS)
    tokens, logps = [], []
    for t in range(L):
        dist = torch.distributions.Categorical(logits=policy(prev, t))
        tok = dist.sample()
        tokens.append(tok)
        logps.append(dist.log_prob(tok))
        prev = tok
    tokens = torch.stack(tokens, 1)
    # Reward only at the end, like a verifier scoring a finished answer.
    reward = (tokens == TARGET).float().mean(1) + REWARD_SHIFT
    return tokens, torch.stack(logps, 1), reward


def train(use_baseline: bool, steps=150):
    policy = TinyPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    baseline = 0.0
    for step in range(steps):
        _, logps, reward = rollout(policy)
        # With gamma=1 and a terminal reward, every token's return-to-go equals the final reward.
        adv = reward - baseline if use_baseline else reward
        loss = -(logps.sum(1) * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # A running mean is the simplest state-independent baseline; it is unbiased because it ignores the sampled actions.
        baseline = 0.9 * baseline + 0.1 * reward.mean().item()
    with torch.no_grad():
        return rollout(policy)[2].mean().item() - REWARD_SHIFT


if __name__ == "__main__":
    no_b = train(use_baseline=False)
    with_b = train(use_baseline=True)
    print(f"fraction of target matched, no baseline:   {no_b:.2f}")
    print(f"fraction of target matched, with baseline: {with_b:.2f}")
    assert with_b > 0.9 and with_b > no_b, "the baseline should solve the toy faster"
```

Without a baseline, every sampled sequence has a reward near 5 and gets pushed up; only the small differences between samples carry signal, so learning is slow and noisy. With the baseline, below-average sequences get a negative weight and the toy is solved within the same budget. Things to try: set `REWARD_SHIFT = 0` (the gap shrinks, since rewards of 0 already act as negatives relative to the good samples), and shrink `BATCH` and compare learning curves over several seeds; replace the running-mean baseline with the mean reward of the *other* samples in the batch (you have just invented RLOO, see 2.8); make the reward 1 only if the whole sequence matches (a harder, sparser reward, like math correctness) and observe the exploration problem.

### Common confusions (RL fundamentals)

- **"The reward is differentiated."** Never. Policy gradient differentiates log-probabilities of sampled actions; the reward is only a weight.
- **"A baseline biases the gradient."** Not if it does not depend on the action being scored. It changes variance only.
- **"$V$ and $Q$ are the same thing for LLMs because transitions are deterministic."** $Q(s, a) = r + V(s')$ when transitions are deterministic, but $V(s)$ still averages over the policy's future sampling. They remain different functions.
- **"PPO's clip limits the KL."** It removes the incentive to move ratios far from 1 on sampled actions; it does not hard-bound the KL, which can still drift over many epochs. Many implementations also monitor KL and stop early.
- **"Discounting is needed."** For finite episodes with terminal rewards, $\gamma = 1$ is standard for LLMs; discounting would penalize long correct answers.
- **"An episode is a token."** An episode is the full response. A token is one action within it.

### Check yourself (RL fundamentals)

1. In the LLM MDP, what are the state, the action, the transition function and the episode? Why is the transition function trivial?
2. Show that the environment dynamics $P(s' \mid s, a)$ do not appear in $\nabla_\theta \log p_\theta(\tau)$. Why is this important?
3. Prove that subtracting a state-dependent baseline leaves the policy gradient unbiased. What goes wrong if the baseline depends on the action?
4. With a terminal reward $r$, $\gamma = 1$ and no critic, what is the REINFORCE weight on each token? What does that imply about credit assignment?
5. In GAE, what do $\lambda = 0$ and $\lambda = 1$ correspond to, and what does each trade off?
6. For an action with $\hat A \gt  0$ whose ratio is already 1.35 with $\varepsilon = 0.2$, what is the PPO gradient for that sample, and why?

<details>
<summary>Answers</summary>

1. State: prompt plus tokens generated so far. Action: the next token. Transition: append the token (deterministic, known). Episode: one complete response up to EOS or max length. Trivial because the "environment" is just string concatenation; all randomness comes from the policy (and from the prompt distribution and reward).
2. $\log p_\theta(\tau) = \log\rho(s_0) + \sum_t \log\pi_\theta(a_t \mid s_t) + \sum_t \log P(s_{t+1} \mid s_t, a_t)$; only the policy terms depend on $\theta$. So we can estimate gradients from samples without modeling the environment (or the reward function).
3. $\mathbb{E}_{a\sim\pi}[\nabla\log\pi(a \mid s) b(s)] = b(s)\nabla\sum_a\pi(a \mid s) = b(s)\nabla 1 = 0$. If $b$ depends on $a$, it cannot be pulled out of the sum and the term no longer vanishes, so the gradient becomes biased.
4. Every token gets weight $r$ (or $r - b$). All tokens in a good response are reinforced equally, including irrelevant or even harmful ones; credit assignment relies on averaging over many samples.
5. $\lambda = 0$: one-step TD advantage $\delta_t$, low variance, biased by critic errors. $\lambda = 1$: Monte Carlo advantage $G_t - V(s_t)$, unbiased but high variance.
6. Zero. With $\hat A \gt  0$ and $\rho \gt  1 + \varepsilon = 1.2$, the min picks the clipped term $1.2\hat A$, which is constant in $\theta$. The update has already moved that action's probability far enough.

</details>

### Resources (RL fundamentals)

**Read**

- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) (start here): concise math of policy gradients, VPG and PPO with clean reference code.
- [Sutton and Barto, Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html): the textbook; chapters 3 (MDPs), 13 (policy gradient) are the essentials here.
- [Lilian Weng, Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/): one-page tour of the policy gradient family with derivations.
- [Schulman et al., High-Dimensional Continuous Control Using GAE (arXiv:1506.02438)](https://arxiv.org/abs/1506.02438): the GAE paper.
- [Schulman et al., Trust Region Policy Optimization (arXiv:1502.05477)](https://arxiv.org/abs/1502.05477): the trust-region argument PPO approximates.
- [Schulman et al., Proximal Policy Optimization Algorithms (arXiv:1707.06347)](https://arxiv.org/abs/1707.06347): the clipped objective.

**Watch**

- [David Silver, RL Course (DeepMind x UCL, 2015)](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ): the classic lecture series; lectures 1-3 and 7 cover this section. Slides at [davidsilver.uk/teaching](https://www.davidsilver.uk/teaching/).
- [David Silver, Lecture 7: Policy Gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs): the single most relevant lecture.
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction): hands-on units including REINFORCE and PPO from scratch.

---

## 02 RLHF

### 2.1 The full RLHF pipeline (InstructGPT)

InstructGPT (Ouyang et al. 2022) is the template every later system modifies. Three stages:

1. **SFT.** Fine-tune a pretrained model (GPT-3) on human-written demonstrations (~13k prompts). Gives a policy that follows instructions reasonably and serves as the starting point and reference.
2. **Reward model.** For ~33k prompts, sample several SFT outputs, have labelers rank them, and train a reward model $r_\phi(x, y)$ with the Bradley-Terry loss you saw in Fundamentals. InstructGPT used a 6B RM.
3. **RL with PPO.** Optimize the policy on ~31k prompts to maximize the RM score, with a KL penalty to the SFT model.

The headline result: labelers preferred outputs from the 1.3B InstructGPT model over the 175B GPT-3. Alignment training is far cheaper than the corresponding scale-up. InstructGPT also mixed in pretraining gradients during PPO ("PPO-ptx") to reduce regressions on public NLP benchmarks (the "alignment tax").

{{< fig "reinforcement-learning/09" >}}

### 2.2 The RLHF objective and the KL penalty

The objective is

$$
\max_{\theta} \; \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(\cdot \mid x)}\left[ r_\phi(x, y) \right] - \beta \, \mathbb{D}_{\text{KL}}\left( \pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x) \right)
$$

- $x$: prompt from the RL prompt set $\mathcal{D}$.
- $y$: sampled response.
- $r_\phi(x, y)$: reward model score.
- $\pi_{\text{ref}}$: the frozen reference policy (the SFT model).
- $\beta$: KL coefficient (often 0.01 to 0.1).

Why the KL term? The reward model is only accurate near the distribution it was trained on (SFT outputs). Without the penalty, the policy drifts to strange regions where the RM gives high scores for bad reasons (section 2.5). The KL keeps the policy close to fluent, sensible text, and it preserves output diversity. This is the same objective DPO solves in closed form; PPO solves it by sampling.

**Per-token KL vs sequence reward.** The sequence-level KL decomposes into a sum of per-token log-ratios: $\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \sum_t \log \frac{\pi_\theta(y_t \mid x, y_{\lt t})}{\pi_{\text{ref}}(y_t \mid x, y_{\lt t})}$. PPO-based RLHF folds the KL into the reward, per token:

$$
r_t = \begin{cases}
-\beta \log \dfrac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)} & t \lt  T \\[2ex]
r_\phi(x, y) - \beta \log \dfrac{\pi_\theta(y_T \mid s_T)}{\pi_{\text{ref}}(y_T \mid s_T)} & t = T
\end{cases}
$$

- $s_t = (x, y_{\lt t})$: state at token $t$; $T$: final token index.

So every token gets a small dense penalty for deviating from the reference, and the last token additionally receives the RM score. GAE then spreads credit backward. The sampled log-ratio is a single-sample estimate of the KL (it can be negative for an individual token; it is positive in expectation).

{{< fig "reinforcement-learning/10" >}}

### 2.3 PPO for LLMs: the four models

A PPO-RLHF step:

1. **Rollout**: sample responses $y$ for a batch of prompts from the current policy (the expensive part; uses inference engines like vLLM in modern systems).
2. **Score**: compute $r_\phi(x, y)$ with the RM, per-token log-probs under the policy and the reference, and values $V_\psi(s_t)$ with the critic.
3. **Advantages**: per-token rewards as above, then GAE.
4. **Update**: several epochs of minibatch PPO on policy (clipped objective) and critic (value regression).

{{< fig "reinforcement-learning/11" >}}

This requires **four models** in memory:

| Model | Role | Trained? |
| --- | --- | --- |
| Policy $\pi_\theta$ | Generates rollouts; being optimized | Yes |
| Reference $\pi_{\text{ref}}$ | KL anchor (frozen SFT model) | No |
| Reward model $r_\phi$ | Scores complete responses | No |
| Value model (critic) $V_\psi$ | Per-token value for GAE | Yes |

The two trained models also carry optimizer states (16 bytes/param, from Chapter 5). For a 70B policy with a 70B critic, this is several times the memory of SFT, plus a generation engine. This memory and engineering burden is a big reason DPO became popular, and why GRPO and RLOO drop the critic.

Practical details that matter: whiten (normalize) advantages per batch; initialize the critic from the RM; mask padding and prompt tokens out of every loss; monitor KL, response length, reward and value loss together; and use a large enough batch of prompts, since gradient noise is high.

### 2.4 Why RL instead of just DPO?

DPO optimizes the same KL-regularized objective using a fixed dataset of preference pairs. Online RL samples from the *current* policy, so it gets feedback on what the model does now, not on what the SFT model did when data was collected. Empirically, well-tuned online methods tend to edge out offline DPO for the best results, at higher cost and complexity, while DPO is a strong, cheap baseline. Tulu 3 used both: DPO for preferences, then RLVR for verifiable skills.

{{< fig "reinforcement-learning/12" >}}

### 2.5 Reward hacking and overoptimization

The RM is a learned proxy for what humans want. Optimize a proxy hard enough and you find its blind spots (Goodhart's law). Classic symptoms: responses get longer and longer (RMs often have a length bias), excessive hedging or flattery, lists and formatting for their own sake, or repetitive phrases the RM happens to like. With verifiable rewards, hacking looks like exploiting the checker: printing the expected answer format without solving, special-casing tests.

**Gao, Schulman and Hilton (2022), "Scaling Laws for Reward Model Overoptimization"**, made this quantitative with a synthetic setup: a large "gold" RM plays the role of humans, and smaller "proxy" RMs are trained on labels from it. They optimize against the proxy and measure the gold score as a function of how far the policy has moved, $d = \sqrt{\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})}$:

$$
R_{\text{BoN}}(d) = d(\alpha_{\text{bon}} - \beta_{\text{bon}} d), \qquad R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d)
$$

- $R(d)$: gold reward improvement over the initial policy.
- $d$: square root of KL from the initial policy.
- $\alpha, \beta$: fitted coefficients.
- BoN: best-of-$n$ sampling against the proxy; RL: PPO against the proxy.

Both curves rise, peak, then fall: the proxy score keeps climbing while true quality degrades. Larger proxy RMs (and more RM data) push the peak further out and higher, with coefficients scaling smoothly with RM size. Practical lessons: the KL penalty (or early stopping) is not optional; bigger and better RMs buy you more optimization headroom; evaluate with something other than the RM you optimized.

{{< fig "reinforcement-learning/13" >}}

### 2.6 RLAIF

Human labels are slow and expensive. **RLAIF** replaces human preference labels with labels from an AI model:

- **Constitutional AI** (Bai et al. 2022, Anthropic): the model critiques and revises its own outputs according to a written set of principles, then an AI labeler produces harmlessness preferences that train the RM for RL.
- **RLAIF vs RLHF** (Lee et al. 2023, Google): an off-the-shelf LLM labeler produced preferences that led to policies rated comparably to RLHF on summarization and dialogue tasks; they also tried using the LLM's score directly as the reward ("direct RLAIF").

Today most preference data in open recipes (Tulu 3, UltraFeedback-style pipelines) is AI-labeled, often with humans in the loop for auditing. The same reward-hacking concerns apply, plus inherited biases from the judge model.

### 2.7 The reasoning turn: RLVR and GRPO

**RLVR** (RL with verifiable rewards; named in Tulu 3) skips the reward model when correctness can be checked programmatically: math answers against a reference, code against tests, instruction constraints against a checker. The reward is typically binary (1 if correct, else 0), sometimes with a small format reward. A verifier cannot be "flattered", which greatly reduces (but does not eliminate) reward hacking, and it scales without labelers.

{{< fig "reinforcement-learning/14" >}}

Combined with long chain-of-thought, this produced the reasoning models. **DeepSeek-R1-Zero** applied RL directly to the DeepSeek-V3 base model with only accuracy and format rewards (answer between tags, reasoning between `<think>` tags). Over training, response length grew from hundreds to thousands of tokens and the model spontaneously learned to re-check and backtrack (the paper's "aha moment"). **DeepSeek-R1** made this usable: cold-start SFT on a few thousand long-CoT examples, reasoning-focused RL (with a language-consistency reward), rejection-sampling SFT on ~800k samples, then a final RL stage covering helpfulness and harmlessness. Their RL algorithm is GRPO.

{{< fig "reinforcement-learning/15" >}}

#### GRPO, derived

**GRPO** (Group Relative Policy Optimization) was introduced in DeepSeekMath (Shao et al. 2024). The motivation: PPO's critic is a model as large as the policy, and with a single terminal reward it is hard to train and gives questionable per-token values anyway. Can we get a baseline without a critic?

Start from the policy gradient with a baseline (section 1.7). For a prompt $q$, the ideal baseline is $V(q)$, the expected reward of the policy on $q$. Estimate it by Monte Carlo: sample a **group** of $G$ responses $\{o_1, \dots, o_G\}$ from $\pi_{\theta_{\text{old}}}(\cdot \mid q)$, score them $r_1, \dots, r_G$, and use their mean. GRPO also divides by the standard deviation:

$$
\hat A_{i,t} = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}
$$

- $i$: index of the response within the group; $t$: token index within response $i$.
- $\hat A_{i,t}$: the same value for every token $t$ of response $i$ (outcome supervision).

{{< fig "reinforcement-learning/16" >}}

Plug this advantage into PPO's clipped surrogate, averaged over tokens of each response and over the group, and put the KL regularizer into the loss instead of the reward:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}}\left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big( \min\big( \rho_{i,t} \hat A_{i,t}, \; \mathrm{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon) \hat A_{i,t} \big) - \beta \, \mathbb{D}_{\text{KL}}^{(i,t)} \Big) \right]
$$

- $\rho_{i,t} = \pi_\theta(o_{i,t} \mid q, o_{i,\lt t}) / \pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,\lt t})$: per-token ratio.
- $|o_i|$: length of response $i$ in tokens.
- $\beta$: KL coefficient (0.04 in DeepSeekMath).
- $\mathbb{D}_{\text{KL}}^{(i,t)}$: a per-token KL estimate against $\pi_{\text{ref}}$, using the "k3" estimator (from John Schulman's note on approximating KL):

$$
\mathbb{D}_{\text{KL}}^{(i,t)} = \frac{\pi_{\text{ref}}(o_{i,t} \mid \cdot)}{\pi_\theta(o_{i,t} \mid \cdot)} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid \cdot)}{\pi_\theta(o_{i,t} \mid \cdot)} - 1
$$

This estimator is always non-negative and unbiased for $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ when sampling from $\pi_\theta$, with lower variance than the naive log-ratio.

**GRPO vs PPO**:

| | PPO (RLHF) | GRPO |
| --- | --- | --- |
| Baseline | Learned critic $V_\psi(s_t)$ | Mean reward of $G$ samples for the same prompt |
| Advantage granularity | Per token (via GAE) | Per response, broadcast to all tokens |
| Models in memory | Policy, ref, RM, critic | Policy, ref, (RM or verifier) |
| KL to reference | Folded into per-token reward | Separate loss term (k3 estimator) |
| Samples per prompt | Often 1 | $G$ (e.g. 8 to 64) |
| Cost | Critic training and memory | More generation per prompt |

GRPO trades the critic for more samples. Generation is costly, but it is embarrassingly parallel on inference engines, and the group gives a baseline that is exactly conditioned on the prompt's difficulty. Note also: if all $G$ responses get the same reward (all right or all wrong), every advantage is zero and the prompt contributes no gradient (only KL). That will matter for DAPO.

{{< fig "reinforcement-learning/17" >}}

### 2.8 RLOO, Dr. GRPO, DAPO

**RLOO** ("Back to Basics", Ahmadian et al. 2024) argues that for RLHF from a strong pretrained/SFT model, PPO's machinery is unnecessary. Treat the whole response as one action (a bandit), sample $k$ responses per prompt, and use the **leave-one-out** mean as each sample's baseline:

$$
\hat A_i = r_i - \frac{1}{k - 1} \sum_{j \ne i} r_j
$$

The baseline for sample $i$ excludes $r_i$, so it is independent of sample $i$'s action and the estimator is exactly unbiased. A plain group mean that includes $r_i$ is not a problem by itself: $r_i - \mathrm{mean}(r) = \frac{k-1}{k}\left(r_i - \frac{1}{k-1}\sum_{j \ne i} r_j\right)$, just a rescaled RLOO advantage. GRPO's departures from an unbiased gradient come from its other normalizations, discussed next. RLOO matched or beat PPO and was more efficient in their experiments, and is plain REINFORCE with a baseline: exactly what you built in 1.11.

{{< fig "reinforcement-learning/18" >}}

**Dr. GRPO** ("Understanding R1-Zero-Like Training: A Critical Perspective", Liu et al. 2025) points out two biases in GRPO's normalizations:

1. **Length bias from $1/|o_i|$.** Each response's token losses are averaged, so a token in a short response gets a larger weight than one in a long response. For positive advantages, this favors short correct answers; for *negative* advantages it penalizes long wrong answers *less* per token, so wrong answers drift longer. This is part of why "response length grows during RL" was observed, and not all of that growth is useful reasoning.
2. **Difficulty bias from dividing by std.** Prompts where almost all samples agree (very easy or very hard) have small std, so their advantages are scaled up, overweighting them relative to medium-difficulty prompts.

{{< fig "reinforcement-learning/19" >}}

The fix: drop the std division and replace $1/|o_i|$ with a constant normalizer (for example the maximum generation length), so the objective is an unbiased policy gradient. They report better token efficiency (shorter wrong answers) at similar accuracy.

**DAPO** (Yu et al. 2025, ByteDance Seed) is a set of four practical changes that reached 50 on AIME 2024 with Qwen2.5-32B base:

1. **Clip-Higher.** Decouple the clip range into $\varepsilon_{\text{low}} = 0.2$ and $\varepsilon_{\text{high}} = 0.28$. With symmetric clipping, low-probability tokens can barely increase (a token at probability 0.01 can only reach 0.012 per update), which collapses entropy; a higher upper bound leaves room for exploration.
2. **Dynamic sampling.** Filter out prompts whose group accuracy is 0 or 1 (zero advantage, no gradient) and keep sampling until the batch is full of informative prompts.
3. **Token-level policy gradient loss.** Average the loss over all tokens in the batch rather than per response then per group, so long responses get weight proportional to their length and long bad patterns are properly penalized.
4. **Overlong reward shaping.** Truncated responses get a soft length penalty (and truncated samples can be masked) instead of being treated as plain wrong answers, reducing reward noise.

{{< fig "reinforcement-learning/20" >}}

{{< fig "reinforcement-learning/21" >}}

DAPO also removes the KL term entirely, arguing that for long-CoT reasoning the policy is *supposed* to move far from the initial model. Many reasoning-RL recipes now drop or shrink the KL penalty, relying on verifiable rewards to limit hacking.

All of these are the same template: $\nabla \log\pi \times \hat A$, with choices for the baseline, the normalization and the trust region.

### 2.9 Code: GRPO advantages and loss

```python
import torch


def grpo_advantages(rewards: torch.Tensor, scale_by_std: bool = True, eps: float = 1e-6):
    """rewards: [num_prompts, G] -> advantages of the same shape."""
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean
    if scale_by_std:  # Dr. GRPO sets this False to avoid over-weighting near-unanimous groups.
        adv = adv / (rewards.std(dim=1, keepdim=True) + eps)
    return adv


def grpo_loss(logp, old_logp, ref_logp, adv, mask,
              clip_low=0.2, clip_high=0.2, beta=0.04, agg="seq_mean", max_len=None):
    """
    logp, old_logp, ref_logp, mask: [B, T] per-token tensors for B = num_prompts * G responses.
    adv: [B] one advantage per response.
    agg: "seq_mean" (GRPO), "token_mean" (DAPO), "constant" (Dr. GRPO).
    """
    adv = adv.unsqueeze(1)  # every token of a response shares its outcome advantage
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * adv
    pg = torch.min(unclipped, clipped)

    # k3 estimator: non-negative per token and lower variance than the raw log-ratio.
    log_r = ref_logp - logp
    kl = torch.exp(log_r) - log_r - 1

    per_token = -(pg - beta * kl) * mask
    if agg == "seq_mean":
        return (per_token.sum(1) / mask.sum(1).clamp(min=1)).mean()
    if agg == "token_mean":
        return per_token.sum() / mask.sum().clamp(min=1)
    if agg == "constant":
        return per_token.sum() / (per_token.shape[0] * (max_len or per_token.shape[1]))
    raise ValueError(agg)


if __name__ == "__main__":
    torch.manual_seed(0)
    P, G, T = 2, 4, 5
    rewards = torch.tensor([[1., 0., 0., 1.], [0., 0., 0., 0.]])
    adv = grpo_advantages(rewards).flatten()
    print("advantages:", adv.view(P, G))
    assert torch.allclose(adv[4:], torch.zeros(4)), "all-equal group gives no signal"

    lengths = torch.tensor([5, 3, 4, 2, 5, 5, 1, 3])
    mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
    logp = (torch.randn(P * G, T) * 0.1 - 1.0).requires_grad_(True)
    old_logp = logp.detach().clone()  # first PPO epoch: policy equals the sampler
    ref_logp = old_logp + 0.05 * torch.randn_like(old_logp)

    for agg in ("seq_mean", "token_mean", "constant"):
        loss = grpo_loss(logp, old_logp, ref_logp, adv, mask, agg=agg, beta=0.0)
        (g,) = torch.autograd.grad(loss, logp)
        # With ratio == 1, the gradient on each token's logp is -adv times its aggregation weight.
        print(agg, "grad on token 0 per response:", [round(v, 3) for v in g[:, 0].tolist()])
        assert (g[:4, 0].sign() == -adv[:4].sign()).all()
```

Read the printed gradients: under `seq_mean`, the second response (length 3) gets a larger per-token gradient than the first (length 5) despite equal-magnitude advantages; under `token_mean` and `constant`, every token is weighted equally. That is the length bias from section 2.8 in two lines of output. In a real trainer, `logp` comes from a forward pass of the policy on the sampled tokens, `old_logp` from the policy at rollout time (often recomputed by the training engine, since inference-engine log-probs differ slightly), and `ref_logp` from the frozen reference.

### 2.10 Process vs outcome reward models

- **Outcome reward model (ORM)**: scores the final answer. Rule-based verifiers are perfect ORMs where available.
- **Process reward model (PRM)**: scores each intermediate reasoning step, giving dense credit assignment.

{{< fig "reinforcement-learning/22" >}}

"Let's Verify Step by Step" (Lightman et al. 2023, OpenAI) collected PRM800K, 800k human step-level labels on MATH solutions, and found process supervision produced a better verifier for best-of-$N$ reranking than outcome supervision. Math-Shepherd (Wang et al. 2023) generated step labels automatically by rolling out completions from each step and checking how often they reach the right answer (a Monte Carlo estimate of the step's value, essentially $V(s_t)$).

In large-scale RL, however, PRMs have been less successful. The DeepSeek-R1 paper lists PRMs among its unsuccessful attempts: steps are hard to define, step labels are noisy, and a learned PRM invites reward hacking and must be retrained as the policy changes. Outcome rewards with a verifier plus group baselines proved simpler and more robust. PRMs remain useful for search and reranking at inference time.

### 2.11 Test-time compute

Reasoning RL is one way to convert extra *inference* compute into accuracy. The family of approaches:

- **Parallel sampling**: generate $N$ answers and pick by majority vote (self-consistency) or by a verifier/RM (best-of-$N$). Accuracy rises roughly log-linearly with $N$ until the verifier's errors dominate.
- **Search**: beam search or tree search guided by a PRM.
- **Sequential, learned**: train the model with RL to think longer, check itself and backtrack, as in o1 and R1. Longer chains use more compute per answer, and accuracy scales with the thinking budget.

{{< fig "reinforcement-learning/23" >}}

Snell et al. (2024) showed that allocating test-time compute adaptively by prompt difficulty can outperform a model 14x larger in FLOPs-matched comparisons on problems where the base model already has some success. The broader point: RL training compute and test-time compute are now both scaling axes, alongside pretraining.

### 2.12 Frameworks

The hard part of RL for LLMs is systems: you need fast generation (an inference engine like vLLM or SGLang), training (FSDP or Megatron), and weight syncing between them every step.

{{< fig "reinforcement-learning/24" >}}

- **TRL** (Hugging Face): `GRPOTrainer`, `PPOTrainer`, `RLOOTrainer`, `DPOTrainer` on top of `transformers`; easiest to start with, supports vLLM for generation. Best for single-node and learning.
- **OpenRLHF**: Ray-based, vLLM for rollouts, DeepSpeed for training; places the four PPO models on separate GPU groups to scale to 70B.
- **verl** (ByteDance, the HybridFlow paper): flexible single-controller/multi-controller design, FSDP or Megatron backends, vLLM or SGLang rollouts; used for DAPO and many reasoning-RL papers.

### Common confusions (RLHF)

- **"GRPO has no KL."** Original GRPO has a KL term in the loss; some later recipes (DAPO) remove it.
- **"GRPO doesn't need a reward model."** It doesn't need a *critic*. It still needs a reward: a learned RM or a verifier.
- **"The reference model and old policy are the same."** $\pi_{\text{ref}}$ is frozen (the SFT model) and anchors the KL. $\pi_{\theta_{\text{old}}}$ is the policy snapshot that generated the current batch and defines the PPO ratio. They coincide only at the very start.
- **"Higher RM score means a better model."** Only up to the overoptimization peak; past it, gold quality falls while the proxy rises.
- **"Longer responses mean better reasoning."** Some length growth is an artifact of per-response normalization (Dr. GRPO).
- **"RLVR can't be hacked."** Weak verifiers can be (format tricks, test special-casing). Verifier quality is part of the reward design.
- **"DPO and PPO optimize different objectives."** Same KL-regularized objective; DPO uses offline pairs and a closed-form reparameterization, PPO samples online.

### Check yourself (RLHF)

1. List the four models in PPO-based RLHF, which are trained, and which GRPO removes.
2. Why does PPO-RLHF put a per-token KL penalty into the reward? What would happen with $\beta = 0$ against a learned RM?
3. Derive GRPO's advantage as a Monte Carlo baseline. What happens to the gradient when all $G$ responses for a prompt are correct?
4. How does RLOO's baseline differ from GRPO's group mean, and why does that make RLOO unbiased?
5. Explain the length bias that Dr. GRPO identifies in the $1/|o_i|$ normalization, for both positive and negative advantages.
6. Name DAPO's four techniques and the problem each addresses.

<details>
<summary>Answers</summary>

1. Policy (trained), reference (frozen), reward model (frozen), critic/value model (trained). GRPO removes the critic.
2. The RM is only reliable near the reference distribution; the per-token KL keeps the policy close and gives a dense signal. With $\beta = 0$ the policy will drift to regions that exploit RM errors (reward hacking: length, flattery, odd patterns), and gold quality will eventually fall (Gao et al.).
3. The ideal baseline is $V(q) = \mathbb{E}_{o\sim\pi}[r(q, o)]$. Sampling $G$ responses and averaging their rewards estimates it; subtracting gives an advantage estimate, normalized by the group std. If all are correct, every $r_i$ equals the mean, so all advantages are zero and only the KL term produces gradient.
4. RLOO uses the mean of the *other* $k-1$ rewards, which does not depend on sample $i$'s action, so the baseline argument from section 1.7 applies exactly. GRPO's mean includes $r_i$ itself, but $r_i - \mathrm{mean}(r)$ equals $\frac{G-1}{G}$ times the leave-one-out advantage, so on its own it only rescales the gradient; GRPO's bias comes from the std division and the $1/|o_i|$ length normalization.
5. Each token's gradient weight is $\hat A_i / |o_i|$. For correct (positive) responses, shorter ones get stronger per-token reinforcement. For wrong (negative) responses, longer ones get weaker per-token penalties, so the model is penalized less for being wrong at length, pushing incorrect answers to grow.
6. Clip-Higher (entropy collapse, lets low-probability tokens rise), dynamic sampling (zero-advantage groups waste batch slots), token-level loss (long responses underweighted, long bad patterns under-penalized), overlong reward shaping (truncation adds reward noise).

</details>

### Resources (RLHF)

**Read**

- [Nathan Lambert, Reinforcement Learning from Human Feedback (rlhfbook.com)](https://rlhfbook.com/) (start here): a free, current book covering RMs, PPO, GRPO, RLOO, DPO and RLVR with consistent notation.
- [Ouyang et al., Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155): InstructGPT, the canonical RLHF pipeline.
- [Gao et al., Scaling Laws for Reward Model Overoptimization (arXiv:2210.10760)](https://arxiv.org/abs/2210.10760): the quantitative account of reward hacking.
- [Shao et al., DeepSeekMath (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300): introduces GRPO and a unified view of SFT, RFT, DPO, PPO and GRPO gradients.
- [DeepSeek-AI, DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948): R1-Zero, the R1 pipeline, and what did not work (PRMs, MCTS).
- [Ahmadian et al., Back to Basics: REINFORCE-style optimization for RLHF (arXiv:2402.14740)](https://arxiv.org/abs/2402.14740): RLOO and the case against PPO's complexity.
- [Liu et al., Understanding R1-Zero-Like Training (arXiv:2503.20783)](https://arxiv.org/abs/2503.20783): Dr. GRPO and GRPO's normalization biases.
- [Yu et al., DAPO (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476): four practical fixes for large-scale reasoning RL.
- [John Schulman, Approximating KL Divergence](http://joschu.net/blog/kl-approx.html): the k1/k2/k3 estimators used in GRPO.
- Also useful: [Illustrating RLHF (Hugging Face blog)](https://huggingface.co/blog/rlhf), [Constitutional AI (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073), [RLAIF vs RLHF (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267), [Let's Verify Step by Step (arXiv:2305.20050)](https://arxiv.org/abs/2305.20050), [Math-Shepherd (arXiv:2312.08935)](https://arxiv.org/abs/2312.08935), [Scaling LLM Test-Time Compute Optimally (arXiv:2408.03314)](https://arxiv.org/abs/2408.03314), [HybridFlow / verl (arXiv:2409.19256)](https://arxiv.org/abs/2409.19256). Code: [TRL](https://github.com/huggingface/trl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [verl](https://github.com/volcengine/verl).

**Watch**

- [Nathan Lambert, Post-Training and RLHF Overview (Lecture 1)](https://www.youtube.com/watch?v=o6l6tJQgUg4): companion lecture series to the RLHF book.
- [Nathan Lambert, Understanding Policy Gradient Algorithms for RL on LLMs (Lecture 3)](https://www.youtube.com/watch?v=K_Sj_-1BUMM): PPO, GRPO and friends in LLM terms.
- [Nathan Lambert, Over-Optimization and RLHF's Bad Reputation (Lecture 9)](https://www.youtube.com/watch?v=y04JhXpiI4s): reward hacking and overoptimization in practice.
- [GRPO Explained: DeepSeekMath, Yannic Kilcher](https://www.youtube.com/watch?v=bAWV_yrqx4w): paper walkthrough of the GRPO derivation.

---

## Project

**Train a small reasoning model with GRPO, then break it on purpose.**

1. **Setup.** Take a small instruct model (0.5B to 1.5B, for example a Qwen2.5 Instruct model) and a math dataset with numeric answers (GSM8K). Write a verifier: extract the final answer from `<answer>...</answer>` and compare to the reference. Reward = 1 if correct, plus 0.1 for correct format.
2. **Baseline.** Measure pass@1 and pass@8 on a held-out split before training.
3. **Train with TRL's `GRPOTrainer`** (or your own loop using the `grpo_loss` above and vLLM for rollouts) with $G = 8$. Log mean reward, response length, KL to reference, entropy, and the fraction of zero-advantage groups.
4. **Ablations**, each a short run with everything else fixed:
   - `scale_by_std` on vs off, and `agg` = `seq_mean` vs `token_mean` vs `constant`. Plot response length for correct vs incorrect answers separately.
   - $\beta = 0.04$ vs $\beta = 0$.
   - Symmetric clip vs Clip-Higher ($\varepsilon_{\text{high}} = 0.28$). Plot entropy.
5. **Reward hacking.** Replace the verifier with a weak one (for example, reward any response that contains a number equal to the answer anywhere). Train, and find what the model learns to exploit. Then fix the verifier.
6. **Report.** Final pass@1 and pass@8 for each setting, plots, and a paragraph answering: did RL improve pass@8 or mainly pass@1 (sharpening vs new capability)? Which normalization would you choose, and why?

---

previous: [linear attention](/posts/learn-ai-05-linear-attention/)
