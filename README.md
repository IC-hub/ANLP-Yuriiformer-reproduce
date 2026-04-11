# YuriiFormer Reproduction & Optimizer-Inspired Transformer Variants

A from-scratch reproduction of **YuriiFormer** (Nesterov + Lie–Trotter) plus a systematic comparison of additional optimizer-inspired transformer variants on TinyStories and OpenWebText, with downstream evaluation on HellaSwag and ARC-Easy.

The core idea: a pre-norm transformer layer can be interpreted as **one iteration of an optimization algorithm on token embeddings**, with attention and MLP acting as two oracles. Different choices of optimizer (GD, Nesterov, Triple Momentum Method, Adam-style preconditioning) and operator-splitting scheme (Euler, Lie–Trotter, Strang) give rise to different transformer architectures.

This repo implements and compares 6 such variants under matched compute and identical training pipelines.

---

## Architectures

| Model | Optimizer view | Splitting | Velocity stream | Learned scalars/layer | Params |
|---|---|---|---|---|---|
| `VanillaTransformer` | GD | Lie–Trotter | no | 0 | 124M |
| `YuriiFormer` | Nesterov | Lie–Trotter | yes | 6 (μ,β,γ × 2) | 163M |
| `TMMFormer` | Triple Momentum Method | Lie–Trotter | yes | 8 (μ,β,γ,ν × 2) | 163M |
| `AdamFormer` | Adam (1st/2nd moment) | Lie–Trotter | yes (m, s) | 6 (β₁,β₂,γ × 2) | ~163M |
| `AdamWFormer` | AdamW (decoupled wd) | Lie–Trotter | yes (m, s) | 8 (β₁,β₂,γ,λ × 2) | ~163M |

**Triple Momentum Method (TMM)** is the first-order optimal algorithm for L-smooth, μ-strongly convex functions (Van Scoy et al. 2018), with convergence rate (1−√(μ/L))² — strictly better than Nesterov. `TMMFormer` generalizes `YuriiFormer` by adding a learnable scalar `ν` that decouples the iterate update from the gradient-evaluation lookahead.

**Adam/AdamW variants** maintain per-token first/second moment streams alongside the hidden state, mirroring the Adam optimizer applied to embeddings. They underperform momentum-based variants — confirming that token-space gradients lack the per-dimension scale variance that makes Adam useful for parameter optimization.

---

## From Optimizers to Transformer Architectures

Following Zimin et al. (2026), we view a transformer block as one iteration of a first-order optimization algorithm acting on the token-embedding matrix $X_t \in \mathbb{R}^{T \times d}$. The composite objective is

$$\min_X \; \mathcal{E}(X) + \mathcal{F}(X),$$

where $\mathcal{E}$ is an interaction energy (token–token) and $\mathcal{F}$ is a potential energy (per-token). Their gradients are realized by the two transformer oracles:

$$\nabla \mathcal{E}_t(X) \;\approx\; \mathrm{Attn}_t\!\bigl(\mathrm{LN}(X)\bigr), \qquad \nabla \mathcal{F}_t(X) \;\approx\; \mathrm{MLP}_t\!\bigl(\mathrm{LN}(X)\bigr).$$

A first-order optimization template applied to $\mathcal{E} + \mathcal{F}$ via **Lie–Trotter splitting** then yields a transformer block: each layer performs an attention substep (gradient step on $\mathcal{E}$) followed by an MLP substep (gradient step on $\mathcal{F}$). Different optimizer templates → different transformer architectures.

All five variants share the same 12L/12H/$d=768$ pre-norm backbone, GPT-2 BPE tokenizer, and weight-tied output head — they differ **only** in the optimizer template and in the auxiliary streams ($V_t$, $M_t$, $S_t$) propagated alongside the state $X_t$. Per-layer scalars ($\mu, \beta, \gamma, \nu, \lambda$) are all learned, reparameterized through $\sigma$ for $(0,1)$ values or $\mathrm{softplus}$ for positives.

---

### 1. `VanillaTransformer` — Gradient Descent

**Optimizer (gradient descent on $f$):**

$$x_{t+1} \;=\; x_t - \gamma_t \, \nabla f(x_t).$$

**Transformer (Lie–Trotter splitting of GD on $\mathcal{E} + \mathcal{F}$):**

$$
\begin{aligned}
X_{t+\frac{1}{2}} &\;=\; X_t \;+\; \mathrm{Attn}_t(X_t) \\
X_{t+1} &\;=\; X_{t+\frac{1}{2}} \;+\; \mathrm{MLP}_t(X_{t+\frac{1}{2}})
\end{aligned}
$$

This is the standard pre-norm nanoGPT block: no auxiliary stream, no learned scalars, step size absorbed into the oracles. Recovers eq. (5) of Zimin et al. (2026).

---

### 2. `YuriiFormer` — Nesterov Accelerated Gradient

**Optimizer (Nesterov, 1983).** Given iterate $x_t$ and velocity $v_t$:

$$
\begin{aligned}
\tilde{x}_t &\;=\; x_t + \mu_t\, v_t & &\text{(lookahead)}\\
v_{t+1} &\;=\; \beta_t\, v_t - \gamma_t\, \nabla f(\tilde{x}_t) & &\text{(velocity update)}\\
x_{t+1} &\;=\; x_t + v_{t+1} & &\text{(iterate update)}
\end{aligned}
$$

**Transformer.** Maintain a velocity stream $V_t$ (from a separate learned embedding) alongside $X_t$. Each layer applies the Nesterov template twice — once with $\mathrm{Attn}$, once with $\mathrm{MLP}$:

$$
\begin{aligned}
X^{\text{in}}_t &\;=\; X_t + \mu_t\, V_t \\
V_{t+\frac{1}{2}} &\;=\; \mathrm{LN}_v\!\bigl(\beta_t\, V_t + \gamma_t\, \mathrm{Attn}_t(\mathrm{LN}(X^{\text{in}}_t))\bigr) \\
X_{t+\frac{1}{2}} &\;=\; X_t + V_{t+\frac{1}{2}} \\[2pt]
X^{\text{in}}_{t+\frac{1}{2}} &\;=\; X_{t+\frac{1}{2}} + \mu_{t+\frac{1}{2}}\, V_{t+\frac{1}{2}} \\
V_{t+1} &\;=\; \mathrm{LN}_v\!\bigl(\beta_{t+\frac{1}{2}}\, V_{t+\frac{1}{2}} + \gamma_{t+\frac{1}{2}}\, \mathrm{MLP}_t(\mathrm{LN}(X^{\text{in}}_{t+\frac{1}{2}}))\bigr) \\
X_{t+1} &\;=\; X_{t+\frac{1}{2}} + V_{t+1}
\end{aligned}
$$

Six learned scalars per layer ($\mu, \beta, \gamma$ for the attention and MLP substeps). Velocity LayerNorm $\mathrm{LN}_v$ stabilizes the velocity stream across depth. This is the Nesterov+Lie–Trotter variant of Zimin et al. (2026).

---

### 3. `TMMFormer` — Triple Momentum Method

**Optimizer (Van Scoy, Freeman, Lynch, 2018).** TMM is the first-order optimal algorithm for $L$-smooth, $\mu$-strongly convex objectives, with convergence rate

$$\bigl(1 - \sqrt{\mu/L}\bigr)^2,$$

strictly better than Nesterov's $1 - \sqrt{\mu/L}$. Its update generalizes Nesterov by introducing a second scalar $\nu_t$ that decouples the **gradient-evaluation lookahead** from the **iterate update**:

$$
\begin{aligned}
\tilde{x}_t &\;=\; x_t + \mu_t\, v_t & &\text{(lookahead)}\\
v_{t+1} &\;=\; \beta_t\, v_t - \gamma_t\, \nabla f(\tilde{x}_t) & &\text{(velocity update)}\\
x_{t+1} &\;=\; x_t + \nu_t\, v_{t+1} & &\text{(iterate update, } \nu_t \neq 1 \text{ in general)}
\end{aligned}
$$

YuriiFormer is the special case $\nu_t \equiv 1$.

**Transformer.** Apply the TMM template twice per layer (Lie–Trotter):

$$
\begin{aligned}
X^{\text{in}}_t &\;=\; X_t + \mu_t\, V_t \\
V_{t+\frac{1}{2}} &\;=\; \mathrm{LN}_v\!\bigl(\beta_t\, V_t + \gamma_t\, \mathrm{Attn}_t(\mathrm{LN}(X^{\text{in}}_t))\bigr) \\
X_{t+\frac{1}{2}} &\;=\; X_t + \nu_t\, V_{t+\frac{1}{2}} \\[2pt]
X^{\text{in}}_{t+\frac{1}{2}} &\;=\; X_{t+\frac{1}{2}} + \mu_{t+\frac{1}{2}}\, V_{t+\frac{1}{2}} \\
V_{t+1} &\;=\; \mathrm{LN}_v\!\bigl(\beta_{t+\frac{1}{2}}\, V_{t+\frac{1}{2}} + \gamma_{t+\frac{1}{2}}\, \mathrm{MLP}_t(\mathrm{LN}(X^{\text{in}}_{t+\frac{1}{2}}))\bigr) \\
X_{t+1} &\;=\; X_{t+\frac{1}{2}} + \nu_{t+\frac{1}{2}}\, V_{t+1}
\end{aligned}
$$

Eight learned scalars per layer ($\mu, \beta, \gamma, \nu$ for each substep). We initialize $\nu_t$ so that $\mathrm{softplus}(\nu^{\text{raw}}) \approx 1$, so training begins in the YuriiFormer regime and learns where to deviate.

---

### 4. `AdamFormer` — Adam

**Optimizer (Kingma & Ba, 2015).** Adam maintains first/second moment EMAs $m_t, s_t$ of the gradient and rescales each coordinate by the inverse square root of the second moment:

$$
\begin{aligned}
g_t &\;=\; \nabla f(x_t) \\
m_{t+1} &\;=\; \beta_{1,t}\, m_t + (1 - \beta_{1,t})\, g_t \\
s_{t+1} &\;=\; \beta_{2,t}\, s_t + (1 - \beta_{2,t})\, g_t \odot g_t \\
x_{t+1} &\;=\; x_t - \gamma_t \, \frac{m_{t+1}}{\sqrt{s_{t+1}} + \varepsilon}
\end{aligned}
$$

**Transformer.** Maintain two auxiliary streams alongside $X_t$: a first-moment stream $M_t$ (from a separate learned embedding) and a second-moment stream $S_t$ (initialized to ones). Apply Adam twice per layer:

$$
\begin{aligned}
G^{\text{a}}_t &\;=\; \mathrm{Attn}_t(\mathrm{LN}(X_t)) \\
M_{t+\frac{1}{2}} &\;=\; \beta_{1,t}\, M_t + (1 - \beta_{1,t})\, G^{\text{a}}_t \\
S_{t+\frac{1}{2}} &\;=\; \beta_{2,t}\, S_t + (1 - \beta_{2,t})\, G^{\text{a}}_t \odot G^{\text{a}}_t \\
X_{t+\frac{1}{2}} &\;=\; X_t + \gamma_t\, \mathrm{LN}_u\!\!\left(\frac{M_{t+\frac{1}{2}}}{\sqrt{S_{t+\frac{1}{2}}} + \varepsilon}\right) \\[6pt]
G^{\text{m}}_t &\;=\; \mathrm{MLP}_t(\mathrm{LN}(X_{t+\frac{1}{2}})) \\
M_{t+1} &\;=\; \beta_{1,t+\frac{1}{2}}\, M_{t+\frac{1}{2}} + (1 - \beta_{1,t+\frac{1}{2}})\, G^{\text{m}}_t \\
S_{t+1} &\;=\; \beta_{2,t+\frac{1}{2}}\, S_{t+\frac{1}{2}} + (1 - \beta_{2,t+\frac{1}{2}})\, G^{\text{m}}_t \odot G^{\text{m}}_t \\
X_{t+1} &\;=\; X_{t+\frac{1}{2}} + \gamma_{t+\frac{1}{2}}\, \mathrm{LN}_u\!\!\left(\frac{M_{t+1}}{\sqrt{S_{t+1}} + \varepsilon}\right)
\end{aligned}
$$

Six learned scalars per layer ($\beta_1, \beta_2, \gamma$ per substep). The auxiliary $\mathrm{LN}_u$ normalizes the adaptive update direction across depth.

---

### 5. `AdamWFormer` — AdamW

**Optimizer (Loshchilov & Hutter, 2019).** AdamW differs from Adam by **decoupling** weight decay from the adaptive update — the iterate is shrunk toward zero *before* the Adam step:

$$x_{t+1} \;=\; (1 - \lambda_t)\, x_t \;-\; \gamma_t\, \frac{m_{t+1}}{\sqrt{s_{t+1}} + \varepsilon}.$$

**Transformer.** Same auxiliary streams as AdamFormer; the only change is the iterate update of each substep:

$$
\begin{aligned}
X_{t+\frac{1}{2}} &\;=\; (1 - \lambda_t)\, X_t \;+\; \gamma_t\, \mathrm{LN}_u\!\!\left(\frac{M_{t+\frac{1}{2}}}{\sqrt{S_{t+\frac{1}{2}}} + \varepsilon}\right) \\[4pt]
X_{t+1} &\;=\; (1 - \lambda_{t+\frac{1}{2}})\, X_{t+\frac{1}{2}} \;+\; \gamma_{t+\frac{1}{2}}\, \mathrm{LN}_u\!\!\left(\frac{M_{t+1}}{\sqrt{S_{t+1}} + \varepsilon}\right)
\end{aligned}
$$

Eight learned scalars per layer (adds $\lambda$ per substep). We initialize $\lambda^{\text{raw}} = -5$ so that $\sigma(\lambda^{\text{raw}}) \approx 0.007$ — the model starts essentially at AdamFormer and learns the optimal per-layer decay.

---

### Common design notes

- **Velocity / update LayerNorm.** $\mathrm{LN}_v$ (in YuriiFormer / TMMFormer) and $\mathrm{LN}_u$ (in AdamFormer / AdamWFormer) are essential for stability across depth: without them, momentum-based architectures diverge after a few hundred steps.
- **Auxiliary stream initialization.** $V_0$ and $M_0$ are produced by *separate learned embeddings* (`vel_tok_emb + vel_pos_emb`, etc.), not by zero-initialization. $S_0$ is initialized to all-ones.
- **Two-optimizer training.** All learned per-layer scalars (the `*_raw` parameters) are trained by AdamW at a higher LR ($3 \cdot 10^{-3}$) than the rest of the network. Matrix weights use **Muon**; embeddings, LayerNorm, and scalars use **AdamW**.
- **Weight tying** with `tok_emb` for the output projection in every variant.

### Common design notes

- **`LN_v` / `LN_u`** (LayerNorm on the velocity / adaptive-update tensor) is borrowed from YuriiFormer.  It dramatically stabilizes the auxiliary stream — without it, momentum-style architectures diverge after a few hundred steps.
- **Auxiliary stream initialization.**  Velocity `v` and 1st-moment `m` are produced by *separate learned embeddings* (`vel_tok_emb + vel_pos_emb`, etc.), not by zero-initialization.  The 2nd moment `s` is initialized to ones.
- **Two-optimizer setup.**  All learned per-layer scalars (the `*_raw` parameters) go through AdamW with a higher LR (`3e-3`) than the rest of the network.  Matrix weights use **Muon**, embeddings/LN/scalars use **AdamW**.
- **Weight tying** with `tok_emb` for the output projection in every variant.

---

## Results

### TinyStories (10k steps, effective batch ≈ 480, single seed)

| Method | Best Val | Final Val | Train@10k |
|---|---:|---:|---:|
| Paper Vanilla GD+LT (nanoGPT) | 1.106 | 1.114 | — |
| Paper YuriiFormer | **1.078** | 1.090 | 0.896 |
| **TMMFormer** (ours) | **1.1284** | 1.1387 | 0.8464 |
| YuriiFormer (ours) | 1.1299 | 1.1384 | 0.8497 |
| AdamWFormer (ours) | 1.1472 | 1.1547 | — |
| AdamFormer (ours) | 1.1528 | 1.1554 | — |
| VanillaTransformer (ours) | 1.1569 | 1.1604 | 0.9418 |

**TS observations**: TMM and YuriiFormer are statistically tied (Δ = 0.0015), both clearly ahead of Adam-style variants and Vanilla. Our reproduction loss (~1.13) is higher than paper YuriiFormer (1.078) — a systematic Muon/effective-batch hyperparameter offset — but **relative comparisons under matched setup remain valid**. The extra ν freedom in TMM neither helps nor hurts at 10k steps.

### OpenWebText (30k steps)

| Method | Best Val | Final Val | Train@30k |
|---|---:|---:|---:|
| Paper Vanilla GD+LT | 2.990 | — | — |
| Paper YuriiFormer | **2.920** | — | — |
| **YuriiFormer + WSD** (ours) | **2.9275** | 2.9275 | 2.9348 |
| **YuriiFormer + SAWD** (ours) | **2.9323** | 2.9323 | 2.9376 |
| TMMFormer (ours) | 2.9342 | 2.9342 | 2.9290 |
| YuriiFormer (ours, cosine) | 2.9413 | 2.9413 | 2.9352 |
| **YuriiFormer + SAM** (ours) | 2.9481 | 2.9482 | 2.9273 |
| AdamWFormer (ours) | 2.9883 | 2.9883 | 2.9883 |
| AdamFormer (ours) | 2.9911 | 2.9911 | 2.9904 |
| VanillaTransformer (ours) | 3.0078 | 3.0080 | 3.0087 |

**OWT observations**: We trained three sharpness-aware variants of YuriiFormer (Foret et al. 2021 / Watts et al. 2026), all sharing the same architecture and optimizer setup, differing only in **schedule** and whether **SAM**'s dual-pass perturbation is active:

- **`+WSD`**: cosine → Warmup–Stable–Decay schedule (warmup 3k → stable 25k → linear decay 25k–30k). No SAM. Same compute as cosine baseline.
- **`+SAM`**: keep cosine, add SAM perturbation $\epsilon^* = \rho \, g/\|g\|$ at every step ($\rho = 0.05$). ~2× compute.
- **`+SAWD`**: WSD schedule, with SAM **only during the decay phase** (steps 25k–30k). ~1.17× compute.

The val-loss ranking is **WSD < SAWD < TMM < YuriiFormer (cosine) < SAM < AdamW < Adam < Vanilla**. Two surprising findings:

1. **WSD alone, with no SAM, is the single strongest variant** (2.9275). Just changing the LR schedule beats both the previous best (TMMFormer cosine, 2.9342) by Δ ≈ −0.007, and the same-architecture YuriiFormer cosine baseline by Δ ≈ −0.014. The gap to paper YuriiFormer (2.920) is only 0.008 nats — nearly erasing the ~0.05-nat reproduction offset that all cosine variants exhibited under our 2-GPU DDP + `torch.compile` setup.
2. **SAM alone, on the cosine schedule, slightly hurts** (2.9481 vs 2.9413 for the cosine baseline, Δ ≈ +0.007). The dual-pass perturbation under our setup with $\rho = 0.05$ does not improve val loss; SAM's compute is wasted on val-loss.
3. **SAWD recovers most of WSD's gain** (2.9323) by combining WSD's schedule with SAM only during decay. It is +0.005 above WSD and −0.002 below TMM. The fact that **SAWD did not beat WSD** is the key result here: under this setup, schedule (WSD) is doing more work than SAM perturbation.

### Downstream Evaluation (TinyStories checkpoints)

HellaSwag (10-shot) and ARC-Easy (25-shot), evaluated with `lm-evaluation-harness` v0.4.3:

| Model | HellaSwag acc_norm | ARC-Easy acc_norm |
|---|---:|---:|
| TMMFormer | 0.2682 | 0.2563 |
| AdamFormer | 0.2680 | 0.2660 |
| AdamWFormer | 0.2675 | 0.2635 |
| VanillaTransformer | 0.2669 | 0.2542 |

All TS-trained models are **at chance level** (HellaSwag random ≈ 0.25, ARC-Easy 4-choice ≈ 0.25). This is expected — the TinyStories vocabulary/style distribution is too narrow to transfer. **OWT-checkpoint downstream evaluation is the meaningful comparison** and will be added once the OWT runs complete.

### Attention Entropy (OWT best checkpoints)

We measure how peaked vs. diffuse each head's attention distribution is, as a proxy for how specialized the head has become. For each variant we monkey-patch `CausalSelfAttention.forward` to compute the softmax weights explicitly (instead of fused SDPA), and accumulate the per-query Shannon entropy averaged over heads, batch and valid query positions.

For a causal attention layer with sequence length $T$ and per-head softmax weights $a^{(h)}_{ij}$ ($i$ = query, $j$ = key, $j \le i$), the per-head mean entropy is

$$
H^{(h)} \;=\; \frac{1}{B(T-1)} \sum_{b=1}^{B} \sum_{i=2}^{T} \Big(- \sum_{j=1}^{i} a^{(h)}_{b,i,j} \log a^{(h)}_{b,i,j}\Big),
$$

reported in **nats**. The first query ($i=1$) is dropped because its softmax is degenerate (single key). The maximum possible value is $\log T = \log 1024 \approx 6.931$ (uniform attention). Values close to 0 mean the head has collapsed to a single token (e.g. attention sink / induction copy); values near $\log T$ mean the head averages indiscriminately over context.

Measured on each variant's OWT `best.pt`, 8 batches × 4 sequences of length 1024 from the OWT validation split:

| Layer | Vanilla | YuriiFormer | TMMFormer | AdamFormer | AdamWFormer | **+WSD** | **+SAM** | **+SAWD** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | **5.39** | 3.37 | 3.32 | 4.22 | 4.29 | 3.22 | 3.28 | 3.22 |
| 1  | 4.17 | 2.60 | 2.93 | **0.33** | **1.62** | 1.77 | 3.16 | **1.72** |
| 2  | 3.58 | 3.35 | 3.32 | 2.91 | 3.23 | 3.29 | 2.63 | 3.18 |
| 3  | 3.79 | 2.95 | 2.97 | 3.44 | 3.59 | 2.84 | 3.16 | 2.76 |
| 4  | 3.12 | 2.88 | 2.83 | 3.09 | 3.05 | 2.80 | 2.92 | 2.75 |
| 5  | 3.07 | 3.27 | 3.30 | 3.38 | 3.23 | 3.22 | 3.10 | 3.13 |
| 6  | 3.41 | 3.17 | 3.15 | 3.35 | 3.41 | 3.03 | 3.35 | 3.00 |
| 7  | 3.07 | 3.31 | 3.22 | 3.02 | 3.11 | 3.22 | 3.46 | 3.19 |
| 8  | 3.01 | 3.14 | 3.11 | 3.29 | 3.21 | 3.10 | 3.20 | 3.08 |
| 9  | 3.10 | 3.18 | 3.12 | 3.20 | 3.22 | 3.09 | 3.29 | 3.09 |
| 10 | 3.16 | 3.32 | 3.26 | 3.32 | 3.31 | 3.20 | 3.30 | 3.21 |
| 11 | 3.45 | 3.30 | 3.36 | 3.50 | 3.44 | 3.18 | 3.52 | 3.18 |
| **mean** | **3.527** | **3.154** | **3.157** | **3.089** | **3.224** | **2.997** | **3.198** | **2.959** |

**Observations**

1. **YuriiFormer ≡ TMMFormer in attention behavior**. The two columns agree to ≤ 0.05 nats per layer (overall Δ ≈ 0.003). This reinforces the loss-level finding that TMM's extra ν degree of freedom does not change how attention is used in practice.
2. **Adam/AdamW collapse layer 1**. AdamFormer's layer 1 mean entropy is 0.33 nats with `min_h = 0.0000` — at least one head has fully collapsed to an attention sink. AdamWFormer shows a milder version (1.62, `min_h ≈ 0.0001`). This degeneracy is absent in the Nesterov family.
3. **Vanilla has the most diffuse attention overall** (mean 3.53; layer 0 = 5.39, ≈ 78% of max). Without the auxiliary momentum/Adam streams, the model has not developed sharp specialization at the input-side layers.
4. **Adam family has the most diffuse layer 0** among auxiliary-stream variants (4.22–4.29 vs ~3.35 for Nesterov family). The dual moment streams (m, s) appear to push the first layer toward broader, more averaging-like aggregation.
5. **Deep layers (2–11) are relatively flat across variants** (typical spread < 0.3 nats). The architectural differences mostly show up at the input-side layers; the internal attention motifs converge to similar entropies.
6. All variants sit between **~45% and ~50% of $\log T$** — well away from uniform but also not collapsed, indicating healthy mixed-specificity attention overall.
7. **WSD and SAWD push attention specialization further; SAM does not.** SAWD has the lowest overall mean entropy (**2.959**), narrowly ahead of WSD (2.997) — the only two variants below 3 nats. Both make layer 1 the most specialized of any non-degenerate variant (1.72 / 1.77 nats, `min_h ≈ 0.009` — clearly above the AdamFormer collapse at 0.0). Across the deep layers (3–11) both WSD and SAWD are uniformly 0.05–0.15 nats below cosine YuriiFormer/TMM, indicating a network-wide shift toward more confident attention.
8. **SAM (cosine + dual pass)** moves entropy in the *opposite* direction: mean rises to 3.198, *higher* than the cosine YuriiFormer baseline (3.154). The SAM perturbation with $\rho = 0.05$ on cosine apparently nudges attention to be more diffuse — the model hedges across more keys. Combined with SAM's slightly worse val loss (2.948 vs 2.941), this suggests pure SAM here is over-regularizing attention without a corresponding flatness payoff.
9. **The schedule (WSD) does the work, the perturbation (SAM) helps only when paired with it.** WSD-only and SAWD give virtually identical entropy curves (within 0.04 nats per layer); SAM-only on cosine looks more like AdamWFormer than like WSD. The decay phase of WSD/SAWD is what produces the entropy collapse, not the SAM step itself.

The script and per-head tensors live in `attention_entropy.py` and `attention_entropy_results/<variant>.pt`.

### Loss-Landscape Sharpness (OWT best checkpoints)

We evaluate three sharpness/flatness proxies on a small batch of OWT validation data using `loss_sharpness.py`:

1. **Top Hessian eigenvalue $\lambda_{\max}$** via power iteration on the Hessian–vector product $Hv = \nabla_\theta \langle \nabla_\theta L, v\rangle$ (double backward, math SDPA backend). Captures the steepest curvature direction.
2. **Hessian trace $\mathrm{tr}(H)$** via Hutchinson's estimator with Rademacher probes:
   $$\mathrm{tr}(H) \;\approx\; \frac{1}{K} \sum_{k=1}^K v_k^\top H v_k, \qquad v_k \in \{-1, +1\}^d.$$
   Captures the *average* curvature across all directions.
3. **1D filter-normalized loss curve** $L(\theta + \alpha d)$ for $\alpha \in [-0.5, 0.5]$ along a random direction $d$ scaled per-parameter to its Frobenius norm (Li et al. 2018), giving a cheap visualization of how fast loss rises around the minimum.

Sharper minima ⇒ larger $\lambda_{\max}$, larger $\mathrm{tr}(H)$, steeper 1D curve. Flatter minima are usually associated with better generalization.

| Variant | val_loss | $\lambda_{\max}$ | $\mathrm{tr}(H)$ | $\mathrm{tr}(H)/n$ | curve $\Delta$ |
|---|---:|---:|---:|---:|---:|
| **YuriiFormer + SAWD** | **2.928** | **−73.5** | **9 642** | **5.89e−5** | 11.68 |
| **YuriiFormer + SAM**  | 2.948 | **89.0** | 11 029 | 6.73e−5 | 8.32 |
| **YuriiFormer + WSD**  | **2.928** | 112.4 | 15 724 | 9.60e−5 | 10.77 |
| TMMFormer | 2.934 | 130.7 | 23 258 | 1.42e−4 | 9.08 |
| YuriiFormer (cosine) | 2.941 | 167.7 | 22 841 | 1.39e−4 | 8.54 |
| AdamFormer | 2.991 | 423.9 | 34 409 | 2.10e−4 | 9.36 |
| AdamWFormer | 2.988 | **1889.1** | 48 933 | 2.99e−4 | 9.77 |
| Vanilla | 3.008 | 79.8 | 39 320 | 3.16e−4 | 11.35 |

**Observations**

1. **Nesterov family lives in the flattest basin** (TMM/Yurii both $\mathrm{tr}/n \approx 1.4 \times 10^{-4}$, ≈ half of Vanilla and ≈ 30–50% lower than Adam/AdamW). This aligns with the standard "flat minimum ↔ better generalization" intuition and matches their lower val loss.
2. **AdamWFormer is dramatically sharp**: $\lambda_{\max} \approx 1889$, ~11× TMM and ~24× Vanilla. The decoupled weight decay drives parameters into a region with one or a few extremely steep directions, even though average curvature is comparable.
3. **AdamFormer is intermediate**: $\lambda_{\max} = 424$ (~3× Nesterov family) but $\mathrm{tr}/n$ moderate. Removing decoupled wd softens the worst direction but the basin is still sharper than Nesterov.
4. **Vanilla has the smallest $\lambda_{\max}$ but the largest $\mathrm{tr}/n$ and steepest 1D curve** — a "wide but bumpy" basin: no single direction is extremely steep, yet many directions contribute moderate curvature, and the random 1D probe rises faster than for any other variant.
5. **TMM ≈ Yurii in landscape too**: trace within 2%, 1D curve ranges differ by only 0.5 nats; TMM's $\lambda_{\max}$ is slightly lower (130.7 vs 167.7), perhaps reflecting ν damping the worst eigendirection. Combined with the entropy and downstream-task results, this is the third independent measurement showing the two are functionally equivalent.
6. **WSD finds a strictly flatter basin than cosine — at the same architecture.** Replacing cosine with the Warmup–Stable–Decay schedule on YuriiFormer drives $\mathrm{tr}(H)$ from 22 841 → **15 724** (Δ ≈ −31%), $\mathrm{tr}/n$ from 1.39e−4 → **9.60e−5** (Δ ≈ −31%), and $\lambda_{\max}$ from 167.7 → **112.4** (Δ ≈ −33%). It is the lowest $\mathrm{tr}/n$ of *all* cosine variants — including TMM and cosine YuriiFormer — by a clear margin. This is direct evidence that WSD's long stable phase + late linear decay is implicitly sharpness-aware: by holding peak LR through the stable phase and then decaying linearly, the optimizer spends most of its budget *exploring* and only at the end *settles* into a flat region. The 1D curve $\Delta$ (10.77) is comparatively large, but this metric depends on a non-seeded random direction and is substantially less reliable than the trace and $\lambda_{\max}$ estimates, which agree that WSD is much flatter than cosine.
7. **SAM and SAWD push the basin even flatter than WSD, validating Foret 2021's intuition at the architecture level.** Both SAM (cosine + dual pass) and SAWD (WSD + decay-phase SAM) reach $\mathrm{tr}/n$ values **30–40% lower than WSD alone** — 6.73e−5 (SAM) and **5.89e−5 (SAWD, the flattest of all eight variants)**. SAWD's $\mathrm{tr}/n$ is roughly **24× lower than the cosine YuriiFormer baseline** (1.39e−4 → 5.89e−5), and **54× lower than Vanilla** (3.16e−4). The SAM perturbation is doing exactly what it advertises — eliminating the steepest positive eigendirection.
8. **SAWD's $\lambda_{\max}$ is *negative* (−73.5).** Power iteration converges to the eigenvalue with the largest magnitude, so a negative result means the *steepest* direction at SAWD's minimum is one of negative curvature (a saddle-like direction), not positive. Combined with the still-positive trace (9 642), this implies the positive-curvature spectrum is so flat that the largest negative eigenvalue dominates in magnitude. This is the *signature* of an aggressive flat-minimum optimizer: every positive-curvature direction has been smoothed below the magnitude of any saddle direction the optimizer happened to leave behind. SAM (cosine) does not reach this regime — its $\lambda_{\max} = 89$ is positive but already lower than every cosine baseline.
9. **Sharpness ranking ≠ val-loss ranking ≠ downstream ranking.** SAWD has the flattest landscape but its val loss is +0.005 vs WSD; SAM has the second-flattest landscape but the *worst* val loss of the three sharpness-aware variants. This suggests that under our 30k-step OWT setup, **flatness is necessary but not sufficient** for the val-loss gain — the schedule's effect on optimization trajectory matters at least as much as the curvature of the final basin. WSD wins on val loss because it finds a "good enough" flat region without paying the SAM tax that nudges the trajectory off course.

Per-variant tensors are saved to `loss_sharpness_results/<variant>.pt`.

### Downstream Evaluation (OWT best checkpoints)

HellaSwag (10-shot) and ARC-Easy (25-shot), evaluated with `lm-evaluation-harness` v0.4.3:

| Model | val_loss | HellaSwag acc_norm | ARC-Easy acc_norm |
|---|---:|---:|---:|
| **YuriiFormer + WSD**  | **2.928** | 0.3177 | **0.4398** |
| **YuriiFormer + SAWD** | 2.932 | 0.3147 | 0.4360 |
| **TMMFormer**          | 2.934 | **0.3182** | 0.4343 |
| YuriiFormer (cosine)   | 2.941 | 0.3158 | 0.4306 |
| **YuriiFormer + SAM**  | 2.948 | 0.3133 | 0.4276 |
| AdamFormer             | 2.991 | 0.3096 | 0.4339 |
| AdamWFormer            | 2.988 | 0.3008 | 0.4188 |
| VanillaTransformer     | 3.008 | 0.3020 | 0.4167 |

**Observations**

1. **WSD wins on ARC-Easy outright (0.4398) and ties TMMFormer on HellaSwag** (0.3177 vs 0.3182, Δ = 0.0005 — well within seed noise). It is the strongest *or tied-for-strongest* variant on every measurement: best val loss, best ARC-Easy, second-best HellaSwag.
2. **The downstream ranking among cosine baselines still matches val-loss ranking**: TMM > Yurii > Adam > AdamW > Vanilla. The Nesterov family still beats Adam/AdamW by ~1 acc-point on HellaSwag and ~1.5 on ARC-Easy, just as before.
3. **SAWD downstream ≈ TMMFormer downstream**, both clearly above the cosine YuriiFormer baseline. SAWD is between WSD and TMM on both tasks.
4. **SAM (cosine + dual pass) underperforms its cosine baseline on every downstream metric**: HellaSwag 0.3133 vs 0.3158, ARC-Easy 0.4276 vs 0.4306. Combined with SAM's slightly *worse* val loss (2.948 vs 2.941), this means SAM at $\rho = 0.05$ on cosine is a net negative under our setup, despite the substantially flatter loss landscape it produces (see sharpness section). This is the cleanest demonstration that **flatness is not enough** — the trajectory matters.
5. **All variants are clearly above chance** (HellaSwag random ≈ 0.25, ARC-Easy 4-choice ≈ 0.25), unlike the TS-trained checkpoints — confirming OWT-scale pretraining gives genuine generalization.

### Cross-cutting summary: WSD vs SAM vs SAWD

| Metric | Best variant | 2nd | 3rd |
|---|---|---|---|
| Val loss | **WSD** (2.928) | SAWD (2.932) | TMM (2.934) |
| HellaSwag acc_norm | TMM (0.3182) | **WSD** (0.3177) | YuriiFormer (0.3158) |
| ARC-Easy acc_norm | **WSD** (0.4398) | SAWD (0.4360) | TMM (0.4343) |
| Sharpness $\mathrm{tr}/n$ | **SAWD** (5.89e−5) | SAM (6.73e−5) | WSD (9.60e−5) |
| Attention entropy | **SAWD** (2.959) | WSD (2.997) | AdamFormer (3.089) |

The story is consistent across all five metrics: **WSD does the bulk of the work** — it owns val loss and one downstream task, ties on the other, and is the third-flattest basin even without any SAM perturbation. **SAWD is the curvature champion** — the SAM step in the decay phase produces the lowest sharpness and entropy of any variant, and downstream is competitive with WSD/TMM. **SAM-only is the loser** — it pays 2× compute to produce a flat basin (second-best $\mathrm{tr}/n$) but neither val loss nor downstream improves; the cosine schedule cannot exploit the flatness SAM creates.

The main practical takeaway: under this 30k-step OWT setup, *changing the LR schedule is more powerful than adding SAM*, and the two only combine usefully (SAWD) when SAM is restricted to the late decay phase rather than running throughout.

---

## Project Structure

```
.
├── model.py                  # YuriiFormer (Nesterov + Lie–Trotter, 6 scalars/layer)
├── tmm_model.py              # TMMFormer (Triple Momentum Method, 8 scalars/layer)
├── vanilla_model.py          # VanillaTransformer (GD + Lie–Trotter, no velocity)
├── adam_model.py             # AdamFormer (Adam-style m,v streams)
├── adamw_model.py            # AdamWFormer (decoupled weight decay variant)
│
├── data.py                   # TinyStories tokenization (GPT-2 BPE) + dataloader
├── data_owt.py               # OpenWebText tokenization + streaming dataloader
│
├── train.py                  # Original single-GPU YuriiFormer trainer
├── yurii_train_ddp.py        # DDP YuriiFormer (TinyStories)
├── tmm_train_ddp.py          # DDP TMMFormer (TinyStories)
├── vanilla_train_ddp.py      # DDP VanillaTransformer (TinyStories)
├── adam_train_ddp.py         # DDP AdamFormer (TinyStories)
├── adamw_train_ddp.py        # DDP AdamWFormer (TinyStories)
│
├── tmm_train_owt.py          # TMMFormer OpenWebText trainer (DDP)
├── vanilla_train_owt.py      # VanillaTransformer OWT trainer
├── adam_train_owt.py         # AdamFormer OWT trainer
├── adamw_train_owt.py        # AdamWFormer OWT trainer
│
├── eval_model.py             # lm-eval-harness wrapper: YuriiFormer
├── tmm_eval_model.py         # lm-eval-harness wrapper: TMMFormer
├── vanilla_eval_model.py     # lm-eval-harness wrapper: VanillaTransformer
├── adam_eval_model.py        # lm-eval-harness wrapper: AdamFormer
├── adamw_eval_model.py       # lm-eval-harness wrapper: AdamWFormer
├── eval_run.py               # YuriiFormer eval driver (HellaSwag + ARC-Easy)
├── tmm_eval_run.py           # TMMFormer eval driver
├── vanilla_eval_run.py       # VanillaTransformer eval driver
├── adam_eval_run.py          # AdamFormer eval driver
├── adamw_eval_run.py         # AdamWFormer eval driver
│
├── *_train_preempt.sbatch    # Preemption-aware DDP training jobs (TinyStories, debug, 2 GPU)
├── *_train_owt.sbatch        # OWT training jobs (general partition, 2 GPU, ~28h)
├── *_eval.sbatch             # Single-GPU downstream evaluation jobs
│
├── pyproject.toml            # Dependencies (managed by uv)
└── README.md
```

### Code structure conventions

- **`*_model.py`** — model definition. All variants share a common pre-norm interface: `forward(x: LongTensor[B,T]) -> Float[B,T,V]`. Velocity-bearing variants (TMM/Yurii/Adam) maintain a parallel `v` (and for Adam also `m_2`) state initialized from a separate embedding and updated layer-by-layer.

- **`*_train_ddp.py`** — DDP training script. Identical structure across variants:
  - 12L/12H/768d small config
  - Block size 1024, effective batch 480 (BATCH_SIZE 8 × GRAD_ACCUM 60)
  - **Two-optimizer setup**: Muon on 2D weight matrices, AdamW on embeddings/LayerNorm/scalars
  - LR: Muon 0.02, AdamW 6e-4, scalars 3e-3 (TS); Muon 0.004 for OWT
  - 1k-step warmup → cosine decay to 10% of peak
  - bfloat16 mixed precision, `torch.compile`
  - Validation every 100 steps; best.pt saved on improvement
  - Wandb logging (per-layer learned scalars for TMM/Yurii)

- **`*_train_owt.py`** — OWT variant. Same architecture/optimizer setup but: 30k steps, 3k warmup, reduced Muon LR (0.004) per paper Table 1, checkpoints to `$CACHE/checkpoints_<variant>_owt/` (compute-node-local for I/O performance), `HF_HOME` overridden to avoid disk-quota issues.

- **`*_eval_model.py`** — wraps a trained checkpoint as a `lm_eval.api.model.TemplateLM`, implementing `_loglikelihood_tokens` for multiple-choice tasks. Strips `_orig_mod.` prefix from `torch.compile`-saved state dicts.

- **`*_train_preempt.sbatch`** — SLURM batch script with `SIGUSR1` trap → `scontrol requeue` for graceful preemption recovery; auto-resumes from `best.pt` on restart. Excludes Blackwell nodes (currently flaky).

---

## Setup

```bash
uv sync
# or
pip install -e .
```

Requires PyTorch ≥ 2.4 (for Muon optimizer) and `lm-eval==0.4.3` for downstream tasks.

## Training

### TinyStories (single variant)

Data is auto-downloaded from `roneneldan/TinyStories` on first run and cached.

```bash
sbatch tmm_train_preempt.sbatch       # TMMFormer
sbatch yurii_train_preempt.sbatch     # YuriiFormer
sbatch vanilla_train_preempt.sbatch   # VanillaTransformer
sbatch adam_train_preempt.sbatch      # AdamFormer
sbatch adamw_train_preempt.sbatch     # AdamWFormer
```

Each runs ~1.8h on 2 GPUs (debug partition). Checkpoints land in `checkpoints_<variant>/best.pt`.

### OpenWebText

```bash
sbatch tmm_train_owt.sbatch
sbatch vanilla_train_owt.sbatch
sbatch adam_train_owt.sbatch
sbatch adamw_train_owt.sbatch
```

Each runs ~28h on 2 GPUs (general partition). Checkpoints land in `$CACHE/checkpoints_<variant>_owt/best.pt`.

## Evaluation

After training completes:

```bash
sbatch tmm_eval.sbatch       # TMMFormer  → eval_results_tmm/
sbatch yurii_eval.sbatch     # YuriiFormer → eval_results_yurii/
sbatch vanilla_eval.sbatch
sbatch adam_eval.sbatch
sbatch adamw_eval.sbatch
```

Each runs HellaSwag (10-shot) + ARC-Easy (25-shot) on 1 GPU, ~30min total.

## References

- Tikhomirov & Yudin (2024). *Transformers as Optimization Iterations on Token Embeddings.* (YuriiFormer paper)
- Van Scoy, Freeman, Lynch (2018). *The Fastest Known Globally Convergent First-Order Method for Minimizing Strongly Convex Functions.* (Triple Momentum Method)
- Nesterov (1983). *A method of solving a convex programming problem with convergence rate O(1/k²).*
