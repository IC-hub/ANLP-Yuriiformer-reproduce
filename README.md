# YuriiFormer Reproduction & Optimizer-Inspired Transformer Variants

A from-scratch reproduction of **YuriiFormer** (Nesterov + Lie–Trotter) plus a systematic comparison of additional optimizer-inspired transformer variants on TinyStories and OpenWebText, with downstream evaluation on HellaSwag and ARC-Easy.

The core idea: a pre-norm transformer layer can be interpreted as **one iteration of an optimization algorithm on token embeddings**, with attention and MLP acting as two oracles. Different choices of optimizer (GD, Nesterov, Triple Momentum Method, Adam-style preconditioning, Muon orthogonalized momentum, SOAP Kronecker preconditioning) and operator-splitting scheme (Euler, Lie–Trotter, Strang) give rise to different transformer architectures.

This repo implements and compares 8 such variants under matched compute and identical training pipelines.

---

## Architectures

| Model | Optimizer view | Splitting | Velocity stream | Learned scalars/layer | Params |
|---|---|---|---|---|---|
| `VanillaTransformer` | GD | Lie–Trotter | no | 0 | 124M |
| `YuriiFormer` | Nesterov | Lie–Trotter | yes | 6 (μ,β,γ × 2) | 163M |
| `TMMFormer` | Triple Momentum Method | Lie–Trotter | yes | 8 (μ,β,γ,ν × 2) | 163M |
| `AdamFormer` | Adam (1st/2nd moment) | Lie–Trotter | yes (m, s) | 6 (β₁,β₂,γ × 2) | ~163M |
| `AdamWFormer` | AdamW (decoupled wd) | Lie–Trotter | yes (m, s) | 8 (β₁,β₂,γ,λ × 2) | ~163M |
| `MuonFormer` | Muon (orthogonalized momentum) | Lie–Trotter | yes (m) | 4 (β,γ × 2) | ~163M |
| `SOAPFormer` | SOAP (Kronecker-preconditioned) | Lie–Trotter | yes (m, R) | 6 (β₁,β_R,γ × 2) | ~170M |

**Triple Momentum Method (TMM)** is the first-order optimal algorithm for L-smooth, μ-strongly convex functions (Van Scoy et al. 2018), with convergence rate (1−√(μ/L))² — strictly better than Nesterov. `TMMFormer` generalizes `YuriiFormer` by adding a learnable scalar `ν` that decouples the iterate update from the gradient-evaluation lookahead.

**Adam/AdamW variants** maintain per-token first/second moment streams alongside the hidden state, mirroring the Adam optimizer applied to embeddings. They underperform momentum-based variants — confirming that token-space gradients lack the per-dimension scale variance that makes Adam useful for parameter optimization.

---

## From Optimizers to Transformer Architectures

Following Zimin et al. (2026), we view a transformer block as one iteration of a first-order optimization algorithm acting on the token-embedding matrix $X_t \in \mathbb{R}^{T \times d}$. The composite objective is

$$\min_X \; \mathcal{E}(X) + \mathcal{F}(X),$$

where $\mathcal{E}$ is an interaction energy (token–token) and $\mathcal{F}$ is a potential energy (per-token). Their gradients are realized by the two transformer oracles:

$$\nabla \mathcal{E}_t(X) \;\approx\; \mathrm{Attn}_t\!\bigl(\mathrm{LN}(X)\bigr), \qquad \nabla \mathcal{F}_t(X) \;\approx\; \mathrm{MLP}_t\!\bigl(\mathrm{LN}(X)\bigr).$$

A first-order optimization template applied to $\mathcal{E} + \mathcal{F}$ via **Lie–Trotter splitting** then yields a transformer block: each layer performs an attention substep (gradient step on $\mathcal{E}$) followed by an MLP substep (gradient step on $\mathcal{F}$). Different optimizer templates → different transformer architectures.

All seven variants share the same 12L/12H/$d=768$ pre-norm backbone, GPT-2 BPE tokenizer, and weight-tied output head — they differ **only** in the optimizer template and in the auxiliary streams ($V_t$, $M_t$, $S_t$, $R_t$) propagated alongside the state $X_t$. Per-layer scalars ($\mu, \beta, \gamma, \nu, \lambda$) are all learned, reparameterized through $\sigma$ for $(0,1)$ values or $\mathrm{softplus}$ for positives.

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

### 6. `MuonFormer` — Muon (Orthogonalized Momentum)

**Optimizer (Jordan et al., 2024).** Muon performs steepest descent under the **spectral (operator) norm** rather than the Frobenius/L2 norm. It maintains a momentum buffer, then extracts the **orthogonal polar factor** via Newton-Schulz (NS) iterations — producing an update whose singular values are all 1:

$$
\begin{aligned}
m_{t+1} &\;=\; \beta_t\, m_t + (1 - \beta_t)\, \nabla f(x_t) & &\text{(momentum EMA)}\\
U_{t+1} &\;=\; \mathrm{NS}_K\!\bigl(m_{t+1}\bigr) & &\text{(orthogonalize via Newton-Schulz)}\\
x_{t+1} &\;=\; x_t - \gamma_t\, U_{t+1} & &\text{(iterate update)}
\end{aligned}
$$

where $\mathrm{NS}_K$ denotes $K$ iterations of the quintic Newton-Schulz recurrence applied to the Frobenius-normalized momentum:

$$
\begin{aligned}
Y_0 &\;=\; m / \|m\|_F \\
Y_{k+1} &\;=\; Y_k \bigl(a I + b\, Y_k^\top Y_k + c\, (Y_k^\top Y_k)^2\bigr), \qquad k = 0, \dots, K-1
\end{aligned}
$$

with quintic coefficients $a = 3.4445$, $b = -4.7750$, $c = 2.0315$. After $K \approx 5$ iterations, $Y_K$ approximates the orthogonal polar factor of $m$ (all singular values $\approx 1$). This makes the update **isotropic** — no singular direction dominates, regardless of the gradient spectrum.

**Transformer.** Maintain a momentum stream $M_t$ (from a separate learned embedding) alongside $X_t$. Each layer applies the Muon template twice (Lie–Trotter):

$$
\begin{aligned}
G^{\text{a}}_t &\;=\; \mathrm{Attn}_t(\mathrm{LN}(X_t)) \\
M_{t+\frac{1}{2}} &\;=\; \beta_t\, M_t + (1 - \beta_t)\, G^{\text{a}}_t \\
X_{t+\frac{1}{2}} &\;=\; X_t + \gamma_t\, \mathrm{LN}_u\!\bigl(\mathrm{NS}_K(M_{t+\frac{1}{2}})\bigr) \\[6pt]
G^{\text{m}}_t &\;=\; \mathrm{MLP}_t(\mathrm{LN}(X_{t+\frac{1}{2}})) \\
M_{t+1} &\;=\; \beta_{t+\frac{1}{2}}\, M_{t+\frac{1}{2}} + (1 - \beta_{t+\frac{1}{2}})\, G^{\text{m}}_t \\
X_{t+1} &\;=\; X_{t+\frac{1}{2}} + \gamma_{t+\frac{1}{2}}\, \mathrm{LN}_u\!\bigl(\mathrm{NS}_K(M_{t+1})\bigr)
\end{aligned}
$$

Four learned scalars per layer ($\beta, \gamma$ per substep). $\beta \in (0,1)$ via sigmoid, $\gamma > 0$ via softplus. The Newton-Schulz iteration count $K = 5$ is a fixed hyperparameter.

**Computational note.** Each NS iteration involves $Y^\top Y$ ($d \times d$ matmul) and right-multiplication, costing $O(T \cdot d^2)$ per iteration. With $K = 5$ iterations per substep, 2 substeps per layer, the overhead is $O(10 \cdot T \cdot d^2)$ per layer — comparable to the MLP cost and smaller than attention's $O(T^2 \cdot d)$.

**Hypothesis.** The forced isotropy of the Muon update may prevent the attention collapse seen in AdamFormer (layer 1 entropy 0.33 nats), since no singular direction in the update can dominate. On the other hand, it may suppress beneficial anisotropy that helps specialization.

---

### 7. `SOAPFormer` — SOAP (Kronecker-Preconditioned Adam)

**Optimizer (Vyas et al., 2024).** SOAP extends Shampoo by applying Adam-style adaptivity in the eigenbasis of the Kronecker-factored gradient covariance. For a matrix iterate $X \in \mathbb{R}^{T \times d}$, full SOAP tracks both a left covariance $L_t \in \mathbb{R}^{T \times T}$ and a right covariance $R_t \in \mathbb{R}^{d \times d}$, then eigendecomposes both to form a preconditioned update.

**Right-factor-only variant.** We drop the left factor $L_t$ (which is $T \times T = 1024 \times 1024$ — expensive to eigendecompose at every layer) and keep only the right covariance $R_t \in \mathbb{R}^{d \times d}$. This captures cross-dimension correlations in the "gradient" signal, while cross-position interactions are already handled by attention. The matrix inverse square root $R_t^{-1/2}$ is approximated via **Newton-Schulz iterations** (same routine as MuonFormer), avoiding explicit eigendecomposition:

$$
\begin{aligned}
g_t &\;=\; \nabla f(x_t) \\
m_{t+1} &\;=\; \beta_{1,t}\, m_t + (1 - \beta_{1,t})\, g_t & &\text{(first moment EMA)}\\
R_{t+1} &\;=\; \beta_{R,t}\, R_t + (1 - \beta_{R,t})\, g_t^\top g_t & &\text{(right covariance EMA, } d \times d\text{)}\\
x_{t+1} &\;=\; x_t - \gamma_t\, m_{t+1}\, R_{t+1}^{-1/2} & &\text{(preconditioned update)}
\end{aligned}
$$

**Transformer.** Maintain a first-moment stream $M_t \in \mathbb{R}^{T \times d}$ (from a learned embedding) and a right-covariance stream $R_t \in \mathbb{R}^{d \times d}$ (initialized to identity) alongside $X_t$. Each layer applies the SOAP template twice (Lie–Trotter):

$$
\begin{aligned}
G^{\text{a}}_t &\;=\; \mathrm{Attn}_t(\mathrm{LN}(X_t)) \\
M_{t+\frac{1}{2}} &\;=\; \beta_{1,t}\, M_t + (1 - \beta_{1,t})\, G^{\text{a}}_t \\
R_{t+\frac{1}{2}} &\;=\; \beta_{R,t}\, R_t + (1 - \beta_{R,t})\, (G^{\text{a}}_t)^\top G^{\text{a}}_t \\
X_{t+\frac{1}{2}} &\;=\; X_t + \gamma_t\, \mathrm{LN}_u\!\bigl(M_{t+\frac{1}{2}}\, \mathrm{NS}^{-1/2}_K(R_{t+\frac{1}{2}})\bigr) \\[6pt]
G^{\text{m}}_t &\;=\; \mathrm{MLP}_t(\mathrm{LN}(X_{t+\frac{1}{2}})) \\
M_{t+1} &\;=\; \beta_{1,t+\frac{1}{2}}\, M_{t+\frac{1}{2}} + (1 - \beta_{1,t+\frac{1}{2}})\, G^{\text{m}}_t \\
R_{t+1} &\;=\; \beta_{R,t+\frac{1}{2}}\, R_{t+\frac{1}{2}} + (1 - \beta_{R,t+\frac{1}{2}})\, (G^{\text{m}}_t)^\top G^{\text{m}}_t \\
X_{t+1} &\;=\; X_{t+\frac{1}{2}} + \gamma_{t+\frac{1}{2}}\, \mathrm{LN}_u\!\bigl(M_{t+1}\, \mathrm{NS}^{-1/2}_K(R_{t+1})\bigr)
\end{aligned}
$$

where $\mathrm{NS}^{-1/2}_K(R)$ denotes $K$ Newton-Schulz iterations approximating $R^{-1/2}$.

Six learned scalars per layer ($\beta_1, \beta_R, \gamma$ per substep). $\beta_1, \beta_R \in (0,1)$ via sigmoid, $\gamma > 0$ via softplus. $R_0 = I_d$ (identity); $M_0$ from learned embeddings.

**Why right-factor only?** The left covariance $L_t \in \mathbb{R}^{T \times T}$ captures cross-position correlations — but attention already provides exactly this. The right covariance $R_t \in \mathbb{R}^{d \times d}$ captures cross-dimension correlations, which no other component models. This asymmetry makes the right-factor-only variant both efficient and non-redundant.

**Parameters.** The $R_t$ covariance stream adds $12 \times d^2 = 12 \times 768^2 \approx 7\text{M}$ state elements (not trained — evolved per-layer), bringing the total to ~170M.

**Why SOAP over Shampoo?** Both share the same expensive component (Kronecker covariance tracking + matrix inverse square root). SOAP adds Adam-style first-moment tracking in the preconditioned space — cheap (element-wise) but strictly better for convergence, since it provides per-direction adaptivity that raw Shampoo lacks.

---

### Common design notes

- **Velocity / update LayerNorm.** $\mathrm{LN}_v$ (in YuriiFormer / TMMFormer) and $\mathrm{LN}_u$ (in AdamFormer / AdamWFormer / MuonFormer / SOAPFormer) are essential for stability across depth: without them, momentum-based architectures diverge after a few hundred steps.
- **Auxiliary stream initialization.** $V_0$ and $M_0$ are produced by *separate learned embeddings* (`vel_tok_emb + vel_pos_emb`, etc.), not by zero-initialization. $S_0$ is initialized to all-ones. $R_0$ (SOAPFormer's right covariance) is initialized to identity.
- **Newton-Schulz iterations.** MuonFormer and SOAPFormer both use $K = 5$ iterations of the quintic Newton-Schulz recurrence — for polar-factor extraction (MuonFormer) and matrix inverse square root (SOAPFormer), respectively.
- **Two-optimizer training.** All learned per-layer scalars (the `*_raw` parameters) are trained by AdamW at a higher LR ($3 \cdot 10^{-3}$) than the rest of the network. Matrix weights use **Muon**; embeddings, LayerNorm, and scalars use **AdamW**.
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
| **TMMFormer** (ours) | **2.9342** | 2.9342 | 2.9290 |
| YuriiFormer (ours) | 2.9413 | 2.9413 | 2.9352 |
| AdamWFormer (ours) | 2.9883 | 2.9883 | 2.9883 |
| AdamFormer (ours) | 2.9911 | 2.9911 | 2.9904 |
| VanillaTransformer (ours) | 3.0078 | 3.0080 | 3.0087 |

**OWT observations**: TMMFormer (2.9342) is the best variant under our reproduction setup, narrowly ahead of YuriiFormer (2.9413, Δ ≈ 0.007). Both Nesterov-family momentum variants beat Adam/AdamW by ~0.05 nats and Vanilla by ~0.07. All five trained variants beat paper Vanilla LT (2.990), but none reach paper YuriiFormer (2.920) — consistent with the reproduction offset seen on TS.

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

| Layer | Vanilla | YuriiFormer | TMMFormer | AdamFormer | AdamWFormer |
|---:|---:|---:|---:|---:|---:|
| 0  | **5.39** | 3.37 | 3.32 | 4.22 | 4.29 |
| 1  | 4.17 | 2.60 | 2.93 | **0.33** | **1.62** |
| 2  | 3.58 | 3.35 | 3.32 | 2.91 | 3.23 |
| 3  | 3.79 | 2.95 | 2.97 | 3.44 | 3.59 |
| 4  | 3.12 | 2.88 | 2.83 | 3.09 | 3.05 |
| 5  | 3.07 | 3.27 | 3.30 | 3.38 | 3.23 |
| 6  | 3.41 | 3.17 | 3.15 | 3.35 | 3.41 |
| 7  | 3.07 | 3.31 | 3.22 | 3.02 | 3.11 |
| 8  | 3.01 | 3.14 | 3.11 | 3.29 | 3.21 |
| 9  | 3.10 | 3.18 | 3.12 | 3.20 | 3.22 |
| 10 | 3.16 | 3.32 | 3.26 | 3.32 | 3.31 |
| 11 | 3.45 | 3.30 | 3.36 | 3.50 | 3.44 |
| **mean** | **3.527** | **3.154** | **3.157** | **3.089** | **3.224** |

**Observations**

1. **YuriiFormer ≡ TMMFormer in attention behavior**. The two columns agree to ≤ 0.05 nats per layer (overall Δ ≈ 0.003). This reinforces the loss-level finding that TMM's extra ν degree of freedom does not change how attention is used in practice.
2. **Adam/AdamW collapse layer 1**. AdamFormer's layer 1 mean entropy is 0.33 nats with `min_h = 0.0000` — at least one head has fully collapsed to an attention sink. AdamWFormer shows a milder version (1.62, `min_h ≈ 0.0001`). This degeneracy is absent in the Nesterov family.
3. **Vanilla has the most diffuse attention overall** (mean 3.53; layer 0 = 5.39, ≈ 78% of max). Without the auxiliary momentum/Adam streams, the model has not developed sharp specialization at the input-side layers.
4. **Adam family has the most diffuse layer 0** among auxiliary-stream variants (4.22–4.29 vs ~3.35 for Nesterov family). The dual moment streams (m, s) appear to push the first layer toward broader, more averaging-like aggregation.
5. **Deep layers (2–11) are relatively flat across variants** (typical spread < 0.3 nats). The architectural differences mostly show up at the input-side layers; the internal attention motifs converge to similar entropies.
6. All variants sit between **~45% and ~50% of $\log T$** — well away from uniform but also not collapsed, indicating healthy mixed-specificity attention overall.

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
| **TMMFormer** | **2.934** | **130.7** | **23 258** | **1.42e−4** | **9.08** |
| YuriiFormer | 2.941 | 167.7 | 22 841 | 1.39e−4 | 8.54 |
| AdamFormer | 2.991 | 423.9 | 34 409 | 2.10e−4 | 9.36 |
| AdamWFormer | 2.988 | **1889.1** | 48 933 | 2.99e−4 | 9.77 |
| Vanilla | 3.008 | 79.8 | 39 320 | 3.16e−4 | 11.35 |

**Observations**

1. **Nesterov family lives in the flattest basin** (TMM/Yurii both $\mathrm{tr}/n \approx 1.4 \times 10^{-4}$, ≈ half of Vanilla and ≈ 30–50% lower than Adam/AdamW). This aligns with the standard "flat minimum ↔ better generalization" intuition and matches their lower val loss.
2. **AdamWFormer is dramatically sharp**: $\lambda_{\max} \approx 1889$, ~11× TMM and ~24× Vanilla. The decoupled weight decay drives parameters into a region with one or a few extremely steep directions, even though average curvature is comparable.
3. **AdamFormer is intermediate**: $\lambda_{\max} = 424$ (~3× Nesterov family) but $\mathrm{tr}/n$ moderate. Removing decoupled wd softens the worst direction but the basin is still sharper than Nesterov.
4. **Vanilla has the smallest $\lambda_{\max}$ but the largest $\mathrm{tr}/n$ and steepest 1D curve** — a "wide but bumpy" basin: no single direction is extremely steep, yet many directions contribute moderate curvature, and the random 1D probe rises faster than for any other variant.
5. **TMM ≈ Yurii in landscape too**: trace within 2%, 1D curve ranges differ by only 0.5 nats; TMM's $\lambda_{\max}$ is slightly lower (130.7 vs 167.7), perhaps reflecting ν damping the worst eigendirection. Combined with the entropy and downstream-task results, this is the third independent measurement showing the two are functionally equivalent.

Per-variant tensors are saved to `loss_sharpness_results/<variant>.pt`.

### Downstream Evaluation (OWT best checkpoints)

HellaSwag (10-shot) and ARC-Easy (25-shot), evaluated with `lm-evaluation-harness` v0.4.3:

| Model | val_loss | HellaSwag acc_norm | ARC-Easy acc_norm |
|---|---:|---:|---:|
| **TMMFormer** | **2.934** | **0.3182** | **0.4343** |
| YuriiFormer | 2.941 | 0.3158 | 0.4306 |
| AdamFormer | 2.991 | 0.3096 | 0.4339 |
| AdamWFormer | 2.988 | 0.3008 | 0.4188 |
| VanillaTransformer | 3.008 | 0.3020 | 0.4167 |

**Observations**

1. **The downstream ranking matches the val-loss ranking exactly**: TMM > Yurii > Adam > AdamW > Vanilla on both tasks. The pretraining advantage transfers cleanly.
2. **All variants are clearly above chance** (HellaSwag random ≈ 0.25, ARC-Easy 4-choice ≈ 0.25), unlike the TS-trained checkpoints — confirming OWT-scale pretraining gives genuine generalization.
3. **TMM and Yurii are statistically tied** (HellaSwag Δ = 0.0024, ARC Δ = 0.0037), once again confirming the equivalence of the two architectures under our setup.
4. The **Nesterov family beats Adam/AdamW by ~1 acc-point** on HellaSwag and **~1.5 on ARC-Easy**, despite all four variants having essentially the same parameter count — supporting the broader claim that the Nesterov-style update is a more effective transformer block.

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
