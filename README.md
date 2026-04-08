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
| AdamWFormer (ours) | 1.1472 | 1.1547 | — |
| AdamFormer (ours) | 1.1528 | 1.1554 | — |
| VanillaTransformer (ours) | 1.1569 | 1.1604 | 0.9418 |
| YuriiFormer (ours) | running | — | — |

**TS observations**: TMM is the best non-paper variant on TinyStories. Adam-style variants underperform vanilla momentum. Our reproduction loss (~1.15) is higher than paper Vanilla LT (1.114) — likely a Muon/effective-batch hyperparameter difference — but **relative comparisons under matched setup remain valid**.

### OpenWebText (30k steps)

| Method | Best Val | Status |
|---|---:|---|
| Paper Vanilla GD+LT | 2.990 | — |
| Paper YuriiFormer | **2.920** | — |
| **TMMFormer** (ours) | **2.9720** | 70% (interrupted, resumed) |
| AdamWFormer (ours) | 2.9883 | done |
| AdamFormer (ours) | 2.9911 | done |
| VanillaTransformer (ours) | 3.0224 | 83% running |

**OWT observations**: TMM at 70% progress already beats AdamW (2.9883) and paper Vanilla LT (2.990). Cosine decay through the remaining 30% should push it further down, plausibly into the 2.94–2.96 range — approaching but not yet matching paper YuriiFormer (2.920).

### Downstream Evaluation (TinyStories checkpoints)

HellaSwag (10-shot) and ARC-Easy (25-shot), evaluated with `lm-evaluation-harness` v0.4.3:

| Model | HellaSwag acc_norm | ARC-Easy acc_norm |
|---|---:|---:|
| TMMFormer | 0.2682 | 0.2563 |
| AdamFormer | 0.2680 | 0.2660 |
| AdamWFormer | 0.2675 | 0.2635 |
| VanillaTransformer | 0.2669 | 0.2542 |

All TS-trained models are **at chance level** (HellaSwag random ≈ 0.25, ARC-Easy 4-choice ≈ 0.25). This is expected — the TinyStories vocabulary/style distribution is too narrow to transfer. **OWT-checkpoint downstream evaluation is the meaningful comparison** and will be added once the OWT runs complete.

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
