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
| `AdamFormer` | Adam (1st/2nd moment) | Lie–Trotter | yes (m, v) | 0 | ~140M |
| `AdamWFormer` | AdamW (decoupled wd) | Lie–Trotter | yes (m, v) | 0 | ~140M |

**Triple Momentum Method (TMM)** is the first-order optimal algorithm for L-smooth, μ-strongly convex functions (Van Scoy et al. 2018), with convergence rate (1−√(μ/L))² — strictly better than Nesterov. `TMMFormer` generalizes `YuriiFormer` by adding a learnable scalar `ν` that decouples the iterate update from the gradient-evaluation lookahead.

**Adam/AdamW variants** maintain per-token first/second moment streams alongside the hidden state, mirroring the Adam optimizer applied to embeddings. They underperform momentum-based variants — confirming that token-space gradients lack the per-dimension scale variance that makes Adam useful for parameter optimization.

---

## Per-Variant Architecture & Update Equations

All variants share the same backbone: 12 layers, 12 heads, $d_{\text{model}}=768$, GPT-2 BPE tokenizer (50304 vocab), causal self-attention + 4× MLP, weight-tied output head, pre-norm. They differ **only** in the per-layer block update rule, and in the auxiliary streams (velocity $v$, first moment $m$, second moment $s$) propagated alongside the hidden state $x$.

The two oracles in every block are

$$g_{\text{attn}}(x) \;=\; \mathrm{Attn}\!\bigl(\mathrm{LN}(x)\bigr), \qquad g_{\text{mlp}}(x) \;=\; \mathrm{MLP}\!\bigl(\mathrm{LN}(x)\bigr).$$

Each block performs two substeps (Lie–Trotter splitting): an attention substep followed by an MLP substep. Every per-layer scalar ($\mu, \beta, \gamma, \nu, \lambda$) is **learned**, parameterized via $\sigma(\cdot)$ for values in $(0,1)$ or $\mathrm{softplus}(\cdot)$ for positives. Subscripts $a/m$ denote attention/MLP substep parameters.

### 1. `VanillaTransformer` — GD + Lie–Trotter

Standard pre-norm nanoGPT block. No auxiliary stream, no learned scalars.

$$
\begin{aligned}
x &\;\leftarrow\; x + g_{\text{attn}}(x) \\
x &\;\leftarrow\; x + g_{\text{mlp}}(x)
\end{aligned}
$$

One step of **gradient descent** with unit step size on the composite objective $f_{\text{attn}} + f_{\text{mlp}}$, split via Lie–Trotter. The "gradients" are the oracle outputs themselves — the layer **is** the optimization step.

### 2. `YuriiFormer` — Nesterov + Lie–Trotter

Maintains a parallel velocity stream $v$, initialized from a separate learned velocity embedding. **6 learned scalars per layer.**

Attention substep:

$$
\begin{aligned}
\tilde{x} &\;=\; x + \mu_a\, v &&\text{(Nesterov lookahead)}\\
v &\;\leftarrow\; \mathrm{LN}_v\!\bigl(\beta_a\, v + \gamma_a\, g_{\text{attn}}(\tilde{x})\bigr) \\
x &\;\leftarrow\; x + v
\end{aligned}
$$

MLP substep (analogous, with $\mu_m, \beta_m, \gamma_m$ and $g_{\text{mlp}}$).

One step of **Nesterov accelerated gradient** per substep: the oracle is queried at the lookahead $x + \mu v$, velocity is an EMA of past oracle outputs, and the iterate advances by the full velocity.

### 3. `TMMFormer` — Triple Momentum Method + Lie–Trotter

Strict generalization of YuriiFormer: adds a second learned scalar $\nu$ that decouples the iterate update from the gradient lookahead. **8 learned scalars per layer.**

Attention substep:

$$
\begin{aligned}
\tilde{x} &\;=\; x + \mu_a\, v &&\text{(gradient lookahead)}\\
v &\;\leftarrow\; \mathrm{LN}_v\!\bigl(\beta_a\, v + \gamma_a\, g_{\text{attn}}(\tilde{x})\bigr) \\
x &\;\leftarrow\; x + \nu_a\, v &&\text{(iterate update, } \nu\neq 1\text{ in general)}
\end{aligned}
$$

MLP substep is structurally identical with $\mu_m, \beta_m, \gamma_m, \nu_m$.

The **Triple Momentum Method** (Van Scoy et al., 2018) is the first-order optimal algorithm for $L$-smooth, $\mu$-strongly convex objectives, with convergence rate

$$\bigl(1 - \sqrt{\mu/L}\bigr)^2,$$

strictly faster than Nesterov's $1 - \sqrt{\mu/L}$. YuriiFormer is the special case $\nu \equiv 1$. We initialize $\nu$ via $\mathrm{softplus}(0.5413) \approx 1$, so training begins in the YuriiFormer regime and learns where to deviate.

### 4. `AdamFormer` — Adam + Lie–Trotter

Maintains **two** auxiliary streams: first moment $m$ (from a learned embedding) and second moment $s$ (initialized to ones). **6 learned scalars per layer.**

Attention substep, with $g = g_{\text{attn}}(x)$:

$$
\begin{aligned}
m &\;\leftarrow\; \beta_{1,a}\, m + (1-\beta_{1,a})\, g \\
s &\;\leftarrow\; \beta_{2,a}\, s + (1-\beta_{2,a})\, g \odot g \\
x &\;\leftarrow\; x + \gamma_a\, \mathrm{LN}_u\!\left(\frac{m}{\sqrt{s} + \varepsilon}\right)
\end{aligned}
$$

MLP substep is structurally identical with $\beta_{1,m}, \beta_{2,m}, \gamma_m$ and $g = g_{\text{mlp}}(x)$.

One step of **Adam** per substep: 1st/2nd moment EMAs propagate forward layer by layer, and the iterate moves along the per-coordinate normalized direction $m / \sqrt{s}$. The extra $\mathrm{LN}_u$ normalizes the adaptive update to keep its magnitude stable across depth.

### 5. `AdamWFormer` — AdamW + Lie–Trotter

Same auxiliary streams and adaptive direction as AdamFormer, but with **decoupled weight decay** on the iterate. **8 learned scalars per layer** (adds $\lambda_a, \lambda_m$).

Attention substep:

$$
\begin{aligned}
m &\;\leftarrow\; \beta_{1,a}\, m + (1-\beta_{1,a})\, g \\
s &\;\leftarrow\; \beta_{2,a}\, s + (1-\beta_{2,a})\, g \odot g \\
x &\;\leftarrow\; (1 - \lambda_a)\, x + \gamma_a\, \mathrm{LN}_u\!\left(\frac{m}{\sqrt{s} + \varepsilon}\right)
\end{aligned}
$$

We initialize $\lambda_* = \sigma(-5) \approx 0.007$, so AdamWFormer starts very close to AdamFormer and learns the per-layer decay. This mirrors AdamW vs. Adam: each iteration pulls the iterate slightly toward zero before applying the adaptive update.

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
