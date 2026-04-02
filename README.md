# YuriiFormer Reproduction

A from-scratch reproduction of the **YuriiFormer** (Nesterov+Lie-Trotter) small model (124M parameters) trained on TinyStories, plus downstream evaluation on HellaSwag and ARC-Easy.

YuriiFormer reinterprets transformer layers as Nesterov-accelerated optimization steps, introducing a velocity stream alongside the standard hidden state. Each layer has 6 learnable scalars (mu, beta, gamma for attention and MLP) that control lookahead, momentum, and step size.

## Reproduction Results

| Metric | Paper | Ours |
|--------|-------|------|
| Best val loss | 1.078 | 1.076 |
| Final val loss | 1.090 | 1.088 |
| Train @10k | 0.896 | 0.879 |

## Project Structure

```
├── model.py         # YuriiFormer architecture (124M + 38.6M velocity params)
├── data.py          # TinyStories tokenization (GPT-2 BPE) and dataloader
├── train.py         # Training loop: Muon + AdamW, cosine LR, gradient accumulation
├── eval_model.py    # lm-evaluation-harness wrapper for the custom model
├── eval_run.py      # Downstream evaluation driver (HellaSwag, ARC-Easy)
├── run.sh           # SLURM job script for training
├── eval.sh          # SLURM job script for evaluation
└── pyproject.toml   # Dependencies managed by uv
```

## Setup

```bash
uv sync
```

## Training

Data is automatically downloaded from HuggingFace (`roneneldan/TinyStories`) on the first run and cached as tokenized `.npy` files under `DATA_DIR` in `data.py`.

Submit the training job (requires single GPU, ~5 hours with H200):

```bash
bash run.sh
```

This trains for 10k steps with 16-step gradient accumulation (effective batch size 480), cosine LR schedule with 1k-step warmup, and bfloat16 mixed precision. Checkpoints are saved to `checkpoints/`. Metrics are logged to Weights & Biases.

## Evaluation

After training, run downstream evaluation on the best checkpoint:

```bash
bash eval.sh
```

This evaluates on HellaSwag (10-shot) and ARC-Easy (25-shot) using lm-evaluation-harness v0.4.3. Results and per-sample predictions are saved to `eval_results/`.
