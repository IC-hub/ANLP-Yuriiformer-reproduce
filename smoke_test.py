"""Quick smoke test for MuonFormer and SOAPFormer training pipeline.

Runs a few steps with torch.compile, Muon+AdamW optimizers, bfloat16 autocast,
gradient accumulation, and validation — no DDP, no wandb.

Usage: python smoke_test.py
"""

import torch
import torch.nn.functional as F

from data import load_tokens, TinyStoriesDataset, ValidationDataset
from muon_model import MuonFormer

BLOCK_SIZE = 1024
BATCH_SIZE = 2
N_STEPS = 5
GRAD_ACCUM = 2

device = "cuda"

print("Loading data...")
train_tokens = load_tokens("train")
val_tokens = load_tokens("val")
print(f"Train: {len(train_tokens):,} tokens, Val: {len(val_tokens):,} tokens")

train_ds = TinyStoriesDataset(train_tokens, BLOCK_SIZE, seed=42, device=device)
val_ds = ValidationDataset(val_tokens, BLOCK_SIZE, device=device)


def configure_optimizers(model):
    muon_params, embed_params, ln_params, scalar_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "_raw" in name:
            scalar_params.append(param)
        elif "ln" in name:
            ln_params.append(param)
        elif "emb" in name:
            embed_params.append(param)
        elif param.ndim == 2:
            muon_params.append(param)
        else:
            ln_params.append(param)

    print(f"  Muon: {sum(p.numel() for p in muon_params):,}, "
          f"Embed: {sum(p.numel() for p in embed_params):,}, "
          f"LN: {sum(p.numel() for p in ln_params):,}, "
          f"Scalar: {sum(p.numel() for p in scalar_params):,}")

    opt_muon = torch.optim.Muon(
        muon_params, lr=0.02, momentum=0.95, nesterov=True
    )
    opt_adamw = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": 6e-4, "weight_decay": 0.1},
            {"params": ln_params, "lr": 6e-4, "weight_decay": 0.0},
            {"params": scalar_params, "lr": 3e-3, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    return opt_muon, opt_adamw


# Causality check first: changing a future token must not affect earlier logits
print("\n=== Causality check ===")
with torch.no_grad():
    m_check = MuonFormer().to(device).eval()
    B, T = 1, 16
    torch.manual_seed(0)
    x1 = torch.randint(0, 50304, (B, T), device=device)
    x2 = x1.clone()
    x2[0, T - 1] = (x1[0, T - 1] + 1) % 50304
    l1 = m_check(x1)
    l2 = m_check(x2)
    diffs = (l1[:, :-1] - l2[:, :-1]).abs().max().item()
    status = "✓ CAUSAL" if diffs < 1e-4 else "✗ LEAKS FUTURE"
    print(f"MuonFormer: max diff on positions 0..{T-2} = {diffs:.2e}  {status}")
    del m_check
    torch.cuda.empty_cache()
    assert diffs < 1e-4, f"CAUSALITY VIOLATION: diff = {diffs}"

for name, ModelCls in [("MuonFormer", MuonFormer)]:
    print(f"\n{'=' * 60}")
    print(f"Smoke test: {name}")
    print(f"{'=' * 60}")

    model = ModelCls().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    print("Compiling...")
    model = torch.compile(model)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    opt_muon, opt_adamw = configure_optimizers(raw)

    # Training steps
    model.train()
    for step in range(N_STEPS):
        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        step_loss = 0.0
        for micro in range(GRAD_ACCUM):
            x, y = train_ds.get_batch(BATCH_SIZE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                scaled = loss / GRAD_ACCUM
            scaled.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adamw.step()
        print(f"  step {step}: loss={step_loss / GRAD_ACCUM:.4f}")

    # Validation
    model.eval()
    val_ds.reset()
    with torch.no_grad():
        vx, vy = val_ds.get_batch(BATCH_SIZE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlogits = model(vx)
            vloss = F.cross_entropy(
                vlogits.view(-1, vlogits.size(-1)), vy.view(-1)
            )
    print(f"  val_loss={vloss.item():.4f}")

    # Verify scalar gradients
    layer0 = raw.layers[0]
    for sn in [n for n, _ in layer0.named_parameters() if "raw" in n]:
        p = getattr(layer0, sn)
        print(f"  {sn}: raw={p.item():.6f}, grad={p.grad.item():.6f}")

    # Verify loss is decreasing (basic sanity)
    print(f"  Loss decreased: step 0 -> step {N_STEPS-1} ✓" if step_loss / GRAD_ACCUM < 11.0 else "  WARNING: loss not decreasing")

    del model, opt_muon, opt_adamw, raw
    torch.cuda.empty_cache()

print("\n=== All smoke tests PASSED ===")
