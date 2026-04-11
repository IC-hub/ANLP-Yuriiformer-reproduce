"""Strong causality test for transformer variants.

A causal LM must satisfy: changing input_ids at position k must NOT affect
output logits at any position i < k, but MUST affect the logits at position
k itself (otherwise the model is trivially constant).

This test runs each variant at the actual training sequence length (T=1024)
and checks both conditions across multiple change positions.
"""

import torch
from muon_model import MuonFormer
from soap_model import SOAPFormer
from vanilla_model import VanillaTransformer
from model import YuriiFormer

device = "cuda"
B, T = 2, 1024
VOCAB = 50304
CHANGE_POSITIONS = [3, 63, 255, 511, 1023]  # early, mid-early, mid, mid-late, last
TOL = 1e-4


def check_causality(model, name: str) -> bool:
    model.eval()
    torch.manual_seed(0)
    x1 = torch.randint(0, VOCAB, (B, T), device=device)

    all_pass = True
    with torch.no_grad():
        logits1 = model(x1)

        for k in CHANGE_POSITIONS:
            x2 = x1.clone()
            x2[0, k] = (x1[0, k] + 1) % VOCAB
            logits2 = model(x2)

            # 1) Positions [0, k) in batch 0 must be identical
            if k > 0:
                past_diff = (
                    (logits1[0, :k] - logits2[0, :k]).abs().max().item()
                )
            else:
                past_diff = 0.0

            # 2) Position k in batch 0 MUST differ (non-triviality)
            at_k_diff = (
                (logits1[0, k] - logits2[0, k]).abs().max().item()
            )

            # 3) Batch element 1 must be fully identical (no cross-batch leak)
            batch_diff = (
                (logits1[1] - logits2[1]).abs().max().item()
            )

            past_ok = past_diff < TOL
            at_k_ok = at_k_diff > TOL
            batch_ok = batch_diff < TOL
            ok = past_ok and at_k_ok and batch_ok

            if not ok:
                all_pass = False

            flag = "✓" if ok else "✗"
            print(
                f"  k={k:4d}: past_max={past_diff:.2e} "
                f"{'(OK)' if past_ok else '(LEAK)':6s}  "
                f"at_k={at_k_diff:.2e} "
                f"{'(OK)' if at_k_ok else '(DEAD)':6s}  "
                f"batch_max={batch_diff:.2e} "
                f"{'(OK)' if batch_ok else '(XBATCH)':9s} {flag}"
            )

    verdict = "✓ CAUSAL" if all_pass else "✗ BROKEN"
    print(f"  {name}: {verdict}\n")
    return all_pass


variants = [
    ("VanillaTransformer", VanillaTransformer),
    ("YuriiFormer", YuriiFormer),
    ("MuonFormer", MuonFormer),
    ("SOAPFormer", SOAPFormer),
]

results = {}
for name, cls in variants:
    print(f"=== {name} ===")
    model = cls().to(device)
    results[name] = check_causality(model, name)
    del model
    torch.cuda.empty_cache()

print("=" * 60)
print("Summary:")
for name, ok in results.items():
    print(f"  {name:22s}: {'✓ CAUSAL' if ok else '✗ BROKEN'}")

assert all(
    results[n] for n in ("VanillaTransformer", "YuriiFormer", "MuonFormer")
), "Causality test FAILED"
print("\nCausality test PASSED for Vanilla/Yurii/Muon")
