"""Plot training curves for MuonFormer and SOAPFormer runs."""

import re
import matplotlib.pyplot as plt

LOGS = {
    "MuonFormer": "logs/muonformer_train_7036576.out",
    "SOAPFormer": "logs/soapformer_train_7036577.out",
}


def parse_log(path):
    steps, losses = [], []
    val_steps, val_losses = [], []
    step_re = re.compile(r"^step\s+(\d+)\s+\|\s+loss\s+([0-9.]+)")
    val_re = re.compile(r"val_loss:\s+([0-9.]+)")
    last_step = 0
    with open(path) as f:
        for line in f:
            m = step_re.match(line)
            if m:
                last_step = int(m.group(1))
                steps.append(last_step)
                losses.append(float(m.group(2)))
                continue
            v = val_re.search(line)
            if v:
                val_steps.append(last_step)
                val_losses.append(float(v.group(1)))
    return steps, losses, val_steps, val_losses


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, path in LOGS.items():
    try:
        steps, losses, val_steps, val_losses = parse_log(path)
    except FileNotFoundError:
        print(f"Missing log: {path}")
        continue
    print(f"{name}: {len(steps)} train points, {len(val_losses)} val points")
    axes[0].plot(steps, losses, label=name, alpha=0.8)
    axes[1].plot(val_steps, val_losses, label=name, marker="o", markersize=3, alpha=0.8)

for ax, title, ylabel in [
    (axes[0], "Training Loss", "train loss"),
    (axes[1], "Validation Loss", "val loss"),
]:
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(1.13, color="gray", linestyle="--", alpha=0.5, label="~TMM final (1.13)")

plt.tight_layout()
plt.savefig("muon_soap_curves.png", dpi=120, bbox_inches="tight")
print("Saved muon_soap_curves.png")
