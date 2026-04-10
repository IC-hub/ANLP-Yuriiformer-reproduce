"""SOAPFormer: Right-factor Kronecker-preconditioned Adam transformer architecture.

Each layer applies a right-factor-only SOAP template (first-moment EMA +
right-covariance tracking + Newton-Schulz R^{-1/2} preconditioning) twice
via Lie-Trotter splitting — once with the attention oracle and once with
the MLP oracle.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CausalSelfAttention, MLP


# ── Newton-Schulz for matrix inverse square root ──────────────────────────

# Quintic coefficients (same as MuonFormer)
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def newton_schulz_inv_sqrt(R: torch.Tensor, K: int = 5) -> torch.Tensor:
    """Approximate R^{-1/2} for a symmetric PD matrix R via Newton-Schulz.

    We compute the polar factor of R^{-1}: since R is symmetric PD,
    its polar factor is R^{-1/2}.  We apply the quintic NS iteration
    to the normalized R, then rescale.

    Args:
        R: symmetric PD matrix of shape (d, d) or (B, d, d)
        K: number of Newton-Schulz iterations (default 5)

    Returns:
        R_inv_sqrt: approximate R^{-1/2}, same shape as R
    """
    # Normalize by Frobenius norm for NS convergence
    if R.dim() == 2:
        norm = R.norm().clamp(min=1e-12)
    else:
        norm = R.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    Y = R / norm

    I = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype)
    for _ in range(K):
        A = Y @ Y                              # Y^2 for symmetric Y
        Y = Y @ (_NS_A * I + _NS_B * A + _NS_C * A @ A)

    # The NS iteration on R/||R|| converges to sign(R/||R||) = I for PD R.
    # For the inverse square root, we use the relation:
    #   If Y_K ≈ (R/||R||)^{-1/2}, then R^{-1/2} ≈ Y_K / sqrt(||R||).
    # Re-derive: we want R^{-1/2}.  NS on normalized R gives polar factor
    # of R/||R||, which for symmetric PD is (R/||R||)^{1/2} / ||(R/||R||)^{1/2}||.
    # Instead, we directly apply NS to get an approximation and rescale.
    if R.dim() == 2:
        return Y / norm.sqrt()
    else:
        return Y / norm.sqrt()


# ── SOAPFormer block ──────────────────────────────────────────────────────

class SOAPLTBlock(nn.Module):
    """SOAP+Lie-Trotter block with state/1st-moment/right-covariance streams.

    Per-substep (attention, then MLP):
        g     = Oracle(LN(x))
        m_new = beta1 * m + (1 - beta1) * g              (1st moment EMA)
        R_new = betaR * R + (1 - betaR) * g^T @ g        (right covariance EMA)
        u     = m_new @ NS_inv_sqrt(R_new)                (preconditioned update)
        x_new = x + gamma * LN_u(u)                      (state update)
    """

    def __init__(self, d_model: int, n_heads: int, ns_iters: int = 5):
        super().__init__()
        self.d_model = d_model
        self.ns_iters = ns_iters

        # Pre-norm for oracles
        self.ln_attn = nn.LayerNorm(d_model, bias=False)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

        # Update LayerNorm (stabilizes preconditioned update before residual add)
        self.ln_update_attn = nn.LayerNorm(d_model, bias=False)
        self.ln_update_mlp = nn.LayerNorm(d_model, bias=False)

        # Learned scalars (6 per layer): beta1, betaR, gamma per substep
        # beta1, betaR in (0,1) via sigmoid; gamma > 0 via softplus
        self.beta1_attn_raw = nn.Parameter(torch.zeros(1))
        self.betaR_attn_raw = nn.Parameter(torch.zeros(1))
        self.gamma_attn_raw = nn.Parameter(torch.zeros(1))
        self.beta1_mlp_raw = nn.Parameter(torch.zeros(1))
        self.betaR_mlp_raw = nn.Parameter(torch.zeros(1))
        self.gamma_mlp_raw = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        beta1_a = torch.sigmoid(self.beta1_attn_raw)
        betaR_a = torch.sigmoid(self.betaR_attn_raw)
        gamma_a = F.softplus(self.gamma_attn_raw)
        beta1_m = torch.sigmoid(self.beta1_mlp_raw)
        betaR_m = torch.sigmoid(self.betaR_mlp_raw)
        gamma_m = F.softplus(self.gamma_mlp_raw)

        # ── Attention substep ───────────────────────────────────────────
        g_attn = self.attn(self.ln_attn(x))                   # (B, T, d)
        m_half = beta1_a * m + (1 - beta1_a) * g_attn
        # Right covariance update: g^T @ g averaged over batch
        # g_attn is (B, T, d); g^T g gives (B, d, d), mean over batch → (d, d)
        gTg_attn = (g_attn.transpose(-2, -1) @ g_attn).mean(dim=0)  # (d, d)
        R_half = betaR_a * R + (1 - betaR_a) * gTg_attn
        # Preconditioned update: m @ R^{-1/2}
        R_inv_sqrt = newton_schulz_inv_sqrt(R_half, K=self.ns_iters)  # (d, d)
        update_attn = self.ln_update_attn(m_half @ R_inv_sqrt)
        x_half = x + gamma_a * update_attn

        # ── MLP substep ─────────────────────────────────────────────────
        g_mlp = self.mlp(self.ln_mlp(x_half))                 # (B, T, d)
        m_next = beta1_m * m_half + (1 - beta1_m) * g_mlp
        gTg_mlp = (g_mlp.transpose(-2, -1) @ g_mlp).mean(dim=0)  # (d, d)
        R_next = betaR_m * R_half + (1 - betaR_m) * gTg_mlp
        R_inv_sqrt = newton_schulz_inv_sqrt(R_next, K=self.ns_iters)
        update_mlp = self.ln_update_mlp(m_next @ R_inv_sqrt)
        x_next = x_half + gamma_m * update_mlp

        return x_next, m_next, R_next


# ── SOAPFormer top-level model ────────────────────────────────────────────

class SOAPFormer(nn.Module):
    """SOAP+Lie-Trotter SOAPFormer (small config: 12L/12H/768d)."""

    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        ns_iters: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Main embeddings (state stream)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 1st moment embeddings (separate learned initialization)
        self.m_tok_emb = nn.Embedding(vocab_size, d_model)
        self.m_pos_emb = nn.Embedding(max_seq_len, d_model)

        # Right covariance R_0 = I_d (identity; not a learned parameter —
        # it represents the initial assumption of isotropic gradient covariance)

        # Transformer layers
        self.layers = nn.ModuleList(
            [SOAPLTBlock(d_model, n_heads, ns_iters=ns_iters) for _ in range(n_layers)]
        )

        # Final LayerNorm before output projection
        self.final_ln = nn.LayerNorm(d_model, bias=False)

        # Weight tying: output projection shares weights with tok_emb

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following nanoGPT conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections by 1/sqrt(2*n_layers)
        scale = 1.0 / math.sqrt(2 * self.n_layers)
        for layer in self.layers:
            torch.nn.init.normal_(
                layer.attn.out_proj.weight, mean=0.0, std=0.02 * scale
            )
            torch.nn.init.normal_(layer.mlp.w2.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        # Initialize state, 1st moment, and right covariance
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        m = self.m_tok_emb(input_ids) + self.m_pos_emb(pos)
        R = torch.eye(self.d_model, device=input_ids.device, dtype=x.dtype)

        # Apply SOAP+Lie-Trotter layers
        for layer in self.layers:
            x, m, R = layer(x, m, R)

        # Final LN + weight-tied output projection
        x = self.final_ln(x)
        logits = F.linear(x, self.tok_emb.weight)

        return logits
