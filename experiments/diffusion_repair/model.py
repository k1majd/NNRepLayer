"""
Score network (MLP) and LoRA adapter for the 2D diffusion repair experiment.

ScoreNet
--------
  Input : (x, t)  where x ∈ R² and t is an integer timestep index.
  Output: predicted noise ε ∈ R² (DDPM ε-parameterisation).
  Architecture: sinusoidal time embedding → MLP with SiLU activations.

LoRALinear
----------
  Wraps a frozen nn.Linear and adds a trainable rank-r delta:
      W' = W + B·A / r        (A ∈ R^{r×in},  B ∈ R^{out×r})

LoRAScoreNet
------------
  Replaces every nn.Linear inside a ScoreNet with LoRALinear.
  Only lora_A / lora_B parameters are trainable; all base weights are frozen.
  Initial delta is exactly 0 (standard LoRA initialisation).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timestep indices to sinusoidal embeddings.

    Args:
        t:   (B,)  long tensor of timestep indices.
        dim: embedding dimension (must be even).

    Returns:
        (B, dim) float tensor.
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device).float()
        / max(half - 1, 1)
    )  # (dim/2,)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, dim/2)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Base score network
# ---------------------------------------------------------------------------


class ScoreNet(nn.Module):
    """Small MLP that predicts the noise ε given (x_t, t)."""

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)

        # Build hidden layers: first layer takes (x ∥ t_emb)
        layers: list = [nn.Linear(data_dim + hidden_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, data_dim)  noisy sample at timestep t.
            t: (B,)           integer timestep indices.
        Returns:
            (B, data_dim) predicted noise.
        """
        t_emb = sinusoidal_embedding(
            t, self.time_embed_dim
        )  # (B, time_embed_dim)
        t_emb = F.silu(self.time_proj(t_emb))  # (B, hidden_dim)
        xt = torch.cat([x, t_emb], dim=-1)  # (B, data_dim+hidden_dim)
        return self.net(xt)  # (B, data_dim)


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank delta W' = W + BA/r."""

    def __init__(self, linear: nn.Linear, rank: int = 8) -> None:
        super().__init__()
        out_features, in_features = linear.weight.shape

        # Keep the base layer; freeze it.
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)

        # Trainable low-rank factors.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is zero → initial delta = 0  (standard LoRA init)

        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = (x @ self.lora_A.T @ self.lora_B.T) / self.rank
        return base + delta

    def weight_delta_linf(self) -> torch.Tensor:
        """L-∞ norm of the current weight delta (BA/r)."""
        with torch.no_grad():
            delta_W = (self.lora_B @ self.lora_A) / self.rank
        return delta_W.abs().max()


# ---------------------------------------------------------------------------
# LoRA-wrapped score network
# ---------------------------------------------------------------------------


class LoRAScoreNet(nn.Module):
    """ScoreNet with LoRA adapters on every Linear layer.

    Shares frozen base weights with the original ScoreNet instance;
    only lora_A and lora_B parameters are trainable.
    """

    def __init__(self, base: ScoreNet, rank: int = 8) -> None:
        super().__init__()
        self.time_embed_dim = base.time_embed_dim
        self.time_proj = LoRALinear(base.time_proj, rank)

        new_layers: list = []
        for layer in base.net:
            if isinstance(layer, nn.Linear):
                new_layers.append(LoRALinear(layer, rank))
            else:
                new_layers.append(layer)  # SiLU passes through unchanged
        self.net = nn.Sequential(*new_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = F.silu(self.time_proj(t_emb))
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)

    def lora_params(self) -> list:
        """Return only the trainable LoRA parameters."""
        return [p for name, p in self.named_parameters() if "lora" in name]

    def max_weight_delta(self) -> torch.Tensor:
        """Maximum L-∞ weight-change norm across all LoRA layers."""
        norms = [
            m.weight_delta_linf()
            for m in self.modules()
            if isinstance(m, LoRALinear)
        ]
        return torch.stack(norms).max()
