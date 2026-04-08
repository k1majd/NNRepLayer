"""
2D Gaussian mixture dataset for the diffusion repair toy experiment.

Layout: 8 components arranged on a circle of given radius.
Approximately half the mass is in the lower half-plane (x_2 < 0) — the
"unsafe" region that the repair step should eliminate.
"""

import math
import torch


def generate_gaussian_mixture(
    n_samples: int,
    n_gaussians: int = 8,
    radius: float = 2.0,
    std: float = 0.3,
    seed: int = None,
) -> torch.Tensor:
    """Return (n_samples, 2) samples from a ring-shaped Gaussian mixture.

    Component centres are evenly spaced on a circle:
        centre_k = (radius·cos(2πk/K), radius·sin(2πk/K))

    With K=8 and radius=2.0:
      - 3 centres lie in the lower half-plane  → ~37 % unsafe at baseline
      - 2 centres sit on  x_2 = 0             → ~50 % unsafe at baseline
    """
    if seed is not None:
        torch.manual_seed(seed)

    angles = torch.linspace(0, 2 * math.pi, n_gaussians + 1)[:-1]  # (K,)
    centres = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)], dim=1
    )  # (K, 2)

    components = torch.randint(0, n_gaussians, (n_samples,))
    means = centres[components]  # (N, 2)
    samples = means + std * torch.randn(n_samples, 2)
    return samples
