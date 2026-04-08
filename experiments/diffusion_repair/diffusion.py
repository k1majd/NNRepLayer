"""
DDPM noise schedule and DDIM sampler for the 2D diffusion repair experiment.

DDPMScheduler
-------------
  Stores pre-computed α̅_t values for a linear β schedule.
  Provides:
    q_sample()           — forward diffusion (used in DDPM training)
    ddim_rollout()       — deterministic sampling, no gradient (evaluation)
    ddim_rollout_grad()  — deterministic sampling, with gradient (repair backprop)

DDIM update rule (η = 0, fully deterministic):
    x̂₀ = (x_t − √(1−α̅_t) · ε) / √α̅_t
    x_{t'} = √α̅_{t'} · x̂₀ + √(1−α̅_{t'}) · ε
"""

import torch


class DDPMScheduler:
    """Linear β schedule with pre-computed cumulative products."""

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clamp_x0_pred: bool = False,
        clamp_min: float = -6.0,
        clamp_max: float = 6.0,
        device: str = "cpu",
    ) -> None:
        self.T = T
        self.device = device
        self.clamp_x0_pred = clamp_x0_pred
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars  # ᾱ_t  (T,)
        self.sqrt_ab = alpha_bars.sqrt()  # √ᾱ_t
        self.sqrt_one_minus_ab = (1.0 - alpha_bars).sqrt()  # √(1−ᾱ_t)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """Sample x_t ~ q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I).

        Returns:
            x_t:   (B, D) noisy sample.
            noise: (B, D) the Gaussian noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_ab[t].view(-1, 1)
        sqrt_1mab = self.sqrt_one_minus_ab[t].view(-1, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Reverse process — evaluation (no gradient)
    # ------------------------------------------------------------------

    def ddim_rollout(
        self,
        model,
        x_T: torch.Tensor,
        n_steps: int = 50,
        clamp_x0: bool = None,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling from x_T → x_0 (no grad).

        Uses evenly spaced timestep sub-sequence of length n_steps.
        """
        step_idx = (
            torch.linspace(self.T - 1, 0, n_steps + 1).long().to(self.device)
        )
        if clamp_x0 is None:
            clamp_x0 = self.clamp_x0_pred
        x = x_T.clone()
        model.eval()
        with torch.no_grad():
            for i in range(n_steps):
                t_cur = step_idx[i]
                t_next = step_idx[i + 1]
                t_batch = t_cur.expand(x.shape[0])

                eps = model(x, t_batch)

                ab_cur = self.alpha_bars[t_cur]
                ab_next = self.alpha_bars[t_next]

                x0_pred = (x - (1.0 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
                if clamp_x0:
                    x0_pred = x0_pred.clamp(self.clamp_min, self.clamp_max)
                x = ab_next.sqrt() * x0_pred + (1.0 - ab_next).sqrt() * eps
        return x  # (B, D)

    # ------------------------------------------------------------------
    # Reverse process — repair (gradient flows through)
    # ------------------------------------------------------------------

    def ddim_rollout_grad(
        self,
        model,
        x_T: torch.Tensor,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling keeping the computation graph.

        This is used during SARP repair so that gradients flow back from
        g(x_0) through the whole denoising trajectory to the LoRA params.
        Clamping is omitted to avoid gradient blocking.
        """
        step_idx = (
            torch.linspace(self.T - 1, 0, n_steps + 1).long().to(self.device)
        )
        x = x_T.clone()
        model.train()
        for i in range(n_steps):
            t_cur = step_idx[i]
            t_next = step_idx[i + 1]
            t_batch = t_cur.expand(x.shape[0])

            eps = model(x, t_batch)

            ab_cur = self.alpha_bars[t_cur]
            ab_next = self.alpha_bars[t_next]

            x0_pred = (x - (1.0 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
            x = ab_next.sqrt() * x0_pred + (1.0 - ab_next).sqrt() * eps
        return x  # (B, D)
