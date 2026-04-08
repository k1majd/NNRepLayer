"""
SARP-inspired Augmented Lagrangian repair loop for diffusion models.

Implements Algorithm 1 from:
    "Repairing Neural Networks for Safety in Robotic Systems
     using Predictive Models" (IROS 2024, Majd et al.)

Mapping from SARP to diffusion:
    policy   π_θ(s)       →  denoiser  ε_θ(x_t, t)
    pred. model  f(s,a)   →  DDIM rollout  Φ_θ(x_T) = x_0
    feature constraints  g_c(z,a) = 0  →  g_k(x_0) ≤ 0
    repair parameter  θ   →  LoRA adapter  Δ

Augmented Lagrangian (SARP Eq. 4):
    L̃(Δ, λ, μ) = L_pres(Δ)  − λ · ĝ(Δ)  + (μ/2) · ĝ(Δ)²

where:
    L_pres(Δ) = ||ε_{θ+Δ}(x_t,t) − ε_θ(x_t,t)||²   (distillation)
    ĝ(Δ)      = mean_i ReLU( g( Φ_{θ+Δ}(x_T^(i)) ) ) (sampled violation)

Dual update (SARP Algorithm 1):
    λ  ←  λ − μ · ĝ(Δ)
    μ  ←  β · μ    (every `penalty_update_every` epochs)
"""

from typing import Callable, Dict
import torch
import torch.nn as nn


class SARPRepair:
    """SARP-inspired repair for diffusion models via augmented Lagrangian.

    Args:
        base_model:   Frozen ScoreNet instance (original denoiser).
        lora_model:   LoRAScoreNet wrapping base_model (adapter to optimise).
        scheduler:    DDPMScheduler with ddim_rollout_grad().
        g_k:          Safety evaluator callable  g_k(x) -> (B,) violations.
        mu_0:         Initial ALM penalty weight (SARP default: 5.0).
        beta_alm:     Penalty scaling factor per update (SARP default: 1.5).
        repair_lr:    Adam learning rate for LoRA params.
        max_weight_bound: L-∞ clip on LoRA weight delta.
        ddim_steps:   DDIM steps used during repair rollouts.
        device:       Torch device string.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_model: nn.Module,
        scheduler,
        g_k: Callable,
        mu_0: float = 5.0,
        beta_alm: float = 1.5,
        repair_lr: float = 1e-3,
        max_weight_bound: float = 2.0,
        ddim_steps: int = 20,
        device: str = "cpu",
    ) -> None:
        self.base_model = base_model.eval()
        self.lora_model = lora_model
        self.scheduler = scheduler
        self.g_k = g_k
        self.mu = float(mu_0)
        self.beta_alm = float(beta_alm)
        self.max_weight_bound = float(max_weight_bound)
        self.ddim_steps = ddim_steps
        self.device = device

        # Lagrange multiplier — starts at 0 (SARP Algorithm 1)
        self.lambda_k: float = 0.0

        self.optimizer = torch.optim.Adam(
            lora_model.lora_params(), lr=repair_lr
        )

        # Training history tracked for later plotting
        self.history: Dict[str, list] = {
            "loss_total": [],
            "loss_preservation": [],
            "violation_mean": [],
            "violation_rate": [],
            "lambda_k": [],
            "mu": [],
            "weight_delta_linf": [],
        }

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def _preservation_loss(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """||ε_{θ+Δ}(x_t, t) − ε_θ(x_t, t)||²  (distillation / preservation).

        Gradient flows only through the LoRA model.
        """
        with torch.no_grad():
            target = self.base_model(x_t, t)
        pred = self.lora_model(x_t, t)
        return ((pred - target) ** 2).mean()

    def _safety_violation(self, x0: torch.Tensor) -> torch.Tensor:
        """ĝ(Δ) = mean_i ReLU( g_k(x_0^(i)) ).

        Converts inequality  g(x) ≤ 0  to equality  ReLU(g(x)) = 0
        (SARP's ReLU trick, Sec. IV).
        """
        return torch.relu(self.g_k(x0))  # (B,) — non-negative

    def _augmented_lagrangian(
        self,
        loss_pres: torch.Tensor,
        violation: torch.Tensor,
    ) -> torch.Tensor:
        """L̃ = L_pres − λ·ĝ + (μ/2)·ĝ²   (SARP Eq. 4)."""
        g_mean = violation.mean()
        return loss_pres - self.lambda_k * g_mean + (self.mu / 2.0) * g_mean**2

    # ------------------------------------------------------------------
    # Single optimisation step
    # ------------------------------------------------------------------

    def _step(
        self,
        seeds: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, float]:
        """One primal update + dual recording (dual update is done outside)."""
        self.optimizer.zero_grad()

        # 1. DDIM rollout WITH gradient to get x_0
        x0 = self.scheduler.ddim_rollout_grad(
            self.lora_model, seeds, self.ddim_steps
        )

        # 2. Safety violation  ĝ(Δ) = mean ReLU(g(x_0))
        violation = self._safety_violation(x0)

        # 3. Preservation loss
        loss_pres = self._preservation_loss(x_t, t)

        # 4. Augmented Lagrangian
        loss = self._augmented_lagrangian(loss_pres, violation)

        # 5. Gradient step on LoRA params
        loss.backward()
        self.optimizer.step()

        g_mean = violation.mean().item()
        viol_rate = (violation > 0).float().mean().item()
        w_delta = self.lora_model.max_weight_delta().item()

        return {
            "loss_total": loss.item(),
            "loss_preservation": loss_pres.item(),
            "violation_mean": g_mean,
            "violation_rate": viol_rate,
            "lambda_k": self.lambda_k,
            "mu": self.mu,
            "weight_delta_linf": w_delta,
        }

    # ------------------------------------------------------------------
    # Safety check (no grad)
    # ------------------------------------------------------------------

    def safety_check(self, seeds: torch.Tensor, tol: float = 0.05) -> bool:
        """Return True if violation rate on `seeds` is <= tol."""
        self.lora_model.eval()
        with torch.no_grad():
            x0 = self.scheduler.ddim_rollout(
                self.lora_model, seeds, self.ddim_steps
            )
            viol = self._safety_violation(x0)
        return (viol > 0).float().mean().item() <= tol

    # ------------------------------------------------------------------
    # Full repair loop — Algorithm 1
    # ------------------------------------------------------------------

    def run(
        self,
        seeds: torch.Tensor,
        x_t_data: torch.Tensor,
        t_data: torch.Tensor,
        max_epochs: int = 300,
        repair_batch_seeds: int = 200,
        repair_batch_data: int = 256,
        penalty_update_every: int = 10,
        safety_tol: float = 0.05,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Full SARP repair loop.

        Args:
            seeds:            (N_seeds, 2) fixed noise vectors x_T ~ N(0,I).
            x_t_data:         (N_data, 2) pre-noised training samples.
            t_data:           (N_data,)   corresponding timestep indices.
            max_epochs:       maximum number of repair epochs.
            repair_batch_seeds: seeds per iteration (random subset of seeds).
            repair_batch_data:  data points per iteration (random subset).
            penalty_update_every: scale μ every this many epochs.
            safety_tol:       target violation rate (early stop).
            verbose:          print progress every 50 epochs.

        Returns:
            Training history dict.
        """
        print("=" * 60)
        print("  SARP Diffusion Repair")
        print(f"  seeds={seeds.shape[0]}  data={x_t_data.shape[0]}")
        print(
            f"  μ₀={self.mu:.1f}  β={self.beta_alm:.1f}  tol={safety_tol:.2f}"
        )
        print("=" * 60)

        for epoch in range(max_epochs):
            # ---- Subsample seeds ------------------------------------------
            s_idx = torch.randperm(seeds.shape[0])[:repair_batch_seeds]
            seed_batch = seeds[s_idx]

            # ---- Subsample preservation data ------------------------------
            d_idx = torch.randperm(x_t_data.shape[0])[:repair_batch_data]
            x_t_batch = x_t_data[d_idx]
            t_batch = t_data[d_idx]

            # ---- Primal step ----------------------------------------------
            stats = self._step(seed_batch, x_t_batch, t_batch)

            # ---- Dual update (SARP Algorithm 1) --------------------------
            # λ ← λ − μ · ĝ(Δ)
            self.lambda_k -= self.mu * stats["violation_mean"]
            stats["lambda_k"] = self.lambda_k  # update logged value

            # ---- Penalty scaling -----------------------------------------
            if (epoch + 1) % penalty_update_every == 0:
                self.mu *= self.beta_alm
                stats["mu"] = self.mu

            # ---- Log ------------------------------------------------------
            for k, v in stats.items():
                self.history[k].append(v)

            # ---- Print progress ------------------------------------------
            if verbose and (epoch + 1) % 50 == 0:
                print(
                    f"  [{epoch+1:4d}/{max_epochs}] "
                    f"loss={stats['loss_total']:8.4f}  "
                    f"pres={stats['loss_preservation']:7.4f}  "
                    f"viol_rate={stats['violation_rate']:.3f}  "
                    f"λ={self.lambda_k:7.3f}  "
                    f"μ={self.mu:6.2f}  "
                    f"‖ΔW‖∞={stats['weight_delta_linf']:.4f}"
                )

            # ---- Early stop when safe ------------------------------------
            if (epoch + 1) % 50 == 0 and self.safety_check(
                seeds, tol=safety_tol
            ):
                print(
                    f"\n  ✓ Safety satisfied (viol_rate ≤ {safety_tol}) "
                    f"at epoch {epoch + 1}. Stopping early."
                )
                break

        return self.history
