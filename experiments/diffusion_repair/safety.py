"""
Safety evaluators for the 2D diffusion repair experiment.

A safety evaluator g_k(x) returns a scalar per sample:
    g_k(x) <= 0   →  sample is SAFE
    g_k(x)  > 0   →  sample VIOLATES the constraint

Following SARP (IROS 2024), inequality constraints g(x) <= 0 are converted
to equality via ReLU:  ReLU(g(x)) = 0  <=>  g(x) <= 0.
"""

import torch


class SafetyEvaluator:
    """Collection of 2D safety-constraint functions."""

    # ------------------------------------------------------------------
    # Constraint definitions
    # ------------------------------------------------------------------

    @staticmethod
    def upper_half_plane(x: torch.Tensor) -> torch.Tensor:
        """Safe set: x_2 >= 0  (upper half-plane).

        g(x) = -x[:,1]    →  g(x) <= 0  iff  x_2 >= 0
        """
        return -x[:, 1]  # (B,)

    @staticmethod
    def circle(x: torch.Tensor, radius: float = 3.5) -> torch.Tensor:
        """Safe set: ‖x‖ <= radius  (disc of given radius).

        g(x) = ‖x‖² - r²
        """
        return x.pow(2).sum(dim=-1) - radius**2  # (B,)

    @staticmethod
    def box(x: torch.Tensor, x_max: float = 3.0) -> torch.Tensor:
        """Safe set: |x_i| <= x_max for every dimension.

        Returns the maximum per-dimension violation.
        """
        return (x.abs() - x_max).clamp(min=0).max(dim=-1).values  # (B,)

    @staticmethod
    def polytope(
        x: torch.Tensor,
        A: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Safe set: Ax <= b  (polytope / half-space intersection).

        Returns the maximum halfspace violation per sample.

        Args:
            x: (B, 2)
            A: (K, 2)  halfspace normals
            b: (K,)    halfspace offsets
        Returns:
            (B,) maximum violation value
        """
        return (x @ A.T - b).max(dim=-1).values  # (B,)

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def __call__(
        self,
        x: torch.Tensor,
        constraint: str = "upper_half_plane",
        **kwargs,
    ) -> torch.Tensor:
        """Evaluate the named constraint and return per-sample violation values.

        Args:
            x:          (B, 2) generated samples.
            constraint: name of constraint method to call.
            **kwargs:   extra arguments forwarded to the constraint (e.g. radius).
        Returns:
            (B,) tensor — positive values indicate violations.
        """
        fn = getattr(self, constraint)
        return fn(x, **kwargs)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def violation_rate(g_values: torch.Tensor) -> float:
        """Fraction of samples with g > 0 (unsafe)."""
        return (g_values > 0).float().mean().item()

    @staticmethod
    def mean_violation(g_values: torch.Tensor) -> float:
        """Mean of ReLU(g) — average magnitude of violations."""
        return g_values.clamp(min=0).mean().item()
