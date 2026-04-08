# Diffusion-Repair v1

> Inspired by **SARP** (Safety-Aware Repair with Predictive Models, IROS 2024)  
> and **NNRepLayer** (CoRL 2022). Both by Majd et al.

Goal: minimally modify a pretrained diffusion model so generated samples stay in a safe truncated set while preserving generation quality.

---

## Analogy: SARP → Diffusion-Repair

| SARP concept | Diffusion-Repair equivalent |
|---|---|
| Policy $\pi_\theta: \mathcal S \to \mathcal A$ | Denoising network $\epsilon_\theta(x_t, t)$ |
| Expert demonstrations $\mathcal D = \{(s_n, a_n)\}$ | Clean data distribution samples |
| Predictive model $f(s, a) \to z$ | Full DDIM trajectory rollout $x_{T} \to x_0$ |
| Feature constraints $\mathcal G(z, a) = 0$ | Safety constraints on $x_0$ (and/or intermediate $x_t$) |
| Repair parameter $\theta$ | LoRA/adapter $\Delta$ on denoiser layers |
| Augmented Lagrangian update | Same — directly applies |
| `SafetyCheck` on $\mathcal D$ | Empirical violation rate check on holdout seeds |

---

## 1. Problem Setup

Let the denoising network be $\epsilon_\theta(x_t, t)$ and let $x_0 = \Phi_\theta(x_T, \xi)$ denote the full deterministic DDIM rollout from noise seed $x_T$ to generated sample $x_0$.

Define the safe set as:

$$
\mathcal S = \{x_0 : g_k(x_0) \le 0,\; k=1,\dots,K\}
$$

where each $g_k$ is a differentiable safety evaluator (e.g. a trained classifier, a norm bound, a polytope halfspace in latent space).

Instead of optimizing full $\theta$, optimize a small parameter block (LoRA/adapter):

$$
\theta' = \theta + \Delta
$$

The predictive model in SARP language is the trajectory rollout itself:

$$
f(x_T; \theta') = \Phi_{\theta'}(x_T) = x_0
$$

---

## 2. SARP-Style Optimization Formulation

Following SARP (Eq. 4–5), define the augmented Lagrangian:

$$
\widetilde{\mathcal L}(\Delta, \lambda, \mu) =
\underbrace{\mathcal L(\pi_{\theta+\Delta})}_{\text{preservation loss}}
- \sum_{k=1}^K \lambda_k \cdot \hat g_k(\Delta)
+ \frac{\mu}{2} \sum_{k=1}^K \hat g_k(\Delta)^2
$$

where:

- $\mathcal L(\pi_{\theta'}) = \frac{1}{N}\sum_{i=1}^N \|\epsilon_{\theta'}(x_t^{(i)}, t) - \epsilon_\theta(x_t^{(i)}, t)\|_2^2$ (distillation / behavior preservation)
- $\hat g_k(\Delta) = \frac{1}{N}\sum_{i=1}^N \text{ReLU}(g_k(\Phi_{\theta'}(x_T^{(i)})))$ (sampled constraint violation, converted from inequality to equality via ReLU as in SARP)
- $\lambda_k \in \mathbb R$ are Lagrange multipliers (initialized to 0)
- $\mu \in \mathbb R^+$ is a penalty scaling parameter

The repair problem is:

$$
\Delta^* = \arg\min_{\Delta,\, \lambda,\, \mu} \widetilde{\mathcal L}(\Delta, \lambda, \mu)
$$

---

## 3. Algorithm (SARP Algorithm 1 Adapted)

```
Input:  ε_θ (frozen base denoiser), safety evaluators g_k,
        sample pool D = {x_T^(i)}, max_weight_bound δ_max
Output: repaired adapter Δ*

Initialize: Δ = 0,  λ = 0,  μ = μ₀ > 0,  k = 0

while SafetyCheck(Δ, g, D) is False:

    1. Sample minibatch from D
    2. Run DDIM rollout:  x_0^(i) = Φ_{θ+Δ}(x_T^(i))
    3. Compute ĝ_k(Δ) = mean_i ReLU(g_k(x_0^(i)))   for each k
    4. Compute preservation loss L = ||ε_{θ+Δ}(x_t,t) - ε_θ(x_t,t)||²
    5. Compute augmented Lagrangian L_tilde
    6. Update: Δ ← Δ - η ∇_Δ L_tilde          (Adam)
    7. Update: λ_k ← λ_k - μ · ĝ_k(Δ)        (dual ascent)
    8. Update: μ ← β · μ                        (penalty scaling, β > 1)
    9. Enforce: ||Δ||_∞ ≤ δ_max                (weight change bound)
    k ← k + 1

return Δ
```

`SafetyCheck`: fraction of $\{x_T^{(i)}\}$ producing unsafe $x_0$ falls below tolerance $\varepsilon$.

Default hyperparameters from SARP: $\mu_0 = 5$, $\eta = 0.001$, $\beta = 1.5$.

---

## 4. Optional: Path Safety (Intermediate Timesteps)

If you want the denoising path itself to stay close to the safe set at selected timesteps $\mathcal T_s$:

$$
\hat g_k^{\text{path}}(\Delta) = \frac{1}{N|\mathcal T_s|}\sum_{i,t\in\mathcal T_s} \text{ReLU}(g_k(x_t^{(i)}))
$$

Add as extra constraint terms to the Lagrangian.  
This is the diffusion analogue of SARP's constraint on action trajectories (action-rate bounding in the prosthesis case).

---

## 5. Why SARP Applies Well Here

1. **No MIP needed.** SARP moved from MIQP (NNRepLayer) to differentiable ALM precisely because composing policy + predictive model is nonlinear. Diffusion has the same structure.
2. **Predictive model is the rollout.** In SARP for prosthetics, $f(s,a)$ predicts future pressure/states. Here, the full DDIM rollout $\Phi_\theta$ is the predictive model — it predicts what $x_0$ the denoiser will produce.
3. **Repair only needed parameters.** SARP repairs all layers; adapting this to LoRA for diffusion is a natural improvement.
4. **Inequality via ReLU.** SARP's trick of converting $g(·) \le 0$ to $\text{ReLU}(g(·)) = 0$ carries over directly.
5. **Dual ascent stabilizes.** The $(\lambda, \mu)$ update prevents the penalty from dominating and killing generation diversity (mode collapse analog).

---

## 6. What Needs to Be Relaxed vs. NNRepLayer MIQP

| NNRepLayer | Must Relax For Diffusion |
|---|---|
| Hard linear output constraints | Soft via ALM (as SARP does) |
| Single-layer repair | Multi-layer adapter (LoRA) |
| ReLU-MIP Big-M encoding | Not needed — use gradient through rollout |
| Bound tightening (IA/LP) | Not needed — use sampled trajectory stats |
| Gurobi MILP solver | Adam gradient steps |
| Provable per-sample guarantee | Empirical violation rate + optional reachability post-hoc |

---

## 7. Metrics to Report

Following SARP's evaluation table:

| Metric | Definition |
|---|---|
| Efficacy (E) | % of unsafe seeds repaired (constraint satisfied) |
| Relative Change (RC) | Mean change of constrained feature vs. original |
| Side Effect (SE) | Change in generation quality on originally-safe seeds |
| Violation rate at $x_0$ | Final sample safety rate |
| Path violation rate | Safety rate across timesteps $\mathcal T_s$ |
| FID / CLIP score delta | Distribution shift from base model |
| Worst-case quantile | $q_{0.95}$ of $\max_k g_k(x_0)$ |

---

## 8. Toy 2D First (Recommended)

Before latent image diffusion, validate on a 2D DDPM/DDIM toy where $\mathcal S$ is a polytope or annulus:

1. Train baseline 2D diffusion.
2. Define explicit $g_k(x) = a_k^T x - b_k$ (halfplane constraints).
3. Apply Algorithm 3 above and confirm truncation.
4. Compare: penalty-only ($\lambda=0$, SARP$_{\text{penalty}}$) vs. full ALM (SARP$_{\text{Lagrangian}}$).

This directly mirrors SARP's two-variant comparison in Table I.

---

## 9. Harder-Guarantee Direction

Following SARP Remark 1: after gradient-based repair, pass repaired adapter through a **neural network reachability tool** to:

1. Find any remaining violating regions beyond the training seed set.
2. Add those seeds back to $\mathcal D$ and re-run repair (counterexample-guided refinement).
3. Certify that no unsafe $x_0$ is reachable from a bounded noise seed set.

Global hard guarantees for full diffusion remain very difficult, but local tube certificates around sampled trajectories are feasible.

---

## 10. Future Extensions

- **Continual repair**: update adapter + predictive model online as distribution shifts (SARP future work).
- **Multi-modal safety**: combine collision-style classifier $g_k$ with bounded-action-rate style norms.
- **Partially observable analog**: repair under missing conditioning signals (mirrors SARP partial-obs prosthesis case).
- **Cross-architecture**: apply same ALM framework to flow-matching / consistency models where rollout is even faster.
