"""
Step 2: Run SARP repair on the baseline model and evaluate results.

Usage (from repo root):
    python experiments/diffusion_repair/run_repair.py

Requires baseline checkpoint from train_baseline.py.

Output:
    experiments/diffusion_repair/results/repaired_lora.pt
    experiments/diffusion_repair/results/repair_results.png
"""

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent.parent / "src"))

import torch
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt

from config import Config, EXPERIMENT_DIR
from data import generate_gaussian_mixture
from model import ScoreNet, LoRAScoreNet
from diffusion import DDPMScheduler
from safety import SafetyEvaluator
from repair import SARPRepair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_baseline(cfg: Config, device: str) -> tuple:
    ckpt_path = cfg.results_dir / cfg.checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found at {ckpt_path}.\n"
            "Run train_baseline.py first."
        )
    # PyTorch 2.6+ defaults to weights_only=True, but this checkpoint stores
    # additional Python objects (e.g., config dataclass), so full load is needed.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ScoreNet(
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_embed_dim=cfg.time_embed_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded baseline from {ckpt_path}")
    return model


def build_preservation_data(
    cfg: Config, scheduler: DDPMScheduler, device: str
) -> tuple:
    """Pre-compute (x_t, t) pairs from the training distribution.

    These are used for the preservation loss during repair.
    """
    x0 = generate_gaussian_mixture(
        cfg.n_repair_data, cfg.n_gaussians, cfg.gauss_radius, cfg.gauss_std
    ).to(device)
    t = torch.randint(1, cfg.T, (cfg.n_repair_data,), device=device)
    x_t, _ = scheduler.q_sample(x0, t)
    return x_t, t


def evaluate(
    cfg: Config,
    model,
    scheduler: DDPMScheduler,
    safety: SafetyEvaluator,
    seeds: torch.Tensor,
    label: str,
) -> dict:
    """Generate samples from fixed seeds and compute safety metrics."""
    x0 = scheduler.ddim_rollout(model, seeds, n_steps=cfg.ddim_steps_eval).cpu()
    g = safety(x0, cfg.constraint)
    return {
        "x0": x0,
        "viol_rate": safety.violation_rate(g),
        "mean_viol": safety.mean_violation(g),
        "g": g,
        "label": label,
    }


def compute_efficacy(g_before: torch.Tensor, g_after: torch.Tensor) -> float:
    """Fraction of originally-unsafe seeds that are safe after repair."""
    originally_unsafe = g_before > 0
    if not originally_unsafe.any():
        return 1.0
    now_safe = g_after[originally_unsafe] <= 0
    return now_safe.float().mean().item()


def compute_side_effect(
    g_before: torch.Tensor,
    g_after: torch.Tensor,
    x0_before: torch.Tensor,
    x0_after: torch.Tensor,
) -> float:
    """Mean prediction change on originally-safe seeds (analogous to SARP SE)."""
    originally_safe = g_before <= 0
    if not originally_safe.any():
        return float("nan")
    diff = (
        (x0_after[originally_safe] - x0_before[originally_safe]).pow(2).mean()
    )
    return diff.item()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    cfg: Config,
    res_before: dict,
    res_after: dict,
    history: dict,
    save_name: str = "repair_results.png",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(
        "SARP Diffusion Repair — 2D Toy\n"
        "Safe set: upper half-plane  {x : x₂ ≥ 0}",
        fontsize=13,
    )

    def scatter_panel(ax, x0, g, title):
        safe = x0[(g <= 0)]
        unsafe = x0[(g > 0)]
        vr = float((g > 0).mean())
        if len(safe):
            ax.scatter(
                safe[:, 0],
                safe[:, 1],
                s=5,
                alpha=0.5,
                color="#2ecc71",
                label=f"safe   ({len(safe)})",
            )
        if len(unsafe):
            ax.scatter(
                unsafe[:, 0],
                unsafe[:, 1],
                s=5,
                alpha=0.5,
                color="#e74c3c",
                label=f"unsafe ({len(unsafe)})",
            )
        ax.axhline(0, color="k", lw=0.8, ls="--", label="safety boundary")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nviol_rate={vr:.1%}")
        ax.legend(markerscale=3, fontsize=8, loc="lower right")

    scatter_panel(
        axes[0, 0],
        res_before["x0"].numpy(),
        res_before["g"].numpy(),
        "Baseline DDIM samples",
    )
    scatter_panel(
        axes[0, 1],
        res_after["x0"].numpy(),
        res_after["g"].numpy(),
        "Repaired DDIM samples (SARP)",
    )

    # Violation rate over epochs
    ax = axes[1, 0]
    ax.plot(history["violation_rate"], color="#e74c3c", lw=1.5)
    ax.axhline(
        cfg.safety_tol,
        color="k",
        ls="--",
        lw=1,
        label=f"target ≤ {cfg.safety_tol:.0%}",
    )
    ax.set_xlabel("Repair epoch")
    ax.set_ylabel("Violation rate")
    ax.set_title("Violation rate during repair")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Preservation loss and total loss
    ax = axes[1, 1]
    ax.plot(
        history["loss_preservation"],
        color="#3498db",
        lw=1.5,
        label="Preservation loss",
    )
    ax.plot(
        history["loss_total"],
        color="#9b59b6",
        lw=1.5,
        alpha=0.8,
        label="Total (ALM) loss",
    )
    ax.set_xlabel("Repair epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves during repair")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = cfg.results_dir / save_name
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def update_cfg_from_args(cfg: Config, args: argparse.Namespace) -> None:
    if args.ddim_steps_repair is not None:
        cfg.ddim_steps_repair = args.ddim_steps_repair
    if args.repair_batch_seeds is not None:
        cfg.repair_batch_seeds = args.repair_batch_seeds
    if args.repair_epochs is not None:
        cfg.repair_epochs = args.repair_epochs
    if args.beta_alm is not None:
        cfg.beta_alm = args.beta_alm
    if args.penalty_update_every is not None:
        cfg.penalty_update_every = args.penalty_update_every
    if args.mu_0 is not None:
        cfg.mu_0 = args.mu_0
    if args.repair_lr is not None:
        cfg.repair_lr = args.repair_lr


def ensure_history_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "# Diffusion Repair Improvement History\n\n"
        "This file records every tested algorithm variant and its metrics.\n\n"
        "## Metric Definitions\n"
        "- before_viol_rate: baseline violation rate before repair\n"
        "- after_viol_rate: violation rate after repair\n"
        "- viol_rate_delta: after_viol_rate - before_viol_rate (negative is better)\n"
        "- efficacy: fraction of originally-unsafe samples repaired\n"
        "- side_effect_mse: MSE shift on originally-safe samples\n\n"
        "## Run Log\n",
        encoding="utf-8",
    )


def append_history_entry(path: Path, metrics: dict, config_changes: dict) -> None:
    ensure_history_file(path)
    lines = [
        "\n### " + metrics["tag"] + "\n",
        f"- timestamp: {metrics['timestamp']}\n",
        f"- change_summary: {metrics['change_summary']}\n",
        f"- config_overrides: {json.dumps(config_changes, sort_keys=True)}\n",
        f"- before_viol_rate: {metrics['before_viol_rate']:.4f}\n",
        f"- after_viol_rate: {metrics['after_viol_rate']:.4f}\n",
        f"- viol_rate_delta: {metrics['viol_rate_delta']:.4f}\n",
        f"- before_mean_viol: {metrics['before_mean_viol']:.4f}\n",
        f"- after_mean_viol: {metrics['after_mean_viol']:.4f}\n",
        f"- efficacy: {metrics['efficacy']:.4f}\n",
        f"- side_effect_mse: {metrics['side_effect_mse']:.4f}\n",
        f"- max_weight_delta_linf: {metrics['max_weight_delta_linf']:.4f}\n",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.writelines(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SARP diffusion repair and log metrics."
    )
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--change-summary", type=str, default="baseline settings")
    parser.add_argument("--repair-epochs", type=int)
    parser.add_argument("--ddim-steps-repair", type=int)
    parser.add_argument("--repair-batch-seeds", type=int)
    parser.add_argument("--beta-alm", type=float)
    parser.add_argument("--penalty-update-every", type=int)
    parser.add_argument("--mu-0", type=float)
    parser.add_argument("--repair-lr", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-file",
        type=str,
        default=str(EXPERIMENT_DIR / "IMPROVEMENT_HISTORY.md"),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = Config()
    update_cfg_from_args(cfg, args)
    device = "cpu"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    safety = SafetyEvaluator()

    # 1. Load baseline ---------------------------------------------------
    base_model = load_baseline(cfg, device)
    scheduler = DDPMScheduler(
        T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=device
    )

    # 2. Fixed evaluation seeds ------------------------------------------
    torch.manual_seed(args.seed)
    seeds = torch.randn(cfg.n_repair_seeds, 2, device=device)

    # 3. Evaluate BEFORE repair ------------------------------------------
    print("\n--- Baseline evaluation ---")
    res_before = evaluate(cfg, base_model, scheduler, safety, seeds, "Baseline")
    print(f"  Violation rate : {res_before['viol_rate']:.1%}")
    print(f"  Mean violation : {res_before['mean_viol']:.4f}")

    # 4. Build LoRA model ------------------------------------------------
    lora_model = LoRAScoreNet(base_model, rank=cfg.lora_rank).to(device)
    lora_params = sum(p.numel() for p in lora_model.lora_params())
    print(
        f"\nLoRA adapter: {lora_params:,} trainable parameters "
        f"(rank={cfg.lora_rank})"
    )

    # 5. Pre-compute preservation data ----------------------------------
    print("\nBuilding preservation dataset …")
    x_t_data, t_data = build_preservation_data(cfg, scheduler, device)

    # 6. SARP repair loop -----------------------------------------------
    repairer = SARPRepair(
        base_model=base_model,
        lora_model=lora_model,
        scheduler=scheduler,
        g_k=lambda x: safety(x, cfg.constraint),
        mu_0=cfg.mu_0,
        beta_alm=cfg.beta_alm,
        repair_lr=cfg.repair_lr,
        max_weight_bound=cfg.max_weight_bound,
        ddim_steps=cfg.ddim_steps_repair,
        device=device,
    )

    print()
    history = repairer.run(
        seeds=seeds,
        x_t_data=x_t_data,
        t_data=t_data,
        max_epochs=cfg.repair_epochs,
        repair_batch_seeds=cfg.repair_batch_seeds,
        repair_batch_data=256,
        penalty_update_every=cfg.penalty_update_every,
        safety_tol=cfg.safety_tol,
        verbose=True,
    )

    # 7. Evaluate AFTER repair ------------------------------------------
    print("\n--- Post-repair evaluation ---")
    res_after = evaluate(cfg, lora_model, scheduler, safety, seeds, "Repaired")
    print(f"  Violation rate : {res_after['viol_rate']:.1%}")
    print(f"  Mean violation : {res_after['mean_viol']:.4f}")

    efficacy = compute_efficacy(res_before["g"], res_after["g"])
    side_effect = compute_side_effect(
        res_before["g"], res_after["g"], res_before["x0"], res_after["x0"]
    )

    # 8. Print summary table (mirrors SARP Table II) --------------------
    print("\n" + "=" * 54)
    print("  Summary (SARP-style metrics)")
    print("=" * 54)
    print(f"  {'Metric':<30} {'Before':>8}  {'After':>8}")
    print("-" * 54)
    print(
        f"  {'Violation rate':30} {res_before['viol_rate']:>8.1%}  "
        f"{res_after['viol_rate']:>8.1%}"
    )
    print(
        f"  {'Mean violation':30} {res_before['mean_viol']:>8.4f}  "
        f"{res_after['mean_viol']:>8.4f}"
    )
    print(f"  {'Efficacy (E)':30} {'—':>8}  {efficacy:>8.1%}")
    print(f"  {'Side effect (SE, MSE)':30} {'—':>8}  {side_effect:>8.4f}")
    max_delta = lora_model.max_weight_delta().item()
    print(f"  {'Max ‖ΔW‖∞ (weight budget)':30} {'—':>8}  {max_delta:>8.4f}")
    print("=" * 54)

    config_overrides = {
        "ddim_steps_repair": cfg.ddim_steps_repair,
        "repair_batch_seeds": cfg.repair_batch_seeds,
        "repair_epochs": cfg.repair_epochs,
        "beta_alm": cfg.beta_alm,
        "penalty_update_every": cfg.penalty_update_every,
        "mu_0": cfg.mu_0,
        "repair_lr": cfg.repair_lr,
        "seed": args.seed,
    }
    metrics = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "tag": args.tag,
        "change_summary": args.change_summary,
        "before_viol_rate": float(res_before["viol_rate"]),
        "after_viol_rate": float(res_after["viol_rate"]),
        "viol_rate_delta": float(res_after["viol_rate"] - res_before["viol_rate"]),
        "before_mean_viol": float(res_before["mean_viol"]),
        "after_mean_viol": float(res_after["mean_viol"]),
        "efficacy": float(efficacy),
        "side_effect_mse": float(side_effect),
        "max_weight_delta_linf": float(max_delta),
    }

    metrics_path = cfg.results_dir / f"metrics_{args.tag}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "config": config_overrides}, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    append_history_entry(Path(args.history_file), metrics, config_overrides)
    print(f"  History updated → {args.history_file}")

    # 9. Save repaired LoRA ------------------------------------------
    lora_stem = Path(cfg.lora_name).stem
    lora_path = cfg.results_dir / f"{lora_stem}_{args.tag}.pt"
    torch.save(
        {
            "lora_state_dict": lora_model.state_dict(),
            "history": history,
            "config": cfg,
        },
        lora_path,
    )
    print(f"\n  Repaired LoRA saved → {lora_path}")

    # 10. Plot -----------------------------------------------------------
    plot_results(
        cfg,
        res_before,
        res_after,
        history,
        save_name=f"repair_results_{args.tag}.png",
    )


if __name__ == "__main__":
    main()
