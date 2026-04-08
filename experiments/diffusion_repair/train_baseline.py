"""
Step 1: Train and save the baseline 2D diffusion model.

Usage (from repo root):
    python experiments/diffusion_repair/train_baseline.py

Output:
    experiments/diffusion_repair/results/baseline_scorenet.pt
    experiments/diffusion_repair/results/baseline_plot.png
"""

import sys
from pathlib import Path

# Allow imports from this experiment package regardless of cwd.
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(
    0, str(EXPERIMENT_DIR.parent.parent / "src")
)  # nnreplayer (not needed here)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import Config, EXPERIMENT_DIR
from data import generate_gaussian_mixture
from model import ScoreNet
from diffusion import DDPMScheduler
from safety import SafetyEvaluator


def train(cfg: Config, device: str = "cpu") -> ScoreNet:
    torch.manual_seed(cfg.train_seed)

    # --- Data -----------------------------------------------------------
    print(f"Generating {cfg.n_train_samples} training samples …")
    x0_train = generate_gaussian_mixture(
        cfg.n_train_samples,
        n_gaussians=cfg.n_gaussians,
        radius=cfg.gauss_radius,
        std=cfg.gauss_std,
        seed=cfg.train_seed,
    ).to(device)

    # --- Model & scheduler ----------------------------------------------
    model = ScoreNet(
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        time_embed_dim=cfg.time_embed_dim,
    ).to(device)

    scheduler = DDPMScheduler(
        T=cfg.T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ScoreNet: {total_params:,} parameters")

    # --- DDPM training loop ---------------------------------------------
    print(f"\nTraining for {cfg.n_train_epochs} epochs …")
    model.train()
    N = x0_train.shape[0]

    for epoch in range(cfg.n_train_epochs):
        perm = torch.randperm(N)
        x0_shuffled = x0_train[perm]
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, cfg.train_batch_size):
            x0_batch = x0_shuffled[i : i + cfg.train_batch_size]
            B = x0_batch.shape[0]

            # Sample random timesteps t ~ Uniform(1, T)
            t = torch.randint(1, cfg.T, (B,), device=device)

            # Forward diffusion: x_t, ε
            x_t, noise = scheduler.q_sample(x0_batch, t)

            # Predict noise
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / n_batches
            print(f"  Epoch {epoch+1:4d}/{cfg.n_train_epochs}  loss={avg:.5f}")

    print("\nTraining complete.")
    return model, scheduler


def evaluate_and_plot(
    cfg: Config, model: ScoreNet, scheduler: DDPMScheduler, device: str = "cpu"
) -> None:
    """Generate samples, compute safety stats, and save a summary plot."""
    safety = SafetyEvaluator()

    # Ground-truth training data
    x0_data = generate_gaussian_mixture(
        cfg.n_eval_samples, cfg.n_gaussians, cfg.gauss_radius, cfg.gauss_std
    )

    # Baseline DDIM samples
    x_T = torch.randn(cfg.n_eval_samples, 2, device=device)
    x0_gen = scheduler.ddim_rollout(
        model, x_T, n_steps=cfg.ddim_steps_eval
    ).cpu()

    # Safety statistics
    g_data = safety(x0_data, cfg.constraint)
    g_gen = safety(x0_gen, cfg.constraint)

    viol_data = safety.violation_rate(g_data)
    viol_gen = safety.violation_rate(g_gen)
    print(f"\n  Training data  violation rate : {viol_data:.1%}")
    print(f"  Baseline DDIM  violation rate : {viol_gen:.1%}")

    # --- Plot -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Baseline 2D Diffusion Model", fontsize=13)

    def scatter_safe_unsafe(ax, samples, g_vals, title):
        safe = samples[(g_vals <= 0).squeeze()]
        unsafe = samples[(g_vals > 0).squeeze()]
        if safe.shape[0]:
            ax.scatter(
                safe[:, 0],
                safe[:, 1],
                s=6,
                alpha=0.5,
                color="#2ecc71",
                label=f"safe ({safe.shape[0]})",
            )
        if unsafe.shape[0]:
            ax.scatter(
                unsafe[:, 0],
                unsafe[:, 1],
                s=6,
                alpha=0.5,
                color="#e74c3c",
                label=f"unsafe ({unsafe.shape[0]})",
            )
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.legend(markerscale=3, fontsize=8)

    scatter_safe_unsafe(
        axes[0],
        x0_data.numpy(),
        g_data.numpy(),
        f"Training data  (viol={viol_data:.1%})",
    )
    scatter_safe_unsafe(
        axes[1],
        x0_gen.numpy(),
        g_gen.numpy(),
        f"Baseline DDIM  (viol={viol_gen:.1%})",
    )

    plt.tight_layout()
    save_path = cfg.results_dir / "baseline_plot.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def main() -> None:
    cfg = Config()
    device = "cpu"

    # Create output directory
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model, scheduler = train(cfg, device)

    # Save checkpoint
    ckpt_path = cfg.results_dir / cfg.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        ckpt_path,
    )
    print(f"\n  Checkpoint saved → {ckpt_path}")

    # Evaluate and plot
    evaluate_and_plot(cfg, model, scheduler, device)


if __name__ == "__main__":
    main()
