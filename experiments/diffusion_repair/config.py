from dataclasses import dataclass
from pathlib import Path

# Absolute path to this experiment directory, regardless of cwd
EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # ------ Data --------------------------------------------------------
    n_gaussians: int = 8
    gauss_radius: float = 2.0
    gauss_std: float = 0.3
    n_train_samples: int = 20000

    # ------ Diffusion ---------------------------------------------------
    T: int = 1000  # total diffusion timesteps
    beta_start: float = 1e-4
    beta_end: float = 0.02
    ddim_steps_eval: int = 50  # DDIM steps used when evaluating
    ddim_steps_repair: int = 20  # DDIM steps used during repair (faster)

    # ------ Score network -----------------------------------------------
    hidden_dim: int = 128
    n_layers: int = 3  # number of hidden layers
    time_embed_dim: int = 64  # sinusoidal time embedding dimension

    # ------ Baseline training -------------------------------------------
    n_train_epochs: int = 200
    train_batch_size: int = 512
    train_lr: float = 1e-3
    train_seed: int = 42

    # ------ Safety ------------------------------------------------------
    # Safe set: upper half-plane  {x : x_2 >= 0}
    # Constraint function: g(x) = -x[:,1]   (g(x) <= 0  <=>  x_2 >= 0)
    constraint: str = "upper_half_plane"

    # ------ Repair (SARP) -----------------------------------------------
    # Mirrors SARP (IROS 2024) Algorithm 1 defaults where available.
    lora_rank: int = 8  # rank of LoRA adapters
    n_repair_seeds: int = (
        500  # fixed noise seeds used across all repair iterations
    )
    repair_batch_seeds: int = 200  # seeds subsampled each iteration
    n_repair_data: int = (
        5000  # pre-computed (x_t, t) pairs for preservation loss
    )
    repair_epochs: int = 300
    repair_lr: float = 1e-3
    mu_0: float = 5.0  # initial ALM penalty weight  (SARP default)
    beta_alm: float = 1.5  # penalty scaling factor      (SARP default)
    penalty_update_every: int = 10  # scale mu every N epochs
    max_weight_bound: float = 2.0  # L-inf bound on LoRA weight delta
    safety_tol: float = 0.05  # acceptable violation rate (5 %)

    # ------ Evaluation --------------------------------------------------
    n_eval_samples: int = 2000

    # ------ Paths -------------------------------------------------------
    results_dir: Path = EXPERIMENT_DIR / "results"
    checkpoint_name: str = "baseline_scorenet.pt"
    lora_name: str = "repaired_lora.pt"
