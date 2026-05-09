import numpy as np
from config import (
    D_NOISE,
    D_TASK,
    seed_hash,
    WORLD_SEED,
    NOISE,
    TASK,
    N,
    SEED,
    TASK_TYPE,
    TEMPERATURE
)


# ============================================================
# Latent generative world (shared latent structure)
# ============================================================
def generate_base_world(n=N, seed=SEED):
    """Draw latent nuisance and task factors from fixed seed."""
    rng = np.random.default_rng(seed)
    Zn = rng.normal(size=(n, D_NOISE))
    Zt = rng.normal(size=(n, D_TASK))
    return Zn, Zt


# ============================================================
# Task readout / signal specification
# ============================================================
def compute_signal(Zt_i, Zn_i, task_type=TASK_TYPE):
    """Define task-dependent readout over latent variables."""
    if task_type == "linear":
        return (
            1.2 * Zt_i[:, 0]
            + 0.7 * Zt_i[:, 1]
            - 0.3 * (Zt_i[:, 2] if Zt_i.shape[1] > 2 else 0)
        )

    if task_type == "rotated":
        return 1.0 * Zt_i[:, 1] - 1.0 * Zt_i[:, 0]

    if task_type == "sparse":
        return 2.0 * Zt_i[:, 0]

    if task_type == "nonlinear":
        return 5.5 * Zt_i[:, 0] * Zt_i[:, 1]

    if task_type == "noise_interaction":
        return Zt_i[:, 0] + 0.5 * Zn_i[:, 0]

    raise ValueError(f"Unknown task_type: {task_type}")


# ============================================================
# Data generation under latent-noise coupling model
# ============================================================
def build_dataset(
    Zn,
    Zt,
    rng,
    noise=NOISE,
    task=TASK,
    seed=SEED,
    task_type=TASK_TYPE,
    temperature = TEMPERATURE
):
    """Generate observed samples from latent variables with noise injection."""

    # Deterministic noise process tied to world configuration
    noise_seed = seed_hash("noise", WORLD_SEED, noise, task)
    noise_rng = np.random.default_rng(noise_seed)
    eps = noise_rng.normal(0, 1, size=Zn.shape[0])

    # Scale nuisance component
    Zn_i = noise * Zn

    # Inject nuisance into task latents
    Zt_i = Zt.copy()
    Zt_i[:, 0] += 0.6 * Zn_i[:, 0]
    Zt_i[:, 1] += 0.4 * Zn_i[:, 1]

    # Task-dependent signal
    signal = compute_signal(Zt_i, Zn_i, task_type=task_type)
    signal = task * signal

    # Add stochastic margin noise
    noise_term = noise * eps
    margin = signal + noise_term

    # Logistic observation model
    prob = 1 / (1 + np.exp(-margin / temperature))
    y = rng.binomial(1, prob)

    # Observed representation (concatenated latent embedding)
    X = np.concatenate([Zn_i, Zt_i], axis=1)
    return X, y


# ============================================================
# Reproducible dataset API (fixed world seed)
# ============================================================
def generate_fixed_dataset(
    Zn,
    Zt,
    seed=SEED,
    noise=NOISE,
    task=TASK,
    task_type=TASK_TYPE,
):
    """Stable wrapper ensuring deterministic dataset generation."""
    rng = np.random.default_rng(seed)
    return build_dataset(
        Zn,
        Zt,
        rng,
        noise=noise,
        task=task,
        seed=seed,
        task_type=task_type,
    )
