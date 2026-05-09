import numpy as np
import hashlib
from pathlib import Path


N_BOOT = 500
N_SPLITS = 5

D_NOISE = 5
D_TASK = 5

SEEDS = list(range(10))

EXPERIMENT_SEED = 42

TEMPERATURE = 1.5

N=500

SEED=0

ALPHA=0.05

IDENT_LOW = 0.15
IDENT_HIGH = 0.85


NOISE=1.5
TASK=0.6

TASK_TYPE = [
    "linear",
    "nonlinear",
    "noise_interaction",
    "rotated",
    "sparse",
]


def make_rng(seed):
    return np.random.default_rng(seed)


def seed_hash(*args):
    s = "_".join(map(str, args))
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)

global WORLD_SEED
WORLD_SEED = seed_hash("world", EXPERIMENT_SEED)


def seed_for_bootstrap(base, s_idx, b):
    return seed_hash("boot", base, s_idx, b)


def seed_for_model(base, x):
    return seed_hash("model", base, x)
