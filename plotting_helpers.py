from viz_config import style_ticks, spines_minimal, crosshair
import numpy as np



SURROGATE_ORDER = ["Mean-only", "IID-marginal", "Cov-Gaussian", "Latent-FA", "Copula", "Decision-Weak", "Decision-Strong"]
REGIME_ORDER = ["identifiable", "saturated", "degenerate"]

def get_ordered_names(available_names):
    return [n for n in SURROGATE_ORDER if n in available_names]

def finalize_panel(ax, boxed=False, enabled=True):
    crosshair(ax, enabled=enabled)
    spines_minimal(ax, boxed=boxed)
    style_ticks(ax)

def smart_xlim_focus(x, lower=2, upper=98, pad=0.4):
    x = np.asarray(x)
    if len(x) == 0:
        return -1, 1

    lo, hi = np.percentile(x, [lower, upper])
    span = hi - lo

    if span == 0:
        span = 1e-6

    return lo - pad * span, hi + pad * span
