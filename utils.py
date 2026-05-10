import numpy as np
from sklearn.metrics import log_loss
from config import ALPHA



def safe_log_loss(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return log_loss(y, p, labels=[0, 1])


def bootstrap_gap_CI(gaps, alpha=ALPHA):
    return np.percentile(gaps, 100 * alpha / 2), np.percentile(gaps, 100 * (1 - alpha / 2))


