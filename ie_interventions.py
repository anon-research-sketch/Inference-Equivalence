import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import FactorAnalysis


# ============================================================
# Identity surrogate (baseline: no transformation)
# ============================================================
def identity_surrogate(Xtr, ytr, Xte, yte, rng):
    return Xtr.copy(), ytr.copy(), Xte.copy(), yte.copy()


# ============================================================
# Mean-only surrogate (class-conditional collapse to mean)
# ============================================================
def mean_only_surrogate(Xtr, ytr, Xte, yte, rng=None):
    Xtr_new = np.zeros_like(Xtr)
    Xte_new = np.zeros_like(Xte)

    for c in [0, 1]:
        mu = Xtr[ytr == c].mean(axis=0)

        idx_tr = np.where(ytr == c)[0]
        idx_te = np.where(yte == c)[0]

        Xtr_new[idx_tr] = mu
        Xte_new[idx_te] = mu

    return Xtr_new, ytr.copy(), Xte_new, yte.copy()


# ============================================================
# IID marginal surrogate (feature-wise resampling)
# ============================================================
def iid_marginal_surrogate(Xtr, ytr, Xte, yte, rng, conditioned=True):

    Xtr_new = np.zeros_like(Xtr)
    Xte_new = np.zeros_like(Xte)

    classes = np.unique(ytr)

    # Unconditioned marginal resampling
    if not conditioned:
        for j in range(Xtr.shape[1]):
            Xtr_new[:, j] = rng.choice(Xtr[:, j], size=len(Xtr), replace=True)
            Xte_new[:, j] = rng.choice(Xtr[:, j], size=len(Xte), replace=True)

        return Xtr_new.astype(np.float32), ytr.copy(), Xte_new.astype(np.float32), yte.copy()

    # Class-conditional marginal resampling
    for c in classes:
        Xc = Xtr[ytr == c]

        idx_tr = np.where(ytr == c)[0]
        idx_te = np.where(yte == c)[0]

        for j in range(Xtr.shape[1]):
            Xtr_new[idx_tr, j] = rng.choice(Xc[:, j], size=len(idx_tr), replace=True)
            Xte_new[idx_te, j] = rng.choice(Xc[:, j], size=len(idx_te), replace=True)

    return Xtr_new.astype(np.float32), ytr.copy(), Xte_new.astype(np.float32), yte.copy()


# ============================================================
# Covariance Gaussian surrogate (class-conditional Gaussian)
# ============================================================
def cov_gaussian_surrogate(Xtr, ytr, Xte, yte, rng):
    Xtr_new = np.zeros_like(Xtr)
    Xte_new = np.zeros_like(Xte)

    for c in [0, 1]:
        Xc = Xtr[ytr == c]
        mu = Xc.mean(axis=0)
        cov = LedoitWolf().fit(Xc).covariance_

        idx_tr = np.where(ytr == c)[0]
        idx_te = np.where(yte == c)[0]

        Xtr_new[idx_tr] = rng.multivariate_normal(mu, cov, size=len(idx_tr))
        Xte_new[idx_te] = rng.multivariate_normal(mu, cov, size=len(idx_te))

    return Xtr_new, ytr.copy(), Xte_new, yte.copy()


# ============================================================
# Latent Factor Analysis surrogate (generative latent model)
# ============================================================
def latent_FA_surrogate(Xtr, ytr, Xte, yte, rng,
                        conditioned=True, n_factors=5):

    n_factors = min(n_factors, Xtr.shape[1])

    Xtr_new = np.zeros_like(Xtr)
    Xte_new = np.zeros_like(Xte)

    classes = np.unique(ytr)

    # Global FA model (unconditional)
    if not conditioned:
        fa = FactorAnalysis(n_components=n_factors, random_state=0).fit(Xtr)

        z_tr = rng.normal(size=(len(Xtr), n_factors))
        z_te = rng.normal(size=(len(Xte), n_factors))

        eps_tr = rng.normal(scale=np.sqrt(fa.noise_variance_), size=Xtr.shape)
        eps_te = rng.normal(scale=np.sqrt(fa.noise_variance_), size=Xte.shape)

        Xtr_new = fa.mean_ + z_tr @ fa.components_ + eps_tr
        Xte_new = fa.mean_ + z_te @ fa.components_ + eps_te

        return Xtr_new.astype(np.float32), ytr.copy(), Xte_new.astype(np.float32), yte.copy()

    # Class-conditional FA model
    for c in classes:
        Xc = Xtr[ytr == c]

        idx_tr = np.where(ytr == c)[0]
        idx_te = np.where(yte == c)[0]

        fa = FactorAnalysis(n_components=n_factors, random_state=0).fit(Xc)

        z_tr = rng.normal(size=(len(idx_tr), n_factors))
        z_te = rng.normal(size=(len(idx_te), n_factors))

        eps_tr = rng.normal(scale=np.sqrt(fa.noise_variance_), size=(len(idx_tr), Xtr.shape[1]))
        eps_te = rng.normal(scale=np.sqrt(fa.noise_variance_), size=(len(idx_te), Xte.shape[1]))

        Xtr_new[idx_tr] = fa.mean_ + z_tr @ fa.components_ + eps_tr
        Xte_new[idx_te] = fa.mean_ + z_te @ fa.components_ + eps_te

    return Xtr_new.astype(np.float32), ytr.copy(), Xte_new.astype(np.float32), yte.copy()


# ============================================================
# Copula surrogate (nonlinear dependency-preserving resampling)
# ============================================================
def copula_surrogate(Xtr, ytr, Xte, yte, rng, conditioned=True):

    import scipy.stats
    from sklearn.covariance import LedoitWolf

    def build_copula(Xref, n_samples):
        n, d = Xref.shape

        # Rank transform → empirical copula
        ranks = scipy.stats.rankdata(Xref, axis=0, method="average")
        U = (ranks - 0.5) / n
        U = np.clip(U, 1e-6, 1 - 1e-6)

        Z = scipy.stats.norm.ppf(U)
        Sigma = LedoitWolf().fit(Z).covariance_

        Znew = rng.multivariate_normal(
            mean=np.zeros(d),
            cov=Sigma,
            size=n_samples
        )

        Unew = scipy.stats.norm.cdf(Znew)

        Ynew = np.zeros((n_samples, d))

        for j in range(d):
            vals = np.sort(Xref[:, j])
            quantiles = (np.arange(n) + 0.5) / n

            y = np.interp(Unew[:, j], quantiles, vals)

            # variance re-alignment to original scale
            mu_real = Xref[:, j].mean()
            std_real = Xref[:, j].std() + 1e-6

            mu_sur = y.mean()
            std_sur = y.std() + 1e-6

            y = (y - mu_sur) / std_sur
            y = y * std_real + mu_real

            Ynew[:, j] = y

        return Ynew.astype(np.float32)

    Xtr_new = np.zeros_like(Xtr)
    Xte_new = np.zeros_like(Xte)

    classes = np.unique(ytr)

    # Unconditional copula
    if not conditioned:
        Xte_new = build_copula(Xtr, len(Xte))
        return Xtr.copy(), ytr.copy(), Xte_new, yte.copy()

    # Class-conditional copula
    for c in classes:
        Xc = Xtr[ytr == c]

        idx_tr = np.where(ytr == c)[0]
        idx_te = np.where(yte == c)[0]

        Xtr_new[idx_tr] = build_copula(Xc, len(idx_tr))
        Xte_new[idx_te] = build_copula(Xc, len(idx_te))

    return Xtr_new.astype(np.float32), ytr.copy(), Xte_new.astype(np.float32), yte.copy()


# ============================================================
# Decision-based perturbation (classifier-aligned noise)
# ============================================================
def decision_model(Xtr, ytr, Xte, yte, rng, strength=1.0, mode="weak"):
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    probe.fit(Xtr, ytr)

    w = probe.named_steps["logisticregression"].coef_[0]
    w = w / (np.linalg.norm(w) + 1e-8)

    def perturb(X):
        margin = X @ w

        if mode == "weak":
            alpha = rng.normal(0, strength, size=(X.shape[0], 1))
        else:
            jitter = rng.normal(0, 0.05 * strength, size=(X.shape[0], 1))
            alpha = -(strength + jitter) * np.sign(margin).reshape(-1, 1)

        return X + alpha * w

    return perturb(Xtr), ytr.copy(), perturb(Xte), yte.copy()


# ============================================================
# IE intervention family
# ============================================================
IE_INTERVENTIONS = {
    "Mean-only": mean_only_surrogate,
    "IID-marginal": iid_marginal_surrogate,
    "Cov-Gaussian": cov_gaussian_surrogate,
    "Latent-FA": latent_FA_surrogate,
    "Copula": copula_surrogate,
    "Decision-Weak": lambda *args: decision_model(*args, strength=0.5, mode="weak"),
    "Decision-Strong": lambda *args: decision_model(*args, strength=1.5, mode="strong"),
}


def mean_var_stats(X, Xs):
    mean_diff = np.mean(np.abs(X.mean(axis=0) - Xs.mean(axis=0)))

    var_diff = np.mean(np.abs(X.var(axis=0) - Xs.var(axis=0)))

    return {
        "mean_diff": float(mean_diff),
        "var_diff": float(var_diff),
    }

def covariance_stats(X, Xs):

    cov_real = np.cov(X, rowvar=False)
    cov_sur  = np.cov(Xs, rowvar=False)

    frob = np.linalg.norm(cov_real - cov_sur, ord="fro")

    corr = np.corrcoef(
        cov_real.flatten(),
        cov_sur.flatten()
    )[0, 1]

    return {
        "cov_frobenius": float(frob),
        "cov_corr": float(corr),
    }

from sklearn.decomposition import PCA


def explained_variance_stats(X, Xs, n_components=10):

    pca_real = PCA(n_components=n_components)
    pca_real.fit(X)

    pca_sur = PCA(n_components=n_components)
    pca_sur.fit(Xs)

    real_evr = pca_real.explained_variance_ratio_
    sur_evr  = pca_sur.explained_variance_ratio_

    ev_diff = np.mean(
        np.abs(real_evr - sur_evr)
    )



    return {
        "ev_diff": float(ev_diff),
    }

def descriptive_benchmark(X, y, builder, rng):

    Xs, ys, _, _ = builder(X, y, X, y, rng)

    out = {}

    out.update(mean_var_stats(X, Xs))
    out.update(covariance_stats(X, Xs))
    out.update(explained_variance_stats(X, Xs))

    return out
