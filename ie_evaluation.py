import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utils import safe_log_loss, bootstrap_gap_CI
from config import seed_hash, IDENT_LOW, IDENT_HIGH, N_SPLITS, N_BOOT, WORLD_SEED, make_rng, seed_for_bootstrap, seed_for_model
from ie_interventions import IE_INTERVENTIONS


# ============================================================
# Decoder factory
# ============================================================
def make_decoder(seed, decoder_type="linear"):
    """Build decoder model used for risk estimation."""
    if decoder_type == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=seed
            )
        )

    elif decoder_type == "quadratic":
        return make_pipeline(
            StandardScaler(),
            QuadraticDiscriminantAnalysis(reg_param=0.5)
        )

    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")


# ============================================================
# Risk normalization
# ============================================================
def compute_normalized_risk(R_real, R_null, R_min):
    """Normalize empirical risk into IE scale."""
    denom = R_null - R_min
    if denom <= 1e-8:
        return np.nan
    return (R_real - R_min) / denom


def classify_regime(R_tilde):
    """Classify identifiability regime from normalized risk."""
    if np.isnan(R_tilde):
        return "undefined"
    elif R_tilde <= IDENT_LOW:
        return "saturated"
    elif R_tilde >= IDENT_HIGH:
        return "degenerate"
    else:
        return "identifiable"


# ============================================================
# Core IE diagnostic (cross-validation + risk comparison)
# ============================================================
def primary_ie_diagnostic(X, y, seed, decoder_type="linear", verbose=True, debug=False, debug_level=1):

    # Cross-validation setup (shared across all interventions)
    fold_seed = seed_hash("cv", WORLD_SEED)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=fold_seed)
    splits = list(cv.split(X, y))

    base_loss_list = []
    train_loss_list = []

    # ========================================================
    # Baseline empirical risk estimation
    # ========================================================
    for fold_id, (tr, te) in enumerate(splits):

        clf = make_decoder(seed + fold_id, decoder_type)
        clf.fit(X[tr], y[tr])

        p_te = clf.predict_proba(X[te])
        loss_te = safe_log_loss(y[te], p_te)

        p_tr = clf.predict_proba(X[tr])
        loss_tr = safe_log_loss(y[tr], p_tr)

        base_loss_list.append(loss_te)
        train_loss_list.append(loss_tr)

    R_real = float(np.mean(base_loss_list))

    # ========================================================
    # Minimum achievable risk (in-sample)
    # ========================================================
    def estimate_r_min(X, y):
        clf = make_decoder(0, decoder_type)
        clf.fit(X, y)
        p = clf.predict_proba(X)
        return safe_log_loss(y, p)

    # ========================================================
    # Null model risk (random guessing baseline)
    # ========================================================
    def compute_null_logloss(y):
        p = np.clip(np.mean(y), 1e-8, 1 - 1e-8)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    R_min = estimate_r_min(X, y)
    R_null = compute_null_logloss(y)

    R_tilde = compute_normalized_risk(R_real, R_null, R_min)
    regime = classify_regime(R_tilde)

    if verbose:
        print(f"\nR_tilde={R_tilde:.4f} | regime={regime}")

    results = {}

    # ========================================================
    # Skip interventions if regime is not identifiable
    # ========================================================
    if regime != "identifiable":

        for name in IE_INTERVENTIONS.keys():
            results[name] = {
                "delta_IE": np.nan,
                "ci": (np.nan, np.nan),
                "ie_status": f"SKIPPED CI | regime={regime}",
                "R_tilde": float(R_tilde),
                "regime": regime
            }

            if verbose:
                print(f"{name:<15} | SKIPPED CI")

        return results

    # ========================================================
    # SURROGATE LOOP (IE interventions)
    # ========================================================
    for s_idx, (name, builder) in enumerate(IE_INTERVENTIONS.items()):

        gaps = []
        loss_real_list = []
        loss_sur_list = []
        transfer_gaps = []
        loss_transfer_list = []

        if verbose:
            print(f"\n[Surrogate] {name}")

        # ====================================================
        # Bootstrap sampling
        # ====================================================
        for b in range(N_BOOT):

            rng = make_rng(seed_for_bootstrap(seed, s_idx, b))

            fold_gaps = []
            fold_real = []
            fold_sur = []
            fold_transfer = []
            fold_transfer_loss = []

            # =================================================
            # Cross-validation evaluation
            # =================================================
            for f, (tr, te) in enumerate(splits):

                Xtr, ytr = X[tr], y[tr]
                Xte, yte = X[te], y[te]

                # Real model
                clf_real = make_decoder(seed_for_model(seed, b + f), decoder_type)
                clf_real.fit(Xtr, ytr)

                loss_real = safe_log_loss(
                    yte,
                    clf_real.predict_proba(Xte)
                )

                # =================================================
                # Surrogate intervention
                # =================================================
                Xs_tr, ys_tr, Xs_te, ys_te = builder(
                    Xtr, ytr, Xte, yte, rng
                )

                clf_sur = make_decoder(seed_for_model(seed, b + f + 999), decoder_type)
                clf_sur.fit(Xs_tr, ys_tr)

                loss_sur = safe_log_loss(
                    ys_te,
                    clf_sur.predict_proba(Xs_te)
                )

                gap = loss_sur - loss_real

                fold_gaps.append(gap)
                fold_real.append(loss_real)
                fold_sur.append(loss_sur)

                # =================================================
                # Transfer test (train on real, test on surrogate)
                # =================================================
                loss_transfer = safe_log_loss(
                    ys_te,
                    clf_real.predict_proba(Xs_te)
                )

                transfer_gap = loss_transfer - loss_real
                fold_transfer.append(transfer_gap)
                fold_transfer_loss.append(loss_transfer)

            gaps.append(np.mean(fold_gaps))
            transfer_gaps.append(np.mean(fold_transfer))
            loss_real_list.append(np.mean(fold_real))
            loss_sur_list.append(np.mean(fold_sur))
            loss_transfer_list.append(np.mean(fold_transfer_loss))

        gaps = np.array(gaps)

        ci_low, ci_high = bootstrap_gap_CI(gaps)
        delta = float(gaps.mean())

        transfer_gaps = np.array(transfer_gaps)

        ci_t_low, ci_t_high = bootstrap_gap_CI(transfer_gaps)
        delta_transfer = float(transfer_gaps.mean())

        # ====================================================
        # IE decision rule
        # ====================================================
        if ci_low <= 0 <= ci_high:
            ie_status = "IE holds"
        elif ci_low > 0:
            ie_status = "IE fails (positive deviation)"
        else:
            ie_status = "IE fails (negative deviation)"

        if ci_t_low <= 0 <= ci_t_high:
            transfer_status = "holds"
        elif ci_t_low > 0:
            transfer_status = "fails (positive deviation)"
        else:
            transfer_status = "fails (negative deviation)"

        results[name] = {
            "delta_IE": delta,
            "ci": (ci_low, ci_high),
            "ie_status": ie_status,
            "transfer_status": transfer_status,
            "R_tilde": float(R_tilde),
            "regime": regime,
            "loss_real_mean": float(np.mean(loss_real_list)),
            "loss_sur_mean": float(np.mean(loss_sur_list)),
            "gap_std": float(np.std(gaps)),
            "delta_transfer": delta_transfer,
            "ci_transfer": (ci_t_low, ci_t_high),
            "loss_transfer_mean": float(np.mean(loss_transfer_list))
        }

        if verbose:
            print(
                f"{name:<15} | "
                f"real={results[name]['loss_real_mean']:.4f} | "
                f"sur={results[name]['loss_sur_mean']:.4f} | "
                f"ΔIE={delta:.4f} | "
                f"CI=({ci_low:.4f}, {ci_high:.4f}) | "
                f"{ie_status} | "
                f"loss_transfer={results[name]['loss_transfer_mean']:.4f} | "
                f"ΔTransfer={delta_transfer:.4f} | CI_Transfer=({ci_t_low:.4f}, {ci_t_high:.4f}) | "
                f"{transfer_status}"
            )

    return results


# ============================================================
# Multi-decoder evaluation wrapper
# ============================================================
def run_ie_both_decoders(
    X,
    y,
    seed,
    decoder_types=None,
    verbose=True
):
    """Run IE diagnostic across multiple decoder families."""

    if decoder_types is None:
        decoder_types = ["linear", "quadratic"]

    all_results = {}

    for dec in decoder_types:
        if verbose:
            print(f"\n===== Decoder: {dec} =====")

        res = primary_ie_diagnostic(
            X,
            y,
            seed=seed,
            decoder_type=dec,
            verbose=verbose,
        )

        all_results[dec] = res

    return {
        "seed": seed,
        "results": all_results,
    }
