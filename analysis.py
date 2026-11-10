"""Reproducible comparison of concrete-strength regression models.

Run from this folder with: python analysis.py
Dependencies: NumPy, pandas, xlrd, Matplotlib. All scaling is fitted within each training
fold. Random holdout and age-range transfer use fixed random seeds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Concrete_Data.xls"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEED = 20261110
TEST_FRACTION = 0.20
AGE_CUTOFF = 90
BOOTSTRAPS = 500

RAW_NAMES = [
    "cement_kg_m3", "slag_kg_m3", "fly_ash_kg_m3", "water_kg_m3",
    "superplasticizer_kg_m3", "coarse_aggregate_kg_m3",
    "fine_aggregate_kg_m3", "age_days",
]
TARGET_NAME = "strength_mpa"
MODELS = ["MEAN", "OLS", "PHYS", "RIDGE", "LASSO", "POLY2_RIDGE"]
ALPHAS = {
    "RIDGE": np.logspace(-3, 3, 13),
    "LASSO": np.logspace(-4, -0.3, 13),
    "POLY2_RIDGE": np.logspace(-5, 4, 19),
}


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw = pd.read_excel(DATA_PATH, engine="xlrd")
    if raw.shape != (1030, 9):
        raise ValueError(f"Expected 1030 rows and 9 columns; got {raw.shape}")
    raw.columns = RAW_NAMES + [TARGET_NAME]
    data = raw.astype(float)
    if data.isna().any().any():
        raise ValueError("Unexpected missing values in the UCI source file")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Dataset contains non-finite values")
    return data


def make_features(data: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    if model == "PHYS":
        binder = data["cement_kg_m3"] + data["slag_kg_m3"] + data["fly_ash_kg_m3"]
        x = pd.DataFrame({
            "binder_total_kg_m3": binder,
            "water_binder_ratio": data["water_kg_m3"] / binder,
            "slag_binder_fraction": data["slag_kg_m3"] / binder,
            "flyash_binder_fraction": data["fly_ash_kg_m3"] / binder,
            "superplasticizer_binder_ratio": data["superplasticizer_kg_m3"] / binder,
            "aggregate_binder_ratio": (data["coarse_aggregate_kg_m3"] + data["fine_aggregate_kg_m3"]) / binder,
            "log_age_days": np.log(data["age_days"]),
        })
        return x.to_numpy(dtype=float), list(x.columns)
    return data[RAW_NAMES].to_numpy(dtype=float), RAW_NAMES.copy()


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-12] = 1.0
    return (x - mean) / scale, mean, scale


def polynomial2(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n, p = x.shape
    cols = [x[:, j] for j in range(p)]
    names = [f"x{j + 1}" for j in range(p)]
    for j in range(p):
        cols.append(x[:, j] ** 2)
        names.append(f"x{j + 1}^2")
        for k in range(j + 1, p):
            cols.append(x[:, j] * x[:, k])
            names.append(f"x{j + 1}:x{k + 1}")
    return np.column_stack(cols).reshape(n, -1), names


def soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def solve_coefficients(x: np.ndarray, y: np.ndarray, kind: str, alpha: float) -> np.ndarray:
    if kind in ("OLS", "PHYS"):
        return np.linalg.lstsq(x, y, rcond=None)[0]
    if kind in ("RIDGE", "POLY2_RIDGE"):
        return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    if kind == "LASSO":
        n, p = x.shape
        gram = (x.T @ x) / n
        rho = (x.T @ y) / n
        beta = np.zeros(p)
        for _ in range(10000):
            largest_change = 0.0
            for j in range(p):
                old = beta[j]
                residual_correlation = rho[j] - gram[j] @ beta + gram[j, j] * old
                beta[j] = soft_threshold(residual_correlation, alpha) / max(gram[j, j], 1e-12)
                largest_change = max(largest_change, abs(beta[j] - old))
            if largest_change < 1e-8:
                break
        return beta
    raise ValueError(f"Unknown model: {kind}")


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    kind: str,
    alpha: float = 0.0,
) -> tuple[np.ndarray, dict]:
    if kind == "MEAN":
        mean_y = float(np.mean(y_train))
        return np.full(len(x_predict), mean_y), {"alpha": 0.0}

    x_train_z, mean_x, scale_x = standardize_fit(x_train)
    x_predict_z = (x_predict - mean_x) / scale_x
    feature_names = None
    if kind == "POLY2_RIDGE":
        x_train_z, feature_names = polynomial2(x_train_z)
        x_predict_z, _ = polynomial2(x_predict_z)
    x_train_z, mean_poly, scale_poly = standardize_fit(x_train_z)
    x_predict_z = (x_predict_z - mean_poly) / scale_poly

    mean_y = float(np.mean(y_train))
    scale_y = float(np.std(y_train)) or 1.0
    y_train_z = (y_train - mean_y) / scale_y
    beta = solve_coefficients(x_train_z, y_train_z, kind, alpha)
    prediction = mean_y + scale_y * (x_predict_z @ beta)
    details = {"alpha": alpha}
    if kind != "POLY2_RIDGE":
        details["coef_per_input_unit"] = beta * scale_y / scale_x / scale_poly
    else:
        details["coef_scaled"] = beta * scale_y / scale_poly
        details["feature_names"] = feature_names
    return prediction, details


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "MAE_MPa": float(np.mean(np.abs(error))),
        "RMSE_MPa": float(np.sqrt(np.mean(error**2))),
        "R2": float(1.0 - np.sum(error**2) / sst) if sst else float("nan"),
    }


def kfolds(indices: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    return [fold for fold in np.array_split(shuffled, k) if len(fold)]


def tune_alpha(x: np.ndarray, y: np.ndarray, kind: str, indices: np.ndarray, seed: int) -> float:
    grid = ALPHAS[kind]
    folds = kfolds(indices, min(4, len(indices)), seed)
    scores = []
    all_indices = np.arange(len(y))
    for alpha in grid:
        fold_errors = []
        for valid in folds:
            train = np.setdiff1d(all_indices, valid, assume_unique=False)
            pred, _ = fit_predict(x[train], y[train], x[valid], kind, float(alpha))
            fold_errors.extend((y[valid] - pred).tolist())
        scores.append(np.sqrt(np.mean(np.square(fold_errors))))
    return float(grid[int(np.argmin(scores))])


def alpha_cv_curve(x: np.ndarray, y: np.ndarray, kind: str, seed: int) -> list[dict]:
    indices = np.arange(len(y))
    folds = kfolds(indices, 4, seed)
    rows = []
    for alpha in ALPHAS[kind]:
        fold_rmses = []
        for valid in folds:
            train = np.setdiff1d(indices, valid, assume_unique=False)
            pred, _ = fit_predict(x[train], y[train], x[valid], kind, float(alpha))
            fold_rmses.append(float(np.sqrt(np.mean((y[valid] - pred) ** 2))))
        rows.append({
            "model": kind,
            "alpha": float(alpha),
            "cv_rmse_mean_MPa": float(np.mean(fold_rmses)),
            "cv_rmse_sd_MPa": float(np.std(fold_rmses, ddof=1)),
        })
    return rows


def nested_cv_predictions(x: np.ndarray, y: np.ndarray, kind: str, seed: int) -> tuple[np.ndarray, list[float]]:
    indices = np.arange(len(y))
    outer = kfolds(indices, 5, seed)
    prediction = np.full(len(y), np.nan)
    chosen_alphas = []
    for fold_number, valid in enumerate(outer):
        train = np.setdiff1d(indices, valid, assume_unique=False)
        alpha = tune_alpha(x[train], y[train], kind, np.arange(len(train)), seed + 100 + fold_number) if kind in ALPHAS else 0.0
        pred, _ = fit_predict(x[train], y[train], x[valid], kind, alpha)
        prediction[valid] = pred
        chosen_alphas.append(alpha)
    return prediction, chosen_alphas


def split_indices(n: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(n * fraction))
    return order[n_test:], order[:n_test]


def fit_final(x: np.ndarray, y: np.ndarray, kind: str, seed: int) -> tuple[np.ndarray, dict]:
    alpha = tune_alpha(x, y, kind, np.arange(len(y)), seed) if kind in ALPHAS else 0.0
    prediction, details = fit_predict(x, y, x, kind, alpha)
    details["alpha"] = alpha
    return prediction, details


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, diagonal: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(x, y, s=22, alpha=0.6, color="#2563eb", edgecolors="none")
    if diagonal:
        low = min(float(np.min(x)), float(np.min(y)))
        high = max(float(np.max(x)), float(np.max(y)))
        ax.plot([low, high], [low, high], color="#dc2626", linestyle="--", linewidth=1.3, label="Идеальное совпадение")
        ax.legend(frameon=False)
    else:
        ax.axhline(0.0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.25)
    fig.savefig(path, format="svg")
    plt.close(fig)


def bootstrap_coefficients(x: np.ndarray, y: np.ndarray, names: list[str], kind: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    coef_samples = []
    effect_samples = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(y), size=len(y))
        _, info = fit_predict(x[sample], y[sample], x[sample], kind)
        coef = np.asarray(info["coef_per_input_unit"])
        coef_samples.append(coef)
        # A standardized effect puts kg/m3 and dimensionless ratios on comparable scale.
        x_sd = np.std(x[sample], axis=0)
        effect_samples.append(coef * np.where(x_sd > 1e-12, x_sd, 0.0))
    coef_samples = np.asarray(coef_samples)
    effect_samples = np.asarray(effect_samples)
    median_effect = np.median(effect_samples, axis=0)
    ranks = np.argsort(np.argsort(-np.abs(effect_samples), axis=1), axis=1) + 1
    out = []
    for j, name in enumerate(names):
        pos = float(np.mean(coef_samples[:, j] > 0))
        neg = float(np.mean(coef_samples[:, j] < 0))
        median_sign = np.sign(np.median(coef_samples[:, j]))
        out.append({
            "model": kind,
            "feature": name,
            "coef_median_per_unit": float(np.median(coef_samples[:, j])),
            "coef_q025": float(np.quantile(coef_samples[:, j], 0.025)),
            "coef_q975": float(np.quantile(coef_samples[:, j], 0.975)),
            "positive_fraction": pos,
            "negative_fraction": neg,
            "sign_agreement_with_median": float(max(pos, neg)),
            "standardized_effect_median": float(median_effect[j]),
            "abs_effect_rank_median": int(np.median(ranks[:, j])),
            "abs_effect_rank_q05": int(np.quantile(ranks[:, j], 0.05)),
            "abs_effect_rank_q95": int(np.quantile(ranks[:, j], 0.95)),
            "median_sign": int(median_sign),
        })
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    data = load_data()
    y_all = data[TARGET_NAME].to_numpy()
    train_idx, test_idx = split_indices(len(data), TEST_FRACTION, SEED)
    metric_rows = []
    prediction_rows = []
    oof_for_plot = {}
    model_fit_details = {}

    for model in MODELS:
        feature_source = "PHYS" if model == "PHYS" else "OLS"
        x_all, feature_names = make_features(data, feature_source)
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_test, y_test = x_all[test_idx], y_all[test_idx]
        cv_pred, cv_alphas = nested_cv_predictions(x_train, y_train, model, SEED + 1)
        train_pred, details = fit_final(x_train, y_train, model, SEED + 2)
        test_pred, _ = fit_predict(x_train, y_train, x_test, model, details["alpha"])
        model_fit_details[model] = details
        oof_for_plot[model] = (y_test, test_pred)
        for split_name, truth, pred in [("train", y_train, train_pred), ("cv_nested", y_train, cv_pred), ("test_random", y_test, test_pred)]:
            row = {"model": model, "split": split_name, "n": int(len(truth)), **metrics(truth, pred)}
            row["selected_alpha"] = details["alpha"] if model in ALPHAS else 0.0
            if split_name == "cv_nested" and cv_alphas:
                row["outer_alpha_median"] = float(np.median(cv_alphas))
            metric_rows.append(row)
        for row_index, (truth, pred) in enumerate(zip(y_test, test_pred)):
            prediction_rows.append({
                "model": model,
                "row_index_in_source": int(test_idx[row_index]),
                "age_days": float(data.iloc[test_idx[row_index]]["age_days"]),
                "observed_mpa": float(truth),
                "predicted_mpa": float(pred),
                "residual_mpa": float(truth - pred),
            })

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(RESULTS / "metrics.csv", index=False)
    write_csv(RESULTS / "predictions_random_test.csv", prediction_rows)
    alpha_rows = []
    for model in ALPHAS:
        source = "PHYS" if model == "PHYS" else "OLS"
        x_all, _ = make_features(data, source)
        alpha_rows.extend(alpha_cv_curve(x_all[train_idx], y_all[train_idx], model, SEED + 2))
    alpha_frame = pd.DataFrame(alpha_rows)
    alpha_frame.to_csv(RESULTS / "regularization_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for model, group in alpha_frame.groupby("model"):
        ax.plot(group["alpha"], group["cv_rmse_mean_MPa"], marker="o", markersize=3, label=model)
    ax.set_xscale("log")
    ax.set(title="CV error vs regularization strength", xlabel="alpha (log scale)", ylabel="4-fold CV RMSE (MPa)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "regularization_curve.svg", format="svg")
    plt.close(fig)

    # Age-range transfer: hold out all samples older than 90 days.
    age_train_mask = data["age_days"].to_numpy() <= AGE_CUTOFF
    age_test_mask = ~age_train_mask
    age_train_idx = np.flatnonzero(age_train_mask)
    age_test_idx = np.flatnonzero(age_test_mask)
    age_subtrain_rel, age_interp_rel = split_indices(len(age_train_idx), TEST_FRACTION, SEED + 20)
    age_subtrain_idx = age_train_idx[age_subtrain_rel]
    age_interp_idx = age_train_idx[age_interp_rel]
    age_rows = []
    age_predictions = []
    for model in MODELS:
        source = "PHYS" if model == "PHYS" else "OLS"
        x_all, _ = make_features(data, source)
        xa_train, ya_train = x_all[age_subtrain_idx], y_all[age_subtrain_idx]
        alpha = tune_alpha(xa_train, ya_train, model, np.arange(len(ya_train)), SEED + 21) if model in ALPHAS else 0.0
        interp_pred, _ = fit_predict(xa_train, ya_train, x_all[age_interp_idx], model, alpha)
        range_pred, _ = fit_predict(xa_train, ya_train, x_all[age_test_idx], model, alpha)
        for split_name, idx, pred in [("interpolation_age_le_90", age_interp_idx, interp_pred), ("extrapolation_age_gt_90", age_test_idx, range_pred)]:
            result = {"model": model, "split": split_name, "n": int(len(idx)), "selected_alpha": alpha, **metrics(y_all[idx], pred)}
            age_rows.append(result)
            for source_idx, observed, estimate in zip(idx, y_all[idx], pred):
                age_predictions.append({
                    "model": model,
                    "split": split_name,
                    "row_index_in_source": int(source_idx),
                    "age_days": float(data.iloc[source_idx]["age_days"]),
                    "observed_mpa": float(observed),
                    "predicted_mpa": float(estimate),
                    "residual_mpa": float(observed - estimate),
                })
    pd.DataFrame(age_rows).to_csv(RESULTS / "age_transfer_metrics.csv", index=False)
    write_csv(RESULTS / "age_transfer_predictions.csv", age_predictions)

    # Final coefficients and bootstrap sign/order stability for OLS and the physical basis.
    coefficient_rows = []
    stability_rows = []
    for model in ["OLS", "PHYS", "RIDGE", "LASSO"]:
        feature_source = "PHYS" if model == "PHYS" else "OLS"
        x_all, names = make_features(data, feature_source)
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        train_prediction, details = fit_final(x_train, y_train, model, SEED + 2)
        if "coef_per_input_unit" not in details:
            continue
        for name, value in zip(names, details["coef_per_input_unit"]):
            coefficient_rows.append({"model": model, "feature": name, "coef_per_input_unit": float(value)})
        if model in ["OLS", "PHYS"]:
            stability_rows.extend(bootstrap_coefficients(x_train, y_train, names, model, SEED + 30))
    write_csv(RESULTS / "coefficients.csv", coefficient_rows)
    write_csv(RESULTS / "coefficient_stability_bootstrap.csv", stability_rows)

    # Simple collinearity diagnostic on the original component quantities.
    corr = data[RAW_NAMES].corr().to_numpy()
    pairs = []
    for i in range(len(RAW_NAMES)):
        for j in range(i + 1, len(RAW_NAMES)):
            pairs.append((abs(float(corr[i, j])), float(corr[i, j]), RAW_NAMES[i], RAW_NAMES[j]))
    max_abs_corr, max_corr, corr_name_1, corr_name_2 = max(pairs)
    raw_x, _ = make_features(data.iloc[train_idx], "OLS")
    raw_xz, _, _ = standardize_fit(raw_x)
    raw_condition_number = float(np.linalg.cond(raw_xz))
    composition_range_exceedance = {}
    for feature in RAW_NAMES[:-1]:
        train_min = float(data.iloc[age_subtrain_idx][feature].min())
        train_max = float(data.iloc[age_subtrain_idx][feature].max())
        heldout_values = data.iloc[age_test_idx][feature].to_numpy()
        composition_range_exceedance[feature] = float(np.mean((heldout_values < train_min) | (heldout_values > train_max)))

    # Random-test diagnostic plots and simple residual-dependence checks.
    binder_all = data["cement_kg_m3"] + data["slag_kg_m3"] + data["fly_ash_kg_m3"]
    water_binder_all = data["water_kg_m3"] / binder_all
    residual_diagnostics = []
    for model in ["OLS", "PHYS"]:
        y_test, test_pred = oof_for_plot[model]
        residual = y_test - test_pred
        tag = model.lower()
        plot_scatter(FIGURES / f"measured_vs_predicted_{tag}.svg", y_test, test_pred,
                    f"Measured vs predicted: {model}, random test split", "Measured strength (MPa)", "Predicted strength (MPa)", diagonal=True)
        plot_scatter(FIGURES / f"residual_vs_prediction_{tag}.svg", test_pred, residual,
                    f"Residuals vs prediction: {model}, random test split", "Predicted strength (MPa)", "Residual (MPa)")
        age_test = data.iloc[test_idx]["age_days"].to_numpy()
        wb_test = water_binder_all.to_numpy()[test_idx]
        plot_scatter(FIGURES / f"residual_vs_age_{tag}.svg", age_test, residual,
                    f"Residuals vs age: {model}, random test split", "Age (days)", "Residual (MPa)")
        plot_scatter(FIGURES / f"residual_vs_water_binder_{tag}.svg", wb_test, residual,
                    f"Residuals vs water/binder: {model}, random test split", "Water / binder ratio", "Residual (MPa)")
        for diagnostic_name, diagnostic_values in [("prediction_mpa", test_pred), ("age_days", age_test), ("water_binder_ratio", wb_test)]:
            corr_value = float(np.corrcoef(diagnostic_values, residual)[0, 1]) if np.std(diagnostic_values) > 0 and np.std(residual) > 0 else 0.0
            residual_diagnostics.append({"model": model, "residual_vs": diagnostic_name, "pearson_correlation": corr_value})
    write_csv(RESULTS / "residual_diagnostics.csv", residual_diagnostics)

    metadata = {
        "source": "UCI Concrete Compressive Strength, dataset id 165",
        "n_observations": int(len(data)),
        "n_train_random": int(len(train_idx)),
        "n_test_random": int(len(test_idx)),
        "target_mean_mpa": float(np.mean(y_all)),
        "target_sd_mpa": float(np.std(y_all, ddof=1)),
        "age_cutoff_days": AGE_CUTOFF,
        "n_age_le_cutoff": int(np.sum(age_train_mask)),
        "n_age_gt_cutoff": int(np.sum(age_test_mask)),
        "n_age_train_subtrain": int(len(age_subtrain_idx)),
        "n_age_interpolation_holdout": int(len(age_interp_idx)),
        "age_holdout_component_out_of_training_min_max_fraction": composition_range_exceedance,
        "largest_abs_raw_feature_correlation": {"feature_1": corr_name_1, "feature_2": corr_name_2, "correlation": max_corr},
        "condition_number_standardized_raw_features_train": raw_condition_number,
        "bootstrap_repetitions": BOOTSTRAPS,
        "seed": SEED,
        "features": RAW_NAMES,
        "models": MODELS,
    }
    (RESULTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("DATASET", json.dumps(metadata, ensure_ascii=False))
    print("RANDOM SPLIT METRICS")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("AGE TRANSFER METRICS")
    print(pd.DataFrame(age_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("BOOTSTRAP PHYSICAL COEFFICIENTS")
    print(pd.DataFrame(stability_rows).query("model == 'PHYS'").to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
