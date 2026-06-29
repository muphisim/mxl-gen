################################################################################
#                      MxL-GEN - ML Plotting Utilities                         #
#                                                                              #
# Small helpers to visualise ML training and evaluation results used in the    #
# surrogate pipeline. Includes summary scatter plots, nested-CV aggregation    #
# plots, and a flexible feature-importance reporter (direct/coef/permutation). #
#                                                                              #
################################################################################
"""
Notes:
    - Lightweight plotting helpers for model evaluation and reporting.
    - Designed to be used after mlModels or any pipeline that provides
      predictions and ground-truth arrays.
    - Feature importance routine will try to map importances back through
      PCA (if present) to the original feature space when possible.
"""

__all__ = ["plot_results", "plot_nested_results", "feature_importance"]

from typing import Dict, Optional, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

from sklearn.inspection import permutation_importance

# import helpers for metrics
import MxL_GEN.surrogate_model.ml_helpers as helpers

# configure large-font plotting defaults (matches original intent)
matplotlib.rcParams.update({"font.size": 35})


def plot_results(results: Dict[str, Any], output_dir: str, prefix: str = "results") -> None:
    """Produce per-model true-vs-pred scatter plots and write a combined figure.

    Writes a PNG file into ``output_dir`` with the given prefix. Each subplot
    shows predicted vs true values, a dashed identity line, and a small metrics
    box with R2 / RMSE / MAE / Spearman correlation computed via
    ``helpers._compute_metrics``.

    Args:
        results (Dict[str, Any]): Mapping model name -> dict containing
            ``"y_true"`` and ``"y_pred"`` (lists or arrays).
        output_dir (str): Directory where the summary image will be written.
        prefix (str, optional): Prefix used for the output PNG filename.
            Defaults to ``"results"``.

    Returns:
        None
    """
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    # hide all axes by default then enable those we need
    for ax in axes_flat:
        ax.set_visible(False)

    for i, (name, info) in enumerate(results.items()):
        ax = axes_flat[i]
        ax.set_visible(True)
        y_true = np.array(info.get("y_true", []))
        y_pred = np.array(info.get("y_pred", []))
        ax.scatter(y_pred, y_true, s=10)
        if y_true.size:
            m = helpers._compute_metrics(y_true, y_pred)
            ax.text(
                0.05,
                0.95,
                f"R2={m['r2']:.3f}\nRMSE={m['rmse']:.3f}\nMAE={m['mae']:.3f}\n\u03C1={m['spearman']:.3f}",
                transform=ax.transAxes,
                va="top",
            )
            mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
            if mn == mx:
                mn -= 1
                mx += 1
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
            ax.set_xlim([mn, mx])
            ax.set_ylim([mn, mx])
        ax.set_title(name)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"{prefix}_summary.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Wrote summary figure to {out_png}")


def plot_nested_results(aggregated: Dict[str, Any], output_dir: str, prefix: str = "nested") -> None:
    """Create scatter plots combining predictions across folds for each model.

    For each model in ``aggregated``, fold predictions are flattened and a
    per-model PNG is written showing predicted vs true values with simple
    metrics annotated.

    Args:
        aggregated (Dict[str, Any]): Mapping model name -> dict that should
            contain a ``"folds"`` key with an iterable of fold dicts. Each fold
            dict should expose ``"y_true"`` and ``"y_pred"``.
        output_dir (str): Directory where PNG outputs will be written.
        prefix (str, optional): Prefix used for output filenames. Defaults to
            ``"nested"``.

    Returns:
        None
    """
    for name, info in aggregated.items():
        ys = []
        yps = []
        for f in info.get("folds", []):
            ys.extend(f.get("y_true", []))
            yps.extend(f.get("y_pred", []))
        ys = np.array(ys)
        yps = np.array(yps)
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.scatter(yps, ys, s=10)
        if ys.size:
            m = helpers._compute_metrics(ys, yps)
            ax.text(0.05, 0.95, f"R2={m['r2']:.3f}\n\u03C1={m['spearman']:.3f}", transform=ax.transAxes, va="top")
            mn, mx = float(min(ys.min(), yps.min())), float(max(ys.max(), yps.max()))
            if mn == mx:
                mn -= 1
                mx += 1
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
            ax.set_xlim([mn, mx])
            ax.set_ylim([mn, mx])
        out_png = os.path.join(output_dir, f"{prefix}_{name}.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote nested figure for {name} to {out_png}")


def feature_importance(
    pipeline,
    output_dir: str,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    label: str = "",
    feature_names: Optional[list] = None,
    use_permutation: bool = False,
    n_repeats: int = 30,
    top_k: int = 20,
    seed: int = 42,
    n_jobs: int = 1,
    create_plots: bool = True,
    csv_name: Optional[str] = None,
    png_name: Optional[str] = None,
) -> pd.DataFrame:
    """Save and plot feature importances for the given sklearn Pipeline.

    Behaviour:
        - Prefer ``feature_importances_`` (tree ensembles) when present.
        - Otherwise prefer ``coef_`` (linear models), using absolute values or
          mean across outputs.
        - If neither are present, compute permutation importance (requires ``y``).
        - If PCA is present in the pipeline the routine attempts to map
          importances back to original feature space by using PCA components.
        - Results are saved as a CSV and (optionally) a PNG bar plot.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Fitted sklearn Pipeline containing preprocessing (e.g. scaler,
            optional PCA) and an estimator (named ``"model"`` or as the last step).
        output_dir (str): Directory to save CSV/PNG outputs.
        X (np.ndarray): Feature matrix used to compute importances (should match
            pipeline preprocessing).
        y (Optional[np.ndarray], optional): Target vector required for permutation
            importances. Defaults to ``None``.
        label (str, optional): Label prefix used for filenames and plot titles.
            Defaults to ``""``.
        feature_names (Optional[list], optional): List of feature names; if
            ``None`` default names ``f0..f{n-1}`` are used. Defaults to ``None``.
        use_permutation (bool, optional): Force permutation importance even if
            model provides direct importances/coefs. Defaults to ``False``.
        n_repeats (int, optional): Number of repeats for permutation importance.
            Defaults to ``30``.
        top_k (int, optional): Number of top features to display in the plot.
            Defaults to ``20``.
        seed (int, optional): RNG seed for permutation calculations. Defaults to ``42``.
        n_jobs (int, optional): Parallel workers used by ``permutation_importance``.
            Defaults to ``1``.
        create_plots (bool, optional): Whether to create and save a PNG plot of
            importances. Defaults to ``True``.
        csv_name (Optional[str], optional): Optional filename for the CSV output.
            If ``None``, an auto-generated timestamped name is used. Defaults to ``None``.
        png_name (Optional[str], optional): Optional filename for the PNG output.
            If ``None``, an auto-generated timestamped name is used. Defaults to ``None``.

    Returns:
        pd.DataFrame: DataFrame with columns ``["feature", "importance"]`` sorted descending.
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must equal number of columns in X")

    # Unwrap model from pipeline (support both named 'model' step and last-step convention)
    if "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
    else:
        model = pipeline.steps[-1][1]

    pca = pipeline.named_steps.get("pca", None)

    importances = None
    reason = None

    # Prefer built-in importances/coefs unless user requested permutation
    if (not use_permutation) and hasattr(model, "feature_importances_"):
        importances = np.array(getattr(model, "feature_importances_")).ravel()
        reason = "feature_importances_"
    elif (not use_permutation) and hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        if coef.ndim == 1:
            importances = np.abs(coef).ravel()
        else:
            importances = np.mean(np.abs(coef), axis=0).ravel()
        reason = "coef_ (abs or mean)"
    else:
        use_permutation = True

    # Permutation importance fallback (requires y)
    if use_permutation:
        if y is None:
            raise RuntimeError("y is not set as input. y is required for permutation calculation.")
        print(f"[feature_importance] Computing permutation importance (n_repeats={n_repeats}). This may be slow.")
        res = permutation_importance(
            pipeline, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs, scoring="r2"
        )
        importances = res.importances_mean
        reason = f"permutation_importance (n_repeats={n_repeats})"

    if importances is None:
        raise RuntimeError("Could not determine importances (unexpected).")

    # Map importances back through PCA to original feature space if PCA is present
    if pca is not None:
        try:
            comp = pca.components_
            # If importances are on PCA components, transform to original space
            if comp.shape[0] != importances.shape[0]:
                if importances.shape[0] == comp.shape[1]:
                    mapped_importances = importances
                else:
                    raise RuntimeError("Dimension mismatch between PCA components and importances.")
            else:
                mapped_importances = np.abs(comp.T.dot(importances))
            final_importances = mapped_importances
            mapping_note = "mapped_from_pca_components"
        except Exception as e:
            # fallback to returning importances directly and rename features as PCs
            final_importances = importances
            feature_names = [f"PC{i}" for i in range(len(importances))]
            mapping_note = f"mapping_failed:{e}"
    else:
        final_importances = importances
        mapping_note = "direct"

    # sanitise and format results
    final_importances = np.asarray(final_importances).ravel()
    final_importances = np.nan_to_num(final_importances, nan=0.0, posinf=0.0, neginf=0.0)

    df = pd.DataFrame({"feature": feature_names, "importance": final_importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    ts = int(time.time())
    if csv_name is None:
        csv_name = f"{label}feature_importances_{ts}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    # create a horizontal bar plot for the top_k features if requested
    if create_plots:
        top = df.head(top_k)
        plt.figure(figsize=(max(12, top_k * 0.6), 12))
        plt.barh(range(len(top)), top["importance"].values[::-1], align="center")
        plt.yticks(range(len(top)), top["feature"].values[::-1])
        plt.xlabel("Importance (arbitrary scale)")
        plt.title(f"Feature importances ({reason}; mapping={mapping_note})")
        plt.tight_layout()
        if png_name is None:
            png_name = f"{label}feature_importances_{ts}.png"
        png_path = os.path.join(output_dir, png_name)
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[feature_importance] Saved plot to {png_path}")

    print(f"[feature_importance] Saved CSV to {csv_path}  (reason={reason}; mapping={mapping_note})")
    return df
