################################################################################
#                    MxL-GEN - ML Model Utilities and Defaults                 #
#                                                                              #
# Helper functions for computing regression metrics and safe correlations,     #
# along with a default set of machine-learning models and parameter grids      #
# used in surrogate training and evaluation.                                   #
#                                                                              #
################################################################################

"""

Provides:
    - _spearman_scorer:     Safe computation of Spearman correlation.
    - _pearson_safe:        Safe computation of Pearson correlation.
    - _compute_metrics:     Standard regression metrics as a dictionary.
    - DEFAULT_MODEL_DICT:   Mapping of model names to (estimator, parameter grid)
                            for use in surrogate training pipelines.


Notes:
    - Small helpers for computing regression metrics and safe
      correlations (Spearman/Pearson) used across the surrogate pipeline.
    - Includes a ready-to-use DEFAULT_MODEL_DICT mapping human names to
      (estimator, hyperparameter-grid) pairs for quick model selection.
    - Metric helpers return ``float('nan')`` when a correlation cannot be
      computed (for example, when inputs are constant).
"""

from typing import Dict, Any
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def _spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation safely.

    Args:
        y_true (np.ndarray): Ground-truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Spearman correlation coefficient, or NaN if it cannot be computed.
    """
    try:
        return spearmanr(y_true, y_pred).correlation
    except Exception:
        return float("nan")


def _pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation safely.

    Args:
        y_true (np.ndarray): Ground-truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Pearson correlation coefficient, or NaN if it cannot be computed.
    """
    try:
        return pearsonr(y_true, y_pred)[0]
    except Exception:
        return float("nan")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute common regression metrics and return them in a dict.

    Metrics included:
      - r2: R-squared
      - mse: mean squared error
      - rmse: root mean squared error
      - mae: mean absolute error
      - pearson: Pearson correlation (safe)
      - spearman: Spearman correlation (safe)

    All values are rounded to 6 decimal places for compact reporting.

    Args:
        y_true (np.ndarray): Ground-truth values.
        y_pred (np.ndarray): Predictions.

    Returns:
        Dict[str, float]: Mapping metric name -> value.
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "r2": float(np.round(r2_score(y_true, y_pred), 6)),
        "mse": float(np.round(mean_squared_error(y_true, y_pred), 6)),
        "rmse": float(np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 6)),
        "mae": float(np.round(mean_absolute_error(y_true, y_pred), 6)),
        "pearson": float(np.round(_pearson_safe(y_true, y_pred), 6)),
        "spearman": float(np.round(_spearman_scorer(y_true, y_pred), 6)),
    }


# ---------------------------------------------------------------------------
# Default model dictionary for ML model selection.
#
# Each entry maps:
#     name -> (estimator_instance, parameter_grid_for_search)
#
# These parameters are intended for use in pipelines where the estimator is
# named 'model', hence parameter keys take the form 'model__param'.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DICT: Dict[str, Any] = {

    "Linear regression": (
        LinearRegression(),
        {
            "model__fit_intercept": [True, False],
            "model__positive": [False]
        }
    ),

    "Ridge": (
        Ridge(),
        {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__fit_intercept": [True, False],
            "model__solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
        }
    ),

    "Lasso": (
        Lasso(max_iter=10000),
        {
            "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "model__fit_intercept": [True, False],
            "model__selection": ["cyclic", "random"]
        }
    ),

    "Bayesian Ridge": (
        BayesianRidge(),
        {
            "model__max_iter": [100, 300, 500],
            "model__tol": [1e-4, 1e-3, 1e-2],
            "model__alpha_1": [1e-6, 1e-5, 1e-4],
            "model__lambda_1": [1e-6, 1e-5, 1e-4]
        }
    ),

    "Decision Tree": (
        DecisionTreeRegressor(),
        {
            "model__criterion": ["squared_error", "friedman_mse", "absolute_error"],
            "model__max_depth": [None, 3, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }
    ),

    "Random Forest": (
        RandomForestRegressor(),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingRegressor(),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [1.0, 0.8]
        }
    ),

    "SVR": (
        SVR(),
        {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "poly"],
            "model__degree": [2, 3]
        }
    ),

    "MLP": (
        MLPRegressor(max_iter=2000),
        {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "model__activation": ["relu", "tanh"],
            "model__solver": ["adam", "lbfgs"],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [1e-3, 1e-2]
        }
    ),
}
