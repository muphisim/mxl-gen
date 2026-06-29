# Surrogate Models

This guide covers how surrogate models are trained, evaluated, inspected, and
saved in MxL-GEN. It also describes the visualisation and plotting tools that
support model selection and validation.

Surrogates approximate the behaviour of the full model and are used to speed
up optimisation while keeping good accuracy.

---

## Inputs to Surrogate Modelling

Surrogate models are trained using the CSV files produced during ground-truth
generation:

- `trainingInput.csv`
- `trainingOutput.csv`

These files live under the `surrogateCreation/` folder and are reused across
EDA, model training, plotting, and optimisation.

---

## Exploratory Analysis Before Training

Before fitting models, it is strongly recommended to inspect the surrogate
dataset using the available EDA tools.

The `data_exploration` helper can be used to:

- plot correlation matrices between inputs
- generate feature histograms and distributions
- report missing values
- perform PCA and inspect explained variance
- detect outliers
- compute a quick proxy for feature importance

All plots and tables are written to `surrogateCreation/` so they can be reviewed
outside the code.

This step helps catch data issues early and informs modelling choices such as
scaling and PCA.

---

## Model Training and Hyperparameter Search

Surrogate model training is handled by the `mlModels` class.

It performs:

- construction of sklearn pipelines (scaler, optional PCA, estimator)
- hyperparameter search using:
  - randomized search
  - grid search
- evaluation using:
  - holdout splits
  - nested cross-validation
  - leave-one-out cross-validation

Candidate models and parameter grids are defined in
`ml_helpers.DEFAULT_MODEL_DICT` and can be extended by the user.

---

## Model Evaluation Metrics

During evaluation, the following metrics are computed and stored:

- R²
- RMSE
- MAE
- Pearson correlation
- Spearman correlation

Metrics are written to JSON and CSV files under `surrogateCreation/` and are
used to track the best-performing model.

---

## Evaluation Plots

Visual evaluation is supported through the `ml_plotter` module.

Available plots include:

- true vs predicted scatter plots for each model
- combined summary figures across multiple models
- nested cross-validation aggregation plots

These plots help identify bias, variance, and failure modes that may not be
obvious from scalar metrics alone.

Plots are saved automatically and do not require interactive sessions.

---

## Feature Importance and Interpretation

Once a surrogate model is trained, feature importance can be analysed using
the `feature_importance` helper.

Depending on the model, importance is computed using:

- built-in feature importances (tree-based models)
- absolute coefficients (linear models)
- permutation importance (model-agnostic fallback)

If PCA is used inside the pipeline, importances are mapped back to the original
feature space when possible.

Results are saved as:

- CSV tables of feature importance
- optional bar plots for visual inspection

---

## Selecting and Saving the Best Model

As evaluations run, `mlModels` keeps track of the best-performing model.

Once selected:

- the full pipeline is retrained on all available data
- the fitted pipeline is saved to disk as a pickle file
- the path to the saved pipeline is recorded for reuse

This saved pipeline is used directly during optimisation.

---

## Using Surrogates in Optimisation

The trained surrogate pipeline is passed into the optimisation stage via the
`Fitness` and `runOptimisation` wrappers.

This allows:

- fast surrogate-based fitness evaluations
- periodic retraining (if enabled)
- final validation using a full ground-truth run

---

## Summary

Surrogate modelling in MxL-GEN is not just about training a regressor.

It includes:

- data inspection and visualisation
- metric-based and visual evaluation
- feature importance analysis
- pipeline persistence and reuse

Together, these tools help ensure that surrogate models are both accurate and
trustworthy before they are used for optimisation.

