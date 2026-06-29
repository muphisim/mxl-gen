# End-to-End MxL-GEN Demo Workflow

This tutorial walks through the full MxL-GEN workflow using the provided
demo pipeline script. It shows how the main components fit together, from
ground-truth generation through surrogate training and optimisation.

The demo is designed to be run as a single script and can be adapted as a
starting point for your own projects.

---

## Overview of the Workflow

The demo pipeline covers the following steps:

1. Define parameter bounds and model templates
2. (Optionally) generate ground-truth simulations
3. Extract surrogate training data
4. Perform exploratory data analysis (EDA)
5. Train and select surrogate ML models
6. Save and reload the trained pipeline
7. Inspect feature importance
8. Run a small optimisation using the surrogate
9. Validate the result with a ground-truth run

Each step corresponds directly to a section in the demo script.

---

## 1. Configuration and User-Provided Encoders

The workflow starts by defining:

- parameter bounds and names
- paths to example model templates
- user-provided encoding functions

These functions live in `encoding.py` and are project-specific:

- `parameter_to_model`: writes model inputs from a parameter vector
- `extract_surrogate_inputs`: extracts surrogate inputs from a run
- `extract_fitness`: extracts the fitness value
- `preprocess_parameters` (optional): filters or modifies parameter vectors

These hooks let MxL-GEN stay model-agnostic.

---

## 2. Ground-Truth Dataset Creation

Ground-truth simulations are created using:

- `start_new_runs`
- `extractData`

This stage:

- copies a full-model template into run folders
- samples parameters within bounds
- launches simulations
- extracts input/output pairs for surrogate training

For quick demos, the number of new runs can be set to zero if data already
exists.

Outputs are written to:

- `groundTruth/` (raw runs)
- `surrogateCreation/` (training CSVs)

---

## 3. Exploratory Data Analysis (EDA)

The `data_exploration` helper provides quick insight into the surrogate data.

In the demo, it is used to:

- plot correlation matrices
- generate feature histograms
- report missing values
- run PCA and inspect explained variance
- detect outliers
- estimate feature importance

All outputs are written to the `surrogateCreation/` folder so they can be
reviewed without rerunning the analysis.

---

## 4. Surrogate Model Training and Selection

Surrogate models are trained using the `mlModels` class.

This step:

- loads the surrogate training CSVs
- runs hyperparameter search (random or grid)
- evaluates models using a holdout split
- records metrics and predictions
- identifies the best-performing model

The demo uses a small number of search iterations to keep runtime short.

Results include:

- JSON summaries
- CSV prediction files
- optional diagnostic plots
- a fitted sklearn pipeline saved to disk

---

## 5. Loading and Using the Saved Pipeline

The trained pipeline is saved as a pickle file and can be reloaded later.

The demo shows how to:

- load the pipeline from disk
- run predictions on new samples
- confirm outputs against training data

This step demonstrates how the surrogate can be used independently of the
training script.

---

## 6. Feature Importance Inspection

Feature importance is computed using `feature_importance`.

Depending on the model, this may use:

- built-in feature importances
- linear coefficients
- permutation importance

If PCA is present in the pipeline, importances are mapped back to the original
feature space where possible.

Both CSV summaries and plots are saved for inspection.

---

## 7. Fitness Wrapper Sanity Check

Before optimisation, the demo instantiates a `Fitness` wrapper directly.

This step:

- reuses the same encoding functions
- runs a few test evaluations
- confirms that the surrogate and fitness logic are wired correctly

This is a quick sanity check before launching optimisation.

---

## 8. Running an Optimisation

Optimisation is performed using the `runOptimisation` wrapper.

The demo:

- configures a MOS optimiser (SHADE + MTS)
- uses the trained surrogate pipeline
- runs a small-budget optimisation
- records logs and intermediate results

This step shows how surrogate-assisted optimisation fits into the workflow.

---

## 9. Ground-Truth Validation of the Optimum

Finally, the demo launches a full-model run using the found optimal parameters.

This step:

- copies the full-model template
- writes the optimal parameters
- launches the run script
- captures output for inspection

This closes the loop by validating the surrogate-guided optimum against the
full model.

---

## Summary

This demo script shows how MxL-GEN components work together as a complete
pipeline:

- ground truth → surrogate data → surrogate model → optimisation → validation

You can adapt individual steps or replace the example templates and encoding
functions to fit your own models and workflows.
