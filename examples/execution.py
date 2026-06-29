################################################################################
#                             MxL-GEN - Demo Pipeline                          #
#                                                                              #
# Top-level script demonstrating the full ASSUAGE workflow:
#   - creates (optionally) ground truth simulations,
#   - extracts surrogate training data,
#   - performs EDA on surrogate inputs/outputs,
#   - runs quick model selection and trains the best pipeline,
#   - demonstrates loading the saved pipeline and computing feature importances,
#   - constructs a Fitness wrapper and runs a quick optimisation realisation.
#
# This file is intended to be used as a drop-in demo runner. Behaviour is kept
# identical to the original file; only documentation and inline comments have
# been added for readability.
################################################################################

# Core / stdlib imports
import os
import pickle

# Numeric / data libs
import numpy as np
import pandas as pd

# Project-specific encoding helpers (user provides these in encoding.py)
from encoding import *

# ---------------------------------------------------------------------
# Basic configuration: parameter bounds, names and template locations
# ---------------------------------------------------------------------
lowerBounds = [0] * 3
upperBounds = [1] * 3
paramNames = [f"radius{i}" for i in range(3)]

# Path to example full-model template shipped with the examples
fullModelTemplate = os.path.join(os.getcwd(), "exampleTemplates/fullModelTemplate")

# Optional callables provided by the user in encoding.py — we attempt to import
# them and fall back with informative messages if absent.
try:
    from encoding import preprocess_parameters as preprocess_func
except Exception:
    print("No preprocess_parameters function found in encoding file")
    preprocess_func = None

try:
    from encoding import parameter_to_model
except Exception:
    print("No parameter_to_model function found in encoding file. This is a necessary function!!")

# ---------------------------------------------------------------------
# Ground truth creation settings
# ---------------------------------------------------------------------
numNewRuns = 0
numCoresPerSim = 1
numCores = 10

# Sanity check: ensure bound vectors match length
assert len(lowerBounds) == len(upperBounds), "Upper and lower bound lists must have the same length."

# ---------------------------------------------------------------------
# Create ground truth dataset (calls into MxL_GEN.create_ground_truth)
# ---------------------------------------------------------------------
from MxL_GEN.create_ground_truth import start_new_runs, extractData
# Uncomment the following line to wipe previous groundTruth/surrogateCreation folders:
# os.system(f"rm -rf groundTruth surrogateCreation")

start_new_runs(
    numNewRuns,
    fullModelTemplate,
    lowerBounds,
    upperBounds,
    numCoresPerSim,
    numCores,
    parameter_to_model,
    groundTruthFolder = "test_gt",
    preprocess_func = preprocess_func,
)

# Extract surrogate inputs/outputs from produced ground-truth runs
# The functions `extract_surrogate_inputs` and `extract_fitness` are expected
# to be provided by the user (typically in encoding.py).
extractData(extract_surrogate_inputs, 
            extract_fitness,
            groundTruthFolder = "test_gt")

# ---------------------------------------------------------------------
# Exploratory Data Analysis (EDA) on the surrogate dataset
# ---------------------------------------------------------------------
from MxL_GEN.surrogate_model.data_exploration import data_exploration

explorer = data_exploration(
    "surrogateCreation/trainingInput.csv",
    "surrogateCreation/trainingOutput.csv",
    scaled_vis=False,
)

# Generate and save a set of diagnostic outputs into surrogateCreation/
explorer.correlation_matrix()
explorer.feature_histograms()
missing_report = explorer.missing_value_report()
reduced_data = explorer.explanatory_dimension(0.95)  # PCA preserving 95% variance
variance_table = explorer.explain_variance_table()
explorer.pairplot(sample=150)
explorer.pairplot(sample=150, hue=0)
outlier_mask = explorer.outlier_detection(method="zscore", thresh=3.0)
importances = explorer.feature_importance_proxy(target_column=0)

print("\nAll demonstration outputs written to 'surrogateCreation/'.")

# ---------------------------------------------------------------------
# ML model fitting, selection and pipeline training
# ---------------------------------------------------------------------
from MxL_GEN.surrogate_model.ml_models import mlModels

modeler = mlModels(
    input_data="surrogateCreation/trainingInput.csv",
    output_data="surrogateCreation/trainingOutput.csv",
    data_label="demo",
    output_dir="surrogateCreation",
    pca=False,            # no PCA for simplicity in the demo
    search_type="random",
    n_iter=5,             # small number of randomized iterations for quick demo
    n_jobs=10,            # level of parallelism for model searches
    seed=42,
    best_model_cutoff=0.95,
)

# Quick holdout evaluation (fast demonstration)
print("\nRunning evaluate_holdout (quick demo)...")
results = modeler.evaluate_holdout(test_size=0.1, inner_cv=5, create_plots=True)

print("\nBest model recorded in the object:")
print(" best_model_name:", getattr(modeler, "best_model_name", None))
print(" best_model_score:", getattr(modeler, "best_model_score", None))
print(" best_model_info:")
print(getattr(modeler, "best_model_info", None))

# Train and save the best pipeline (auto-selected)
print("\nRunning train_best_pipeline() (uses recorded best model)...")
pipeline = modeler.train_best_pipeline(cv=5)  # pipeline saved into modeler.output_dir

print("\nPipeline saved to:", modeler.best_pipeline_path)
print("Best pipeline object type:", type(pipeline))


# ---------------------------------------------------------------------
# Demonstrate loading the saved pipeline and predicting on a few samples
# ---------------------------------------------------------------------
with open(modeler.best_pipeline_path, "rb") as fh:
    loaded_pipe = pickle.load(fh)

sample_X = pd.read_csv("surrogateCreation/trainingInput.csv", header=None).iloc[:5].to_numpy()
preds = loaded_pipe.predict(sample_X)
print("\nPredictions on first 5 samples (loaded pipeline):")
print(preds, pd.read_csv("surrogateCreation/trainingOutput.csv", header=None).iloc[:5].to_numpy())

# Compute and save feature importances for the trained pipeline
from MxL_GEN.surrogate_model.ml_plotter import feature_importance
feature_importance(loaded_pipe, output_dir=modeler.output_dir, X=modeler.X, y=modeler.y)
print("All outputs written into 'surrogateCreation' folder.")

# ---------------------------------------------------------------------
# Quick sanity-check of the Fitness wrapper and a small optimisation run
# ---------------------------------------------------------------------
from MxL_GEN.optimisation.optimisation_fitness import Fitness

# Instantiate a Fitness wrapper (re-using user-provided encoders)
f = Fitness(
    parameter_to_model,
    extract_surrogate_inputs,
    "exampleTemplates/reducedModelTemplate",
    bounds=(lowerBounds, upperBounds),
    parameter_names=paramNames,
    preprocess_func=preprocess_parameters,
    simulation_folder="SimulationTest",
    clean_dir=False,
)

# Two sample surrogate evaluations (demonstration only)
print(f.fitness(np.random.random(size=(24)), id=0).fitness)
print(f.fitness(np.random.random(size=(24)), id=1).fitness)

# ---------------------------------------------------------------------
# Configure and run an optimisation using the runOptimisation wrapper
# ---------------------------------------------------------------------
from MxL_GEN.optimisation.run_optimisation import runOptimisation

optimiser = runOptimisation(
    parameter_to_model=parameter_to_model,
    extract_surrogate_func=extract_surrogate_inputs,
    surrogate_template_folder="exampleTemplates/reducedModelTemplate",
    surrogate_file = modeler.best_pipeline_path,
    bounds=(lowerBounds, upperBounds),
    parameter_names=paramNames,
    preprocess_func=preprocess_parameters,
)

# Note: the run_opt_realisation signature used below mirrors the original call
# in the project. Some wrappers accept a log_folder kwarg; if your local
# implementation does not, remove it when calling.
values = optimiser.run_opt_realisation(
    seed=42,
    budget=50,
    n_jobs=10,
    simulation_folder="run_optimisation_test",
    log_folder="optimisation_logs",
    clean_dir=True,
)

# Launch the full model using the found values and run a ground-truth check.
optimiser.run_ground_truth(
    log_folder="optimisation_logs",
    fullModelTemplate=fullModelTemplate,
    parameter_to_model=parameter_to_model,
    # values = values,   # uncomment to pass the found optimum explicitly
    preprocess_func=preprocess_func,
    seed=42,
)


