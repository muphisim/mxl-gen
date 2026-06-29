# Ground Truth Generation

This guide explains how MxL-GEN generates ground-truth data and prepares it
for surrogate model training.

Ground truth refers to running the full (high-fidelity) model to obtain
input/output pairs that are later used to train surrogate models.

---

## Key Components

Ground truth generation is handled by:

- `start_new_runs`
- `extractData`

Both live in `create_ground_truth.py`.

These functions are model-agnostic and rely on user-provided encoding logic.

---

## Required User Functions

You must provide the following callables (usually in `encoding.py`):

- `parameter_to_model(values, run_folder)`
  - Writes model input files from a parameter vector.
- `extract_surrogate_inputs(run_folder)`
  - Reads surrogate input features from a completed run.
- `extract_fitness(run_folder)`
  - Reads the target output (fitness) from a completed run.
- `preprocess_parameters(values)` (optional)
  - Rejects or modifies parameter vectors before simulation.

---

## Launching Ground-Truth Runs

`start_new_runs` performs the following steps:

- samples parameters within provided bounds
- copies a full-model template into a new run folder
- writes the sampled parameters to disk
- launches the model run script (`run.sh`)
- respects a global CPU core budget

Runs are placed under a `groundTruth/` directory by default.

You can set `numNewRuns = 0` to skip generation if data already exists.

---

## Extracting Training Data

Once runs are complete, `extractData`:

- walks through all run folders
- applies the user extraction functions
- aggregates inputs and outputs
- writes:
  - `trainingInput.csv`
  - `trainingOutput.csv`

These files are written to the `surrogateCreation/` folder and form the
training dataset for surrogate models.

---

## Outputs

After this stage you should have:

- raw simulation folders under `groundTruth/`
- surrogate-ready CSV files under `surrogateCreation/`

These outputs are consumed by the EDA and surrogate modelling steps.

