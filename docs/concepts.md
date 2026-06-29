# Core Concepts

This page introduces the main ideas behind MxL-GEN and how the pieces fit
together. Understanding these concepts makes it easier to adapt the workflow
to your own models.

---

## Ground Truth

Ground truth refers to running the **full, high-fidelity model**.

These runs are assumed to be:

- expensive
- slow
- accurate

MxL-GEN treats the full model as a black box and interacts with it only through
user-provided encoding and extraction functions.

---

## Surrogate Models

A surrogate model is a fast approximation of the full model.

In MxL-GEN, surrogates are:

- trained from ground-truth input/output pairs
- implemented as sklearn pipelines
- optionally updated during optimisation

Surrogates trade some accuracy for large gains in speed.

---

## Encoding Functions

Encoding functions connect MxL-GEN to your model.

They are responsible for:

- writing model input files from parameter vectors
- extracting surrogate inputs
- extracting fitness values

This separation allows MxL-GEN to remain model-agnostic.

---

## Fitness Evaluation

The fitness function wraps model execution and returns a scalar value used
by the optimiser.

Depending on the stage, fitness may be computed using:

- the full model (ground truth)
- a reduced model
- a surrogate model

The same interface is used in all cases.

---

## Optimisation

Optimisation searches for parameter values that minimise or maximise fitness.

MxL-GEN uses:

- evolutionary and local-search strategies
- surrogate-assisted evaluations
- a budget-based stopping criterion

Optimisation can be run with or without surrogates, depending on configuration.

---

## Parallel Execution

Parallelism is handled using Dask.

This allows:

- multiple fitness evaluations to run at once
- better use of available CPU resources
- easy scaling from laptops to clusters

Core limits are respected to avoid oversubscription.

---

## Validation Loop

A key concept in MxL-GEN is closing the loop:

1. Train a surrogate
2. Optimise using the surrogate
3. Validate the result with the full model

This ensures that surrogate-driven decisions remain grounded in reality.

---

## Putting It All Together

The full MxL-GEN workflow looks like:

- ground truth → surrogate data
- surrogate data → trained surrogate model
- surrogate model → optimisation
- optimisation result → ground-truth validation

Each stage is modular and can be replaced or extended as needed.

