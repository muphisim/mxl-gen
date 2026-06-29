# Optimisation

This guide explains how optimisation is performed in MxL-GEN using
surrogate-assisted search.

Optimisation combines:
- a fitness wrapper
- surrogate models
- a multi-strategy evolutionary optimiser

---

## Key Components

Optimisation is driven by:

- `Fitness`
- `runOptimisation`

These components live under `optimisation/`.

---

## The Fitness Wrapper

`Fitness` connects optimisation parameters to model evaluation.

It:
- receives a parameter vector
- writes model inputs
- runs either a surrogate or reduced model
- returns a scalar fitness value

The same encoding functions used for ground truth are reused here, ensuring
consistency across the workflow.

---

## runOptimisation Wrapper

`runOptimisation` is a convenience wrapper that wires together:

- a MOS optimiser (SHADE + MTS)
- a `SurrogateManager`
- a Dask `LocalCluster`
- the `Fitness` evaluator

It handles setup, logging, and cleanup so users can focus on configuration.

---

## Running an Optimisation

A typical optimisation run:

- loads a trained surrogate pipeline
- configures bounds and parameter names
- runs a limited-budget optimisation
- records logs and results
- returns the best parameter vector found

This vector can then be validated using a full ground-truth run.

---

## Ground-Truth Validation

After optimisation, `run_ground_truth` can be used to:

- copy the full-model template
- write the optimal parameters
- launch a final full-model simulation

This closes the optimisation loop by validating surrogate-guided results.

