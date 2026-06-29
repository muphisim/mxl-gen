################################################################################
#                         MxL-GEN - Ground truth builder                       #
#                                                                              #
# Utilities for constructing ground-truth datasets by launching and managing   #
# multiple full-model simulations and extracting their inputs/outputs for      #
# surrogate training.                                                          #
#                                                                              #
################################################################################
"""
Notes:
    - Provides utilities to launch full-model (ground-truth) simulations in
      parallel while respecting (user-implemented) CPU core limits.
    - Manages run folder creation, parameter sampling, and invocation of model
      run scripts using the user-provided, full model, template directory.
    - Includes helpers to traverse completed runs and extract input/output
      arrays into CSV files suitable for surrogate model training.
"""

__all__ = ["start_new_runs", "extractData"]

import os
import time
import subprocess
from typing import Callable, Optional, Tuple, Sequence

import numpy as np
import pandas as pd


def start_new_runs(
    numNewRuns: int,
    fullModelTemplate: str,
    lowerBounds: Sequence[float],
    upperBounds: Sequence[float],
    numCoresPerSim: int,
    numCores: int,
    parameter_to_model: Callable[[np.ndarray, str], None],
    groundTruthFolder: Optional[str] = None,
    preprocess_func: Optional[Callable[[np.ndarray], Tuple[bool, np.ndarray]]] = None,
    seed: int = 42,
) -> None:
    """Launch a set of new ground-truth simulations in parallel.

    The function:
        - ensures a `groundTruth` folder exists (or uses `groundTruthFolder`),
        - identifies the next run id, copies `fullModelTemplate` into run
          folders named `run{N}`, and writes `design_parameters.csv`,
        - calls `parameter_to_model(values, run_folder)` to populate model inputs,
        - launches `./run.sh` in each run folder and streams output to `nohup.out`,
        - optionally runs `preprocess_func(values)` to reject/transform sampled vectors,
        - blocks starting additional runs until sufficient CPU cores are available.

    Args:
        numNewRuns (int): Number of simulations to start.
        fullModelTemplate (str): Path to the folder that will be copied for
            each simulation run.
        lowerBounds (Sequence[float]): Lower bounds for random parameter
            generation (1-D sequence).
        upperBounds (Sequence[float]): Upper bounds for random parameter
            generation (1-D sequence).
        numCoresPerSim (int): CPU cores consumed by each simulation.
        numCores (int): Total available CPU cores (global budget).
        parameter_to_model (Callable[[np.ndarray, str], None]): Callable that
            accepts a 1D numpy array of parameter values and the destination
            run folder path, and writes model input files there.
        groundTruthFolder (Optional[str], optional): Path to the groundTruth
            base folder. If None, uses ``os.path.join(os.getcwd(), "groundTruth")``.
            Defaults to None.
        preprocess_func (Optional[Callable[[np.ndarray], Tuple[bool, np.ndarray]]], optional):
            Optional callable that receives a sampled parameter vector and
            returns a tuple ``(discard_flag, values)``. If ``discard_flag`` is
            True the sample is ignored and a new one is drawn. Defaults to None.
        seed (int, optional): RNG seed for reproducible draws. Defaults to 42.

    Returns:
        None
    """
    # reproducibility
    np.random.seed(seed)

    gt_folder = (
        os.path.join(os.getcwd(), "groundTruth")
        if groundTruthFolder is None
        else groundTruthFolder
    )
    os.makedirs(gt_folder, exist_ok=True)

    # find next numeric run id by scanning existing run folders
    existing = [d for d in os.listdir(gt_folder) if d.startswith("run") and d[3:].isdigit()]
    nextId = max([int(d[3:]) for d in existing], default=-1) + 1

    print("Starting new set of runs with first id", nextId, ". Number of new runs", numNewRuns)
    runsLaunched = 0

    def _num_cores_remaining() -> int:
        """Compute number of CPU cores currently free for new simulations.

        Returns:
            int: Remaining available cores (may be negative if budget exceeded).
        """
        folders = [f for f in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, f))]
        active_count = sum(
            not os.path.exists(os.path.join(gt_folder, f, "endedSim.txt")) for f in folders
        )
        return numCores - numCoresPerSim * active_count

    while runsLaunched < numNewRuns:
        # sample parameters uniformly within provided bounds
        values = np.random.uniform(low=lowerBounds, high=upperBounds)

        # optional preprocess step (user may reject or transform the sample)
        if preprocess_func:
            discard, values = preprocess_func(values)
            if discard:
                continue

        runFolder = os.path.join(gt_folder, f"run{nextId}")
        print(f"Starting simulation in {runFolder}")

        # copy template folder into runFolder
        try:
            subprocess.run(["cp", "-r", fullModelTemplate, runFolder], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error copying folder: {e}")
            break

        # save design parameters (traceability)
        os.system(f"touch {os.path.join(runFolder, 'design_parameters.csv')}")
        np.savetxt(os.path.join(runFolder, "design_parameters.csv"), values.reshape(1, -1), delimiter=",", fmt="%.3f")

        # let user-provided function populate the run folder with model inputs
        parameter_to_model(values, runFolder)

        # launch the model run script in the background, streaming stdout/stderr to nohup.out
        with open(os.path.join(runFolder, "nohup.out"), "ab") as out:
            subprocess.Popen(
                ["./run.sh", "."],
                cwd=runFolder,
                stdout=out,
                stderr=subprocess.STDOUT,
            )

        # small delay and then block until there are enough cores for another run
        time.sleep(1)
        while _num_cores_remaining() < numCoresPerSim:
            time.sleep(10)

        runsLaunched += 1
        nextId += 1


def extractData(
    input_func: Optional[Callable[[str], np.ndarray]],
    output_func: Optional[Callable[[str], np.ndarray]],
    groundTruthFolder: Optional[str] = None,
) -> None:
    """Process run folders in groundTruth and collect inputs/outputs.

    This function:
        - creates (if missing) a surrogateCreation folder,
        - iterates over run folders sorted by name,
        - calls input_func(folder_path) and output_func(folder_path) for each run,
        - collects successful results and writes CSVs for downstream training.

    Args:
        input_func (Optional[Callable[[str], numpy.ndarray]]): Callable that
            accepts a run folder path and returns a 1D numpy array of input features.
        output_func (Optional[Callable[[str], numpy.ndarray]]): Callable that
            accepts a run folder path and returns a 1D numpy array of outputs/targets.
        groundTruthFolder (Optional[str], optional): Path to the groundTruth
            base folder. If None, uses ``os.path.join(os.getcwd(), "groundTruth")``.
            Defaults to None.

    Returns:
        None
    """
    gt_folder = (
        os.path.join(os.getcwd(), "groundTruth")
        if groundTruthFolder is None
        else groundTruthFolder
    )
    surrogate_folder = os.path.join(os.getcwd(), "surrogateCreation")
    os.makedirs(surrogate_folder, exist_ok=True)

    input_arr = []
    output_arr = []

    # iterate deterministically over sorted run folder names
    run_folders = sorted([f for f in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, f))])
    for folder_name in run_folders:
        folder_path = os.path.join(gt_folder, folder_name)
        print(f"Starting to extract data from {folder_name}")
        try:
            # call the user-supplied extraction functions
            input_vector = input_func(folder_path) if input_func is not None else None
            output_vector = output_func(folder_path) if output_func is not None else None

            if input_vector is None or output_vector is None:
                raise ValueError("input_func and output_func must both return arrays; got None")

            input_arr.append(input_vector)
            output_arr.append(output_vector)
            print(f"Successfully extracted data from {folder_name}")
        except Exception as e:
            # report and skip problematic folders
            print(f"Warning: Skipping folder {folder_name} due to error: {e}")

    # convert to numpy arrays and persist for downstream training
    input_np = np.array(input_arr)
    output_np = np.array(output_arr)

    print(f"Input shape: {input_np.shape}")
    pd.DataFrame(input_np).to_csv(os.path.join(surrogate_folder, "trainingInput.csv"), index=False, header=False)
    pd.DataFrame(output_np).to_csv(os.path.join(surrogate_folder, "trainingOutput.csv"), index=False, header=False)

