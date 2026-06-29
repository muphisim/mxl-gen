################################################################################
#                           MxL-GEN - Optimisation Runner                      #
#                                                                              #
# Small helper wrapper that launches surrogate-enabled optimisations using the #
# lurtis_eoe framework. This file exposes a runOptimisation class which wraps  #
# cluster setup, optimiser and surrogate manager configuration, and provides  #
# convenience methods to run a final ground-truth simulation for the found    #
# optimum.                                                                     #
################################################################################

"""
Module: optimisation_runner

Purpose:
    Provide a small, friendly wrapper around the lurtis_eoe optimisation
    pipeline to make launching, logging and post-run verification (ground truth)
    easier for users.

Notes:
    - Convenience runner wrapping lurtis_eoe optimisation pipelines, surrogate
      management and Dask orchestration.
    - Exposes a compact `runOptimisation` class that wires up:
        - a multi-strategy MOS optimiser (SHADE + MTS),
        - a SurrogateManager for online surrogate training,
        - a Dask LocalCluster for parallel fitness evaluations.
    - The wrapper focuses on convenience and does not modify behaviour of the
      underlying lurtis_eoe components.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import os
import numpy as np
import json
import time
import datetime
import subprocess

from distributed import Client, LocalCluster

from lurtis_eoe.Optimisers.Algorithms.SHADE import SHADE
from lurtis_eoe.Optimisers.Algorithms.MTS_LS import MTS
from lurtis_eoe.Optimisers.Algorithms.Operators.Elitism.PairwiseElitism import PairwiseElitism
from lurtis_eoe.Surrogates.SurrogateManager import SurrogateManager
from lurtis_eoe.Surrogates.Models.Classifiers import DecisionTreeClassifier
from lurtis_eoe.Surrogates.Models.Regressors import XGBoostRegressor
from lurtis_eoe.OptimisationProcess import OptimisationProcess
from lurtis_eoe.Optimisers.MOS.MOS import MOSOptimiser
from lurtis_eoe.Fitness.BudgetCounter import FFEBudgetCounter
from lurtis_eoe.Fitness.FitnessModule import DaskFitnessModule

from MxL_GEN.optimisation.optimisation_fitness import Fitness

__all__ = ["runOptimisation"]


class runOptimisation:
    """Convenience wrapper to configure and run surrogate-enabled optimisations.

    What it does:
        - builds a Fitness wrapper that translates parameter vectors into
          simulator runs,
        - configures a MOS optimiser combining SHADE and MTS strategies,
        - creates a SurrogateManager for online training of surrogate models,
        - starts a LocalCluster + Dask client to parallelise evaluations,
        - offers a helper to launch a final full-model (ground-truth) run
          using the found optimal parameters.

    Args:
        parameter_to_model (Callable): maps a parameter vector to model inputs;
            typically writes input files into a run folder.
        extract_surrogate_func (Callable): runs the reduced model and returns
            surrogate inputs for the ML layer.
        surrogate_template_folder (str): path to the surrogate template folder
            used by the Fitness wrapper.
        bounds (Tuple[List[float], List[float]]): lower and upper bounds for the
            optimisation parameters.
        parameter_names (List[str]): human-friendly names of the parameters.
        preprocess_func (Optional[Callable], optional): optional preprocessing
            function applied to candidate vectors before launching runs.
            Defaults to ``None``.
        surrogate_file (Optional[str], optional): path to a pre-trained surrogate
            file to seed the fitness wrapper. Defaults to ``None``.
    """

    def __init__(
        self,
        parameter_to_model: Callable,
        extract_surrogate_func: Callable,
        surrogate_template_folder: str,
        bounds: Tuple[List[float], List[float]],
        parameter_names: List[str],
        preprocess_func: Optional[Callable] = None,
        surrogate_file: Optional[str] = None,
    ):
        # Store user-supplied callables and configuration for later use.
        self.parameter_to_model = parameter_to_model
        self.preprocess_func = preprocess_func
        self.extract_surrogate_func = extract_surrogate_func
        self.surrogate_template_folder = surrogate_template_folder
        self.bounds = bounds
        self.parameter_names = parameter_names
        self.surrogate_file = surrogate_file

    def run_opt_realisation(
        self,
        seed,
        simulation_folder,
        log_folder: str = None,
        clean_dir: bool = True,
        n_jobs: int = 1,
        num_steps: int = 1,
        budget: int = 1000,
        pop_size: int = 15,
    ) -> np.ndarray:
        """Configure and run a single optimisation realisation.

        This method:
            - creates a timestamped log folder (if none supplied),
            - instantiates the Fitness wrapper that ties parameters -> simulator,
            - starts a LocalCluster and Dask client,
            - configures the MOS optimiser and SurrogateManager,
            - runs the optimisation and returns the best-found parameters.

        Args:
            seed (int): Random seed for reproducibility.
            simulation_folder (str): Folder used by the fitness wrapper for
                temporary simulation files.
            log_folder (str, optional): Directory where logs are written. If
                ``None``, a timestamped folder is created. Defaults to ``None``.
            clean_dir (bool, optional): If True, the simulation folder will
                be reset before running. Defaults to ``True``.
            n_jobs (int, optional): Threads per Dask worker (controls parallelism).
                Defaults to ``1``.
            num_steps (int, optional): Number of optimisation steps passed to
                the MOS optimiser. Defaults to ``1``.
            budget (int, optional): Evaluation budget for the FFE budget counter.
                Defaults to ``1000``.
            pop_size (int, optional): Population size for the optimiser.
                Defaults to ``15``.

        Returns:
            numpy.ndarray: Array of optimal parameter values found by the optimisation.
        """
        # record start time for run-time reporting
        timeNow = datetime.datetime.now()

        # create a short log folder name with timestamp
        if log_folder is None:
            log_folder = f'{timeNow.strftime("%m-%d-%H:%M")}-Logs'

        # create log folder, and recreate simulation folder (delete if exists)
        os.system(f"rm -rf {simulation_folder} {log_folder}; mkdir {log_folder}; mkdir {simulation_folder}")

        # instantiate the problem / fitness wrapper
        problem = Fitness(
            self.parameter_to_model,
            self.extract_surrogate_func,
            self.surrogate_template_folder,
            bounds=self.bounds,
            parameter_names=self.parameter_names,
            preprocess_func=self.preprocess_func,
            simulation_folder=simulation_folder,
            clean_dir=clean_dir,
            log_folder=log_folder,
            surrogate_file=self.surrogate_file,
        )

        # seeded RNG for reproducibility
        np.random.seed(seed)

        # configure a local dask cluster for parallel evaluations
        cluster = LocalCluster(threads_per_worker=n_jobs, processes=False)
        cluster.scale(1)  # intentionally fixed to one worker for this wrapper

        # use the distributed Client as a context manager for automatic cleanup
        with Client(cluster) as client:
            optimiser = MOSOptimiser(
                starting_policy={
                    SHADE('SHADE_1', elitism_operator=PairwiseElitism(use_surrogate_models=True)): 0.5,
                    MTS('MTS_1', elitism_operator=PairwiseElitism(use_surrogate_models=True)): 0.5
                },
                num_steps=num_steps
            )

            # create a surrogate manager that will train models online
            surrogate_manager = SurrogateManager(
                warm_up=30,
                trail_size=45,
                models=[XGBoostRegressor(), DecisionTreeClassifier()],
                strategies=[],
                training_schedule='step',
                dask_client=client
            )

            # assemble the optimisation process with dask-backed fitness executor
            op = OptimisationProcess(
                fitness_function=problem,
                fitness_executor=DaskFitnessModule(client, FFEBudgetCounter(budget)),
                optimiser=optimiser,
                surrogates_manager=surrogate_manager,
                output_folder=Path(log_folder),
                seed=seed
            )

            # run the optimiser and retrieve results
            result = op.solve(population_size=pop_size)
            algorithm_info = op.optimiser.tracking_info  # kept for potential inspection

        # small pause to ensure file handles flushed
        time.sleep(.2)
        print("SUCCESSFUL OPTIMISATION FINISH")

        # load the saved Result.json to extract stored optimal values
        result = json.load(open(f"{log_folder}/Result.json",))
        values = np.array([float(v) for v in result["values"]])
        print("Optimal hyperparameters: ", values)

        # print elapsed time
        print("Run in time: ", str((datetime.datetime.now() - timeNow)))
        return values

    def run_ground_truth(
        self,
        log_folder: str,
        fullModelTemplate: str,
        parameter_to_model: Callable[[np.ndarray, str], None],
        values: Optional[np.ndarray] = None,
        preprocess_func: Optional[Callable[[np.ndarray], Tuple[bool, np.ndarray]]] = None,
        seed: int = 42,
    ) -> None:
        """Launch a ground-truth (full) model run for given parameter values.

        The helper copies the provided ``fullModelTemplate`` into a subfolder of
        the simulation folder, writes parameters (via ``parameter_to_model``)
        and starts the run script in the background (nohup style), capturing
        stdout/stderr to ``nohup.out`` inside the run folder.

        Args:
            log_folder (str): Parent folder containing the optimisation's simulation outputs.
            fullModelTemplate (str): Path to the full model template directory to copy.
            parameter_to_model (Callable[[np.ndarray, str], None]): Callable that maps
                a parameter vector to the model inputs in the run folder.
            values (Optional[np.ndarray], optional): Optional parameter vector to use.
                If ``None``, the function will attempt to load values from
                ``log_folder/Result.json``. Defaults to ``None``.
            preprocess_func (Optional[Callable[[np.ndarray], Tuple[bool, np.ndarray]]], optional):
                Optional preprocessing function that can veto or modify provided values.
                Defaults to ``None``.
            seed (int, optional): Seed value for reproducibility (reserved for future use).
                Defaults to ``42``.

        Returns:
            None
        """
        # If no explicit values provided, attempt to load them from Result.json
        if values is None:
            with open(os.path.join(log_folder, "Result.json"), "r") as f:
                values = np.array([float(v) for v in json.load(f)["values"]])

        print("Starting full model with optimal design parameters: ", values)

        # optional preprocessing check — allow the preprocess_func to reject the values
        if preprocess_func:
            discard, values = preprocess_func(values)
            if discard:
                # keep same behaviour as original code: print message but continue
                print("Optimal hyperparameters fail the preprocess. This should not happen!")

        # prepare a run folder inside the simulation folder to hold the full-model template
        runFolder = os.path.join(os.getcwd(), log_folder, "fullModelTemplate")

        try:
            # copy the provided fullModelTemplate into the run folder
            subprocess.run(["cp", "-r", fullModelTemplate, runFolder], check=True)
        except subprocess.CalledProcessError as e:
            # report copy failures but do not alter behaviour
            print(f"Error copying folder: {e}")

        # call the user-supplied parameter_to_model to write parameter files into runFolder
        parameter_to_model(values, runFolder)

        # launch the full-model run script in the background and capture output to nohup.out
        with open(os.path.join(runFolder, "nohup.out"), "ab") as out:
            subprocess.Popen(
                ["./run.sh", "."],
                cwd=runFolder,
                stdout=out,
                stderr=subprocess.STDOUT
            )

        print("Full model run for optimal hyperparameters in ", runFolder)
