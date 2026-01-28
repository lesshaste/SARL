# To be run on a SLURM login node.
#
# Based on advice from:
# - https://ax.dev/docs/0.5.0/tutorials/submitit/
# - https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials

# %% Setup
# Imports
import os
from profile import run
import time
from itertools import product
import warnings
from statistics import mean

import numpy as np
from tqdm import tqdm
from submitit import AutoExecutor, LocalJob, DebugJob
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

from sarl.train import main

warnings.filterwarnings("ignore")  # TODO: Ensure works

# Constants
# - set every time:
LOCAL_DEBUG_MODE = True  # TODO: Set to True for local debugging
SUBMITIT_DIR = "submitit"
HYDRA_CONFIG_PATH = "../../config"
# - computation:
CPU_CORES_PER_TASK = 4
# ---[Toy settings]---
MAX_TRIALS = 40 # Big effect on duration
PARALLEL_LIMIT = 2
# TRAIN_EPISODES = 2
# LEARNING_STEPS = 10 # per episode  # Multiple of on_policy_params.n_steps
# CYCLES = 2
# ---[Proper settings]---
# MAX_TRIALS = 40  # Big effect on duration
# PARALLEL_LIMIT = 40
TRAIN_EPISODES = 1000000
LEARNING_STEPS = 400000  # Multiple of on_policy_params.n_steps
CYCLES = 128
# ---[on-policy algs]---
ON_POLICY_PARAMS = {"n_steps": 100}
# ---[common bounds]---
BOUNDS_LR = (1e-6, 1e-3)
BOUNDS_UPDATE_RATIO = (0.01, 0.99)
# - misc:
SEEDS = [42]
ENVS = ["platform"]  # TODO: Set correct final ENVS
# ["platform", "goal"]

# TODO: Set correct final *_ALGS
DISCRETE_ALGS = ["ppo"]
CONTINUOUS_ALGS = ["ppo"]  # TODO: Vary discrete alg choice first
# DISCRETE_ALGS = ["a2c", "dqn", "ppo"]
# CONTINUOUS_ALGS = ["a2c", "ddpg", "ppo", "sac", "td3"]

## %% ---- SCRIPT START ----
# TODO: REMOVE THIS LINE
# quit()
cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"
width = os.get_terminal_size().columns

update_ratio_param = RangeParameterConfig(
    name="update_ratio",
    bounds=BOUNDS_UPDATE_RATIO,
    parameter_type="float",
    scaling="linear"
)

def get_params_by_alg(label:str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=(BOUNDS_LR[0], BOUNDS_LR[1]),
            parameter_type="float",
            scaling="linear"
        )
    ]
    return {
        "a2c": shared_params + [],
        "dqn": shared_params + [],
        "ppo": shared_params + [],  # INFO: Consider batch size
        "ddpg": shared_params + [],
        "sac": shared_params + [],
        "td3": shared_params + [],
    }
# pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(get_params_by_alg().keys(), repeat=2)]
pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(DISCRETE_ALGS, CONTINUOUS_ALGS)]


# %% Optimisation
# INFO: From source code of ax: the aqcuisition function used is qLogNoisyExpectedImprovement in BoTorch.
# See line 29 of https://github.com/facebook/Ax/blob/main/ax/generators/torch/botorch_modular/utils.py
def optimise():
    for pair, env in list(product(pairs, ENVS)):
        def get_client():
            # Setup Experiment via Ax
            client = Client()
            alg1, alg2 = pair.split("-")
            params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
            params = params + [update_ratio_param]
            client.configure_experiment(name="sarl_opt", parameters=params)  # INFO: What to guess [A]
            client.configure_optimization(objective="mean_reward")  # INFO: What to optimise [B]
            return client
        client = get_client()

        def get_executor():
            # Setup SubmitIt
            executor = AutoExecutor(folder=SUBMITIT_DIR, cluster=cluster)
            executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.
            executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)
            return executor
        executor = get_executor()

        # def objective_function(params: dict[str, list[RangeParameterConfig]]):
        def objective_function(params: dict[str, float]):
            # Define the Function to Optimise.
            # Calls main() from train.py with an automated hydra config.
            GlobalHydra.instance().clear()  # critical reset
            with initialize(config_path=HYDRA_CONFIG_PATH, job_name=(f"{pair.replace('-', '_')}-{env}-{SEEDS}")):
                cfg = compose(config_name="sarl", return_hydra_config=True, overrides=[
                        f"algorithm={pair}",
                        f"environment={env}",
                        f"parameters.seeds={SEEDS}",
                        f"parameters.train_episodes={TRAIN_EPISODES}",
                        f"parameters.learning_steps={LEARNING_STEPS}",
                        f"parameters.cycles={CYCLES}",
                        f"parameters.alg_params.discrete_learning_rate={params['discrete_learning_rate']}",
                        f"parameters.alg_params.continuous_learning_rate={params['continuous_learning_rate']}",
                        f"parameters.alg_params.update_ratio={params['update_ratio']}",
                        f"parameters.alg_params.on_policy_params.n_steps={ON_POLICY_PARAMS['n_steps']}",

                    ])
                HydraConfig.instance().set_config(cfg)  # manually register config
                mean_reward = main(cfg)  # TODO: [1] main() returns mean_reward
            return {"mean_reward": mean_reward}#, "std_reward": std_reward}

        # This demonstrates the objective_function works
        # def toy_func():
        #     params = {"learning_rate": 0.5}
        #     print(objective_function(params))
        #     return True
        # toy_func()

        def run_parallel_exps():
            # Run the Experiment
            """
            Returns list of:
                - The parameters predicted to have the best optimization value without
                    violating any outcome constraints.
                - The metric values for the best parameterization. Uses model prediction if
                    ``use_model_predictions=True``, otherwise returns observed data.
                - The trial which most recently ran the best parameterization
                - The name of the best arm (each trial has a unique name associated with
                    each parameterization)
            """
            jobs = []
            global submitted_jobs
            submitted_jobs = 0
            results = []
            while submitted_jobs < MAX_TRIALS or jobs:
                def run_trials():
                    global submitted_jobs
                    trial_index_to_param = client.get_next_trials(  # INFO: Ax makes guesses [C]
                        min(PARALLEL_LIMIT - len(jobs), MAX_TRIALS - submitted_jobs)
                    )
                    for trial_index, parameters in trial_index_to_param.items():
                        job = executor.submit(objective_function, parameters)
                        submitted_jobs += 1
                        jobs.append((job, trial_index))
                        time.sleep(1)
                def learn_from_any_previous_trials():
                    for job, trial_index in jobs[:]:  # INFO: Ax learns how any previous guesses went [D]
                        # Monitor for completed jobs
                        if job.done() or type(job) in [LocalJob, DebugJob]:
                            result = job.result()
                            print(f"\n[JOB RESULT]: {result}")  # TODO: Check working well
                            print("-" * width)
                            _ = client.complete_trial(trial_index=trial_index, raw_data=result)
                            results.append(results)
                            _ = jobs.remove((job, trial_index))
                        # WARN: Reintroduce sleep() for Slurm
                        # time.sleep(1)
                run_trials()
                learn_from_any_previous_trials()
                # WARN: Reintroduce sleep() for Slurm
                # time.sleep(30)
            best = client.get_best_parameterization()  # TODO: Save best parameterisations & corresponding mean rewards
            # best_param, best_mean_reward = 0, 0
            return {"best": best, "results": results}
        outcome = run_parallel_exps()  # TODO: 2025-12-19 Query the job list to help visualise (also look online for best practices)
        print(f"\n[RESULT] {outcome['best'][0]} results in {outcome['best'][1]} observed on trial {outcome['best'][2]}")
        print("-" * width)

        def visualise():  # TODO: Save visualisations
            return
            sorted_values = sorted(zip(learning_rates, mean_rewards))
            sorted_lr, sorted_mean_rewards = zip(*sorted_values)
            plt.plot(sorted_lr, sorted_mean_rewards)
            plt.xlabel("Learning Rate")
            plt.ylabel("Mean Reward")
            plt.savefig("mean_rewards_rl.png")
optimise()
