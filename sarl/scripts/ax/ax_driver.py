# To be run on a SLURM login node.
#
# Based on advice from:
# - https://ax.dev/docs/0.5.0/tutorials/submitit/
# - https://ax.dev/docs/0.5.0/bayesopt/#tradeoff-between-parallelism-and-total-number-of-trials

# %% Setup
import os
import time
from itertools import product
import warnings
from pathlib import Path
import csv
from typing import Optional, List

import numpy as np
from submitit import AutoExecutor, LocalJob, DebugJob
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from sarl.train import main

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
LOCAL_DEBUG_MODE = True  # set False on slurm
SUBMITIT_DIR = "submitit"
HYDRA_CONFIG_PATH = "../../config"

CPU_CORES_PER_TASK = 4

# --- experiment size / runtime ---
MAX_TRIALS = 80
PARALLEL_LIMIT = 2

# --- training settings ---
TRAIN_EPISODES = 1_000_000  # (note: you commented "does nothing currently")
CYCLES = 10
LEARNING_STEPS = 10000 * CYCLES

# --- on-policy ---
ON_POLICY_PARAMS = {"n_steps": 100}

# --- search bounds ---
BOUNDS_LR = (1e-6, 1e-3)
BOUNDS_UPDATE_RATIO = (0.01, 0.99)

# --- misc ---
SEEDS = [42]
ENVS = ["platform"]

DISCRETE_ALGS = ["a2c"]
CONTINUOUS_ALGS = ["ppo"]

cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"
width = os.get_terminal_size().columns


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def configure_objective_maximize(client: Client, metric_name: str) -> None:
    """
    Configure objective for many Ax Client API variants.
    Ax maximizes by default, so we only try "maximize metric_name".
    """
    if not hasattr(client, "configure_optimization"):
        raise RuntimeError(
            "This Ax Client does not have configure_optimization(). "
            "Please paste the output of `dir(Client())` and I’ll adapt the call."
        )

    # Try the most common variants, without minimize flags.
    attempts = [
        lambda: client.configure_optimization(objective=metric_name),
        lambda: client.configure_optimization(objective_name=metric_name),
        lambda: client.configure_optimization(metric_name),
    ]

    last_err: Optional[Exception] = None
    for fn in attempts:
        try:
            fn()
            return
        except TypeError as e:
            last_err = e
        except Exception as e:
            # Some versions may raise ValueError etc. if arguments are wrong.
            last_err = e

    raise RuntimeError(
        f"Could not configure Ax objective with metric '{metric_name}'. "
        f"Last error: {last_err}"
    )


def _read_eval_csv_mean_reward(eval_csv: Path, *, mode: str = "max") -> Optional[float]:
    """
    Read mean reward from converter-produced eval.csv.

    Expected column (your converter): mean_eval_episode_return
    mode:
      - 'max': peak over all eval points
      - 'last': last row only
    """
    if not eval_csv.exists():
        return None

    rewards: List[float] = []
    with eval_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            key = "mean_eval_episode_return" if "mean_eval_episode_return" in row else None
            if key is None:
                for k in ("mean_reward", "mean_return", "reward", "mean"):
                    if k in row:
                        key = k
                        break
            if key is None:
                continue

            try:
                rewards.append(float(row[key]))
            except (TypeError, ValueError):
                continue

    if not rewards:
        return None

    return max(rewards) if mode == "max" else rewards[-1]


def _safe_float(x) -> float:
    """Convert output to a finite float for Ax."""
    if isinstance(x, dict):
        vals = [v for v in x.values() if v is not None]
        x = float(np.mean(vals)) if vals else float("nan")
    try:
        x = float(x)
    except Exception:
        x = float("nan")

    if not np.isfinite(x):
        return -1e9
    return x


# -----------------------------------------------------------------------------
# Parameter space
# -----------------------------------------------------------------------------
update_ratio_param = RangeParameterConfig(
    name="update_ratio",
    bounds=BOUNDS_UPDATE_RATIO,
    parameter_type="float",
    scaling="linear",
)


def get_params_by_alg(label: str = ""):
    shared_params = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=(BOUNDS_LR[0], BOUNDS_LR[1]),
            parameter_type="float",
            scaling="log",
        )
    ]
    return {
        "a2c": shared_params,
        "dqn": shared_params,
        "ppo": shared_params,
        "ddpg": shared_params,
        "sac": shared_params,
        "td3": shared_params,
    }


pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(DISCRETE_ALGS, CONTINUOUS_ALGS)]


# -----------------------------------------------------------------------------
# Optimisation
# -----------------------------------------------------------------------------
def optimise():
    for pair, env in list(product(pairs, ENVS)):

        def get_client():
            client = Client()
            alg1, alg2 = pair.split("-")
            params = get_params_by_alg("discrete")[alg1] + get_params_by_alg("continuous")[alg2]
            params = params + [update_ratio_param]

            client.configure_experiment(name="sarl_opt", parameters=params)
            # ✅ Ax maximizes by default; do NOT pass minimize flags
            configure_objective_maximize(client, "mean_reward")
            client.configure_generation_strategy(method="quality")
            return client

        client = get_client()

        def get_executor():
            executor = AutoExecutor(folder=SUBMITIT_DIR, cluster=cluster)
            executor.update_parameters(timeout_min=60)
            executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)
            return executor

        executor = get_executor()

        def objective_function(params: dict[str, float], trial_index: int):
            print(f"[AX][TRIAL {trial_index}] params={params}")
            # Run a training job with a composed Hydra config.
            GlobalHydra.instance().clear()

            with initialize(config_path=HYDRA_CONFIG_PATH, job_name=(f"{pair.replace('-', '_')}-{env}-{SEEDS}")):
                alg_discrete, alg_continuous = pair.split("-")

                overrides = [
                    f"algorithm={pair}",
                    f"environment={env}",
                    f"parameters.seeds={SEEDS}",
                    f"parameters.train_episodes={TRAIN_EPISODES}",
                    f"parameters.learning_steps={LEARNING_STEPS}",
                    f"parameters.cycles={CYCLES}",

                    # ✅ FIX: these must go under alg_params.{role}.learning_rate
                    f"+parameters.alg_params.discrete.learning_rate={params['discrete_learning_rate']}",
                    f"+parameters.alg_params.continuous.learning_rate={params['continuous_learning_rate']}",
                    f"parameters.alg_params.update_ratio={params['update_ratio']}",
                ]

                # ✅ FIX: converter_use reads n_steps from alg_params.{role}.n_steps (for on-policy only)
                if alg_discrete.lower() in {"ppo", "a2c"}:
                    overrides.append(f"+parameters.alg_params.discrete.n_steps={ON_POLICY_PARAMS['n_steps']}")
                if alg_continuous.lower() in {"ppo", "a2c"}:
                    overrides.append(f"+parameters.alg_params.continuous.n_steps={ON_POLICY_PARAMS['n_steps']}")

                overrides.append(f"hydra.run.dir=outputs/ax/{pair}/{env}/trial_{trial_index}") 
                cfg = compose(config_name="sarl", return_hydra_config=True, overrides=overrides)
                HydraConfig.instance().set_config(cfg)

                # Ensure output directory exists
                try:
                    Path(HydraConfig.get().runtime.output_dir).mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                mean_reward_from_main = main(cfg)

                # Prefer eval.csv if present
                mean_reward = mean_reward_from_main
                try:
                    output_dir = Path(HydraConfig.get().runtime.output_dir)
                except Exception:
                    output_dir = None

                if output_dir is not None:
                    csv_reward = _read_eval_csv_mean_reward(output_dir / "eval.csv", mode="max")
                    if csv_reward is not None:
                        mean_reward = csv_reward
                        print(f"[AX][METRIC] Using mean_reward from eval.csv: {mean_reward:.6f} (output_dir={output_dir})")

            return {"mean_reward": _safe_float(mean_reward)}

        def run_parallel_exps():
            jobs = []
            submitted_jobs = 0
            results = []

            while submitted_jobs < MAX_TRIALS or jobs:

                # submit more, up to parallel limit
                n = min(PARALLEL_LIMIT - len(jobs), MAX_TRIALS - submitted_jobs)
                if n > 0:
                    trial_index_to_param = client.get_next_trials(n)
                    for trial_index, parameters in trial_index_to_param.items():
                        job = executor.submit(objective_function, parameters, trial_index)
                        submitted_jobs += 1
                        jobs.append((job, trial_index))
                        time.sleep(1)

                # collect finished jobs
                for job, trial_index in jobs[:]:
                    if job.done() or type(job) in [LocalJob, DebugJob]:
                        result = job.result()
                        print(f"\n[JOB RESULT]: {result}")
                        print("-" * width)

                        client.complete_trial(trial_index=trial_index, raw_data=result)
                        results.append(result)
                        jobs.remove((job, trial_index))

                        try:
                            best_params, best_metrics, best_trial_index, _ = client.get_best_parameterization()
                            print(f"\n>>> BEST SO FAR (Trial {best_trial_index}) <<<")
                            print(f"Best Mean Reward: {best_metrics}")
                            print(f"Best Parameters:  {best_params}")
                            print("-" * width)
                        except Exception as e:
                            print(f"[INFO] Could not determine best parameters yet: {e}")

            best = client.get_best_parameterization()
            return {"best": best, "results": results}

        outcome = run_parallel_exps()
        print(f"\n[RESULT] {outcome['best'][0]} results in {outcome['best'][1]} observed on trial {outcome['best'][2]}")
        print("-" * width)


optimise()
