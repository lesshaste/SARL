#!/usr/bin/env python3
"""
Ax + Submitit driver for SARL hyperparameter optimization.

Local (default, no args):
    python sarl/scripts/ax/ax_driver_slurm.py

Slurm controller job (recommended):
    poetry run python sarl/scripts/ax/ax_driver.py --slurm --pair a2c-ppo --parallel-limit 8

Notes:
- This script runs an Ax optimization *controller* process.
- Each trial (training run) is launched via submitit as a separate job (local debug) or Slurm job.
- To safely run trials in parallel, each trial uses a unique Hydra output directory:
    outputs/ax/<run_id>/<pair>/<env>/trial_<trial_index>

Defaults (when run locally with no args) match your requested original settings.
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from submitit import AutoExecutor, DebugJob, LocalJob
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from sarl.train import main


# -----------------------------------------------------------------------------
# Defaults (requested)
# -----------------------------------------------------------------------------
DEFAULT_LOCAL_DEBUG_MODE = True  # set False on slurm
DEFAULT_SUBMITIT_DIR = "submitit"
DEFAULT_HYDRA_CONFIG_PATH = "../../config"

DEFAULT_CPU_CORES_PER_TASK = 4

DEFAULT_MAX_TRIALS = 80
DEFAULT_PARALLEL_LIMIT = 2

DEFAULT_TRAIN_EPISODES = 1_000_000
DEFAULT_CYCLES = 10
DEFAULT_LEARNING_STEPS = 10000 * DEFAULT_CYCLES

DEFAULT_ON_POLICY_PARAMS = {"n_steps": 100}

DEFAULT_BOUNDS_LR = (1e-6, 1e-3)
DEFAULT_BOUNDS_UPDATE_RATIO = (0.01, 0.99)

DEFAULT_SEEDS = [42]
DEFAULT_ENVS = ["platform"]

DEFAULT_DISCRETE_ALGS = ["a2c"]
DEFAULT_CONTINUOUS_ALGS = ["ppo"]


ALL_ALGS = ["a2c", "dqn", "ppo", "ddpg", "sac", "td3"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _run_id() -> str:
    # Make a run id that is stable-ish and unique enough across local + slurm.
    slurm_job = os.environ.get("SLURM_JOB_ID")
    ts = time.strftime("%Y%m%d_%H%M%S")
    if slurm_job:
        return f"slurm{slurm_job}_{ts}"
    return f"local_{ts}"


def _read_eval_csv(eval_csv: Path) -> Optional[List[Tuple[float, float]]]:
    """
    Read converter.py eval.csv:
        "training_timesteps","mean_eval_episode_return"
    Returns list[(training_timesteps, mean_return)] or None.
    """
    if not eval_csv.exists():
        return None
    rows: List[Tuple[float, float]] = []
    with eval_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if "training_timesteps" not in row or "mean_eval_episode_return" not in row:
                continue
            try:
                t = float(row["training_timesteps"])
                r = float(row["mean_eval_episode_return"])
            except Exception:
                continue
            rows.append((t, r))
    return rows or None


def _metric_from_eval_csv(
    eval_rows: List[Tuple[float, float]],
    mode: str = "last",
    last_k: int = 5,
) -> float:
    rewards = [r for _, r in eval_rows]
    if not rewards:
        return 0.0
    if mode == "last":
        return float(rewards[-1])
    if mode == "max":
        return float(max(rewards))
    if mode in ("mean_last_k", "last_k_mean"):
        k = max(1, min(int(last_k), len(rewards)))
        return float(np.mean(rewards[-k:]))
    raise ValueError(f"Unknown metric mode: {mode!r}")


def _coerce_seed_results(x: Union[float, int, Dict[int, float]]) -> List[float]:
    """
    train.main(cfg) can return:
      - float/int objective
      - dict[seed] -> float objective
    Convert to a list of floats.
    """
    if isinstance(x, dict):
        vals = []
        for v in x.values():
            try:
                vals.append(float(v))
            except Exception:
                pass
        return vals
    try:
        return [float(x)]
    except Exception:
        return []


def _mean_and_sem(values: List[float]) -> Tuple[float, float]:
    """
    Return (mean, SEM). If only 1 value, SEM=0.0.
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), 0.0)
    m = mean(values)
    # Use population stdev for stability; SEM is heuristic here anyway.
    sd = pstdev(values)
    sem = sd / (len(values) ** 0.5)
    return (float(m), float(sem))


def configure_objective_maximize(client: Client, metric_name: str) -> None:
    """
    Your Client exposes configure_optimization. Use the most common signature:
        client.configure_optimization(objective="mean_reward")
    """
    try:
        client.configure_optimization(objective=metric_name)
    except TypeError:
        # Some versions accept objective_name
        client.configure_optimization(objective_name=metric_name)


def get_params_by_alg(label: str, bounds_lr: Tuple[float, float]) -> Dict[str, List[RangeParameterConfig]]:
    shared = [
        RangeParameterConfig(
            name=f"{label}_learning_rate",
            bounds=(bounds_lr[0], bounds_lr[1]),
            parameter_type="float",
            scaling="log",  # learning rates should be searched on log scale
        )
    ]
    return {alg: shared for alg in ALL_ALGS}


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
@dataclass
class Settings:
    # Requested defaults:
    local_debug_mode: bool = DEFAULT_LOCAL_DEBUG_MODE
    submitit_dir: str = DEFAULT_SUBMITIT_DIR
    hydra_config_path: str = DEFAULT_HYDRA_CONFIG_PATH

    cpu_cores_per_task: int = DEFAULT_CPU_CORES_PER_TASK

    max_trials: int = DEFAULT_MAX_TRIALS
    parallel_limit: int = DEFAULT_PARALLEL_LIMIT

    train_episodes: int = DEFAULT_TRAIN_EPISODES
    cycles: int = DEFAULT_CYCLES
    learning_steps: int = DEFAULT_LEARNING_STEPS

    on_policy_n_steps: int = DEFAULT_ON_POLICY_PARAMS["n_steps"]

    bounds_lr: Tuple[float, float] = DEFAULT_BOUNDS_LR
    bounds_update_ratio: Tuple[float, float] = DEFAULT_BOUNDS_UPDATE_RATIO

    seeds: List[int] = None
    envs: List[str] = None

    discrete_algs: List[str] = None
    continuous_algs: List[str] = None

    # Slurm options for trial jobs:
    slurm_partition: Optional[str] = None
    slurm_mem: Optional[str] = None
    timeout_min: int = 24 * 60  # default 24h for trial jobs

    # Which pair to run (optional)
    pair: Optional[str] = None

    # Metric source / mode
    metric_source: str = "return"  # "return" or "eval_csv"
    metric_mode: str = "last"      # "last" | "max" | "mean_last_k"
    metric_last_k: int = 5

    # Ax init knobs (optional)
    initialization_budget: Optional[int] = None
    min_observed_init_trials: Optional[int] = None


def parse_args() -> Settings:
    p = argparse.ArgumentParser(add_help=True)

    # Execution mode
    p.add_argument("--slurm", action="store_true", help="Force Slurm mode for trial jobs (cluster='slurm').")
    p.add_argument("--force-local", action="store_true", help="Force local debug mode even if SLURM_JOB_ID is set.")

    # What to run
    p.add_argument("--pair", type=str, default=None, help="Run a single pair, e.g. a2c-ppo. If omitted, runs product of alg lists.")
    p.add_argument("--envs", type=str, default=",".join(DEFAULT_ENVS), help="Comma-separated env list.")
    p.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)), help="Comma-separated seed list.")

    p.add_argument("--discrete-algs", type=str, default=",".join(DEFAULT_DISCRETE_ALGS),
                   help="Comma-separated discrete alg list.")
    p.add_argument("--continuous-algs", type=str, default=",".join(DEFAULT_CONTINUOUS_ALGS),
                   help="Comma-separated continuous alg list.")
    p.add_argument("--all-algs", action="store_true", help="Use all algs for both discrete and continuous (6x6 pairs).")

    # Experiment size
    p.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    p.add_argument("--parallel-limit", type=int, default=DEFAULT_PARALLEL_LIMIT)

    # Training budget
    p.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    p.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    p.add_argument("--learning-steps", type=int, default=DEFAULT_LEARNING_STEPS)
    p.add_argument("--on-policy-n-steps", type=int, default=DEFAULT_ON_POLICY_PARAMS["n_steps"])

    # Search space
    p.add_argument("--lr-min", type=float, default=DEFAULT_BOUNDS_LR[0])
    p.add_argument("--lr-max", type=float, default=DEFAULT_BOUNDS_LR[1])
    p.add_argument("--update-ratio-min", type=float, default=DEFAULT_BOUNDS_UPDATE_RATIO[0])
    p.add_argument("--update-ratio-max", type=float, default=DEFAULT_BOUNDS_UPDATE_RATIO[1])

    # Resources for trial jobs
    p.add_argument("--cpus-per-trial", type=int, default=DEFAULT_CPU_CORES_PER_TASK)
    p.add_argument("--timeout-min", type=int, default=24 * 60)
    p.add_argument("--slurm-partition", type=str, default=None)
    p.add_argument("--slurm-mem", type=str, default=None)

    # Paths
    p.add_argument("--submitit-dir", type=str, default=DEFAULT_SUBMITIT_DIR)
    p.add_argument("--hydra-config-path", type=str, default=DEFAULT_HYDRA_CONFIG_PATH)

    # Metric handling
    p.add_argument("--metric-source", type=str, default="return", choices=["return", "eval_csv"],
                   help="Use train.main(cfg) return ('return') or parse eval.csv ('eval_csv').")
    p.add_argument("--metric-mode", type=str, default="last", choices=["last", "max", "mean_last_k"])
    p.add_argument("--metric-last-k", type=int, default=5)

    # Ax init
    p.add_argument("--initialization-budget", type=int, default=None)
    p.add_argument("--min-observed-init-trials", type=int, default=None)

    a = p.parse_args()

    s = Settings()
    s.submitit_dir = a.submitit_dir
    s.hydra_config_path = a.hydra_config_path

    s.max_trials = a.max_trials
    s.parallel_limit = a.parallel_limit

    s.train_episodes = a.train_episodes
    s.cycles = a.cycles
    s.learning_steps = a.learning_steps
    s.on_policy_n_steps = a.on_policy_n_steps

    s.bounds_lr = (a.lr_min, a.lr_max)
    s.bounds_update_ratio = (a.update_ratio_min, a.update_ratio_max)

    s.cpu_cores_per_task = a.cpus_per_trial
    s.timeout_min = a.timeout_min
    s.slurm_partition = a.slurm_partition
    s.slurm_mem = a.slurm_mem

    s.metric_source = a.metric_source
    s.metric_mode = a.metric_mode
    s.metric_last_k = a.metric_last_k

    s.initialization_budget = a.initialization_budget
    s.min_observed_init_trials = a.min_observed_init_trials

    s.seeds = _parse_int_list(a.seeds)
    s.envs = _parse_csv_list(a.envs)

    if a.all_algs:
        s.discrete_algs = ALL_ALGS[:]
        s.continuous_algs = ALL_ALGS[:]
    else:
        s.discrete_algs = _parse_csv_list(a.discrete_algs)
        s.continuous_algs = _parse_csv_list(a.continuous_algs)

    s.pair = a.pair

    # Decide local vs slurm:
    slurm_detected = os.environ.get("SLURM_JOB_ID") is not None
    if a.force_local:
        s.local_debug_mode = True
    elif a.slurm or slurm_detected:
        s.local_debug_mode = False
    else:
        s.local_debug_mode = DEFAULT_LOCAL_DEBUG_MODE

    return s


# -----------------------------------------------------------------------------
# Main optimization logic
# -----------------------------------------------------------------------------
def make_executor(settings: Settings, run_id: str, pair_slug: str, env: str) -> AutoExecutor:
    # Use a unique folder so multiple controllers don't collide.
    folder = Path(settings.submitit_dir) / run_id / pair_slug / env
    folder.mkdir(parents=True, exist_ok=True)

    cluster = "debug" if settings.local_debug_mode else "slurm"
    ex = AutoExecutor(folder=str(folder), cluster=cluster)

    # Submitit generic:
    ex.update_parameters(timeout_min=int(settings.timeout_min))
    ex.update_parameters(cpus_per_task=int(settings.cpu_cores_per_task))

    # Slurm-specific knobs for trial jobs:
    if not settings.local_debug_mode:
        if settings.slurm_partition:
            ex.update_parameters(slurm_partition=settings.slurm_partition)
        if settings.slurm_mem:
            ex.update_parameters(slurm_mem=settings.slurm_mem)

    return ex


def make_client(settings: Settings, pair: str) -> Client:
    alg_d, alg_c = pair.split("-")

    params = (
        get_params_by_alg("discrete", settings.bounds_lr)[alg_d]
        + get_params_by_alg("continuous", settings.bounds_lr)[alg_c]
    )

    update_ratio_param = RangeParameterConfig(
        name="update_ratio",
        bounds=(settings.bounds_update_ratio[0], settings.bounds_update_ratio[1]),
        parameter_type="float",
        scaling="linear",
    )
    params = params + [update_ratio_param]

    client = Client()
    client.configure_experiment(name=f"sarl_opt_{pair}", parameters=params)
    configure_objective_maximize(client, "mean_reward")

    # Keep your original "quality" strategy, but allow init knobs if provided.
    if settings.initialization_budget is not None or settings.min_observed_init_trials is not None:
        kwargs = {"method": "quality"}
        if settings.initialization_budget is not None:
            kwargs["initialization_budget"] = int(settings.initialization_budget)
        if settings.min_observed_init_trials is not None:
            kwargs["min_observed_initialization_trials"] = int(settings.min_observed_init_trials)
        client.configure_generation_strategy(**kwargs)
    else:
        client.configure_generation_strategy(method="quality")

    return client


def objective_function(
    params: Dict[str, float],
    trial_index: int,
    *,
    settings: Settings,
    pair: str,
    env: str,
    run_id: str,
) -> Dict[str, Union[float, Tuple[float, float]]]:
    """
    Run one training job with the suggested hyperparameters, returning:
        {"mean_reward": float}  or  {"mean_reward": (mean, sem)}
    """
    pair_slug = pair.replace("-", "_")

    # Make logs readable when parallel:
    print(f"[AX][RUN {run_id}][TRIAL {trial_index}][{pair}][{env}] params={params}", flush=True)

    GlobalHydra.instance().clear()
    with initialize(config_path=settings.hydra_config_path, job_name=f"ax_{pair_slug}_{env}_{trial_index}"):

        overrides: List[str] = [
            f"algorithm={pair}",
            f"environment={env}",
            f"parameters.seeds={settings.seeds}",
            f"parameters.train_episodes={settings.train_episodes}",
            f"parameters.learning_steps={settings.learning_steps}",
            f"parameters.cycles={settings.cycles}",

            # Unique output dir per trial (prevents collisions, esp. under parallelism)
            f"hydra.run.dir=outputs/ax/{run_id}/{pair_slug}/{env}/trial_{trial_index}",

            # Hyperparameters (use ++ so it works whether key exists or not)
            f"++parameters.alg_params.discrete.learning_rate={params['discrete_learning_rate']}",
            f"++parameters.alg_params.continuous.learning_rate={params['continuous_learning_rate']}",
            f"++parameters.alg_params.update_ratio={params['update_ratio']}",
        ]

        # On-policy n_steps (A2C/PPO)
        alg_d, alg_c = pair.split("-")
        if alg_d.lower() in {"ppo", "a2c"}:
            overrides.append(f"++parameters.alg_params.discrete.n_steps={settings.on_policy_n_steps}")
        if alg_c.lower() in {"ppo", "a2c"}:
            overrides.append(f"++parameters.alg_params.continuous.n_steps={settings.on_policy_n_steps}")

        cfg = compose(config_name="sarl", return_hydra_config=True, overrides=overrides)
        HydraConfig.instance().set_config(cfg)

        # Ensure output dir exists
        try:
            Path(HydraConfig.get().runtime.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        result_from_main = main(cfg)

        # Metric extraction
        seed_values = _coerce_seed_results(result_from_main)

        if settings.metric_source == "eval_csv":
            # Use eval.csv if present (primarily for debugging).
            # Note: this reads the single trial output dir, which is safe because hydra.run.dir is unique.
            out_dir = Path(HydraConfig.get().runtime.output_dir)
            rows = _read_eval_csv(out_dir / "eval.csv")
            if rows:
                metric = _metric_from_eval_csv(rows, mode=settings.metric_mode, last_k=settings.metric_last_k)
                seed_values = [metric]  # treat as single measurement
                print(f"[AX][TRIAL {trial_index}] metric from eval.csv ({settings.metric_mode}) = {metric}", flush=True)
            else:
                print(f"[AX][TRIAL {trial_index}] eval.csv missing; falling back to return value.", flush=True)

        if not seed_values:
            # If nothing usable, return a very poor score (but not extreme) so Ax can proceed.
            return {"mean_reward": -1e6}

        m, sem = _mean_and_sem(seed_values)

        # If multiple seeds, return (mean, SEM) to help Ax model noise.
        if len(seed_values) > 1:
            return {"mean_reward": (m, sem)}
        return {"mean_reward": m}


def run_one_pair(settings: Settings, pair: str, env: str, run_id: str) -> None:
    pair_slug = pair.replace("-", "_")
    client = make_client(settings, pair)
    executor = make_executor(settings, run_id, pair_slug, env)

    jobs: List[Tuple[object, int]] = []
    submitted = 0

    print(
        f"\n=== RUN {run_id} | pair={pair} env={env} | max_trials={settings.max_trials} "
        f"parallel_limit={settings.parallel_limit} | mode={'LOCAL' if settings.local_debug_mode else 'SLURM'} ===\n",
        flush=True,
    )

    while submitted < settings.max_trials or jobs:
        # Launch more trials if we have capacity.
        n = min(settings.parallel_limit - len(jobs), settings.max_trials - submitted)
        if n > 0:
            trial_index_to_param = client.get_next_trials(n)
            for trial_index, parameters in trial_index_to_param.items():
                job = executor.submit(
                    objective_function,
                    parameters,
                    trial_index,
                    settings=settings,
                    pair=pair,
                    env=env,
                    run_id=run_id,
                )
                jobs.append((job, trial_index))
                submitted += 1
                time.sleep(0.2)

        # Collect completed jobs.
        for job, trial_index in jobs[:]:
            if job.done() or isinstance(job, (LocalJob, DebugJob)):
                try:
                    raw = job.result()
                except Exception as e:
                    # Mark trial as failed in a way that doesn't nuke the model with extreme outliers.
                    print(f"[AX][TRIAL {trial_index}] JOB FAILED: {e}", flush=True)
                    # If complete_trial fails with None on your version, you can switch to {"mean_reward": -1e6}.
                    try:
                        client.complete_trial(trial_index=trial_index, raw_data={})
                    except Exception:
                        client.complete_trial(trial_index=trial_index, raw_data={"mean_reward": -1e6})
                    jobs.remove((job, trial_index))
                    continue

                print(f"\n[AX][TRIAL {trial_index}] RESULT: {raw}", flush=True)
                client.complete_trial(trial_index=trial_index, raw_data=raw)
                jobs.remove((job, trial_index))

                try:
                    best_params, best_metrics, best_trial_index, _ = client.get_best_parameterization()
                    print("\n>>> BEST SO FAR <<<", flush=True)
                    print(f"Trial:  {best_trial_index}", flush=True)
                    print(f"Metric: {best_metrics}", flush=True)
                    print(f"Params:  {best_params}", flush=True)
                    print("-" * 80, flush=True)
                except Exception as e:
                    print(f"[AX] Best point not available yet: {e}", flush=True)

        time.sleep(0.2)

    best_params, best_metrics, best_trial_index, _ = client.get_best_parameterization()
    print(f"\n[AX][DONE] {pair} {env} best_trial={best_trial_index} best_metrics={best_metrics} best_params={best_params}\n", flush=True)


def main_driver() -> None:
    settings = parse_args()
    run_id = _run_id()

    # Determine which pairs to run
    if settings.pair:
        pairs = [settings.pair]
    else:
        pairs = [f"{d}-{c}" for d, c in product(settings.discrete_algs, settings.continuous_algs)]

    # Basic validation
    for pair in pairs:
        if "-" not in pair:
            raise ValueError(f"Invalid --pair {pair!r}. Expected format like 'a2c-ppo'.")
        d, c = pair.split("-")
        if d not in ALL_ALGS or c not in ALL_ALGS:
            raise ValueError(f"Unknown alg in pair {pair!r}. Allowed: {ALL_ALGS}")

    # Run sequentially over envs/pairs in this controller process.
    # (Parallelize across pairs using Slurm job arrays; see your submit.sh example.)
    for pair, env in product(pairs, settings.envs):
        run_one_pair(settings, pair, env, run_id)


if __name__ == "__main__":
    main_driver()
