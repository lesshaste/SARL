# To be run on a SLURM login node (or locally in debug mode).
#
# Single-run version: runs ONE fixed parameter set (no Ax / no optimisation).

# %% Setup
import os
import time
from itertools import product
import warnings
from pathlib import Path
import csv
from typing import Optional, List, Dict

import numpy as np
from submitit import AutoExecutor
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra

from sarl.train import main

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
LOCAL_DEBUG_MODE = True  # set False on slurm
SUBMITIT_DIR = "submitit"
HYDRA_CONFIG_PATH = "../../config"

CPU_CORES_PER_TASK = 4

# --- training settings ---
TRAIN_EPISODES = 1_000_000
CYCLES = 4
LEARNING_STEPS = 40000

# --- on-policy ---
ON_POLICY_PARAMS = {"n_steps": 100}

# --- misc ---
SEEDS = [42]
ENVS = ["platform"]

DISCRETE_ALGS = ["ppo"]
CONTINUOUS_ALGS = ["ppo"]

# ✅ ONE parameter set (edit these values)
FIXED_PARAMS = {
    "discrete_learning_rate": 1.0e-2,
    "continuous_learning_rate": 4.0e-4,
    "update_ratio": 0.05,
}

cluster = "debug" if LOCAL_DEBUG_MODE else "slurm"

try:
    width = os.get_terminal_size().columns
except OSError:
    width = 120

pairs = [f"{alg1}-{alg2}" for alg1, alg2 in product(DISCRETE_ALGS, CONTINUOUS_ALGS)]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_eval_csv_rewards(eval_csv: Path) -> List[float]:
    """Read rewards from converter-produced eval.csv (best-effort)."""
    if not eval_csv.exists():
        return []

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

    return rewards


def _safe_float(x) -> float:
    """Convert output to a finite float."""
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


def _run_training(pair: str, env: str, params: Dict[str, float], run_id: str) -> Dict[str, float]:
    print(f"[RUN {run_id}] pair={pair} env={env} params={params}")

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

            # learning rates + update ratio
            f"+parameters.alg_params.discrete.learning_rate={params['discrete_learning_rate']}",
            f"+parameters.alg_params.continuous.learning_rate={params['continuous_learning_rate']}",
            f"parameters.alg_params.update_ratio={params['update_ratio']}",
        ]

        # on-policy settings
        if alg_discrete.lower() in {"ppo", "a2c"}:
            overrides.append(f"+parameters.alg_params.discrete.n_steps={ON_POLICY_PARAMS['n_steps']}")
        if alg_continuous.lower() in {"ppo", "a2c"}:
            overrides.append(f"+parameters.alg_params.continuous.n_steps={ON_POLICY_PARAMS['n_steps']}")

        # output directory (avoid overwrite)
        overrides.append(f"hydra.run.dir=outputs/manual/{pair}/{env}/{run_id}")

        cfg = compose(config_name="sarl", return_hydra_config=True, overrides=overrides)
        HydraConfig.instance().set_config(cfg)

        # Ensure output directory exists
        try:
            Path(HydraConfig.get().runtime.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        mean_reward_from_main = main(cfg)
        mean_reward = mean_reward_from_main

        # Prefer eval.csv if present
        output_dir: Optional[Path]
        try:
            output_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            output_dir = None

        if output_dir is not None:
            rewards = _read_eval_csv_rewards(output_dir / "eval.csv")
            if rewards:
                last_r = rewards[-1]
                max_r = max(rewards)
                print(f"[METRIC] eval.csv last={last_r:.6f} max={max_r:.6f} (output_dir={output_dir})")
                mean_reward = last_r

    return {"mean_reward": _safe_float(mean_reward)}


# -----------------------------------------------------------------------------
# Single-run driver
# -----------------------------------------------------------------------------
def run_once():
    # unique run id so repeated runs don't overwrite outputs
    run_id = time.strftime("run_%Y%m%d_%H%M%S")

    for pair, env in product(pairs, ENVS):
        if LOCAL_DEBUG_MODE:
            # run directly (fast iteration locally)
            result = _run_training(pair, env, FIXED_PARAMS, run_id)
        else:
            # run through submitit on slurm
            executor = AutoExecutor(folder=SUBMITIT_DIR, cluster=cluster)
            executor.update_parameters(timeout_min=60)
            executor.update_parameters(cpus_per_task=CPU_CORES_PER_TASK)

            job = executor.submit(_run_training, pair, env, FIXED_PARAMS, run_id)
            result = job.result()

        print(f"\n[RESULT] pair={pair} env={env} params={FIXED_PARAMS} -> {result}")
        print("-" * width)


if __name__ == "__main__":
    run_once()
