# sarl/snippets/converter_use.py
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from sarl.environments.wrappers.converter import HybridPolicy, PamdpToMdp

try:
    from sarl.agents.callbacks.data_callback import DataCallback
except Exception:  # pragma: no cover
    DataCallback = None  # type: ignore


ALG_CLS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3,
}

DISCRETE_ALGS = {"PPO", "A2C", "DQN"}
CONTINUOUS_ALGS = {"PPO", "A2C", "DDPG", "SAC", "TD3"}


def _to_int_list(xs: Any) -> List[int]:
    if xs is None:
        return []
    if isinstance(xs, (list, tuple)):
        return [int(x) for x in xs]
    try:
        return [int(x) for x in list(xs)]
    except Exception:
        return [int(xs)]


def _register_env(env_name: str) -> None:
    if env_name.lower().startswith("platform"):
        import sarl.common.bester.environments.gym_platform  # noqa: F401
    elif env_name.lower().startswith("goal"):
        import sarl.common.bester.environments.gym_goal  # noqa: F401


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


# --------------------------------------------------------------------------------------
# Observation wrapper: force Box(float32) observations for SB3
# (handles Tuple(Box, Discrete(n)) and "env returns only Box" mismatches)
# --------------------------------------------------------------------------------------

class ObsToBoxWrapper(gym.ObservationWrapper):
    """
    Converts observations to a flat float32 vector with a Box observation_space.

    Supports:
      - Box -> flattened float32
      - Tuple(Box, Discrete(n)) -> [flattened_box, normalized_discrete_scalar]
        If env returns only the Box (common bug), we synthesize the discrete part as a normalized timestep.
    """

    def __init__(self, env: gym.Env, max_steps_hint: Optional[int] = None):
        super().__init__(env)

        self._t = 0  # timestep counter (used if discrete component missing)
        self._specs: List[Dict[str, Any]] = []
        self._disc_denoms: List[float] = []

        space = env.observation_space
        if isinstance(space, spaces.Box):
            # simplest case
            low = np.asarray(space.low, dtype=np.float32).ravel()
            high = np.asarray(space.high, dtype=np.float32).ravel()
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self._specs = [{"kind": "box", "space": space}]
            return

        if isinstance(space, spaces.Tuple):
            lows: List[np.ndarray] = []
            highs: List[np.ndarray] = []
            for sub in space.spaces:
                if isinstance(sub, spaces.Box):
                    lows.append(np.asarray(sub.low, dtype=np.float32).ravel())
                    highs.append(np.asarray(sub.high, dtype=np.float32).ravel())
                    self._specs.append({"kind": "box", "space": sub})
                elif isinstance(sub, spaces.Discrete):
                    # represent Discrete(n) as a single normalized scalar in [0, 1]
                    lows.append(np.array([0.0], dtype=np.float32))
                    highs.append(np.array([1.0], dtype=np.float32))
                    denom = float(max(sub.n - 1, 1))
                    self._disc_denoms.append(denom)
                    self._specs.append({"kind": "discrete", "space": sub})
                else:
                    raise NotImplementedError(
                        f"ObsToBoxWrapper only supports Box/Discrete inside Tuple, got {sub}"
                    )

            low = np.concatenate(lows).astype(np.float32)
            high = np.concatenate(highs).astype(np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

            # optional: if Tuple contains a Discrete(n) and max_steps_hint given,
            # use it as the scaling reference when synthesizing discrete.
            self._max_steps_hint = int(max_steps_hint) if max_steps_hint else None
            return

        raise NotImplementedError(f"ObsToBoxWrapper does not support observation space: {space}")

    def reset(self, **kwargs):
        self._t = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._t += 1
        return obs, reward, terminated, truncated, info

    def observation(self, obs):
        # If env already returns tuple matching its Tuple space, great.
        # If env returns only the Box while space is Tuple(Box, Discrete),
        # we synthesize the discrete component from timestep.
        parts: List[np.ndarray] = []

        if isinstance(self.env.observation_space, spaces.Box):
            arr = np.asarray(obs, dtype=np.float32).ravel()
            return arr

        # Tuple space case
        if isinstance(obs, tuple):
            obs_items = list(obs)
        else:
            # env returned a non-tuple; treat as first component (usually the Box)
            obs_items = [obs]

        disc_idx = 0
        obs_idx = 0
        for spec in self._specs:
            if spec["kind"] == "box":
                if obs_idx < len(obs_items) and not isinstance(obs_items[obs_idx], (int, np.integer)):
                    arr = np.asarray(obs_items[obs_idx], dtype=np.float32).ravel()
                else:
                    # missing -> zeros
                    sub: spaces.Box = spec["space"]
                    arr = np.zeros(int(np.prod(sub.shape)), dtype=np.float32)
                parts.append(arr)
                obs_idx += 1
            else:
                # Discrete -> normalized scalar
                sub: spaces.Discrete = spec["space"]
                denom = self._disc_denoms[disc_idx]
                disc_idx += 1

                if obs_idx < len(obs_items) and isinstance(obs_items[obs_idx], (int, np.integer)):
                    val = float(int(obs_items[obs_idx]))
                    obs_idx += 1
                else:
                    # synthesize from timestep if missing
                    # assume it behaves like a step counter in [0, n-1]
                    n = int(sub.n)
                    # if max_steps_hint is provided and differs from n, still clamp to n
                    val = float(min(max(self._t, 0), n - 1))

                norm = np.array([val / denom], dtype=np.float32)
                parts.append(norm)

        out = np.concatenate(parts).astype(np.float32)
        return out


# --------------------------------------------------------------------------------------
# SB3 agent creation
# --------------------------------------------------------------------------------------

def _alg_kwargs_for(role: str, alg_params: Dict[str, Any], seed: int, tensorboard_log: Optional[str]) -> Dict[str, Any]:
    """
    alg_params structure (all optional):
      {
        "update_ratio": 0.5,
        "common": {...},
        "discrete": {...},
        "continuous": {...},
      }
    """
    common = dict(alg_params.get("common", {}) or {})
    role_params = dict(alg_params.get(role, {}) or {})

    out: Dict[str, Any] = {"seed": int(seed), "verbose": 0}
    if tensorboard_log:
        out["tensorboard_log"] = tensorboard_log

    out.update(common)
    out.update(role_params)

    out.pop("policy", None)
    out.pop("env", None)
    return out


def _make_agent(alg_name: str, env: gym.Env, role: str, alg_params: Dict[str, Any], seed: int, tensorboard_log: Optional[str]) -> BaseAlgorithm:
    cls = ALG_CLS[alg_name]
    kwargs = _alg_kwargs_for(role=role, alg_params=alg_params, seed=seed, tensorboard_log=tensorboard_log)
    print(f"[{alg_name}][{role}] SB3 kwargs = {kwargs}")
    return cls("MlpPolicy", env, **kwargs)


# --------------------------------------------------------------------------------------
# Env builder
# --------------------------------------------------------------------------------------

def _make_base_env(env_name: str, seed: int, max_steps: int) -> gym.Env:
    _register_env(env_name)
    env = gym.make(env_name)

    if max_steps is not None and int(max_steps) > 0:
        env = TimeLimit(env, max_episode_steps=int(max_steps))

    # Force SB3-compatible Box(float32) observations
    env = ObsToBoxWrapper(env, max_steps_hint=max_steps)

    # For evaluation returns via info["episode"]["r"]
    env = RecordEpisodeStatistics(env)

    env.reset(seed=seed)
    return env


# --------------------------------------------------------------------------------------
# Converter runner
# --------------------------------------------------------------------------------------

def runConverter(
    *,
    discreteAlg: str,
    continuousAlg: str,
    env_name: str,
    max_steps: int,
    learning_steps: int,
    cycles: int,
    seeds: Sequence[int],
    eval_episodes: int,
    use_tensorboard: bool,
    write_csv: bool,
    origin_log_dir: str,
    alg_params: Optional[Dict[str, Any]] = None,
) -> Dict[int, float]:
    alg_params = dict(alg_params or {})

    if discreteAlg not in DISCRETE_ALGS:
        raise ValueError(f"discreteAlg must be one of {sorted(DISCRETE_ALGS)}; got {discreteAlg}")
    if continuousAlg not in CONTINUOUS_ALGS:
        raise ValueError(f"continuousAlg must be one of {sorted(CONTINUOUS_ALGS)}; got {continuousAlg}")

    cycles = int(cycles)
    learning_steps = int(learning_steps)
    if cycles <= 0:
        raise ValueError("cycles must be >= 1")
    if learning_steps <= 0:
        raise ValueError("learning_steps must be >= 1")

    seeds_list = _to_int_list(seeds) or [0]

    # Alignment quantum: ensure total_timesteps divisible by cycles and (if PPO) not too tiny batches.
    # PPO default n_steps is 2048; allow overrides via alg_params.{discrete|continuous}.n_steps
    disc_n_steps = int((alg_params.get("discrete", {}) or {}).get("n_steps", 2048 if discreteAlg == "PPO" else 1) or 1)
    cont_n_steps = int((alg_params.get("continuous", {}) or {}).get("n_steps", 2048 if continuousAlg == "PPO" else 1) or 1)
    rollout_align = max(disc_n_steps, cont_n_steps, 1)

    total_timesteps = int(2 * learning_steps)  # converted MDP steps
    quantum = _lcm(cycles, rollout_align)
    quantum = max(quantum, cycles)

    total_timesteps = (total_timesteps // quantum) * quantum
    if total_timesteps < quantum:
        raise ValueError(f"Budget too small after alignment: total_timesteps={total_timesteps}, quantum={quantum}")

    update_ratio = float(alg_params.get("update_ratio", 0.5))

    results: Dict[int, float] = {}

    for seed in seeds_list:
        run_dir = origin_log_dir if len(seeds_list) == 1 else _ensure_dir(os.path.join(origin_log_dir, f"seed_{seed}"))
        tb_dir = _ensure_dir(os.path.join(run_dir, "tb")) if use_tensorboard else None
        csv_dir = run_dir if write_csv else None

        # Build converted env + views
        base_env = _make_base_env(env_name, seed=seed, max_steps=max_steps)
        mdp = PamdpToMdp(base_env)

        discrete_view = mdp.getComponentMdp(
            action_space_is_discrete=True,
            internal_policy=None,
            combine_continuous_actions=False,
        )
        continuous_view = mdp.getComponentMdp(
            action_space_is_discrete=False,
            internal_policy=None,
            combine_continuous_actions=True,
        )

        # Separate eval env
        eval_env = _make_base_env(env_name, seed=seed + 10_000, max_steps=max_steps)
        eval_mdp = PamdpToMdp(eval_env)

        # Agents
        discrete_agent = _make_agent(discreteAlg, discrete_view, role="discrete", alg_params=alg_params, seed=seed, tensorboard_log=tb_dir)
        continuous_agent = _make_agent(continuousAlg, continuous_view, role="continuous", alg_params=alg_params, seed=seed, tensorboard_log=tb_dir)

        # Wire internal policies so each view can complete a full env step
        def _continuous_policy(obs):
            a, _ = continuous_agent.predict(obs, deterministic=True)
            return a

        def _discrete_policy(obs):
            a, _ = discrete_agent.predict(obs, deterministic=True)
            return int(np.asarray(a).squeeze())

        discrete_view.internal_policy = _continuous_policy
        continuous_view.internal_policy = _discrete_policy

        callbacks: List[BaseCallback] = []
        if DataCallback is not None:
            callbacks.append(DataCallback())
        cb = CallbackList(callbacks) if callbacks else None

        agent = HybridPolicy(
            discreteAgent=discrete_agent,
            continuousAgent=continuous_agent,
            name=f"{discreteAlg}-{continuousAlg}",
            env_name=env_name,
            seed=seed,
        )

        out = agent.learn(
            total_timesteps,
            cycles=cycles,
            callback=cb,
            evaluation_interval=1,
            eval_mdp=eval_mdp,
            eval_episodes=int(eval_episodes),
            log_dir=csv_dir,
            tb_log_name=f"{discreteAlg}-{continuousAlg}",
            update_ratio=update_ratio,
            progress_bar=True,
        )

        results[int(seed)] = float(out)

        try:
            mdp.close()
        except Exception:
            pass
        try:
            eval_mdp.close()
        except Exception:
            pass

    return results


# --------------------------------------------------------------------------------------
# Entrypoints expected by sarl/train.py (auto-generated to avoid import errors)
# --------------------------------------------------------------------------------------

_ENV_MAP = {"platform": "Platform-v0", "goal": "Goal-v0"}
_DISC_MAP = {"ppo": "PPO", "a2c": "A2C", "dqn": "DQN"}
_CONT_MAP = {"ppo": "PPO", "a2c": "A2C", "ddpg": "DDPG", "sac": "SAC", "td3": "TD3"}


def _make_entrypoint(disc_key: str, cont_key: str, env_key: str):
    disc = _DISC_MAP[disc_key]
    cont = _CONT_MAP[cont_key]
    env_name = _ENV_MAP[env_key]

    def _fn(
        *,
        max_steps: int = 200,
        learning_steps: int = 20000,
        cycles: int = 4,
        seeds: Sequence[int] = (1,),
        eval_episodes: int = 120,
        alg_params: Optional[Dict[str, Any]] = None,
        output_dir: str = ".",
        **_,
    ):
        return runConverter(
            discreteAlg=disc,
            continuousAlg=cont,
            env_name=env_name,
            max_steps=max_steps,
            learning_steps=learning_steps,
            cycles=cycles,
            seeds=seeds,
            eval_episodes=eval_episodes,
            use_tensorboard=False,
            write_csv=True,
            origin_log_dir=output_dir,
            alg_params=alg_params,
        )

    _fn.__name__ = f"{disc_key}_{cont_key}_{env_key}"
    return _fn


for _env_key in _ENV_MAP:
    for _disc_key in _DISC_MAP:
        for _cont_key in _CONT_MAP:
            globals()[f"{_disc_key}_{_cont_key}_{_env_key}"] = _make_entrypoint(_disc_key, _cont_key, _env_key)
