from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ObsType
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList


# --------------------------------------------------------------------------------------
# Hybrid policy: coordinates a discrete-policy and a continuous-parameters policy
# on the converted MDP (PamdpToMdp).
# --------------------------------------------------------------------------------------

class HybridPolicy:
    """
    Combines two policies (or SB3 agents) to act on the converted MDP.

    Convention:
      - If obs[0] == -1 => environment is expecting the DISCRETE action id
      - else            => environment is expecting the CONTINUOUS parameters
    """

    def __init__(
        self,
        discretePolicy: Optional[Callable[[Any], Any]] = None,
        continuousPolicy: Optional[Callable[[Any], Any]] = None,
        discreteAgent: Optional[BaseAlgorithm] = None,
        continuousAgent: Optional[BaseAlgorithm] = None,
        name: Optional[str] = None,
        env_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.agent = {key: None for key in ["discrete", "continuous"]}
        self.name = name
        self.timestep = 0
        self.cycle = 0
        self.env_name = env_name
        self.seed = seed

        if discretePolicy is not None:
            self.discretePolicy = discretePolicy
        elif discreteAgent is not None:
            self.agent["discrete"] = discreteAgent
            self.discretePolicy = discreteAgent.predict
        else:
            raise ValueError("Provide either discretePolicy or discreteAgent")

        if continuousPolicy is not None:
            self.continuousPolicy = continuousPolicy
        elif continuousAgent is not None:
            self.agent["continuous"] = continuousAgent
            self.continuousPolicy = continuousAgent.predict
        else:
            raise ValueError("Provide either continuousPolicy or continuousAgent")

    def _call_policy(self, policy: Callable, obs_inner):
        """
        Handles SB3 predict() and plain callables uniformly.
        SB3 predict returns (action, state). Custom callables return action.
        """
        try:
            out = policy(obs_inner)
        except TypeError:
            # If someone passes SB3 policy expecting (obs, state, episode_start, deterministic)
            out = policy(obs_inner, deterministic=False)
        if isinstance(out, tuple):
            return out[0]
        return out

    def predict(self, obs):
        # obs is (indicator, original_obs)
        if obs[0] == -1:
            action = self._call_policy(self.discretePolicy, obs[1])
            # SB3 often returns np.array([...]) for discrete
            if isinstance(action, (np.ndarray, list)):
                action = int(np.asarray(action).squeeze())
            else:
                action = int(action)
            return action

        # continuous parameters
        assert obs[0] > -1
        action = self._call_policy(self.continuousPolicy, obs[1])
        return action

    def _evaluate(self, eval_mdp, evaluation_returns, cycle, eval_episodes, log_dir):
        returns = []
        base_seed = 0 if self.seed is None else int(self.seed)
        for i in range(eval_episodes):
            obs, info = eval_mdp.reset(seed=base_seed + cycle + i)
            done = False
            last_info = info
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, last_info = eval_mdp.step(action)
                done = bool(terminated or truncated)
            # RecordEpisodeStatistics puts episode return in info["episode"]["r"] on terminal step
            ep_ret = last_info.get("episode", {}).get("r", None)
            if ep_ret is None:
                # Fallback: if wrapper not present, accumulate would be needed. Here just use 0.
                ep_ret = 0.0
            returns.append(float(ep_ret))

        mean_return = (float(self.timestep), float(np.mean(returns)))
        evaluation_returns.append(mean_return)

        if log_dir is not None:
            # Make sure the output directory exists before writing.
            from pathlib import Path

            log_path = Path(log_dir)
            file_path = log_path if log_path.suffix else (log_path / "eval.csv")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[REWARD]: Mean reward = {mean_return[1]}")
            print(f"[OUTPUT]: Writing to {file_path}")
            np.savetxt(
                fname=str(file_path),
                X=np.array(evaluation_returns, dtype=np.float64),
                header='"training_timesteps","mean_eval_episode_return"',
                delimiter=",",
                fmt="%1.6f",
            )
        else:
            print(f"[REWARD]: Mean reward = {mean_return[1]}")

        return evaluation_returns

    def learn(
        self,
        total_timesteps: int,
        evaluation_interval: Optional[int] = None,
        eval_mdp=None,
        cycles: int = 1,
        callback=None,
        log_interval: int = 1,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
        eval_episodes: int = 15,
        log_dir: Optional[str] = None,
        rollout_length: Optional[int] = None,
        update_ratio: float = 0.5,
        objective_aggregation: str = "last",  # 'last' | 'max' | 'mean_last_k'
        objective_last_k: int = 3,
    ):
        """
        Alternate training between discrete and continuous agents.

        total_timesteps is in *converted MDP* timesteps (i.e., partial steps).

        This version uses Option A budget allocation:
          - Allocate a fixed per-cycle budget: per_cycle = total_timesteps / cycles
          - Split per_cycle between discrete/continuous according to update_ratio
          - Ensures discrete_steps + continuous_steps == per_cycle (so total adds up cleanly)
        """
        if cycles < 1:
            raise ValueError("cycles must be >= 1")

        # Keep strict divisibility to make accounting clean/predictable
        if total_timesteps % cycles != 0:
            raise ValueError(f"total_timesteps ({total_timesteps}) must be divisible by cycles ({cycles})")

        # Clamp ratio to [0, 1]
        update_ratio = float(update_ratio)
        if update_ratio < 0.0:
            update_ratio = 0.0
        if update_ratio > 1.0:
            update_ratio = 1.0

        # Attach bookkeeping used by DataCallback and elsewhere
        self.timestep = 0
        for agent_type in self.agent.keys():
            agent = self.agent[agent_type]
            if agent is None:
                continue
            agent.agent_type = agent_type
            agent.parent = self

        evaluation_returns = []

        # Fixed per-cycle budget
        per_cycle = int(total_timesteps // cycles)

        # If requested, align per-cycle budget to rollout length (e.g. PPO n_steps) to avoid 0 updates.
        # This will reduce total trained timesteps slightly unless you also pre-align total_timesteps.
        if rollout_length:
            rollout_length = int(rollout_length)
            if rollout_length > 0:
                per_cycle = (per_cycle // rollout_length) * rollout_length
            else:
                rollout_length = None

        if per_cycle <= 0:
            raise ValueError(
                f"per_cycle budget became {per_cycle}. Increase total_timesteps or reduce cycles/rollout_length."
            )

        for cycle in range(cycles):
            self.cycle = cycle

            # Split per-cycle budget across agents
            discrete_steps = int(update_ratio * per_cycle)
            continuous_steps = int(per_cycle - discrete_steps)  # ensures sum == per_cycle

            for agent_type, ratioed_timesteps in (("discrete", discrete_steps), ("continuous", continuous_steps)):
                agent = self.agent.get(agent_type)
                if agent is None:
                    continue

                if ratioed_timesteps <= 0:
                    print(
                        f"[{self.name}][Seed {getattr(agent, 'seed', None)}]"
                        f"[Timestep {self.timestep}/{total_timesteps}]"
                        f"[Cycle {cycle+1}/{cycles}][{agent_type}]"
                        f": Skipping learn() (0 timesteps)."
                    )
                    continue

                print(
                    f"[{self.name}][Seed {getattr(agent, 'seed', None)}]"
                    f"[Timestep {self.timestep}/{total_timesteps}]"
                    f"[Cycle {cycle+1}/{cycles}][{agent_type}]"
                    f": Learning for {ratioed_timesteps}+ timesteps..."
                )

                if not isinstance(agent, BaseAlgorithm):
                    raise NotImplementedError("Only SB3 BaseAlgorithm agents are supported here.")

                # Avoid wrapping the same callback repeatedly
                cb = callback
                if cb is not None and not isinstance(cb, CallbackList):
                    cb = CallbackList([cb])

                agent.learn(
                    ratioed_timesteps,
                    callback=cb,
                    log_interval=log_interval,
                    tb_log_name=f"{tb_log_name}_{agent_type}",
                    reset_num_timesteps=reset_num_timesteps,
                    progress_bar=progress_bar,
                )

                # Advance bookkeeping counter so logs/eval reflect actual trained timesteps
                self.timestep += int(ratioed_timesteps)

            # Evaluate at end of cycle
            if evaluation_interval is not None and eval_mdp is not None:
                if (cycle + 1) % int(evaluation_interval) == 0:
                    evaluation_returns = self._evaluate(
                        eval_mdp, evaluation_returns, cycle, eval_episodes, log_dir
                    )

        if len(evaluation_returns) == 0:
            return 0.0

        rewards = [ret[1] for ret in evaluation_returns]
        if objective_aggregation == "last":
            objective = rewards[-1]
        elif objective_aggregation == "max":
            objective = max(rewards)
        elif objective_aggregation in ("mean_last_k", "last_k_mean"):
            k = max(1, min(int(objective_last_k), len(rewards)))
            objective = float(np.mean(rewards[-k:]))
        else:
            raise ValueError(
                f"Unknown objective_aggregation={objective_aggregation!r}. "
                "Use 'last', 'max', or 'mean_last_k'."
            )

        print(
            f"[REWARD][OBJECTIVE]: aggregation={objective_aggregation} value={objective} "
            f"(evaluations={len(rewards)})"
        )
        return float(objective)

# --------------------------------------------------------------------------------------
# View wrappers: expose only discrete OR only continuous actions while an internal policy
# supplies the other part.
# --------------------------------------------------------------------------------------

class PamdpToMdpView(Env):
    def __init__(
        self,
        parent: Env,
        action_space_is_discrete: bool,
        internal_policy: Optional[Callable[[Any], Any]] = None,
        combine_continuous_actions: bool = False,
    ) -> None:
        """
        Provide an MDP that only accepts either discrete actions or continuous parameters,
        while an internal policy supplies the other component.
        """
        super().__init__()
        self.parent = parent
        self.combine_continuous_actions = bool(combine_continuous_actions)
        self.action_space_is_discrete = bool(action_space_is_discrete)

        # Expose only the original env observation (not the indicator)
        self.observation_space = parent.observation_space[1]
        self.reward_range = parent.reward_range
        self.spec = parent.spec
        self.metadata = parent.metadata
        self.np_random = parent.np_random

        if self.action_space_is_discrete:
            self.action_space = parent.discrete_action_space
        else:
            self.action_space = parent.action_parameter_space
            if self.combine_continuous_actions:
                self.action_parameter_indices_mapping = self.parent.action_parameter_indices_mapping
                self.action_space = self.parent.combine(self.action_space)

        # Default internal policy is random over the *other* component
        if internal_policy is None:
            if self.action_space_is_discrete:
                self.internal_policy = lambda _obs: parent.action_parameter_space.sample()
            else:
                self.internal_policy = lambda _obs: parent.discrete_action_space.sample()
        else:
            self.internal_policy = internal_policy

    def step(self, action):
        # Always return original obs (no indicator) to the learner
        if self.action_space_is_discrete:
            # agent chooses discrete, internal chooses continuous
            obs, reward, terminated, truncated, info = self.parent.step(action)
            view_obs = obs[1]
            obs, reward, terminated, truncated, info = self.parent.step(self.internal_policy(view_obs))
            view_obs = obs[1]
            return view_obs, reward, terminated, truncated, info

        # agent chooses continuous, internal chooses discrete
        view_obs = self.parent.previous_step_output["obs"][1]
        obs, _r0, term0, trunc0, _info0 = self.parent.step(self.internal_policy(view_obs))
        # discrete half-step should not terminate, but be defensive:
        if term0 or trunc0:
            return obs[1], 0.0, term0, trunc0, _info0

        if self.combine_continuous_actions:
            action = self.parent.uncombineAction(action)

        obs, reward, terminated, truncated, info = self.parent.step(action)
        return obs[1], reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.parent.reset(seed=seed, options=options)
        # Return the *original* env observation for both views
        return obs[1], info

    def render(self):
        return self.parent.render()

    def close(self):
        return self.parent.close()


# --------------------------------------------------------------------------------------
# Core converter: turns PAMDP step (discrete + parameters) into a 2-step MDP.
# --------------------------------------------------------------------------------------

STEP_KEYS = ["obs", "reward", "terminated", "truncated", "info"]


class PamdpToMdp(Wrapper):
    """
    Converts a parameterized action MDP (PAMDP) into a 2-step MDP:

      Step A (discrete): choose action id (reward=0, no transition)
      Step B (continuous): choose parameters (executes real env step)

    Observation is a tuple: (indicator, original_obs)
      - indicator == -1  => expecting discrete action
      - indicator >= 0   => expecting continuous parameters for that action id
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.discrete_action_space = self.action_space[0]
        self.action_parameter_space = self.action_space[1]

        self.action_parameter_indices_mapping = self._getParamIndices()

        original_observation_space = self.observation_space
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(2),  # (kept for legacy; indicator stored in obs[0] anyway)
                original_observation_space,
            )
        )

        # IMPORTANT: Initialise with non-terminal defaults (prevents "done" leaking across resets)
        self.previous_step_output = {
            "obs": (-1, None),
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
        }
        self.discrete_action_choice = None

    def getComponentMdp(
        self,
        action_space_is_discrete: bool,
        internal_policy=None,
        combine_continuous_actions: bool = False,
    ) -> Env:
        return PamdpToMdpView(
            self,
            action_space_is_discrete=action_space_is_discrete,
            internal_policy=internal_policy,
            combine_continuous_actions=combine_continuous_actions,
        )

    def expectingDiscreteAction(self) -> bool:
        assert -1 not in self.discrete_action_space
        return self.previous_step_output["obs"][0] == -1

    def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        converted_obs = (-1, obs)

        # FULL reset of cached step output (critical)
        self.discrete_action_choice = None
        self.previous_step_output = {
            "obs": converted_obs,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
        }
        return converted_obs, info

    def step(self, partial_action):
        # --- Step A: discrete choice (no env transition, reward=0, never terminal) ---
        if self.expectingDiscreteAction():
            assert partial_action in self.discrete_action_space
            self.discrete_action_choice = int(partial_action)

            obs = (self.discrete_action_choice, self.previous_step_output["obs"][1])
            reward = 0.0
            terminated, truncated = False, False
            info = {}

            step_output = (obs, reward, terminated, truncated, info)
            self.previous_step_output = dict(zip(STEP_KEYS, step_output))
            return step_output

        # --- Step B: continuous parameters (executes the real env step) ---
        # Convert combined continuous vector -> tuple-of-arrays if needed
        params = partial_action

        if not self.action_parameter_space.contains(params):
            # Likely received a combined vector (e.g. PPO on a single Box)
            indices = self.action_parameter_indices_mapping
            # Build tuple-of-arrays (one per discrete action), clip/cast to each Box bounds
            out = []
            for a in range(len(self.action_parameter_space)):
                box = self.action_parameter_space[a]
                arr = np.asarray(partial_action[indices[a]], dtype=box.dtype)
                arr = np.clip(arr, box.low, box.high)
                out.append(arr)
            params = tuple(out)

        # Be tolerant to dtype/bounds noise
        if isinstance(params, tuple):
            fixed = []
            for a in range(len(self.action_parameter_space)):
                box = self.action_parameter_space[a]
                arr = np.asarray(params[a], dtype=box.dtype)
                arr = np.clip(arr, box.low, box.high)
                fixed.append(arr)
            params = tuple(fixed)

        action = (np.int64(self.discrete_action_choice), params)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # After real step, go back to expecting discrete next
        converted_obs = (-1, obs)

        # reset discrete choice (optional, but safer)
        self.discrete_action_choice = None

        if info is None:
            info = {}

        step_output = (converted_obs, reward, terminated, truncated, info)
        self.previous_step_output = dict(zip(STEP_KEYS, step_output))
        return step_output

    def uncombineAction(self, action):
        """Partition a combined action vector into the Tuple(Box, Box, ...) expected by the PAMDP."""
        output = []
        for i, box in enumerate(self.action_parameter_space.spaces):
            arr = np.asarray(action[self.action_parameter_indices_mapping[i]], dtype=box.dtype)
            arr = np.clip(arr, box.low, box.high)
            output.append(arr)
        return tuple(output)

    def combine(self, space):
        """Combine Tuple(Box, Box, ...) into a single Box for SB3 compatibility."""
        return spaces.Box(
            low=np.array(self.param_lows, dtype=np.float32),
            high=np.array(self.param_highs, dtype=np.float32),
            dtype=np.float32,
        )

    def _getParamIndices(self):
        """
        Build mapping from each action's parameter Box indices into a single concatenated vector.
        """
        indices = {a: [] for a in range(self.discrete_action_space.n)}
        space = self.action_parameter_space

        self.param_lows = []
        self.param_highs = []

        for action in range(len(space)):
            box = space[action]
            for j in range(int(box.shape[0])):
                indices[action].append(len(self.param_highs))
                self.param_highs.append(float(box.high[j]))
                self.param_lows.append(float(box.low[j]))
            indices[action] = np.array(indices[action], dtype=np.int64)

        return indices
