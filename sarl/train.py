# python3 sarl/train.py +experiment=pdqn-platform
# python3 sarl/train.py +experiment=pdqn-goal
# python3 sarl/train.py +experiment=pdqn-platform parameters.train_episodes=50000
# python3 sarl/train.py +experiment=ppo-ppo-platform-1h-15seeds parameters.seeds=[1,2]
# python3 sarl/train.py +experiment=pdqn-platform hydra.job.name="pdqn-platform" parameters.train_episodes=2500
#
# python3 sarl/train.py algorithm=ppo-ppo environment=platform hydra.job.name="ppo-ppo-platform" parameters.train_episodes=10000
# python3 sarl/train.py algorithm=ppo-ppo environment=platform parameters.alg_params.update_ratio=0.4 hydra.job.name="ppo-ppo-platform" parameters.train_episodes=10000

# tensorboard --logdir=outputs

from functools import partial
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import warnings
import logging
from stable_baselines3.common.utils import get_device

# For cleaner output, mutes unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")


@hydra.main(version_base=None, config_path="config", config_name="sarl")
def main(job_config: DictConfig):
    # Identify relevant script
    from sarl.snippets.pdqn_use import pdqn_platform, pdqn_goal  # TODO: Refactor to avoid script-per-combination
    from sarl.snippets.qpamdp_use import qpamdp_platform, qpamdp_goal
    from sarl.snippets.converter_use import runConverter  # TODO: Use this via the partial method
    from sarl.snippets.converter_use import ppo_ppo_platform, a2c_ppo_platform, dqn_ppo_platform, ppo_ppo_goal, a2c_ppo_goal, dqn_ppo_goal
    from sarl.snippets.converter_use import ppo_a2c_platform, a2c_a2c_platform, dqn_a2c_platform, ppo_a2c_goal, a2c_a2c_goal, dqn_a2c_goal
    from sarl.snippets.converter_use import ppo_ddpg_platform, a2c_ddpg_platform, dqn_ddpg_platform, ppo_ddpg_goal, a2c_ddpg_goal, dqn_ddpg_goal
    from sarl.snippets.converter_use import ppo_sac_platform, a2c_sac_platform, dqn_sac_platform, ppo_sac_goal, a2c_sac_goal, dqn_sac_goal
    from sarl.snippets.converter_use import ppo_td3_platform, a2c_td3_platform, dqn_td3_platform, ppo_td3_goal, a2c_td3_goal, dqn_td3_goal
    try:
        chosen_script = {  # (Dict mapping config terms to functions)
            "qpamdp": {
                "platform": qpamdp_platform,
                "goal": qpamdp_goal
            },

            "pdqn": {
                "platform": pdqn_platform,
                "goal": pdqn_goal
            },

            "ppo-ppo": {
                "platform": ppo_ppo_platform,
                "goal": ppo_ppo_goal
            },
            "a2c-ppo": {
                "platform": a2c_ppo_platform,
                "goal": a2c_ppo_goal
            },
            "dqn-ppo": {
                "platform": dqn_ppo_platform,
                "goal": dqn_ppo_goal
            },

            "ppo-a2c": {
                "platform": ppo_a2c_platform,
                "goal": ppo_a2c_goal
            },
            "a2c-a2c": {
                "platform": a2c_a2c_platform,
                "goal": a2c_a2c_goal
            },
            "dqn-a2c": {
                "platform": dqn_a2c_platform,
                "goal": dqn_a2c_goal
            },

            "ppo-ddpg": {
                "platform": ppo_ddpg_platform,
                "goal": ppo_ddpg_goal
            },
            "a2c-ddpg": {
                "platform": a2c_ddpg_platform,
                "goal": a2c_ddpg_goal
            },
            "dqn-ddpg": {
                "platform": dqn_ddpg_platform,
                "goal": dqn_ddpg_goal
            },

            "ppo-sac": {
                "platform": ppo_sac_platform,
                "goal": ppo_sac_goal
            },
            "a2c-sac": {
                "platform": a2c_sac_platform,
                "goal": a2c_sac_goal
            },
            "dqn-sac": {
                "platform": dqn_sac_platform,
                "goal": dqn_sac_goal
            },

            "ppo-td3": {
                "platform": ppo_td3_platform,
                "goal": ppo_td3_goal
            },
            "a2c-td3": {
                "platform": a2c_td3_platform,
                "goal": a2c_td3_goal
            },
            "dqn-td3": {
                "platform": dqn_td3_platform,
                "goal": dqn_td3_goal
            },
        }[job_config["algorithm"]][job_config["environment"]]
    except:
        raise NotImplementedError

    # enable useful log messages, saved to /outputs
    hydra_config = HydraConfig.get()
    logger = logging.getLogger(hydra_config.job.name)
    logger.setLevel(getattr(logging, job_config.get("verbose", "info").upper()))
    logger.info("Log started.")
    logger.info(f"SB3 get_device() output:{get_device()}")
    logger.info(f'Running {job_config["algorithm"]} on {job_config["environment"]} with seeds: {job_config["parameters"]["seeds"]}')

    output_dir = HydraConfig.get().runtime.output_dir  # See /.hydra in relevant folder for config
    logger.info(f"Writing to {output_dir}")

    # def toy_func_to_optimise():  # TODO: Replace with mean & std reward
    #     alg_params = job_config["parameters"]["alg_params"]
    #     print(alg_params)
    #     return -((alg_params["discrete_learning_rate"] - 0.5) ** 2)
    # return toy_func_to_optimise()
    import inspect

    params = dict(job_config["parameters"])
    sig = inspect.signature(chosen_script)

    # If the function doesn't accept **kwargs, only pass what it declares
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not accepts_kwargs:
        params = {k: v for k, v in params.items() if k in sig.parameters}

    return chosen_script(**params, output_dir=output_dir)

    # return chosen_script(**job_config["parameters"], output_dir=output_dir)


if __name__ == "__main__":
    main()
