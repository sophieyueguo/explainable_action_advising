import argparse
import config
import model
import numpy as np
import pacman.gym_wrapper
import pickle
import ray
import torch
import torch.nn.functional as F

from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune import register_env
from ray.tune.logger import pretty_print


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="train",
    choices=["train", "debug", "evaluate", "export", "experiments"],
    help="Running mode. Train to train a new model from scratch. Debug to run a minimal overhead training loop for debugging purposes."
         "Evaluate to rollout a trained pytorch model. Export to export a trained pytorch model from a checkpoint."
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default=None,
    help="Path to the directory containing an RLlib checkpoint. Used with the 'export' mode.",
)
parser.add_argument(
    "--import-path",
    type=str,
    default=None,
    help="Path to a pytorch saved model. Used with the 'evaluate' mode.",
)
parser.add_argument(
    "--export-path",
    type=str,
    default=None,
    help="Path to export a pytorch saved model. Used with the 'export' mode."
)
parser.add_argument(
    "--eval-episodes",
    type=int,
    default=1,
    help="Number of episodes to rollout for evaluation."
)
parser.add_argument(
    "--save-eval-rollouts",
    action="store_true",
    help="Whether to save (state, action, importance) triplets from evaluation rollouts.",
)
parser.add_argument(
    "--model-type",
    default="torch",
    choices=["torch", "tree"],
    help="The type of model to be imported. Options are 'torch' for a pytorch model or 'tree' for an sklearn tree classifier."
)


def env_maker(config):
    env = pacman.gym_wrapper.GymPacman(config)

    return env


def rollout_episode(loaded_model, env, max_steps = 500):
    obs = env.reset()
    done = False

    states = []

    step_idx = 0
    total_reward = 0
    while not done:
        action, _, importance = loaded_model.get_action(obs)

        states.append([obs, action, importance])

        obs, reward, done, info = env.step(action)
        step_idx += 1
        total_reward += reward

        if step_idx > max_steps:
            break

    return states, step_idx, total_reward


def rollout_episodes(loaded_model, env, num_episodes = 1, save_rollouts = False, max_steps = 500):
    all_episode_states = []
    num_steps = []
    rewards = []

    for _ in range(num_episodes):
        states, steps, reward = rollout_episode(loaded_model, env, max_steps = max_steps)
        all_episode_states.append(states)
        num_steps.append(steps)
        rewards.append(reward)

    if save_rollouts:
        with open('model_trajectories.pickle', 'wb') as f:
            pickle.dump(all_episode_states, f)

    return np.mean(rewards), np.mean(num_steps)


def rollout_steps(loaded_model, env, num_steps = 100, save_rollouts = False, max_steps = 500):
    steps_collected = 0

    all_episode_states = []

    while steps_collected < num_steps:
        states, steps, _ = rollout_episode(loaded_model, env, max_steps = max_steps)
        all_episode_states.extend(states)
        steps_collected += steps

    all_episode_states = all_episode_states[:num_steps]

    return all_episode_states


def export_model(checkpoint_dir, export_path):
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config.update(config.CONFIG)
    trainer = ppo.PPOTrainer(config=ppo_config, env="pacman")

    trainer.restore(checkpoint_dir)

    policy = trainer.get_policy(DEFAULT_POLICY_ID)
    model = policy.model

    torch.save(model.state_dict(), export_path)


def train_model(mode = "train", checkpoint_dir = None, export_path = None, import_path = None):
    ray.init(local_mode = False)

    ModelCatalog.register_custom_model("pacman_ac", model.PacmanActorCritic)
    register_env("pacman", env_maker)

    model_type = model.ModelType.TORCH
    if args.model_type == "tree":
        model_type = model.ModelType.TREE

    if mode == "debug":
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config.CONFIG)
        trainer = ppo.PPOTrainer(config=ppo_config, env="pacman")

        # run manual training loop and print results after each iteration
        for _ in range(5):
            result = trainer.train()
            print(pretty_print(result))

    elif mode == "train":
                # automated run with Tune and grid search and TensorBoard
                print("Training automatically with Ray Tune")
                stop = {
                    "training_iteration": 200,
                }

                results = tune.run(
                    ppo.PPOTrainer,
                    config=config.CONFIG,
                    checkpoint_freq=5,
                    checkpoint_at_end=True,
                    # num_samples=20,
                    stop=stop)

    elif mode == "experiments":
        # budget_config = [10, 20, 1000]
        # advice_config = [["aa", "e"], ["aa", "a"], ["aa", "i"], ["aa", "m"],
        #                  ["eaa", "e"], ["eaa", "a"], ["eaa", "i"], ["eaa", "m"],
        #                  [None, None]]

        # for budget in budget_config:
        #     for advice_mode, advice_strategy in advice_config:
        config_copy = config.CONFIG.copy()

        # config_copy["env_config"]["advice_budget"] = budget
        # config_copy["env_config"]["advice_mode"] = advice_mode
        # config_copy["env_config"]["advice_strategy"] = advice_strategy

        stop = {
            "training_iteration": 150,
        }

        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        results = tune.run(
            ppo.PPOTrainer,
            config=config_copy,
            checkpoint_freq=5,
            checkpoint_at_end=True,
            num_samples = 5,
            stop=stop)

    elif mode == "evaluate":
        if import_path is None:
            raise("import_path must be specified for the 'evaluate' mode.")

        env = env_maker(config.CONFIG["env_config"])
        loaded_model = model.ModelWrapper(model_type)
        loaded_model.load(import_path, env.action_space, env.observation_space, config.CONFIG["model"])

        reward, steps = rollout_episodes(loaded_model, env, num_episodes=args.eval_episodes, save_rollouts=args.save_eval_rollouts)

        print("Evaluated {} episodes. Average reward: {}. Average num steps: {}".format(args.eval_episodes, reward, steps))

    elif mode == "export":
        if checkpoint_dir is None:
            raise("checkpoint_dir must be specified for the 'export' mode.")
        if export_path is None:
            raise("export_path must be specified for the 'export' mode.")

        export_model(checkpoint_dir, export_path)

    ray.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    train_model(args.mode, args.checkpoint_dir, args.export_path, args.import_path)
