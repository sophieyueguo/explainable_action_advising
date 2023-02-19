import argparse
import config
import model
import numpy as np
import pickle
from sklearn import tree
import train


parser = argparse.ArgumentParser()
parser.add_argument(
    "--import-path",
    type=str,
    default=None,
    help="Path to the oracle model which is to be distilled with VIPER.",
)
parser.add_argument(
    "--max-iters",
    type=int,
    default=10,
    help="The maximum number of VIPER iterations to run.",
)
parser.add_argument(
    "--steps-per-iter",
    type=int,
    default=10000,
    help="The number of new steps to sample per VIPER iteration",
)
parser.add_argument(
    "--resampled-steps",
    type=int,
    default=5000,
    help="The number of steps to resample each VIPER iteration.",
)
parser.add_argument(
    "--eval-episodes",
    type=int,
    default=100,
    help="The number of evaluation episodes to rollout each VIPER iteration.",
)
parser.add_argument(
    "--export-path",
    type=str,
    default="distilled_tree.pickle",
    help="Path to export the distilled decision tree model.",
)


def fit_decision_tree(data):
    clf = tree.DecisionTreeClassifier()

    obs = np.stack(data[:, 0], axis=0)
    actions = data[:, 1].astype(int)

    clf = clf.fit(obs, actions)

    return clf


def resample(data, n_steps_to_sample=500):
    # Resample the data weighted by the normalized state importance values
    probabilities = data[:, 2].astype(float)
    probabilities = (probabilities / np.sum(probabilities))

    resample_idx = np.random.choice(data.shape[0], n_steps_to_sample, p=probabilities)

    return data[resample_idx, :]


def run_viper(oracle_model_path, max_iters = 10, steps_per_iter = 1000, resampled_steps = 500, eval_episodes = 50, export_path = "distilled_tree.pickle"):
    data = None
    best_reward = -np.inf
    best_model = None

    env = train.env_maker(config.CONFIG["env_config"])
    oracle_model = model.ModelWrapper(model.ModelType.TORCH)
    oracle_model.load(oracle_model_path, env.action_space, env.observation_space, config.CONFIG["model"])

    distilled_model = model.ModelWrapper(model.ModelType.TREE)

    # For each iteration...
    for iter_idx in range(max_iters):
        # Collect new oracle steps with the given model
        new_data = np.array(train.rollout_steps(oracle_model, env, num_steps = steps_per_iter))

        if data is None:
            data = new_data
        else:
            data = np.concatenate([data, new_data], axis = 0)

        # Resample the data according to the probability
        data = resample(data, n_steps_to_sample = resampled_steps)

        clf = fit_decision_tree(data)
        distilled_model.set(clf)

        reward, _ = train.rollout_episodes(distilled_model, env, eval_episodes)

        if reward > best_reward:
            best_reward = reward
            best_model = clf

        print("Iteration {} distilled model with reward {}.".format(iter_idx, reward))

    print("Exporting best found model with reward {} to {}.".format(best_reward, export_path))
    # Save the best found model
    pickle.dump(best_model, open(export_path, 'wb'))


if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    run_viper(args.import_path, max_iters=args.max_iters, steps_per_iter=args.steps_per_iter, resampled_steps=args.resampled_steps, eval_episodes=args.eval_episodes, export_path=args.export_path)