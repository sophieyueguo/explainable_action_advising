from callbacks import LoggingCallbacks
from ray import tune

CONFIG = {
    # "env": "MsPacmanNoFrameskip-v4",
    "env": "pacman",
    "env_config": {
        # "disable_env_checking": True,
        "teacher_model_path": "trained_model3.pth",
        "teacher_dt_path": "distilled_tree3.pickle",
        "advice_budget": tune.grid_search([10, 20, 1000]),
        "advice_mode": tune.grid_search(["aa", "eaa"]), # in the set of {None, "aa", "eaa", "fixed"}
        "advice_strategy": tune.grid_search(["e", "a", "i", "m"]), # in the set of {e, a, i, m}
        "introspection_decay_rate": 1.0,
        "fixed_advise_type": tune.grid_search(["reuse_budget", "decay_reuse", "q_change"]), # in the set of {'reuse_budget', 'decay_reuse', 'q_change'}

        "multiagent": False,
    },
    "callbacks": LoggingCallbacks,
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": 0,
    "num_cpus_per_worker": 1,
    "num_cpus_for_driver": 2,
    "model": {
        # "vf_share_layers": True,

        "custom_model": "pacman_ac",
        "custom_model_config": {
            "input_layer_sizes": [(13, 256)],
            "actor_layer_sizes": [(256, 4)],
            "hidden_layer_sizes": [(256, 256)],
            "value_layer_sizes": [(256, 1)],
        },
    },
    "framework": "torch", # Only torch supported

    "lr": 0.0005, #tune.grid_search([0.001, 0.0005, 0.0001]), #0.0001, # 0.001 is good for random ghost
    "lambda": 0.8, #tune.grid_search([0.8, 0.9, 0.99]),
    "kl_coeff": 0.5, #tune.grid_search([0.1, 0.5, 0.9]),
    "clip_rewards": False,
    "clip_param": 0.2, #tune.grid_search([0.1, 0.2, 0.3]), #0.1,
    "vf_clip_param": 10.0,
    "vf_loss_coeff": 0.5, #tune.grid_search([0.5, 0.75, 1.0]), #1.0
    "entropy_coeff": 0.01, #tune.grid_search([0.1, 0.01, 0.0]),
    "train_batch_size": 5000,
    "rollout_fragment_length": 100,
    "sgd_minibatch_size": 500,
    "num_sgd_iter": 20, #tune.grid_search([3, 10, 20]), #10,

    # "lambda": 0.95,
    # "kl_coeff": 0.5,
    # "clip_rewards": False,
    # "clip_param": 0.1,
    # "vf_clip_param": 10.0,
    # "entropy_coeff": 0.01,
    # "train_batch_size": 5000,
    # "rollout_fragment_length": 100,
    # "sgd_minibatch_size": 500,
    # "num_sgd_iter": 10,

    "num_workers": 1,
    "num_envs_per_worker": 1,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",

    "horizon": 500,
}
