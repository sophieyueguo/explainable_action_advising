from typing import Dict, Tuple

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class LoggingCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        if "action_advice" not in episode.custom_metrics:
            episode.custom_metrics["action_advice"] = 0

        if "action_student" not in episode.custom_metrics:
            episode.custom_metrics["action_student"] = 0

        if "action_introspection" not in episode.custom_metrics:
            episode.custom_metrics["action_introspection"] = 0

        episode_info = episode.last_info_for()

        episode.custom_metrics["action_advice"] += episode_info["action_advice"]
        episode.custom_metrics["action_student"] += episode_info["action_student"]
        episode.custom_metrics["action_introspection"] += episode_info["action_introspection"]

