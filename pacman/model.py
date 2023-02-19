import gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer


class ModelType(Enum):
    TORCH = 1
    TREE = 2


def load_torch_model(import_path, action_space, observation_space, config):
    loaded_model = PacmanActorCritic(action_space, observation_space, action_space.n, config, "PacmanActorCritic")
    loaded_model.load_state_dict(torch.load(import_path, map_location=torch.device('cpu')))
    loaded_model.eval()

    return loaded_model


class PacmanActorCritic(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.input_layer_sizes = model_config["custom_model_config"]["input_layer_sizes"]
        self.hidden_layer_sizes = model_config["custom_model_config"]["hidden_layer_sizes"]
        self.value_layer_sizes = model_config["custom_model_config"]["value_layer_sizes"]
        self.actor_layer_sizes = model_config["custom_model_config"]["actor_layer_sizes"]
        
        self.input_layers = nn.ModuleList(self._create_layers(self.input_layer_sizes))
        self.hidden_layers = nn.ModuleList(self._create_layers(self.hidden_layer_sizes) + [nn.Dropout(0.5)])
        self.actor_layers = nn.ModuleList(self._create_layers(self.actor_layer_sizes, activation_type=None))
        self.value_layers = nn.ModuleList(self._create_layers(self.value_layer_sizes, activation_type=None))

        self._features = None

    def _create_layers(self, sizes, layer_type = nn.Linear, activation_type = nn.ReLU, initializer = normc_initializer):
        layers = []
        for in_size, out_size in sizes:
            layers.append(layer_type(in_size, out_size))

            if initializer is not None:
                initializer(layers[-1].weight)

            if activation_type is not None:
                layers.append(activation_type())

        return layers

    def compute_actor(self, x):
        for layer in self.actor_layers:
            x = layer(x)
        
        return x

    def compute_input(self, x):
        x = x.to(torch.float)
        for layer in self.input_layers:
            x = layer(x)
        
        return x

    def compute_hidden(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        
        return x

    def compute_value(self, x):
        for layer in self.value_layers:
            x = layer(x)
        
        return x

    def forward(self, input_dict, state, seq_lens):
        x = self.compute_input(input_dict["obs"])
        x = self.compute_hidden(x)

        self._features = x

        x = self.compute_actor(x)

        # Make sure to return logits. The output gets placed into a torch categorical distribution as logits (not probs). So do not softmax!
        return x, state

    def value_function(self):
        return self.compute_value(self._features).squeeze(1)


class ModelWrapper():
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def load(self, import_path, action_space, observation_space, config):
        if self.model_type == ModelType.TORCH:
            self.model = load_torch_model(import_path, action_space, observation_space, config)
        elif self.model_type == ModelType.TREE:
            self.model = pickle.load(open(import_path, 'rb'))

    def set(self, in_model):
        self.model = in_model

    def get_action(self, obs):
        if self.model_type == ModelType.TORCH:
            last_state_features = torch.tensor(obs, requires_grad = False)
            action_logit = self.model({"obs": last_state_features})[0]
            action_prob = F.softmax(action_logit, dim=0).cpu().detach().numpy()
            log_action_prob = np.log(action_prob)
            action = np.argmax(action_prob)

            # Use maximum entropy formulation to estimate q values
            importance = np.max(log_action_prob) - np.min(log_action_prob)
        elif self.model_type == ModelType.TREE:
            action = self.model.predict([obs])[0]
            action_prob = self.model.predict_proba([obs])[0]
            importance = None

        return action, action_prob, importance

    def get_explanation(self, obs, action):
        explanation = None

        if self.model_type == ModelType.TREE:
            feature = self.model.tree_.feature
            threshold = self.model.tree_.threshold

            node_indicator = self.model.decision_path([obs])
            leaf_id = self.model.apply([obs])[0]

            # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
            node_index = node_indicator.indices[
                node_indicator.indptr[0] : node_indicator.indptr[1]
            ]

            explanation = []
            for node_id in node_index:
                # continue to the next node if it is a leaf node
                if leaf_id == node_id:
                    continue

                # check if value of the split feature for sample 0 is below threshold
                if obs[feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                explanation.append({'node': node_id,
                            'feature': feature[node_id],
                            'value': obs[feature[node_id]],
                            'inequality': threshold_sign,
                            'threshold': threshold[node_id],
                            'is_leaf': False})

            explanation.append({'node': leaf_id,
                        'value': action,
                        'is_leaf': True})

        return explanation

