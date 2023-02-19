import numpy as np
import torch
from torch.distributions import Categorical

class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        if args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        if self.args.alg == 'coma':
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        state_importance = 0
        if self.args.alg == 'coma':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
            state_importance = self._calc_state_importance(q_value, avail_actions)
        else:
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)
            else:
                action = torch.argmax(q_value)

        # return action, state_importance, q_value.detach().numpy()[0]
        # to be used on gpu
        return action, state_importance, q_value.detach().cpu().numpy()[0]

    def _calc_state_importance(self, q_value, avail_actions):
        return float(torch.max(q_value[avail_actions == 1])) - float(torch.min(q_value[avail_actions == 1]))





    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        prob = torch.nn.functional.softmax(inputs, dim=-1)

        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)


    def load_model(self, path_rnn, path_coma):
        print ('path_rnn', path_rnn, '\npath_come', path_coma)
        map_location = 'cuda:0' if self.policy.args.cuda else 'cpu'
        self.policy.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
        self.policy.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
        print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
