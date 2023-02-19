import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import experiment_parameter as parameter

import torch


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)

        ##################################################
        # pretrained model
        if parameter.pretrain_model:
            self.agents.policy.model_dir = './model/coma/'
            print ('self.model_dir', self.agents.policy.model_dir)
            if os.path.exists(self.agents.policy.model_dir + '/' + parameter.teacher_rnn_path + '/rnn_params.pkl'):
                path_rnn = self.agents.policy.model_dir + '/' + parameter.teacher_rnn_path + '/rnn_params.pkl'
                path_coma = self.agents.policy.model_dir + '/' + parameter.teacher_rnn_path + '/critic_params.pkl'

                print ('path_rnn', path_rnn, '\npath_come', path_coma)
                map_location = 'cuda:0' if self.agents.policy.args.cuda else 'cpu'
                self.agents.policy.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.agents.policy.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
            else:
                raise Exception("No model!")

            self.agents.policy.target_critic.load_state_dict(self.agents.policy.eval_critic.state_dict())

            self.agents.policy.rnn_parameters = list(self.agents.policy.eval_rnn.parameters())
            self.agents.policy.critic_parameters = list(self.agents.policy.eval_critic.parameters())

            if args.optimizer == "RMS":
                self.agents.policy.critic_optimizer = torch.optim.RMSprop(self.agents.policy.critic_parameters, lr=args.lr_critic)
                self.agents.policy.rnn_optimizer = torch.optim.RMSprop(self.agents.policy.rnn_parameters, lr=args.lr_actor)


        if not args.evaluate and args.alg.find('coma') == -1:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []


    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1

        trial_index = 0
        episode_rewards = []
        win_rates = []

        win_rate, episode_reward = self.evaluate()
        print('initial win rate before the training starts!', 'win_rate is ', win_rate, 'episode_reward', episode_reward)
        win_rates.append(win_rate)
        episode_rewards.append(episode_reward)

        if parameter.does_save_result:
            np.save(parameter.save_path + '/win_rates.npy', win_rates)
            np.save(parameter.save_path + '/episode_rewards', episode_rewards)


        while trial_index < parameter.max_trial_index:

            trial_index += 1
            print ('trial_index', trial_index, end="\r")
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                evaluate_steps += 1
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                if parameter.student_train:
                    self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                    if trial_index % parameter.save_model_trial_cycle == 0:
                        self.agents.policy.save_model(trial_index)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    if parameter.student_train:
                        self.agents.train(mini_batch, train_steps)
                        if trial_index % parameter.save_model_trial_cycle == 0:
                            self.agents.policy.save_model(trial_index)
                    train_steps += 1

            if trial_index % parameter.evaluate_trial_cycle == 0:
                win_rate, episode_reward = self.evaluate()
                print('trial_index', trial_index, 'time_steps', time_steps, 'win_rate is ', win_rate, 'episode_reward', episode_reward)
                win_rates.append(win_rate)
                episode_rewards.append(episode_reward)
                if parameter.does_save_result:
                    np.save(parameter.save_path + '/win_rates.npy', win_rates)
                    np.save(parameter.save_path + '/episode_rewards', episode_rewards)

        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate, 'episode_reward', episode_reward)
        print ('len(win_rates)', len(win_rates), 'len(episode_rewards)', len(episode_rewards))
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch
