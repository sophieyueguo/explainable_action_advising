import numpy as np
import os
import torch
from torch.distributions import one_hot_categorical
import time

from common.arguments import get_common_args, get_coma_args
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer

from env.hetro_usar_simple_4_room import Hetro_USAR_Simple_4_Room_Env
from env.hetro_usar_14_room import Hetro_USAR_14_Room_Env
from teacher import Teacher
import experiment_parameter as parameter



class RolloutWorker:
    def __init__(self, env, args):
        self.env = env
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon


        # code for teacher agent
        self.teacher = Teacher(self.args)
        self.student = Agents(args)
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        a, si = [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        self.teacher.agents.policy.init_hidden(1)


        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            state_importances = []


            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if self.args.alg == 'maven':
                    print ('not handled algo, stop')
                else:
                    action, state_importance, _ = self.teacher.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                           avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
                state_importances.append(state_importance)
            a.append(actions)
            si.append(state_importances)

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # self.agents.save_model()
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()

        # print ('end state', state)
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding - don't pad
        # for i in range(step, self.episode_limit):
        #     o.append(np.zeros((self.n_agents, self.obs_shape)))
        #     u.append(np.zeros([self.n_agents, 1]))
        #     s.append(np.zeros(self.state_shape))
        #     r.append([0.])
        #     o_next.append(np.zeros((self.n_agents, self.obs_shape)))
        #     s_next.append(np.zeros(self.state_shape))
        #     u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
        #     avail_u.append(np.zeros((self.n_agents, self.n_actions)))
        #     avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
        #     padded.append([1.])
        #     terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()


        return episode_reward, win_tag, episode, o, a, si, step








class Runner:
    def __init__(self, env, args):
        self.env = env
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            print ('commnet agent, not implemented')
        else:  # no communication agent
            self.rolloutWorker = RolloutWorker(env, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

    def run(self):
        time_steps, train_steps, evaluate_steps = 0, 0, -1

        trial_index = 0
        episode_rewards = []
        win_rates = []
        D = []


        while time_steps < self.args.n_steps_to_sample:
            trial_index += 1
            print ('trial_index', trial_index, end="\r")
            # print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                evaluate_steps += 1
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                # print ('episode_idx', episode_idx)
                episode_reward, win_tag, episode, o, a, si, steps = self.rolloutWorker.generate_episode(episode_idx)
                # print ('episode_reward', episode_reward, 'len(o)', len(o))
                assert len(o) == len(a)


                # Collecting D...
                if parameter.env_type == 'Simple_4_Room':
                    if len(o) < self.env.episode_limit * self.args.data_filter_ratio: # only sample the efficient ones:
                        for i in range(len(o)):
                            D.append([o[i], a[i], si[i]])
                elif parameter.env_type == '14_Room':
                    if episode_reward == 20:
                        for i in range(len(o)):
                            D.append([o[i], a[i], si[i]])
                else:
                    assert False==True, 'Unknown Environment'



                episodes.append(episode)
                time_steps += steps
                # print(_)
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    train_steps += 1

            if trial_index % parameter.evaluate_trial_cycle == 0:
                win_rate, episode_reward = self.evaluate()
                print('trial_index', trial_index, 'time_steps', time_steps, 'win_rate is ', win_rate, 'episode_reward', episode_reward)
                win_rates.append(win_rate)
                episode_rewards.append(episode_reward)
                # np.save('win_rates.npy', win_rates)
                # np.save('episode_rewards', episode_rewards)

        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate, 'episode_reward', episode_reward)
        print ('len(win_rates)', len(win_rates), 'len(episode_rewards)', len(episode_rewards))
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)


        return D, self.rolloutWorker

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            episode_reward, win_tag, episode, o, a, si, steps = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            # print ('single episode reward', episode_reward)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch


def sample_from_teacher(data_filter_ratio, n_steps_to_sample=1000):
    args = get_common_args()
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
        print ('COMA agent!')

    if parameter.env_type == 'Simple_4_Room':
        env = Hetro_USAR_Simple_4_Room_Env()
    elif parameter.env_type == '14_Room':
        env = Hetro_USAR_14_Room_Env()

    else:
        assert False==True, 'Unknown Environment'


    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    args.n_steps_to_sample = n_steps_to_sample
    args.data_filter_ratio = data_filter_ratio

    print ('env_info', env_info)
    runner = Runner(env, args)
    if not args.evaluate:
        D, rolloutWorker = runner.run()
    else:
        win_rate, _ = runner.evaluate()
        print('The win rate of {} is  {}'.format(args.alg, win_rate))

    env.close()

    return D, rolloutWorker
