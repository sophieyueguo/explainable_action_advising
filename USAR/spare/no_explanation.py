'''Compare with benchmarks...'''
import numpy as np

from agent.agent import Agents
from policy_extraction import simulate
from policy_extraction import util
import import experiment_parameter as parameter


def generate_student_with_memory_episode(rolloutWorker, student, ask_budget, fetch_budget, reuse_prob, decay_rate, q_sa_dict, advice_dict, prob_ask, maximum_budget, episode_num=None, evaluate=False):
    # print ('start episode')
    if rolloutWorker.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
        rolloutWorker.env.close()
    o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
    a, si = [], []
    rolloutWorker.env.reset()
    terminated = False
    win_tag = False
    step = 0
    episode_reward = 0  # cumulative rewards
    last_action = np.zeros((rolloutWorker.args.n_agents, rolloutWorker.args.n_actions))

    rolloutWorker.teacher.agents.policy.init_hidden(1)
    student.policy.init_hidden(1)

    # epsilon
    epsilon = 0 if evaluate else rolloutWorker.epsilon
    if rolloutWorker.args.epsilon_anneal_scale == 'episode':
        epsilon = epsilon - rolloutWorker.anneal_epsilon if epsilon > rolloutWorker.min_epsilon else epsilon

    # sample z for maven
    if rolloutWorker.args.alg == 'maven':
        state = rolloutWorker.env.get_state()
        state = torch.tensor(state, dtype=torch.float32)
        if rolloutWorker.args.cuda:
            state = state.cuda()
        z_prob = rolloutWorker.agents.policy.z_policy(state)
        maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
        maven_z = list(maven_z.cpu())


    agreed_obs = [[] for agent_id in range(rolloutWorker.n_agents)]
    agreed_act = [[] for agent_id in range(rolloutWorker.n_agents)]

    while not terminated and step < rolloutWorker.episode_limit:
        obs = rolloutWorker.env.get_obs()
        state = rolloutWorker.env.get_state()
        actions, avail_actions, actions_onehot = [], [], []
        state_importances = []


        for agent_id in range(rolloutWorker.n_agents):
            avail_action = rolloutWorker.env.get_avail_agent_actions(agent_id)
            if rolloutWorker.args.alg == 'maven':
                print ('not handled algo, stop')
            else:
                teacher_action, state_importance, _ = rolloutWorker.teacher.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                student_action, state_importance, q_value = student.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)

                action = None

                if str(obs[agent_id]) in advice_dict:
                    if parameter.benchmark_algo == 'Reusing Budget':
                        if fetch_budget[str(obs[agent_id])] > 0:
                            action = advice_dict[str(obs[agent_id])]
                            if not evaluate:
                                fetch_budget[str(obs[agent_id])] -= 1

                    elif parameter.benchmark_algo == 'Decay Reusing Probability':
                        if reuse_prob[str(obs[agent_id])] > np.random.uniform(low=0.0, high=1.0, size=None):
                            action = advice_dict[str(obs[agent_id])]
                            if not evaluate:
                                reuse_prob[str(obs[agent_id])] *= decay_rate

                    elif parameter.benchmark_algo == 'Q-change Per Step':
                        action = advice_dict[str(obs[agent_id])]
                    else:
                        assert False==True, 'Unknown Benchmark Algorithms'


                elif ask_budget > 0:
                    if prob_ask > np.random.uniform(low=0.0, high=1.0, size=None):
                        action = teacher_action
                        if not evaluate:
                            ask_budget -= 1
                            # print ('calling minus')

                            advice_dict[str(obs[agent_id])] = teacher_action

                            fetch_budget[str(obs[agent_id])] = maximum_budget
                            reuse_prob[str(obs[agent_id])] = 1.


                if action == None:
                    action = student_action

                if not evaluate:
                    if parameter.benchmark_algo == 'Q-change Per Step':
                        if len(q_value) == rolloutWorker.n_actions:
                            key = (str(obs[agent_id]), action)
                            if key in q_sa_dict:
                                old_value = q_sa_dict[key]
                                thre = 0.001
                                if q_value[action] - old_value <= 0.001:
                                    del advice_dict[str(obs[agent_id])]
                            else:
                                q_sa_dict[key] = q_value[action]

                state_importance = 0.1

            # generate onehot vector of th action
            action_onehot = np.zeros(rolloutWorker.args.n_actions)
            action_onehot[action] = 1
            actions.append(int(action))
            actions_onehot.append(action_onehot)
            avail_actions.append(avail_action)
            last_action[agent_id] = action_onehot
            state_importances.append(state_importance)
        a.append(actions)
        si.append(state_importances)

        reward, terminated, info = rolloutWorker.env.step(actions)
        win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
        o.append(obs)
        s.append(state)
        u.append(np.reshape(actions, [rolloutWorker.n_agents, 1]))
        u_onehot.append(actions_onehot)
        avail_u.append(avail_actions)
        r.append([reward])
        terminate.append([terminated])
        padded.append([0.])
        episode_reward += reward
        step += 1
        if rolloutWorker.args.epsilon_anneal_scale == 'step':
            epsilon = epsilon - rolloutWorker.anneal_epsilon if epsilon > rolloutWorker.min_epsilon else epsilon

    # print ('140')
    # last obs
    obs = rolloutWorker.env.get_obs()
    state = rolloutWorker.env.get_state()

    o.append(obs)
    s.append(state)
    o_next = o[1:]
    s_next = s[1:]
    o = o[:-1]
    s = s[:-1]
    # get avail_action for last obs，because target_q needs avail_action in training
    avail_actions = []
    for agent_id in range(rolloutWorker.n_agents):
        avail_action = rolloutWorker.env.get_avail_agent_actions(agent_id)
        avail_actions.append(avail_action)
    avail_u.append(avail_actions)
    avail_u_next = avail_u[1:]
    avail_u = avail_u[:-1]

    for i in range(step, rolloutWorker.episode_limit):
        o.append(np.zeros((rolloutWorker.n_agents, rolloutWorker.obs_shape)))
        u.append(np.zeros([rolloutWorker.n_agents, 1]))
        s.append(np.zeros(rolloutWorker.state_shape))
        r.append([0.])
        o_next.append(np.zeros((rolloutWorker.n_agents, rolloutWorker.obs_shape)))
        s_next.append(np.zeros(rolloutWorker.state_shape))
        u_onehot.append(np.zeros((rolloutWorker.n_agents, rolloutWorker.n_actions)))
        avail_u.append(np.zeros((rolloutWorker.n_agents, rolloutWorker.n_actions)))
        avail_u_next.append(np.zeros((rolloutWorker.n_agents, rolloutWorker.n_actions)))
        padded.append([1.])
        terminate.append([1.])

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
        rolloutWorker.epsilon = epsilon
    if rolloutWorker.args.alg == 'maven':
        episode['z'] = np.array([maven_z.copy()])
    if evaluate and episode_num == rolloutWorker.args.evaluate_epoch - 1 and rolloutWorker.args.replay_dir != '':
        rolloutWorker.env.save_replay()
        rolloutWorker.env.close()

    # print ('end episode')
    return episode, episode_reward, o, a, si, step, ask_budget, fetch_budget, reuse_prob, advice_dict








if __name__ == '__main__':
    _, rolloutWorker = simulate.sample_from_teacher(data_filter_ratio=1.0, n_steps_to_sample=1)

    student = Agents(rolloutWorker.args)

    time_steps, train_steps, evaluate_steps = 0, 0, -1
    trial_index = 0
    episode_rewards = []
    win_rates = []

    ask_budget = parameter.teacher_max_advice_budget

    fetch_budget = {}
    reuse_prob = {}
    q_sa_dict = {}
    advice_dict = {}

    # some of hand-tune parameters for this 3 algorithms
    prob_ask = 0.2
    # prob_ask = 0.6
    decay_rate = 0.98 # old: 0.95
    maximum_budget = 100 # can be bigger?
    # maximum_budget = 10


    while trial_index < parameter.max_trial_index:
        if trial_index % parameter.evaluate_trial_cycle == 0:

            avg_reward = 0
            for i in range(100):
                _, episode_reward, _, _, _, _, ask_budget, fetch_budget, reuse_prob, advice_dict = generate_student_with_memory_episode(rolloutWorker, student, ask_budget, fetch_budget, reuse_prob, decay_rate, q_sa_dict, advice_dict, prob_ask, maximum_budget, episode_num=None, evaluate=True)
                avg_reward += episode_reward
            avg_reward /= 100
            episode_rewards.append(avg_reward)

            # note that version difference...
            np.save(parameter.benchmak_save_path + 'episode_rewards', episode_rewards)
            print ('trial_index', trial_index)
            print ('evaluate, avg_reward', avg_reward)
            print ('ask_budget', ask_budget)
            print ('len(fetch_budget)', len(fetch_budget))
            print ()
            print ()


        trial_index += 1
        print ('trial_index', trial_index, 'time_steps', time_steps)
        episodes = []
        for episode_idx in range(rolloutWorker.args.n_episodes):

            episode, _, _, _, _, steps, ask_budget, fetch_budget, reuse_prob, advice_dict = generate_student_with_memory_episode(rolloutWorker, student, ask_budget, fetch_budget, reuse_prob, decay_rate, q_sa_dict, advice_dict, prob_ask, maximum_budget, episode_num=None, evaluate=False)
            episodes.append(episode)
            time_steps += steps

        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
        if rolloutWorker.args.alg.find('coma') > -1 or rolloutWorker.args.alg.find('central_v') > -1 or rolloutWorker.args.alg.find('reinforce') > -1:
            student.train(episode_batch, train_steps, rolloutWorker.epsilon)
            train_steps += 1
        else:
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                student.train(mini_batch, train_steps)
                train_steps += 1
