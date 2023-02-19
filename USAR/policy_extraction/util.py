import numpy as np
import pickle
from sklearn import tree

import experiment_parameter as parameter


def train_decision_tree_policy(D_prime):
    clfs = [tree.DecisionTreeClassifier() for j in range(len(D_prime[0][0]))] # each agent maintains a tree
    for j in range(len(clfs)):
        X = [D_prime[i][0][j] for i in range(len(D_prime))]
        Y = [D_prime[i][1][j] for i in range(len(D_prime))]

        if parameter.env_type == 'Simple_4_Room':
            if j == 0:
                X.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y.append(5)
            if j == 1:
                X.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y.append(4)

        elif parameter.env_type == '14_Room':
            if j == 0:
                X.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y.append(9)
            if j == 1:
                X.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y.append(8)
        else:
            print ('For new environment, please check this action edge case before starting...')
            assert False==True, 'Unknown Environment'

        clfs[j] = clfs[j].fit(X, Y)
    return clfs




def generate_decision_tree_episode(rolloutWorker, clfs, episode_num=None, evaluate=False):
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
                _, state_importance, _ = rolloutWorker.teacher.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)

                clf_prob = clfs[agent_id].predict_proba([obs[agent_id]])

                action_prob = [clf_prob[0][ai] * avail_action[ai] for ai in range(rolloutWorker.args.n_actions)]
                if sum(action_prob) == 0:
                    action = np.random.choice(rolloutWorker.args.n_actions, 1)[0]
                else:
                    if sum(action_prob) != 1:
                        action_prob = action_prob/sum(action_prob)
                    action = np.random.choice(rolloutWorker.args.n_actions, 1, p=action_prob)[0]

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

    return episode_reward, o, a, si




def process_loaded_data(rolloutWorker, D):
    X = [[] for ai in range(rolloutWorker.n_agents)]
    Y = [[] for ai in range(rolloutWorker.n_agents)]
    X_Y = [[] for ai in range(rolloutWorker.n_agents)]
    for ai in range(rolloutWorker.n_agents):
        X[ai] = [D[i][0][ai] for i in range(len(D))]
        Y[ai] = [D[i][1][ai] for i in range(len(D))]
        ################### HANDLE TREE EDGE CASE...
        ############give different roles unuseable actions to make tree complete

        if parameter.env_type == 'Simple_4_Room':
            if ai == 0:
                X[ai].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y[ai].append(5)
            if ai == 1:
                X[ai].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y[ai].append(4)

        elif parameter.env_type == '14_Room':
            if ai == 0:
                X[ai].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y[ai].append(9)
            if ai == 1:
                X[ai].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                Y[ai].append(8)
        else:
            print ('For new environment, please check this action edge case before starting...')
            assert False==True, 'Unknown Environment'


        for i in range(len(X[ai])):
            X_Y[ai].append([X[ai][i], Y[ai][i]])


    return X, Y, X_Y
