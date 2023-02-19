import experiment_parameter as parameter
'''Core code of EAA'''

import numpy as np

def generate_student_with_memory_episode(rolloutWorker, clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=None, evaluate=False):
    if rolloutWorker.args.replay_dir != '' and evaluate and episode_num == 0:
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

    advisor_agree_rate = 0
    advisor_agree_count = 0
    memory_advisor_agree_rate = 0
    memory_advisor_count = 0

    epsilon = 0 if evaluate else rolloutWorker.epsilon
    if rolloutWorker.args.epsilon_anneal_scale == 'episode':
        epsilon = epsilon - rolloutWorker.anneal_epsilon if epsilon > rolloutWorker.min_epsilon else epsilon

    agreed_obs = [[] for agent_id in range(rolloutWorker.n_agents)]
    agreed_act = [[] for agent_id in range(rolloutWorker.n_agents)]

    while not terminated and step < rolloutWorker.episode_limit:
        obs = rolloutWorker.env.get_obs()
        state = rolloutWorker.env.get_state()
        actions, avail_actions, actions_onehot = [], [], []

        state_importances = []
        teacher_actions = []

        for agent_id in range(rolloutWorker.n_agents):
            avail_action = rolloutWorker.env.get_avail_agent_actions(agent_id)
            if parameter.exp_type == '1-dt+nn' or parameter.exp_type == '2-no_advice_source':
                teacher_action, state_importance, _ = rolloutWorker.teacher.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                   avail_action, epsilon, evaluate)
                state_importances.append(state_importance)
                teacher_actions.append(teacher_action)


        for agent_id in range(rolloutWorker.n_agents):
            avail_action = rolloutWorker.env.get_avail_agent_actions(agent_id)
            if rolloutWorker.args.alg != 'coma':
                print ('not handled algo, stop')
            else:
                if parameter.exp_type == '1-dt+nn' or parameter.exp_type == '2-no_advice_source':
                    teacher_action = teacher_actions[agent_id]
                    state_importance = state_importances[agent_id]
                    student_action, _, _ = student.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                else:
                    assert parameter.exp_type == '3-dt' or parameter.exp_type == '2-no_advice_target'
                    student_action, state_importance, _ = student.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                    state_importances.append(state_importance)

                action = student_action #by default

                if parameter.advice_algo == 'heuristic_with_tree_memory':

                    if parameter.exp_type != '2-no_advice_target':
                        # advice clf_action from the dt teacher
                        clf_prob = clfs[agent_id].predict_proba([obs[agent_id]])
                        action_prob = [clf_prob[0][ai] * avail_action[ai] for ai in range(rolloutWorker.args.n_actions)]
                        if sum(action_prob) == 0:
                            clf_action = np.random.choice(rolloutWorker.args.n_actions, 1)[0]
                        else:
                            if sum(action_prob) != 1:
                                action_prob = action_prob/sum(action_prob)
                            clf_action = np.random.choice(rolloutWorker.args.n_actions, 1, p=action_prob)[0]

                    # advice from the internal dt
                    if len(new_clfs) > 0:
                        new_clf_action = new_clfs[agent_id].predict(obs[agent_id])
                        if new_clf_action != 'undecided' and np.random.rand() < follow_dt_prob:
                            action = new_clf_action
                            if not evaluate:
                                follow_dt_prob *= parameter.decay_ratio

                    # otherwise ask the teacher
                    elif rolloutWorker.teacher.advice_budget > 0 and parameter.exp_type != '2-no_advice_target':

                        ignore_advice = False
                        if not parameter.always_take_advice:
                            if rolloutWorker.env.agent_loc[agent_id] in rolloutWorker.env.rubble_loc:
                                ignore_advice = True

                        if not ignore_advice:
                            if parameter.exp_type == '1-dt+nn' or parameter.exp_type == '2-no_advice_source':
                                if rolloutWorker.teacher.determine_give_advice(state_importance, teacher_action, obs[agent_id], agent_id, student_action, evaluate):
                                    action = teacher_action
                                    if int(teacher_action) == clf_action:
                                        if np.max(action_prob) > 0.5 and clf_action == np.argmax(action_prob): #confident on this selected action
                                            agreed_obs[agent_id].append(obs[agent_id])
                                            agreed_act[agent_id].append(clf_action)

                            elif parameter.exp_type == '3-dt':
                                if rolloutWorker.teacher.determine_give_advice(state_importance, clf_action, obs[agent_id], agent_id, student_action, evaluate):
                                    action = clf_action
                                    if np.max(action_prob) > 0.5 and clf_action == np.argmax(action_prob): #confident on this selected action
                                        agreed_obs[agent_id].append(obs[agent_id])
                                        agreed_act[agent_id].append(clf_action)

                    if parameter.allow_student_explpre_initially:
                        if parameter.exp_type == '2-no_advice_target' and rolloutWorker.env.agent_loc[agent_id] in rolloutWorker.env.rubble_loc:#give random explore for the second case.
                            if episode_num != None:
                                if episode_num < 20500:#simple room pretrain is 20000.
                                    action = np.random.choice(rolloutWorker.args.n_actions, 1)[0]

                elif parameter.advice_algo == 'heuristic_with_no_memory':
                   action = student_action
                   teacher_action = teacher_actions[agent_id]
                   if rolloutWorker.teacher.determine_give_advice(state_importance, teacher_action, obs[agent_id], agent_id, student_action, evaluate):
                       action = teacher_action

                elif parameter.advice_algo == 'random action':
                    action = np.random.choice(rolloutWorker.args.n_actions, 1)[0]
                    state_importance = 0.1

                else:
                    print ('unknown advising algorithm, stop...')

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
    if evaluate and episode_num == rolloutWorker.args.evaluate_epoch - 1 and rolloutWorker.args.replay_dir != '':
        rolloutWorker.env.save_replay()
        rolloutWorker.env.close()

    # print("Teacher: " + str(teacher_count) + ". Student: " + str(student_count) + ". Ignore: " + str(ignore_count) + ". Policy: " + str(policy_count) + ". Out of budget1: " + str(out_of_budget_count) + ". Out of budget2: " + str(out_of_budget2))

    return episode, episode_reward, o, a, si, advisor_agree_rate, agreed_obs, \
    agreed_act, 1.0, step, follow_dt_prob
