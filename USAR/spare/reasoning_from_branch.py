from agent.agent import Agents
from policy_extraction import simulate
from policy_extraction import util
from policy_extraction import tree_helper
import import experiment_parameter as parameter


import numpy as np
import pickle
from sklearn import tree





def generate_student_with_memory_episode(rolloutWorker, clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=None, evaluate=False):
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

    advisor_agree_rate = 0
    advisor_agree_count = 0
    memory_advisor_agree_rate = 0
    memory_advisor_count = 0

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
        #teacher_actions = []

        for agent_id in range(rolloutWorker.n_agents):
            avail_action = rolloutWorker.env.get_avail_agent_actions(agent_id)
            #teacher_action, state_importance, _ = rolloutWorker.teacher.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
            #                                       avail_action, epsilon, evaluate)
            # state_importances.append(state_importance)
            #teacher_actions.append(teacher_action)


        for agent_id in range(rolloutWorker.n_agents):
            avail_action = rolloutWorker.env.get_avail_agent_actions(agent_id)
            if rolloutWorker.args.alg == 'maven':
                print ('not handled algo, stop')
            else:
                # teacher_action = teacher_actions[agent_id]
                # state_importance = state_importances[agent_id]
                student_action, state_importance, _ = student.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                state_importances.append(state_importance)
                action = student_action

                if parameter.advice_algo == 'heuristic_with_tree_memory':

                    clf_prob = clfs[agent_id].predict_proba([obs[agent_id]])

                    action_prob = [clf_prob[0][ai] * avail_action[ai] for ai in range(rolloutWorker.args.n_actions)]
                    if sum(action_prob) == 0:
                        clf_action = np.random.choice(rolloutWorker.args.n_actions, 1)[0]
                    else:
                        if sum(action_prob) != 1:
                            action_prob = action_prob/sum(action_prob)
                        clf_action = np.random.choice(rolloutWorker.args.n_actions, 1, p=action_prob)[0]



                    ############# try out ########################
                    if len(new_clfs) > 0:
                        new_clf_action = new_clfs[agent_id].predict(obs[agent_id])
                        if new_clf_action != 'undecided' and np.random.rand() < follow_dt_prob:
                            action = new_clf_action
                            if not evaluate:
                                follow_dt_prob *= parameter.decay_ratio
                    elif rolloutWorker.teacher.advice_budget > 0:
                        #if rolloutWorker.env.roles[agent_id] == 'engineer' and rolloutWorker.env.agent_loc[agent_id] in rolloutWorker.env.rubble_loc:
                            #action = student_action
                        ignore_advice = False
                        if rolloutWorker.teacher.advice_budget > 0:
                            if rolloutWorker.env.roles[agent_id] == 'engineer':
                                if [obs[agent_id], clf_action] in X_Y[agent_id]:

                                    agreed_obs_act_ind = [X_Y[agent_id].index([obs[agent_id], clf_action])]
                                    paths = tree_helper.collect_clf_paths(clfs[agent_id], agreed_obs_act_ind, X[agent_id], Y[agent_id], printline=False)
                                    room_f = rolloutWorker.env.agent_loc[agent_id] * 4 + 1
                                    for node in paths[1][0]:
                                            # no rubble in agent's room
                                        if not node['is_leaf']:
                                            if node['feature'] == room_f and node['value'] == 1:
                                                    # print ('shoud ignore...')
                                                ignore_advice = True
                        if ignore_advice:
                            action = student_action

                        else:
                            if rolloutWorker.teacher.determine_give_advice(state_importance, clf_action, obs[agent_id], agent_id, student_action, evaluate):
                                action = clf_action
                                if np.max(action_prob) > 0.5 and clf_action == np.argmax(action_prob): #confident on this selected action
                                    agreed_obs[agent_id].append(obs[agent_id])
                                    agreed_act[agent_id].append(clf_action)

                            else:
                                action = student_action
                    else:
                        action = student_action
                    #############################################

                    #if len(new_clfs) > 0:

                        # internal tree knowledge should also be rejected if transferred from new env.
                    #    ignore_advice = False
                    #    new_clf_action = new_clfs[agent_id].predict(obs[agent_id])

                        # if rolloutWorker.env.roles[agent_id] == 'engineer':
                        #    if [obs[agent_id], new_clf_action] in X_Y[agent_id]:

                        #        agreed_obs_act_ind = [X_Y[agent_id].index([obs[agent_id], new_clf_action])]
                        #        paths = tree_helper.collect_clf_paths(clfs[agent_id], agreed_obs_act_ind, X[agent_id], Y[agent_id], printline=False)
                        #        room_f = rolloutWorker.env.agent_loc[agent_id] * 4 + 1
                        #        for node in paths[1][0]:
                                    # no rubble in agent's room
                        #            if not node['is_leaf']:
                        #                if node['feature'] == room_f and node['value'] == 1:
                                            # print ('shoud ignore...')
                        #                    ignore_advice = True


                        # the percentage of this obs has not been seen but maybe used to output an action
                        # that is the same as the teacher.
                   #     if not ignore_advice and new_clf_action != 'undecided' and np.random.rand() < follow_dt_prob: # check memory first
                   #         action = new_clf_action
                   #         count_calc_advice_subdt[agent_id] += 1

                   #         if str(obs[agent_id]) not in d_observed_state[agent_id]:
                   #             count_calc_advice_unobserved[agent_id] += 1
                   #         if not evaluate:
                   #             follow_dt_prob *= parameter.decay_ratio


                   #     else: # otherwise, see if should ask teacher for advice:
                   #         ignore_advice = False
                   #         if rolloutWorker.teacher.advice_budget > 0:
                   #             if rolloutWorker.env.roles[agent_id] == 'engineer':
                   #                 if [obs[agent_id], clf_action] in X_Y[agent_id]:

                   #                     agreed_obs_act_ind = [X_Y[agent_id].index([obs[agent_id], clf_action])]
                   #                     paths = tree_helper.collect_clf_paths(clfs[agent_id], agreed_obs_act_ind, X[agent_id], Y[agent_id], printline=False)
                   #                     room_f = rolloutWorker.env.agent_loc[agent_id] * 4 + 1
                   #                     for node in paths[1][0]:
                                            # no rubble in agent's room
                   #                         if not node['is_leaf']:
                   #                             if node['feature'] == room_f and node['value'] == 1:
                   #                                 # print ('shoud ignore...')
                   #                                 ignore_advice = True
                   #         if ignore_advice:
                   #             action = student_action

                                        # if not node['is_leaf']:
                                        #     print(
                                        #         "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
                                        #         "{inequality} {threshold})".format(
                                        #             node=node['node'],
                                        #             sample=node['sample'],
                                        #             feature=node['feature'],
                                        #             value=node['value'],
                                        #             inequality=node['inequality'],
                                        #             threshold=node['threshold'],
                                        #         )
                                        #
                                        #     )
                                        # else:
                                        #     print ("leaf node {node} : Y[{sample}] representing value {value}".format(
                                        #                 node=node['node'],
                                        #                 sample=node['sample'],
                                        #                 value=node['value']
                                        #         )
                                        #     )

                            # if rolloutWorker.env.roles[agent_id] == 'engineer' and \
                            # rolloutWorker.env.agent_loc[agent_id] in rolloutWorker.env.rubble_loc:
                            #     # print ('advice on rubble...')
                            #     # print ('agent_id', agent_id)
                            #     # print ('rolloutWorker.env.agent_loc[agent_id]', rolloutWorker.env.agent_loc[agent_id])
                            #     # print ('rolloutWorker.env.rubble_loc', rolloutWorker.env.rubble_loc)
                            #     # print ('teacher_action', teacher_action)
                            #
                            #
                            #     # action = np.random.choice(rolloutWorker.args.n_actions, 1)[0]
                            #     # or student action?
                            #     # action = 5
                            #     action = student_action

                 #           elif parameter.take_teacher_advice and rolloutWorker.teacher.determine_give_advice(state_importance, teacher_actions, obs[agent_id], agent_id, student_action, evaluate):
                                # action = teacher_action
                                # if int(teacher_action) == clf_action:
                 #               action = clf_action
                 #               if True:

                 #                   if np.max(action_prob) > 0.3 and clf_action == np.argmax(action_prob): #confident on this selected action
                 #                       agreed_obs[agent_id].append(obs[agent_id])
                 #                       agreed_act[agent_id].append(clf_action)

                                        # #########################################
                                        # # examine how long this path is?
                                        # if [obs[agent_id], clf_action] in X_Y[agent_id]:
                                        #     agreed_obs_act_ind = [X_Y[agent_id].index([obs[agent_id], clf_action])]
                                        #     _, paths = tree_helper.collect_clf_paths(clfs[agent_id], agreed_obs_act_ind, X[agent_id], Y[agent_id], printline=False)
                                        #     # remove more budget count...
                                        #     rolloutWorker.teacher.advice_budget -= (len(paths[0])-1)
                                        #     # don't do that...
                                        #     if rolloutWorker.teacher.advice_budget < 0:
                                        #         print ('advice budget used up!')
                                        #         rolloutWorker.teacher.advice_budget = 0

                #                        d_observed_state[agent_id].add(str(obs[agent_id]))
                                        #########################################
                #            else:
                #                action = student_action
                #    else:
                #        action = student_action

                #elif parameter.advice_algo == 'heuristic_with_no_memory':
                #    action = student_action
                #    if rolloutWorker.teacher.determine_give_advice(state_importance, teacher_actions, obs[agent_id], agent_id, student_action, evaluate):
                #        action = teacher_action

                elif parameter.advice_algo == 'random action':
                    action = np.random.choice(rolloutWorker.args.n_actions, 1)[0]
                    state_importance = 0.1

                else:
                    print ('unknown advising algorithm, stop...')









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

    return episode, episode_reward, o, a, si, advisor_agree_rate, agreed_obs, \
    agreed_act, 1.0, step, count_calc_advice_subdt, count_calc_advice_unobserved, \
    d_observed_state, follow_dt_prob




















if __name__ == '__main__':
    _, rolloutWorker = simulate.sample_from_teacher(data_filter_ratio=1.0, n_steps_to_sample=1)
    with open(parameter.VIPER_data_path, "rb") as fp:   # Unpickling
        D = pickle.load(fp)

    if not parameter.resume_training:
        best_clfs = None
        best_avg_reward = 0
        for i in range(5):
            # pick the relatively best tree to recover the best clf
            clfs = util.train_decision_tree_policy(D)
            avg_reward = 0
            num_test = 50
            for t in range(num_test):
                reward, o, a, si = util.generate_decision_tree_episode(rolloutWorker, clfs, episode_num=None, evaluate=False)
                avg_reward += reward
            avg_reward /= num_test
            print ('select the best clf i', i, 'avg_reward', avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_clfs = clfs
        print ('best_avg_reward', best_avg_reward)

        best_clf_file = open(parameter.save_path + "best_clfs.txt", "wb")
        pickle.dump(best_clfs, best_clf_file)
        best_clf_file.close
    else:
        with open(parameter.save_path + 'best_clfs.txt', 'rb') as f:
            best_clfs = pickle.load(f)



    print_tree = False
    if print_tree:
        for ai in range(rolloutWorker.n_agents):
            tree_helper.vis_decision_tree(best_clfs[ai])
            print ('********************')
            tree_helper.basic_info(best_clfs[ai])





    #------------------------------------------------------------------------

    all_agreed_obs_act_ind = [[] for agent_id in range(rolloutWorker.n_agents)]

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





    time_steps, train_steps, evaluate_steps = 0, 0, -1
    trial_index = 0
    episode_rewards = []
    win_rates = []
    new_clfs = []
    tree_node_counts = [[] for ai in range(rolloutWorker.n_agents)]

    # to count the subtree functionality
    count_calc_advice_subdt = [0] * rolloutWorker.n_agents
    count_calc_advice_unobserved = [0] * rolloutWorker.n_agents
    d_observed_state = [set([]) for ai in range(rolloutWorker.n_agents)]


    follow_dt_prob = 1.



    student = Agents(rolloutWorker.args)
    if parameter.resume_training:
        path_rnn = parameter.resume_model_path + '_rnn_params.pkl'
        path_coma = parameter.resume_model_path + '_critic_params.pkl'
        student.load_model(path_rnn, path_coma)

        for ai in range(rolloutWorker.n_agents):
            tree_node_counts[ai] = list(np.load(parameter.save_path + 'node_counts_'+str(ai) + '.npy'))
        episode_rewards = list(np.load(parameter.save_path + 'episode_rewards.npy', episode_rewards))
        print ('loaded episode_rewards', episode_rewards)


        rolloutWorker.teacher.advice_budget = np.load(parameter.save_path + 'advice_budget.npy')
        follow_dt_prob = np.load(parameter.save_path + 'follow_dt_prob.npy')

        with open(parameter.save_path + 'all_agreed_obs_act_ind.txt', 'rb') as f:
            all_agreed_obs_act_ind = pickle.load(f)
        time_steps, train_steps, evaluate_steps, trial_index = np.load(parameter.save_path + 'steps_indices.npy')

        with open(parameter.save_path + 'new_clfs.txt', 'rb') as f:
            new_clfs = pickle.load(f)



    else:

        # test init performance...
        avg_reward = 0
        for i in range(50):
            _, episode_reward, _, _, _, _, _, _, _, _, _, _, _, _ = generate_student_with_memory_episode(rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=None, evaluate=True)
            avg_reward += episode_reward
        avg_reward /= 50
        episode_rewards.append(avg_reward)
        np.save(parameter.save_path + 'episode_rewards', episode_rewards)

        if rolloutWorker.teacher.advice_budget > 0:
            np.save(parameter.save_path + 'trials_budget_used_up', [trial_index])
        print ('init evaluate, avg_reward', avg_reward)




    while trial_index < parameter.max_trial_index:
        if trial_index % parameter.evaluate_trial_cycle == 0:
            # Update the tree memory
            if trial_index > 1 and parameter.advice_algo == 'heuristic_with_tree_memory':
                if rolloutWorker.teacher.advice_budget > 0: # otherwise don't update the tree when budget is used up...
                    print ('training a new subtree...')
                    new_clfs = []
                    for ai in range(rolloutWorker.n_agents):
                        node_dict, paths = tree_helper.collect_clf_paths(best_clfs[ai], all_agreed_obs_act_ind[ai], X[ai], Y[ai])
                        new_clf = tree_helper.build_sub_tree(node_dict, paths)
                        new_clfs.append(new_clf)
                        print ('tree size', new_clfs[ai].n_nodes)

                        tree_node_counts[ai].append(new_clfs[ai].n_nodes)
                        np.save(parameter.save_path + 'node_counts_'+str(ai), tree_node_counts[ai])

                    # save the D for internal dt.
                    all_agreed_obs_act_ind_file = open(parameter.save_path + "all_agreed_obs_act_ind.txt", "wb")
                    pickle.dump(all_agreed_obs_act_ind, all_agreed_obs_act_ind_file)
                    all_agreed_obs_act_ind_file.close

                    new_clf_file = open(parameter.save_path + "new_clfs.txt", "wb")
                    pickle.dump(new_clfs, new_clf_file)
                    new_clf_file.close



            # Test performance
            avg_reward = 0
            for i in range(50):
                _, episode_reward, _, _, _, _, _, _, _, _, _, _, _, _ = generate_student_with_memory_episode(rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=None, evaluate=True)
                avg_reward += episode_reward
            avg_reward /= 50
            episode_rewards.append(avg_reward)

            np.save(parameter.save_path + 'episode_rewards', episode_rewards)
            np.save(parameter.save_path + 'advice_budget', rolloutWorker.teacher.advice_budget)
            np.save(parameter.save_path + 'follow_dt_prob', follow_dt_prob)
            if rolloutWorker.teacher.advice_budget > 0:
                np.save(parameter.save_path + 'trials_budget_used_up', [trial_index])

            if trial_index % parameter.save_model_trial_cycle == 0:
                student.policy.save_model(trial_index)

            print ('\nevaluate, avg_reward', avg_reward)
            print ('trial_index', trial_index, 'budget left', rolloutWorker.teacher.advice_budget, 'data for tree len(all_agreed_obs_act_ind[0])', len(all_agreed_obs_act_ind[0]))
            print ('len(d_observed_state)', len(d_observed_state))
            # print ('count_calc_advice_subdt', count_calc_advice_subdt)
            # print ('count_calc_advice_unobserved', count_calc_advice_unobserved)
            print ('follow_dt_prob', follow_dt_prob)
            print ()
            print ()


        trial_index += 1
        print('trial_index', trial_index, end="\r")
        np.save(parameter.save_path + 'steps_indices', [time_steps, train_steps, evaluate_steps, trial_index])


        episodes = []
        for episode_idx in range(rolloutWorker.args.n_episodes):
            episode, _, _, _, _, _, agreed_obs, agreed_act, _, steps, count_calc_advice_subdt, count_calc_advice_unobserved, d_observed_state, follow_dt_prob = generate_student_with_memory_episode(rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=None, evaluate=False)
            episodes.append(episode)
            time_steps += steps

            if rolloutWorker.teacher.advice_budget > 0 and parameter.advice_algo == 'heuristic_with_tree_memory': # add info to the memory of student
                for ai in range(rolloutWorker.n_agents):
                    for t in range(len(agreed_obs[ai])):
                        o, a = agreed_obs[ai][t], agreed_act[ai][t]
                        if [o, a] in X_Y[ai]:
                            all_agreed_obs_act_ind[ai].append(X_Y[ai].index([o, a]))

        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
        if rolloutWorker.args.alg.find('coma') > -1 or rolloutWorker.args.alg.find('central_v') > -1 or rolloutWorker.args.alg.find('reinforce') > -1:
            student.train(episode_batch, train_steps, rolloutWorker.epsilon)
            train_steps += 1
        else:
            print ('Not coma, stop...')
            break
