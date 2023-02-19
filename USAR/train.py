from agent.agent import Agents
from policy_extraction import simulate
from policy_extraction import util
from policy_extraction import tree_helper
import experiment_parameter as parameter

from EAA import *
from eval import *

import numpy as np
import pickle
from sklearn import tree


if __name__ == '__main__':
    _, rolloutWorker = simulate.sample_from_teacher(data_filter_ratio=1.0, n_steps_to_sample=1)
    with open(parameter.VIPER_data_path, "rb") as fp:   # Unpickling
        D = pickle.load(fp)

    if not parameter.resume_training and not parameter.load_pretrain_saved_reults:
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
        if parameter.resume_training:
            with open(parameter.load_path + 'best_clfs.txt', 'rb') as f:
                best_clfs = pickle.load(f)
        if parameter.load_pretrain_saved_reults:
            with open(parameter.pretrain_saved_reults_data_path + 'best_clfs.txt', 'rb') as f:
                best_clfs = pickle.load(f)



    print_tree = False
    if print_tree:
        for ai in range(rolloutWorker.n_agents):
            tree_helper.vis_decision_tree(best_clfs[ai])
            print ('********************')
            tree_helper.basic_info(best_clfs[ai])

    #---------------------------------------------
    X, Y, X_Y = util.process_loaded_data(rolloutWorker, D)
    all_agreed_obs_act_ind = [[] for agent_id in range(rolloutWorker.n_agents)]

    time_steps, train_steps, evaluate_steps = 0, 0, -1
    trial_index = 0
    episode_rewards = []
    win_rates = []
    new_clfs = []
    tree_node_counts = [[] for ai in range(rolloutWorker.n_agents)]


    follow_dt_prob = 1.
    student = Agents(rolloutWorker.args)
    if parameter.resume_training and not parameter.load_pretrain_saved_reults:
        path_rnn = parameter.resume_model_path + '_rnn_params.pkl'
        path_coma = parameter.resume_model_path + '_critic_params.pkl'
        student.load_model(path_rnn, path_coma)

        for ai in range(rolloutWorker.n_agents):
            tree_node_counts[ai] = list(np.load(parameter.load_path + 'node_counts_'+str(ai) + '.npy'))
        episode_rewards = list(np.load(parameter.load_path + 'episode_rewards.npy', episode_rewards))
        print ('loaded episode_rewards', episode_rewards)


        rolloutWorker.teacher.advice_budget = np.load(parameter.load_path + 'advice_budget.npy')
        follow_dt_prob = np.load(parameter.load_path + 'follow_dt_prob.npy')

        with open(parameter.load_path + 'all_agreed_obs_act_ind.txt', 'rb') as f:
            all_agreed_obs_act_ind = pickle.load(f)
        time_steps, train_steps, evaluate_steps, trial_index = np.load(parameter.load_path + 'steps_indices.npy')

        with open(parameter.load_path + 'new_clfs.txt', 'rb') as f:
            new_clfs = pickle.load(f)


    elif parameter.load_pretrain_saved_reults:
        path_rnn = parameter.pretrain_saved_reults_model_path + '_rnn_params.pkl'
        path_coma = parameter.pretrain_saved_reults_model_path + '_critic_params.pkl'
        student.load_model(path_rnn, path_coma)

        for ai in range(rolloutWorker.n_agents):
            tree_node_counts[ai] = list(np.load(parameter.pretrain_saved_reults_data_path + 'node_counts_'+str(ai) + '.npy'))
        episode_rewards = list(np.load(parameter.pretrain_saved_reults_data_path + 'episode_rewards.npy', episode_rewards))
        print ('loaded episode_rewards', episode_rewards)
        episode_rewards = [] #clean up the previous ones and save the current

        rolloutWorker.teacher.advice_budget = np.load(parameter.pretrain_saved_reults_data_path + 'advice_budget.npy')
        follow_dt_prob = np.load(parameter.pretrain_saved_reults_data_path + 'follow_dt_prob.npy')

        with open(parameter.pretrain_saved_reults_data_path + 'all_agreed_obs_act_ind.txt', 'rb') as f:
            all_agreed_obs_act_ind = pickle.load(f)

        with open(parameter.pretrain_saved_reults_data_path + 'new_clfs.txt', 'rb') as f:
            new_clfs = pickle.load(f)
    else:
        # initial evaluate here.
        test_init_performance(episode_rewards, all_agreed_obs_act_ind, rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, trial_index)


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

            # test here, for each evaluation cycle
            test_performance(episode_rewards, all_agreed_obs_act_ind, rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, trial_index)

        trial_index += 1
        print('trial_index', trial_index, end="\r")
        np.save(parameter.save_path + 'steps_indices', [time_steps, train_steps, evaluate_steps, trial_index])


        episodes = []
        for episode_idx in range(rolloutWorker.args.n_episodes):
            episode, _, _, _, _, _, agreed_obs, agreed_act, _, steps, follow_dt_prob = generate_student_with_memory_episode(rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=trial_index, evaluate=False)
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
