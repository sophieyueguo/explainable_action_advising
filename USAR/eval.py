from EAA import *

import numpy as np

# Test performance
def test_performance(episode_rewards, all_agreed_obs_act_ind, rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, trial_index):
    avg_reward = 0
    for i in range(50):
        _, episode_reward, _, _, _, _, _, _, _, _, _  = generate_student_with_memory_episode(rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=trial_index, evaluate=True)
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
    print ('follow_dt_prob', follow_dt_prob)
    print ()
    print ()

# test init performance...
def test_init_performance(episode_rewards, all_agreed_obs_act_ind, rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, trial_index):
    avg_reward = 0
    for i in range(50):
        _, episode_reward, _, _, _, _, _, _, _, _, _ = generate_student_with_memory_episode(rolloutWorker, best_clfs, new_clfs, student, follow_dt_prob, X, Y, X_Y, episode_num=trial_index, evaluate=True)
        avg_reward += episode_reward
    avg_reward /= 50
    episode_rewards.append(avg_reward)
    np.save(parameter.save_path + 'episode_rewards', episode_rewards)

    if rolloutWorker.teacher.advice_budget > 0:
        np.save(parameter.save_path + 'trials_budget_used_up', [trial_index])
    print ('init evaluate, avg_reward', avg_reward)
