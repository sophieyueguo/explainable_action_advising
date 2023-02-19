import numpy as np
import pickle
from sklearn import tree

from policy_extraction import simulate
from policy_extraction import util
import experiment_parameter as parameter



def resample(D, teacher, n_steps_to_sample=2000):
    loss = []
    prob = [[] for j in range(len(D[0][0]))] # each agent maintains a probability
    for i in range(len(D)):
        oi, ai, si = D[i][0], D[i][1], D[i][2]
        for j in range(len(si)):
            prob[j].append(si[j])

    D_prime = [[[] for k in range(3) ] for i in range (n_steps_to_sample)]
    for j in range(len(prob)):
        norm = [float(i)/sum(prob[j]) for i in prob[j]]
        prob[j] = norm

        resample_idx = np.random.choice(len(D), n_steps_to_sample, p=prob[j])
        for i in range(n_steps_to_sample):
            for k in range(3):
                D_prime[i][k].append(D[resample_idx[i]][k][j])
    return D_prime



def run_viper(use_saved_data=False, data_filter_ratio=parameter.data_filter_ratio):
    D = []
    best_avg_reward = 0
    for iter in range(parameter.VIPER_max_iter):
        print ()
        print ('iter', iter)
        if iter == 0:
            if not use_saved_data:
                D_0, rolloutWorker = simulate.sample_from_teacher(data_filter_ratio=data_filter_ratio, n_steps_to_sample=100000) #10000 for simple mission room
                print ('len(D_0)', len(D_0))
                with open("policy_extraction/D/D_0.txt", "wb") as fp:   #Pickling
                    pickle.dump(D_0, fp)

            _, rolloutWorker = simulate.sample_from_teacher(data_filter_ratio=data_filter_ratio, n_steps_to_sample=1)

            with open("policy_extraction/D/D_0.txt", "rb") as fp:   # Unpickling
                D = pickle.load(fp)

        D_prime = resample(D, rolloutWorker.teacher, n_steps_to_sample=parameter.VIPER_n_steps_to_sample)
        clfs = util.train_decision_tree_policy(D_prime)

        avg_reward = 0
        num_test = 100
        D_i = []
        for t in range(num_test):
            reward, o, a, si = util.generate_decision_tree_episode(rolloutWorker, clfs, episode_num=None, evaluate=False)
            avg_reward += reward

            if parameter.env_type == 'Simple_4_Room':
                if len(o) <= rolloutWorker.env.episode_limit * data_filter_ratio: # only sample the efficient ones:
                    for i in range(len(o)):
                        D_i.append([o[i], a[i], si[i]])
            elif parameter.env_type == '14_Room':
                if reward > 20:
                    for i in range(len(o)):
                        D_i.append([o[i], a[i], si[i]])
            else:
                assert False==True, 'Unknown Environment'

        avg_reward /= num_test
        print ('avg_reward', avg_reward)
        if avg_reward >= best_avg_reward:
            best_avg_reward = avg_reward

        D = D + D_i
        print('len(D)', len(D))
        with open("policy_extraction/D/D_iter_" + str(iter) + ".txt", "wb") as fp:   #Pickling
            pickle.dump(D_i, fp) #save D_i instead of the whole D, and thus saving the good clfs.

    print ('best_avg_reward', best_avg_reward)


if __name__ == '__main__':
    run_viper(use_saved_data=False)
