#!/usr/bin/env python3
# encoding: utf-8

from hfo import *
import argparse
import numpy as np
import sys, os
import shutil
import pickle

import tree_helper

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'sarsa_libraries','python_wrapper'))
from py_wrapper import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sarsa_util import *



def sample_from_teacher(hfo, discFac, teacher, NF, NOT, NOO, data_filter_ratio=0, n_steps_to_sample=10):
    reward = 0
    D_i = []
    # for episode in range(1,2):
    episode = 0
    while len(D_i) < n_steps_to_sample:
        status, reward, D_epi = gen_teacher_episode(reward, hfo, discFac, teacher, NF, NOT, NOO)
        if reward == 1:
            D_i += D_epi
            print ('good episode', episode, 'reward', reward)
            episode += 1

        # Quit if the server goes down
        if status == SERVER_DOWN:
            hfo.act(QUIT)
            print ('episode quit')
            break
    return D_i




def resample(D, n_steps_to_sample=0):
    # print ()
    # print ('D')
    # for d in D:
    #     print (d)

    loss = []
    prob = [] # each agent maintains a probability
    for i in range(len(D)):
        oi, ai, si = D[i][0], D[i][1], D[i][2]
        prob.append(si)

    D_prime = [[] for i in range (n_steps_to_sample)]
    norm = [float(i)/sum(prob) for i in prob]
    prob = norm

    resample_idx = np.random.choice(len(D), n_steps_to_sample, p=prob)
    for i in range(n_steps_to_sample):
        for k in range(3):
            D_prime[i].append(D[resample_idx[i]][k])
    # print ()
    # print ('D_prime')
    # for d in D_prime:
    #     print (d)
    return D_prime






def run_viper(hfo, discFac, teacher, NF, NOT, NOO, NA, save_D_dir, use_saved_data=False, VIPER_max_iter=10):
    D = []
    best_avg_reward = 0
    for iter in range(VIPER_max_iter):
        print ()
        print ('iter', iter)
        if iter == 0:
            if not use_saved_data:
                D_0 = sample_from_teacher(hfo, discFac, teacher, NF, NOT, NOO, data_filter_ratio=0, n_steps_to_sample=1000) #10000 for simple mission room
                print ('len(D_0)', len(D_0))


                with open(save_D_dir + "/D_0.txt", "wb") as fp:   #Pickling
                    pickle.dump(D_0, fp)

            with open(save_D_dir + "/D_0.txt", "rb") as fp:   # Unpickling
            # each has a separate txt
                D = pickle.load(fp)
                # print ('len(D)', len(D))
                # for d in D:
                #     print (d)

        D_prime = resample(D, n_steps_to_sample=500)
        # D_prime = D
        # print ('len(D_prime)', len(D_prime))
        clf = tree_helper.train_decision_tree_policy(D_prime, NA)
        print ('trained a dt')

        rewards = []
        num_test = 20
        D_i = []

        reward = 0
        for t in range(num_test):
            status, reward, D_epi = gen_dt_episode(clf, reward, hfo, discFac, teacher, NF, NOT, NOO, NA, episode_num=None, evaluate=False)
            print ('reward', reward)
            rewards.append(reward)
            if reward == 1:
                D_i += D_epi

            # Quit if the server goes down
            if status == SERVER_DOWN:
              hfo.act(QUIT)
              # break

        print ('iter', iter, 'reward mean', np.mean(rewards))

        if np.mean(rewards) >= best_avg_reward:
            best_avg_reward = np.mean(rewards)

        D = D + D_i
        print('len(D)', len(D))
        with open(save_D_dir + "/D_iter_" + str(iter) + ".txt", "wb") as fp:   #Pickling
            pickle.dump(D_i, fp) #save D_i instead of the whole D, and thus saving the good clfs.

    print ('best_avg_reward', best_avg_reward)






if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000)
  parser.add_argument('--numTeammates', type=int, default=0)
  parser.add_argument('--numOpponents', type=int, default=1)
  parser.add_argument('--numEpisodes', type=int, default=1)
  parser.add_argument('--learnRate', type=float, default=0.1)
  parser.add_argument('--suffix', type=int, default=0)

  args=parser.parse_args()

  # Create the HFO Environment
  hfo = HFOEnvironment()
  #now connect to the server
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,'bin/teams/base/config/formations-dt',args.port,'localhost','base_left',False)
  # global NF,NA,NOT,NOO
  NOO=args.numOpponents
  if args.numOpponents >0:
    NF=4+4*args.numTeammates
  else:
    NF=3+3*args.numTeammates
  NOT=args.numTeammates
  NA=NOT+2 #PASS to each teammate, SHOOT, DRIBBLE
  learnR=args.learnRate
  #CMAC parameters
  resolution=0.1
  Range=[2]*NF
  Min=[-1]*NF
  Res=[resolution]*NF
  #Sarsa Agent Parameters
  wt_filename="weights_"+str(NOT+1)+"v"+str(NOO)+'_'+str(args.suffix)
  print ('wt_filename', wt_filename)
  discFac=1
  Lambda=0
  eps=0.01
  #initialize the function approximator and the sarsa agent
  FA=CMAC(NF, NA, Range, Min, Res)

  # load weights alternatively to simulate a teacher and a student
  teacher_file = 'teacher_weights/2v2orig_script/'+wt_filename
  save_file = wt_filename+'_viper'

  teacher = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, teacher_file, save_file)
  print ('created a teacher done')
  print ('args.numEpisodes: ', args.numEpisodes)

  save_D_dir = 'example/sarsa_offense/policy_extraction/D_p' + str(args.suffix)
  run_viper(hfo, discFac, teacher, NF, NOT, NOO, NA, save_D_dir, use_saved_data=False)
