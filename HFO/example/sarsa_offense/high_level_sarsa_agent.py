#!/usr/bin/env python3
# encoding: utf-8

from hfo import *
import argparse
import numpy as np
import sys, os
import shutil
import pickle

from policy_extraction import tree_helper

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sarsa_libraries','python_wrapper'))
from py_wrapper import *
from sarsa_util import *


def load_dt_clf(args, NA, use_saved=False):
    best_clf_path = "example/sarsa_offense/policy_extraction/best_clf_p" + str(args.suffix) + ".txt"
    VIPER_data_path = 'example/sarsa_offense/policy_extraction/D_p' + str(args.suffix) + '/D_iter_1.txt'
    with open(VIPER_data_path, "rb") as fp:   # Unpickling
        D = pickle.load(fp)

    if use_saved:
        with open(best_clf_path, "rb") as fp:   # Unpickling
            best_clf = pickle.load(fp)
    else: #generate one from the D file given by the viper.py and save it
        # can pick the best clf trained to improve, but not most important
        clf = tree_helper.train_decision_tree_policy(D, NA)
        best_clf = clf
        best_clf_file = open(best_clf_path, "wb")
        pickle.dump(best_clf, best_clf_file)
        best_clf_file.close

    return best_clf, D





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000)
  parser.add_argument('--numTeammates', type=int, default=0)
  parser.add_argument('--numOpponents', type=int, default=1)
  parser.add_argument('--numEpisodes', type=int, default=1)
  parser.add_argument('--learnRate', type=float, default=0.1)
  parser.add_argument('--suffix', type=int, default=0)

  parser.add_argument('--advice_budget', type=int, default=300) #300 ###set to zero can be training from scratch
  parser.add_argument('--advice_strategy', type=str, default='Early') #'Early', 'Alternative', 'Importance', 'MistakeCorrecting', 'NoAdvise', 'AlwaysAdvise'
  parser.add_argument('--use_EAA', type=bool, default=False)


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

  # student load with no weight file...
  # student = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, '', wt_filename+args.advice_strategy+'_student_training')
  # student.singleSaveWeights()

  # load weights alternatively to simulate a teacher and a student
  tmp_load_file = wt_filename + '_tmp_load'
  teacher_file = 'teacher_weights/2v2orig_script/'+wt_filename
  save_file = wt_filename+args.advice_strategy+'_student_training'

  shutil.copyfile('student_init_weights/' + wt_filename + '_student_init', tmp_load_file)
  student = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, tmp_load_file, save_file)

  print ('args.numEpisodes: ', args.numEpisodes)
  advice_budget = np.copy(args.advice_budget)

  if args.use_EAA:
    best_clf, D = load_dt_clf(args, NA, use_saved=True)
    # # test to see if this dt clf is good or not
    # rewards = []
    # num_test = 100
    # reward = 0
    # for t in range(num_test):
    #     status, reward, _ = gen_dt_episode(best_clf, reward, hfo, discFac, student, NF, NOT, NOO, NA, episode_num=None, evaluate=False)
    #     # print ('reward', reward)
    #     rewards.append(reward)
    # print ('np.mean(rewards)', np.mean(rewards))
    follow_dt_prob = 1.0
    new_clf = None
    paths = []


  reward = 0
  rewards = []
  call_counter = 0

  record_count = 0

  for episode in range(1,args.numEpisodes+1):
    if advice_budget > 0:
        np.save(wt_filename+args.advice_strategy+'_budget_used_up', np.array([episode]))

    if args.use_EAA:
        video_delay = False
        # if episode > 500:
        # if advice_budget == 0:
        # if advice_budget > 0:
        if True:
            record_count += 1
            video_delay = True
            if record_count > 10:
                break
        status, reward, student, advice_budget, call_counter, \
        follow_dt_prob, new_dt_paths = gen_EAA_episode(reward,
        hfo, discFac, student, advice_budget, args.advice_strategy,
        tmp_load_file, teacher_file, save_file, call_counter, NF, NOT, NOO, NA,
        best_clf, new_clf, follow_dt_prob, eval=False, video_delay=video_delay)

        paths += new_dt_paths

        if len(new_dt_paths) > 0:
            node_dict = tree_helper.collect_clf_paths(paths)
            new_clf = tree_helper.build_sub_tree(node_dict, paths)
            # print ('new_clf.n_nodes', new_clf.n_nodes)
    else:
        status, reward, student, advice_budget, call_counter = gen_AA_episode(reward,
        hfo, discFac, student, advice_budget, args.advice_strategy,
        tmp_load_file, teacher_file, save_file, call_counter, NF, NOT, NOO, eval=False)

    print ('episode', episode, 'reward', reward)
    rewards.append(reward)

    np.save(wt_filename+args.advice_strategy+'_rewards', np.array(rewards))
    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      print ('server down')
      break
