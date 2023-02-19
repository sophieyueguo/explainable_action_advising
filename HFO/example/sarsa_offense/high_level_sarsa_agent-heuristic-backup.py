#!/usr/bin/env python3
# encoding: utf-8

from hfo import *
import argparse
import numpy as np
import sys, os

import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sarsa_libraries','python_wrapper'))
from py_wrapper import *







def getReward(s):
  reward=0
  #---------------------------
  if s==GOAL:
    reward=1
  #---------------------------
  elif s==CAPTURED_BY_DEFENSE:
    reward=-1
  #---------------------------
  elif s==OUT_OF_BOUNDS:
    reward=-1
  #---------------------------
  #Cause Unknown Do Nothing
  elif s==OUT_OF_TIME:
    reward=0
  #---------------------------
  elif s==IN_GAME:
    reward=0
  #---------------------------
  elif s==SERVER_DOWN:
    reward=0
  #---------------------------
  else:
    print("Error: Unknown GameState", s)
  return reward

def purge_features(state):
  st=np.empty(NF,dtype=np.float64)
  stateIndex=0
  tmpIndex= 9 + 3*NOT
  for i in range(len(state)):
    # Ignore first six features and teammate proximity to opponent(when opponent is absent)and opponent features
    if(i < 6 or i>9+6*NOT or (NOO==0 and ((i>9+NOT and i<=9+2*NOT) or i==9)) ):
      continue;
    #Ignore Angle and Uniform Number of Teammates
    temp =  i-tmpIndex;
    if(temp > 0 and (temp % 3 == 2 or temp % 3 == 0)):
       continue;
    if (i > 9+6*NOT):
      continue;
    st[stateIndex] = state[i];
    stateIndex+=1;
  return st




# def take_teacher_action(student):







def determine_give_advice(advice_budget, advice_strategy, state_importance,
teacher_action, student_action, call_counter, eval):
    if advice_budget == 0:
        return False, 0

    if advice_budget > 0:
        if advice_strategy == 'Early':
            if not eval:
                advice_budget -= 1
            if advice_budget == 0:
                print ('advice budget used up!')
            return True, advice_budget

        elif advice_strategy == 'Alternative':
            alternative_advice_freq = 5
            if call_counter % alternative_advice_freq == 0:
                if not eval:
                    advice_budget -= 1
                if advice_budget == 0:
                    print ('advice budget used up!')
                return True, advice_budget
            else:
                return False, advice_budget

        elif advice_strategy == 'Importance':
            state_importance_threshold = 0.6
            if state_importance >= state_importance_threshold:
                if not eval:
                    advice_budget -= 1
                if advice_budget == 0:
                    print ('advice budget used up!')
                return True, advice_budget
            else:
                return False, advice_budget

        elif advice_strategy == 'MistakeCorrecting':
            state_importance_threshold = 0.3
            if state_importance >= state_importance_threshold and teacher_action != student_action:
                if not eval:
                    advice_budget -= 1
                if advice_budget == 0:
                    print ('advice budget used up!')
                return True, advice_budget
            else:
                return False, advice_budget
        else:
            print ('unknown advice strategy!')
            return False, advice_budget



# SARSA agent in c++ can only make once here
# let the student temporarily load teacher's weight and decide - still action advising
# to simulate the teacher's action advising
def get_teacher_action_advise(agent, st, tmp_load_file, teacher_file, save_file, calc_si=False):
  agent.singleSaveWeights()
  shutil.copyfile(teacher_file, tmp_load_file)
  agent.singleLoadWeights()
  teacher_action = agent.selectAction(st)
  state_importance = 0
  if calc_si:
      state_importance = agent.computeStateImportance(st)
  shutil.copyfile(save_file, tmp_load_file)
  agent.singleLoadWeights()
  return agent, teacher_action, state_importance





#episode rollouts
def gen_episode(reward, hfo, discFac, student, advice_budget, advice_strategy, use_EAA, tmp_load_file, teacher_file, save_file, call_counter, eval=False):
  st = np.empty(NF,dtype=np.float64)

  count=0
  status=IN_GAME
  action=-1



  while status==IN_GAME:
    count=count+1
    # Grab the state features from the environment
    state = hfo.getState()
    if int(state[5])==1:
      if action != -1:
        #print(st)
        reward=getReward(status)
        #fb.SA.update(state,action,reward,discFac)
        # SA.update(st,action,reward,discFac)
        if not eval:
          student.update(st,action,reward,discFac)
      st=purge_features(state)

      #take an action --old
      #action = fb.SA.selectAction(state)
      # action = SA.selectAction(st)
      #print("Action:", action)

      #'Early', 'Alternative', 'Importance', 'MistakeCorrecting', 'NoAdvise', 'AlwaysAdvise'
      if advice_strategy == 'NoAdvise':
          # train from scratch
        # print ('no advise')
        action = student.selectAction(st)

      elif advice_strategy == 'AlwaysAdvise':
          #let the student always takes teacher's action
        # action = teacher.selectAction(st)
        student, teacher_action = get_teacher_action_advise(student, st, tmp_load_file, teacher_file, save_file)
        action = teacher_action

      elif advice_strategy == 'Early' or advice_strategy == 'Alternative':
          # does not need state importance or teacher action or student action
        call_counter += 1
        should_advise, advice_budget = determine_give_advice(advice_budget, advice_strategy, 0, 0, 0, call_counter, eval)
        if should_advise:
            student, teacher_action, _ = get_teacher_action_advise(student, st, tmp_load_file, teacher_file, save_file, calc_si=False)
            action = teacher_action
        else:
            action = student.selectAction(st)

      elif advice_strategy == 'Importance':
          if advice_budget > 0:
              student, teacher_action, state_importance = get_teacher_action_advise(student, st, tmp_load_file, teacher_file, save_file, calc_si=True)
              # state_importance = teacher.computeStateImportance(st)
              # print ('high_level state_importance', state_importance)
              should_advise, advice_budget = determine_give_advice(advice_budget, 'Importance', state_importance, 0, 0, 0, eval)
          else:
              should_advise = False
          if should_advise:
              action = teacher_action
          else:
              action = student.selectAction(st)

      elif advice_strategy == 'MistakeCorrecting':
        student_action = student.selectAction(st) # student report its intended action
        if advice_budget > 0:
            # teacher_action = teacher.selectAction(st)
            # state_importance = teacher.computeStateImportance(st)
            # print ('high_level state_importance', state_importance)
            student, teacher_action, state_importance = get_teacher_action_advise(student, st, tmp_load_file, teacher_file, save_file, calc_si=True)
            should_advise, advice_budget = determine_give_advice(advice_budget, 'MistakeCorrecting', state_importance, teacher_action, student_action, 0, eval)
        else:
            should_advise = False
        if should_advise:
            action = teacher_action
        else:
            action = student_action


      if action == 0:
        hfo.act(SHOOT)
      elif action == 1:
        hfo.act(DRIBBLE)
      else:
        hfo.act(PASS,state[(9+6*NOT)-(action-2)*3])
    else:
      hfo.act(MOVE)

    status = hfo.step()
  #--------------- end of while loop ------------------------------------------------------

  ############# EPISODE ENDS ###################################################################################
  # Check the outcome of the episode
  if action != -1:
    reward=getReward(status)
    # SA.update(st, action, reward, discFac)
    # SA.endEpisode()
    if not eval:
      student.update(st, action, reward, discFac)
    student.endEpisode()


  # print ('student reward', reward)
  return status, reward, student, advice_budget, call_counter











if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=6000)
  parser.add_argument('--numTeammates', type=int, default=0)
  parser.add_argument('--numOpponents', type=int, default=1)
  parser.add_argument('--numEpisodes', type=int, default=1)
  parser.add_argument('--learnRate', type=float, default=0.1)
  parser.add_argument('--suffix', type=int, default=0)

  parser.add_argument('--advice_budget', type=int, default=300)
  parser.add_argument('--advice_strategy', type=str, default='Early') #'Early', 'Alternative', 'Importance', 'MistakeCorrecting', 'NoAdvise', 'AlwaysAdvise'
  parser.add_argument('--use_EAA', type=bool, default=False)


  args=parser.parse_args()

  # Create the HFO Environment
  hfo = HFOEnvironment()
  #now connect to the server
  hfo.connectToServer(HIGH_LEVEL_FEATURE_SET,'bin/teams/base/config/formations-dt',args.port,'localhost','base_left',False)
  global NF,NA,NOT,NOO
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
  # SA=SarsaAgent(NF, NA, learnR, eps, Lambda, FA, wt_filename, wt_filename+'saved')

  # teacher loads a good weight, student loads an intial weight
  # student = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, 'student_init_weights/'+wt_filename+'_bad', wt_filename+args.advice_strategy+'_student_training')
  # teacher = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, 'teacher_weights/3v3_2000epi/'+wt_filename+'_student_training', wt_filename+'_teacher')


  # student load with no weight file...
  # student = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, '', wt_filename+args.advice_strategy+'_student_training')
  # teacher = copy.deepcopy(SarsaAgent(NF, NA, learnR, eps, Lambda, FA, 'teacher_weights/2v2orig_script/'+wt_filename, wt_filename+'_teacher'))
  # teacher = None
  # TODO: load weights alternatively to simulate a teacher and a student

  # student = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, 'teacher_weights/2v2orig_script/'+wt_filename, wt_filename+args.advice_strategy+'_student_training')
  # student.singleSaveWeights()
  # student.singleLoadWeights()

  tmp_load_file = wt_filename + '_tmp_load'
  teacher_file = 'teacher_weights/2v2orig_script/'+wt_filename
  save_file = wt_filename+args.advice_strategy+'_student_training'

  shutil.copyfile('student_init_weights/' + wt_filename + '_student_init', tmp_load_file)
  student = SarsaAgent(NF, NA, learnR, eps, Lambda, FA, tmp_load_file, save_file)



  print ('args.numEpisodes: ', args.numEpisodes)
  # reward_means = []
  # reward_stds = []
  advice_budget = np.copy(args.advice_budget)

  reward = 0
  rewards = []
  call_counter = 0
  for episode in range(1,args.numEpisodes+1):
    # print ('episode', episode)
    # student.singleSaveWeights()

    if advice_budget > 0:
        np.save(wt_filename+args.advice_strategy+'_budget_used_up', np.array([episode]))
        # print ('episode', episode)
    status, reward, student, advice_budget, call_counter = gen_episode(reward, hfo, discFac, student, advice_budget, args.advice_strategy, args.use_EAA, tmp_load_file, teacher_file, save_file, call_counter, eval=False)
    print ('episode', episode, 'reward', reward)
    rewards.append(reward)

    np.save(wt_filename+args.advice_strategy+'_rewards', np.array(rewards))



    ############################################################################################################
    # eval every few training episodes, don't trian in evaluation

    # if (episode-1) % 100 == 0:
    #     rewards = []
    #     for j in range (100):
    #         print ('j', j)
    #         status, reward, _, advice_budget = gen_episode(reward, hfo, discFac, student, advice_budget, args.advice_strategy, args.use_EAA, tmp_load_file, teacher_file, save_file, eval=True)
    #         print ('reward', reward)
    #         rewards.append(reward)
    #     print ('episode', episode, 'np.mean(rewards)', np.mean(rewards), 'np.std(rewards)', np.std(rewards))
    #     print ('advice_budget', advice_budget)
    #     print ()
    #     reward_means.append(np.mean(rewards))
    #     reward_stds.append(np.std(rewards))
    #     np.save(wt_filename+args.advice_strategy+'_reward_means', np.array(reward_means))
    #     np.save(wt_filename+args.advice_strategy+'_reward_stds', np.array(reward_stds))
    ############################################################################################################
    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      break

  # Save the training process
