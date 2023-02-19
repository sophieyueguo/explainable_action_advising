from hfo import *
import numpy as np
import shutil
import time

from policy_extraction import tree_helper

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

def purge_features(state, NF, NOT, NOO):
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
            state_importance_threshold = 0.7
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
def gen_AA_episode(reward, hfo, discFac, student, advice_budget, advice_strategy,
tmp_load_file, teacher_file, save_file, call_counter, NF, NOT, NOO, eval=False):

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
        reward=getReward(status)
        if not eval:
          student.update(st,action,reward,discFac)
      st=purge_features(state, NF, NOT, NOO)

      if advice_strategy == 'NoAdvise':
        # train from scratch
        action = student.selectAction(st)

      elif advice_strategy == 'AlwaysAdvise':
        #let the student always takes teacher's action
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
    if not eval:
      student.update(st, action, reward, discFac)
    student.endEpisode()
  return status, reward, student, advice_budget, call_counter





def gen_teacher_episode(reward, hfo, discFac, teacher, NF, NOT, NOO):
    st = np.empty(NF,dtype=np.float64)

    count=0
    status=IN_GAME
    action=-1

    D = []
    while status==IN_GAME:
      count=count+1
      # Grab the state features from the environment
      state = hfo.getState()
      if int(state[5])==1:
        if action != -1:
          reward=getReward(status)
          # teacher does not need to update
          D.append([st,action,teacher.computeStateImportance(st)])

        st=purge_features(state, NF, NOT, NOO)
        action = teacher.selectAction(st)

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
      D.append([st,action,teacher.computeStateImportance(st)])
       # teacher does not need to update
      teacher.endEpisode()
    return status, reward, D




def gen_dt_episode(clf, reward, hfo, discFac, agent, NF, NOT, NOO, NA,
episode_num=None, evaluate=False):
    # agent won't take action, here is a placeholder
    st = np.empty(NF,dtype=np.float64)

    count=0
    status=IN_GAME
    action=-1

    D = []
    while status==IN_GAME:
      count=count+1
      # Grab the state features from the environment
      state = hfo.getState()
      if int(state[5])==1:
        if action != -1:
          reward=getReward(status)
          # agent does not need to update
          D.append([st,action,agent.computeStateImportance(st)])

        st=purge_features(state, NF, NOT, NOO)
        clf_prob = clf.predict_proba([st])
        action_prob = clf_prob[0]
        if sum(action_prob) == 0:
            action = np.random.choice(NA, 1)[0]
            print ('random action', action)
        else:
            if sum(action_prob) != 1:
                action_prob = action_prob/sum(action_prob)
            action = np.random.choice(NA, 1, p=action_prob)[0]

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
      D.append([st,action,agent.computeStateImportance(st)])
      # agent does not need to update
      agent.endEpisode()
    return status, reward, D






def gen_EAA_episode(reward, hfo, discFac, student, advice_budget, advice_strategy,
tmp_load_file, teacher_file, save_file, call_counter, NF, NOT, NOO, NA,
best_clf, new_clf, follow_dt_prob, eval=False, decay_ratio=0.997, video_delay=False):

  st = np.empty(NF,dtype=np.float64)

  count=0
  status=IN_GAME
  action=-1

  paths = []

  count_issue_teacher_advice = 0
  count_issue_dt_advice = 0
  count_take_actions = 0

  while status==IN_GAME:
    if video_delay:
        time.sleep(0.1)

    count=count+1
    # Grab the state features from the environment
    state = hfo.getState()
    if int(state[5])==1:
      if action != -1:
        reward=getReward(status)
        if not eval:
          student.update(st,action,reward,discFac)

    # Choose actions here
      st=purge_features(state, NF, NOT, NOO)

      # advice from the internal dt
      count_take_actions += 1
      new_clf_action = 'undecided'
      if new_clf:
          new_clf_action = new_clf.predict(st) #check with real reconstructed dt

      if new_clf_action != 'undecided' and np.random.rand() < follow_dt_prob:
          action = int(new_clf_action)
          count_issue_dt_advice += 1
          # print ('calling new_clf then...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
          if not eval:
              follow_dt_prob *= decay_ratio
      else:
          # otherwise ask the teacher
          if advice_strategy == 'NoAdvise':
            # train from scratch (AA)
            action = student.selectAction(st)

          elif advice_strategy == 'AlwaysAdvise':
            #let the student always takes teacher's action (AA)
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
                student, teacher_action, state_importance = get_teacher_action_advise(student, st, tmp_load_file, teacher_file, save_file, calc_si=True)
                should_advise, advice_budget = determine_give_advice(advice_budget, 'MistakeCorrecting', state_importance, teacher_action, student_action, 0, eval)
            else:
                should_advise = False
            if should_advise:
                action = teacher_action
            else:
                action = student_action

          if should_advise:  #if teacher gives advice, also give explanation?
            count_issue_teacher_advice += 1
            clf_prob = best_clf.predict_proba([st])
            action_prob = clf_prob[0]
            if sum(action_prob) == 0:
                clf_action = np.random.choice(NA, 1)[0]
            else:
                if sum(action_prob) != 1:
                    action_prob = action_prob/sum(action_prob)
                clf_action = np.random.choice(NA, 1, p=action_prob)[0]

            if int(action) == int(clf_action): # reasonable explanation is stored
                if np.max(action_prob) > 0.3 and clf_action == np.argmax(action_prob): #confident on this selected action
                    paths.append(tree_helper.retrieve_path(best_clf, [st], 0, [clf_action], printline=False))

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
  sum_advice = count_issue_dt_advice + count_issue_teacher_advice
  dt_ratio = 'N/A'
  if sum_advice > 0:
      dt_ratio = count_issue_dt_advice/sum_advice

  dt_ratio_all_actions = 'N/A'
  if count_take_actions > 0:
     dt_ratio_all_actions = count_issue_dt_advice/count_take_actions

  # print ('count_take_actions', count_take_actions,
  #        'count_issue_dt_advice', count_issue_dt_advice,
  #        'count_issue_teacher_advice', count_issue_teacher_advice,
  #        'dt_ratio', dt_ratio,
  #        'dt_ratio_all_actions', dt_ratio_all_actions,
  #        'follow_dt_prob', follow_dt_prob)
  print ('advice_budget', advice_budget)

  ############# EPISODE ENDS ###################################################################################
  # Check the outcome of the episode
  if action != -1:
    reward=getReward(status)
    if not eval:
      student.update(st, action, reward, discFac)
    student.endEpisode()
  return status, reward, student, advice_budget, call_counter, follow_dt_prob, paths
