'''For the experiment, set all the parameters that need for tuning'''

#env_type = '14_Room'
env_type = 'Simple_4_Room'

#teacher_advice_strategy = 'Early'
#teacher_advice_strategy = 'Alternative'
#teacher_advice_strategy = 'Importance'
teacher_advice_strategy = 'Mistake Correcting'

#teacher_advice_strategy = 'NNtransfer'
#teacher_advice_strategy = 'pretrain'
# teacher_advice_strategy = 'scratch'
#teacher_advice_strategy = 'non-expert_teacher'

# CHANGE EXPERIMENT INDICES IF RUN MULTIPLE TRIALS
exp_ind = '5/' #0/,1,2,3,4,5 to save different trials

exp_type = '1-dt+nn'
#exp_type = '2-no_advice_source'
#exp_type = '2-no_advice_target'
# exp_type = '3-dt'

allow_student_explpre_initially = False
always_take_advice = True

# NO NEED TO CHANGE #
#############################################################################
if teacher_advice_strategy in ['Early', 'Alternative', 'Importance', 'Mistake Correcting']:
    advice_algo = 'heuristic_with_tree_memory'
elif teacher_advice_strategy in ['NNtransfer', 'pretrain', 'scratch', 'non-expert_teacher']:
    advice_algo = ''
    if teacher_advice_strategy in ['NNtransfer', 'pretrain']:
        pretrain_model = True
    elif teacher_advice_strategy in ['scratch', 'non-expert_teacher']:
        pretrain_model = False
    if teacher_advice_strategy == 'NNtransfer':
        student_train = False
    elif teacher_advice_strategy in ['pretrain', 'scratch', 'non-expert_teacher']:
        student_train = True
else:
    print ('Stop')
print ('teacher_advice_strategy', teacher_advice_strategy, 'advice_algo', advice_algo)

# advice_algo = 'heuristic_with_no_memory' # ignore
# advice_algo = 'random action' # ignore

# benchmark_algo = 'Q-change Per Step' # ignore
# benchmark_algo = 'Reusing Budget' # ignore
# benchmark_algo = 'Decay Reusing Probability'# ignore




# SAVING RESULTS AND MODELS
###############################################################################
evaluate_trial_cycle = 1000

#save_path = 'publish_results/' + env_type + '/' + teacher_advice_strategy + '/' + advice_algo + '/' + exp_ind
does_save_result = False
save_path = 'publish_results/' + env_type + '/' + teacher_advice_strategy + '/' + exp_ind
print ('rewards save_path', save_path)
# benchmak_save_path = 'publish_results/' + env_type + '/' + benchmark_algo + '/' # ignore

does_save_model = False
save_model_dir = 'model/coma/student/' + env_type + '/' + teacher_advice_strategy + '/' + exp_ind
save_model_trial_cycle = 3 * evaluate_trial_cycle





# PARAMETERS CAN BE TUNED
###############################################################################
if env_type == '14_Room':
    # VIPER_data_path = "policy_extraction/VIPER_data/D_14_room_no_rubble.txt"
    # VIPER_data_path = "policy_extraction/VIPER_data/D_14_room_seen_rubble.txt"
    VIPER_n_steps_to_sample = 200000
    VIPER_max_iter = 15
    data_filter_ratio = 0.9

    teacher_max_advice_budget = 30000 # parameter to tune, around 30000 - 100000 maybe
    # teacher_rnn_path = "student/A_Saturn_Section/non-expert_teacher/"
    teacher_rnn_path = "student/14_Room/non-expert_teacher/old_rand_engineer_teacher/"
    teacher_state_importance_threshold = 2 # parameter to tune 2-6, bigger for mistake-correcting, small for importance

    max_trial_index = 40000 # parameter to tune, if we want to see more training
    decay_ratio = 1.0 # parameter to tune, if forget about the internal dt, note that it might not make a difference if internal teacher is bad/never asked


elif env_type == 'Simple_4_Room':
    # VIPER_data_path = "policy_extraction/VIPER_data/D_simple_4_room_no_rubble.txt"
    # VIPER_data_path = "policy_extraction/VIPER_data/D_simple_4_room_seen_rubble.txt"
    VIPER_n_steps_to_sample = 1000
    VIPER_max_iter = 15
    data_filter_ratio = 0.6

    teacher_max_advice_budget = 10000
    teacher_rnn_path = "student/Simple_4_Room/non-expert_teacher/"
    if teacher_advice_strategy == 'Importance':
        teacher_state_importance_threshold = 1.5
    elif teacher_advice_strategy == 'Mistake Correcting':
        teacher_state_importance_threshold = 3
    else:
        teacher_state_importance_threshold = None

    max_trial_index = 30000
    decay_ratio = 1.0


# manually load the model that needs resuming training
# now only for this exp
if exp_type == '2-no_advice_target' and teacher_advice_strategy in ['Early', 'Alternative', 'Importance', 'Mistake Correcting']:
    resume_training = True
    max_trial_index = 20000 +  max_trial_index #plus the 20000 train trials in the source environment
    teacher_max_advice_budget = 0
    resume_model_path = 'model/coma/student/' + env_type + '/' + teacher_advice_strategy + '/student_trained_source/18000'
    load_path = 'publish_results/' + env_type + '/' + teacher_advice_strategy + '/student_trained_source/'
    print ('resume_model_path', resume_model_path)
    print ('load_path', load_path)
else:
    resume_training = False




#resume_model_path = save_model_dir + '0' #number
