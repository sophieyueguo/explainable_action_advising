# from agent.agent import Agents, CommAgents
from agent.agent import Agents
import os
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

import experiment_parameter as parameter


class Teacher:
    def __init__(self, args):
        # init a teacher agent and load model
        print ('Init teacher')
        self.agents = Agents(args)

        self.agents.policy.model_dir = './model/coma/'
        print ('self.model_dir', self.agents.policy.model_dir)
        if os.path.exists(self.agents.policy.model_dir + '/' + parameter.teacher_rnn_path + '/rnn_params.pkl'):
            path_rnn = self.agents.policy.model_dir + '/' + parameter.teacher_rnn_path + '/rnn_params.pkl'
            path_coma = self.agents.policy.model_dir + '/' + parameter.teacher_rnn_path + '/critic_params.pkl'

            print ('path_rnn', path_rnn, '\npath_come', path_coma)
            map_location = 'cuda:0' if self.agents.policy.args.cuda else 'cpu'
            self.agents.policy.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
            self.agents.policy.eval_critic.load_state_dict(torch.load(path_coma, map_location=map_location))
            print('Successfully load the model: {} and {}'.format(path_rnn, path_coma))
        else:
            raise Exception("No model!")

        self.agents.policy.target_critic.load_state_dict(self.agents.policy.eval_critic.state_dict())

        self.agents.policy.rnn_parameters = list(self.agents.policy.eval_rnn.parameters())
        self.agents.policy.critic_parameters = list(self.agents.policy.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.agents.policy.critic_optimizer = torch.optim.RMSprop(self.agents.policy.critic_parameters, lr=args.lr_critic)
            self.agents.policy.rnn_optimizer = torch.optim.RMSprop(self.agents.policy.rnn_parameters, lr=args.lr_actor)
        self.advice_budget = parameter.teacher_max_advice_budget

        self.advice_strategy = parameter.teacher_advice_strategy
        self.called_counter = 0
        self.alternative_advice_freq = 4 #10
        self.state_importance_threshold = parameter.teacher_state_importance_threshold

        self.pred_action_data_size = 1000

        self.clf = svm.SVC(kernel='precomputed')
        self.train_states = []
        self.train_actions = []



    def determine_give_advice(self, state_importance, teacher_action, curr_state, agent_id, action, evaluate):
        if self.advice_budget == 0:
            return False

        self.called_counter += 1
        if self.advice_budget > 0:

            if self.advice_strategy == 'Early':
                if not evaluate:
                    self.advice_budget -= 1
                if self.advice_budget == 0:
                    print ('advice budget used up!')
                return True

            elif self.advice_strategy == 'Alternative':
                if self.called_counter % self.alternative_advice_freq == 0:
                    if not evaluate:
                        self.advice_budget -= 1
                    if self.advice_budget == 0:
                        print ('advice budget used up!')
                    return True

            elif self.advice_strategy == 'Importance':
                if state_importance >= self.state_importance_threshold:
                    if not evaluate:
                        self.advice_budget -= 1
                    if self.advice_budget == 0:
                        print ('advice budget used up!')
                    return True

            elif self.advice_strategy == 'Mistake Correcting':
                if state_importance >= self.state_importance_threshold:
                    if teacher_action != action:
                        if not evaluate:
                            self.advice_budget -= 1
                        if self.advice_budget == 0:
                            print ('advice budget used up!')
                        return True
            else:
                print ('unknown advice strategy!')

        return False


    def modify_pred_action_model(self, train_states, train_actions):
        # takes in the most recent chunk of state-action pairs, and fit model based on this
        # so don't init the prediction until size of collected data is reached
        self.train_states, self.train_actions = train_states, train_actions
        if self.advice_budget > 0:
            if self.advice_budget % 100 == 0:
                if len(self.train_states) >= self.pred_action_data_size:
                    gram_train = np.dot(self.train_states, self.train_states.T)
                    self.clf.fit(gram_train, np.array(self.train_actions))


    def action_array_to_int(self, array, n_actions):
        num_player = len(array)
        ind = array[0]
        for ai in range(1, num_player):
            ind += array[ai] * n_actions
        return ind
