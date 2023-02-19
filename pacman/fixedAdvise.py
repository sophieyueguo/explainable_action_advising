import numpy as np
import torch
import torch.nn.functional as F

class FixedAdvise():
    def __init__(self, fixed_advise_type, advice_budget, teacher):
        self.advice_budget = advice_budget
        self.advice_prob = 0.9
        self.advise_dataset = {}
        self.fixed_advise_type = fixed_advise_type
        self.teacher = teacher

        if fixed_advise_type == 'reuse_budget':
            self.budget_over_state = {}
            self.single_state_maximum_budget = 20
        elif fixed_advise_type == 'decay_reuse':
            self.decay_value = 0.97
            self.reuse_prob_over_state = {}
        elif fixed_advise_type == 'q_change':
            self.q_over_state = {}
            self.q_change_threshold = -20
        
    def reuse_advice(self, state):
        if str(state) in self.advise_dataset:
            if self.fixed_advise_type == 'reuse_budget': 
                if self.budget_over_state[str(state)] > 0:
                    self.budget_over_state[str(state)] -= 1
                    return self.advise_dataset[str(state)]
            elif self.fixed_advise_type == 'q_change':
                return self.advise_dataset[str(state)]
            elif self.fixed_advise_type == 'decay_reuse':
                if self.reuse_prob_over_state[str(state)] > np.random.rand():
                    self.reuse_prob_over_state[str(state)] *= self.decay_value
                    return self.advise_dataset[str(state)]
                else:
                    return None
        else:
            return None
          

    def handle_no_reuse_advice(self, state):
        if self.advice_prob > np.random.rand():
            self.advice_budget -= 1
            if self.advice_budget <= 0:
                print ('advice_budget smaller than zero!')
            teacher_action_int, _, _ = self.teacher.get_action(state)
            self.advise_dataset[str(state)] = teacher_action_int
          
            if self.fixed_advise_type == 'reuse_budget':
                self.budget_over_state[str(state)] = self.single_state_maximum_budget
            elif self.fixed_advise_type == 'decay_reuse':
                self.reuse_prob_over_state[str(state)] = 1.
    
    def compute_q_change(self, state, action, score):
        if self.fixed_advise_type == 'q_change':
            if str(state) in self.advise_dataset:
                if (str(state), action) not in self.q_over_state:
                    self.q_over_state[(str(state), action)] = score
                else:
                    diff = self.q_over_state[(str(state), action)] - score
                    if diff < self.q_change_threshold:# tune shreshold
                        self.advise_dataset.pop(str(state))
                        # print (diff, len(self.advise_dataset))
                    self.q_over_state[(str(state), action)] = score
            
            
            
            
                       






