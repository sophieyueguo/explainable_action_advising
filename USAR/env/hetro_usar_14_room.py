import numpy as np
import experiment_parameter as parameter

N_RUBBLE = 0

class Hetro_USAR_14_Room_Env():

    def __init__(self):
        self.n_actions = 10 # move up, right, down, left, triage, dig,
        self.n_agents = 2
        self.n_rooms = 14
        # real info, what is in what room: (self.n_agents + 2) * self.n_rooms -critical victim, regular, engineer
        # observation, if the player sees some thing: self.n_agents * (2 + self.n_agents)
        # belief, player's mind of the whole environment: (self.n_agents + 2) * self.n_rooms * self.n_agents
        # cannot see regular victim until rubble removed.
        self.state_shape = (self.n_agents + 2) * self.n_rooms + self.n_agents * (2 + self.n_agents) + \
        (self.n_agents + 2) * self.n_rooms * self.n_agents

        self.obs_shape = (self.n_agents + 2) * self.n_rooms # belief. coding convenience. Policy maps this to actions.
        self.episode_limit = 50 #TODO tune

        #self.n_victim = 2
        # think in the new environment just give 1 victim????-switched to 4
        self.n_victim = 4

        self.n_rubble = N_RUBBLE # rooms having rubble while hallway not
        print ('self.n_rubble', self.n_rubble)
        self.roles = ['medic', 'engineer']

    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.n_actions
        env_info["n_agents"] = self.n_agents
        env_info["state_shape"] = self.state_shape
        env_info["obs_shape"] = self.obs_shape
        env_info["episode_limit"] = self.episode_limit
        return env_info

    def make_state_obs(self):
        # state -> observation (partial/full obs) -> belief on self and teammate -> desire-opt action(self.obs, in code)
        self.obs = [] # coding convenience, here assume intent is observable

        # update physical info, what is in what room
        physical_info = []
        for ri in range(self.n_rooms):
            room_vec = [0] * (2 + self.n_agents)
            if ri in self.victim_loc:
                room_vec[0] = 1
            if ri in self.rubble_loc:
                room_vec[1] = 1
            for ai in range(self.n_agents):
                if self.agent_loc[ai] == ri:
                    room_vec[2 + ai] = 1
            physical_info += room_vec

        # update observation
        observation = []
        observation_dic = []
        for ai in range(self.n_agents):
            curr_room = self.agent_loc[ai]
            observation_i = [0] * (2 + self.n_agents)
            if curr_room in self.victim_loc:
                if curr_room not in self.rubble_loc: # victim burried, cannot be seen
                    observation_i[0] = 1
            if curr_room in self.rubble_loc:
                observation_i[1] = 1

            observation_i[2 + ai] = 1
            for aj in range(self.n_agents):
                if self.agent_loc[aj] == curr_room:
                    observation_i[2 + aj] = 1

            observation += observation_i
            observation_dic.append(observation_i)


        # update belief, only update when teammate is observed in the curr_room, now just fully observation...
        # based on observation and previous belief, update belief.
        # can fail to update?
        # belief_update_prob = 1.0 # can teacher identify this failure? - then advice is, the correct belief is...
        belief_vec = []
        belief = self.belief
        for ai in range(self.n_agents):
            belief_ai = []
            for ri in range(self.n_rooms):
                # 0 for victim, 1 for rubble, rest for other players
                # only update for the current room and last room
                # if np.random.uniform(0, 1, 1)[0] <= belief_update_prob:
                if belief[ai][ri][2 + ai] == 1: # clean up the previous existence
                    for aj in range(self.n_agents):
                        belief[ai][ri][2 + aj] = 0
                if self.agent_loc[ai] == ri:
                    belief[ai][ri] = observation_dic[ai]
                belief_vec += belief[ai][ri]
                belief_ai += belief[ai][ri]
            self.obs.append(belief_ai)
        self.belief = belief
        self.state = physical_info + observation + belief_vec



    def reset(self):
        # np.random.seed(0)

        # self.agent_loc = list(np.random.choice(self.n_rooms, self.n_agents))
        self.agent_loc = [9, 9] #agents start from hallway

        # having victims in random rooms instead of hallway nodes
        self.victim_loc = list(np.random.choice(self.n_rooms, self.n_victim, replace=False))
        #while 8 in self.victim_loc or 9 in self.victim_loc:
        #    self.victim_loc = list(np.random.choice(self.n_rooms, self.n_victim, replace=False))

        self.rubble_loc = list(np.random.choice(self.n_rooms, self.n_rubble, replace=False))
        # self.rubble_loc = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13] # having rubble in rooms instead of hallway nodes

        self.belief = [[[0] * (self.n_agents + 2) for i in range(self.n_rooms)] for j in range(self.n_agents)] # what in what rooms, observed so far, originally has nothing
        self.make_state_obs()

    def get_obs(self):
        return self.obs

    def get_state(self):
        return self.state

    def get_avail_agent_actions(self, agent_id):
        #0 is medic and 1 is engineer
        if self.roles[agent_id] == 'medic':
            if self.agent_loc[agent_id] == 0:
                return [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 1:
                return [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 2:
                return [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 3:
                return [1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 4:
                return [0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
            if self.agent_loc[agent_id] == 5:
                return [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 6:
                return [0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
            if self.agent_loc[agent_id] == 7:
                return [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 8:
                return [0, 1, 0, 1, 0, 1, 1, 1, 1, 0]
            if self.agent_loc[agent_id] == 9:
                return [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
            if self.agent_loc[agent_id] == 10:
                return [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 11:
                return [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 12:
                return [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
            if self.agent_loc[agent_id] == 13:
                return [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]


        if self.roles[agent_id] == 'engineer':
            if self.agent_loc[agent_id] == 0:
                return [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 1:
                return [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 2:
                return [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 3:
                return [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 4:
                return [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
            if self.agent_loc[agent_id] == 5:
                return [0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 6:
                return [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
            if self.agent_loc[agent_id] == 7:
                return [0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 8:
                return [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
            if self.agent_loc[agent_id] == 9:
                return [0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
            if self.agent_loc[agent_id] == 10:
                return [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 11:
                return [0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 12:
                return [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
            if self.agent_loc[agent_id] == 13:
                return [0, 0, 0, 1, 0, 0, 0, 0, 0, 1]

    def step(self, actions):
        reward = 0
        info = {'battle_won': False}

        # cannot triage victim when there is a rubble
        for i in range(self.n_agents): # location changes
            if self.agent_loc[i] == 0:
                if actions[i] == 0:
                    self.agent_loc[i] = 4
            if self.agent_loc[i] == 1:
                if actions[i] == 0:
                    self.agent_loc[i] = 5
            if self.agent_loc[i] == 2:
                if actions[i] == 0:
                    self.agent_loc[i] = 6
            if self.agent_loc[i] == 3:
                if actions[i] == 0:
                    self.agent_loc[i] = 7
            if self.agent_loc[i] == 4:
                if actions[i] == 4:
                    self.agent_loc[i] = 0
                if actions[i] == 7:
                    self.agent_loc[i] = 8
            if self.agent_loc[i] == 5:
                if actions[i] == 4:
                    self.agent_loc[i] = 1
                if actions[i] == 1:
                    self.agent_loc[i] = 8
            if self.agent_loc[i] == 6:
                if actions[i] == 4:
                    self.agent_loc[i] = 2
                if actions[i] == 7:
                    self.agent_loc[i] = 9
            if self.agent_loc[i] == 7:
                if actions[i] == 4:
                    self.agent_loc[i] = 3
                if actions[i] == 1:
                    self.agent_loc[i] = 9
            if self.agent_loc[i] == 8:
                if actions[i] == 1:
                    self.agent_loc[i] = 10
                if actions[i] == 3:
                    self.agent_loc[i] = 4
                if actions[i] == 5:
                    self.agent_loc[i] = 5
                if actions[i] == 6:
                    self.agent_loc[i] = 9
                if actions[i] == 7:
                    self.agent_loc[i] = 11
            if self.agent_loc[i] == 9:
                if actions[i] == 1:
                    self.agent_loc[i] = 12
                if actions[i] == 2:
                    self.agent_loc[i] = 8
                if actions[i] == 3:
                    self.agent_loc[i] = 6
                if actions[i] == 5:
                    self.agent_loc[i] = 7
                if actions[i] == 7:
                    self.agent_loc[i] = 13
            if self.agent_loc[i] == 10:
                if actions[i] == 5:
                    self.agent_loc[i] = 8
            if self.agent_loc[i] == 11:
                if actions[i] == 3:
                    self.agent_loc[i] = 8
            if self.agent_loc[i] == 12:
                if actions[i] == 5:
                    self.agent_loc[i] = 9
            if self.agent_loc[i] == 13:
                if actions[i] == 3:
                    self.agent_loc[i] = 9


            if actions[i] == 8 and self.roles[i] == 'medic':
                if self.agent_loc[i] in self.victim_loc and self.agent_loc[i] not in self.rubble_loc:
                    # can triage
                    #TODO: for new setting
                    reward += 10
                    info['battle_won'] = True
                    self.victim_loc.remove(self.agent_loc[i])
                    #if self.agent_loc[0] == self.agent_loc[1]:
                    #    reward += 10
                    #    info['battle_won'] = True
                    #    self.victim_loc.remove(self.agent_loc[i])

            if actions[i] == 9 and self.roles[i] == 'engineer':
                if self.agent_loc[i] in self.rubble_loc:
                    # info['battle_won'] = True
                    self.rubble_loc.remove(self.agent_loc[i])

        terminated = False
        if len(self.victim_loc) == 0 and len(self.rubble_loc) == 0:
            terminated = True

        self.make_state_obs()
        return reward, terminated, info

    def close(self):
        pass
