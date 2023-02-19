# game.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# game.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import numpy as np
import time, os
import torch
import torch.nn.functional as F
import traceback
import sys

from . import featureExtractors
from .advise import *
from .definitions import Directions, Actions
from .util import *

#######################
# Parts worth reading #
#######################

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()



class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.

    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def getPosition(self):
        return (self.pos)

    def getDirection(self):
        return self.direction

    def isInteger(self):
        x,y = self.pos
        return x == int(x) and y == int(y)

    def __eq__(self, other):
        if other == None: return False
        return (self.pos == other.pos and self.direction == other.direction)

    def __hash__(self):
        x = hash(self.pos)
        y = hash(self.direction)
        return hash(x + 13 * y)

    def __str__(self):
        return "(x,y)="+str(self.pos)+", "+str(self.direction)

    def generateSuccessor(self, vector):
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.

        Actions are movement vectors.
        """
        x, y= self.pos
        dx, dy = vector
        direction = Actions.vectorToDirection(vector)
        if direction == Directions.STOP:
            direction = self.direction # There is no stop direction
        return Configuration((x + dx, y+dy), direction)

class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, scared, etc).
    """

    def __init__( self, startConfiguration, isPacman ):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.isPacman = isPacman
        self.scaredTimer = 0
        self.numCarrying = 0
        self.numReturned = 0

    def __str__( self ):
        if self.isPacman:
            return "Pacman: " + str( self.configuration )
        else:
            return "Ghost: " + str( self.configuration )

    def __eq__( self, other ):
        if other == None:
            return False
        return self.configuration == other.configuration and self.scaredTimer == other.scaredTimer

    def __hash__(self):
        return hash(hash(self.configuration) + 13 * hash(self.scaredTimer))

    def copy( self ):
        state = AgentState( self.start, self.isPacman )
        state.configuration = self.configuration
        state.scaredTimer = self.scaredTimer
        state.numCarrying = self.numCarrying
        state.numReturned = self.numReturned
        return state

    def getPosition(self):
        if self.configuration == None: return None
        return self.configuration.getPosition()

    def getDirection(self):
        return self.configuration.getDirection()

class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are positions on a Pacman map with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented like a pacman board.
    """
    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]: raise Exception('Grids can only contain booleans')
        self.CELLS_PER_INT = 30

        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __str__(self):
        out = [[str(self.data[x][y])[0] for x in range(self.width)] for y in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        # return hash(str(self))
        base = 1
        h = 0
        for l in self.data:
            for i in l:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item =True ):
        return sum([x.count(item) for x in self.data])

    def asList(self, key = True):
        list = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key: list.append( (x,y) )
        return list

    def packBits(self):
        """
        Returns an efficient int list representation

        (width, height, bitPackedInts...)
        """
        bits = [self.width, self.height]
        currentInt = 0
        for i in range(self.height * self.width):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = self._cellIndexToPosition(i)
            if self[x][y]:
                currentInt += 2 ** bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(currentInt)
                currentInt = 0
        bits.append(currentInt)
        return tuple(bits)

    def _cellIndexToPosition(self, index):
        x = index / self.height
        y = index % self.height
        return x, y

    def _unpackBits(self, bits):
        """
        Fills in data from a bit-level representation
        """
        cell = 0
        for packed in bits:
            for bit in self._unpackInt(packed, self.CELLS_PER_INT):
                if cell == self.width * self.height: break
                x, y = self._cellIndexToPosition(cell)
                self[x][y] = bit
                cell += 1

    def _unpackInt(self, packed, size):
        bools = []
        if packed < 0: raise ValueError
        for i in range(size):
            n = 2 ** (self.CELLS_PER_INT - i - 1)
            if packed >= n:
                bools.append(True)
                packed -= n
            else:
                bools.append(False)
        return bools

def reconstituteGrid(bitRep):
    if type(bitRep) is not type((1,2)):
        return bitRep
    width, height = bitRep[:2]
    return Grid(width, height, bitRepresentation= bitRep[2:])

####################################
# Parts you shouldn't have to read #
####################################



class GameStateData:
    """

    """
    def __init__( self, prevState = None ):
        """
        Generates a new data packet by copying information from its predecessor.
        """
        if prevState != None:
            self.food = prevState.food.shallowCopy()
            self.capsules = prevState.capsules[:]
            self.agentStates = self.copyAgentStates( prevState.agentStates )
            self.layout = prevState.layout
            self._eaten = prevState._eaten
            self.score = prevState.score

        self._foodEaten = None
        self._foodAdded = None
        self._capsuleEaten = None
        self._agentMoved = None
        self._lose = False
        self._win = False
        self.scoreChange = 0

    def deepCopy( self ):
        state = GameStateData( self )
        state.food = self.food.deepCopy()
        state.layout = self.layout.deepCopy()
        state._agentMoved = self._agentMoved
        state._foodEaten = self._foodEaten
        state._foodAdded = self._foodAdded
        state._capsuleEaten = self._capsuleEaten
        return state

    def copyAgentStates( self, agentStates ):
        copiedStates = []
        for agentState in agentStates:
            copiedStates.append( agentState.copy() )
        return copiedStates

    def __eq__( self, other ):
        """
        Allows two states to be compared.
        """
        if other == None: return False
        # TODO Check for type of other
        if not self.agentStates == other.agentStates: return False
        if not self.food == other.food: return False
        if not self.capsules == other.capsules: return False
        if not self.score == other.score: return False
        return True

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.
        """
        for i, state in enumerate( self.agentStates ):
            try:
                int(hash(state))
            except TypeError as e:
                print(e)
                #hash(state)
        return int((hash(tuple(self.agentStates)) + 13*hash(self.food) + 113* hash(tuple(self.capsules)) + 7 * hash(self.score)) % 1048575 )

    def __str__( self ):
        width, height = self.layout.width, self.layout.height
        map = Grid(width, height)
        if type(self.food) == type((1,2)):
            self.food = reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

        for agentState in self.agentStates:
            if agentState == None: continue
            if agentState.configuration == None: continue
            x,y = [int( i ) for i in nearestPoint( agentState.configuration.pos )]
            agent_dir = agentState.configuration.direction
            if agentState.isPacman:
                map[x][y] = self._pacStr( agent_dir )
            else:
                map[x][y] = self._ghostStr( agent_dir )

        for x, y in self.capsules:
            map[x][y] = 'o'

        return str(map) + ("\nScore: %d\n" % self.score)

    def _foodWallStr( self, hasFood, hasWall ):
        if hasFood:
            return '.'
        elif hasWall:
            return '%'
        else:
            return ' '

    def _pacStr( self, dir ):
        if dir == Directions.NORTH:
            return 'v'
        if dir == Directions.SOUTH:
            return '^'
        if dir == Directions.WEST:
            return '>'
        return '<'

    def _ghostStr( self, dir ):
        return 'G'
        if dir == Directions.NORTH:
            return 'M'
        if dir == Directions.SOUTH:
            return 'W'
        if dir == Directions.WEST:
            return '3'
        return 'E'

    def initialize( self, layout, numGhostAgents ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.food = layout.food.copy()
        #self.capsules = []
        self.capsules = layout.capsules[:]
        self.layout = layout
        self.score = 0
        self.scoreChange = 0

        self.agentStates = []
        numGhosts = 0
        for isPacman, pos in layout.agentPositions:
            if not isPacman:
                if numGhosts == numGhostAgents: continue # Max ghosts reached already
                else: numGhosts += 1
            self.agentStates.append( AgentState( Configuration( pos, Directions.STOP), isPacman) )
        self._eaten = [False for a in self.agentStates]


class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    # Old decay rate: 0.9999999
    def __init__(self, agents, rules, teacher, teacher_dt, advice_budget, advice_mode, advice_strategy, introspection_model, introspection_prob, introspection_decay_rate, fixed_advise_teacher, multiagent=False):
        self.agentCrashed = False
        self.agents = agents
        self.rules = rules
        self.gameOver = False

        self.advice_budget = advice_budget
        self.advice_mode = advice_mode
        self.advice_strategy = advice_strategy
        self.advice_call_counter = 0
        self.introspection_model = introspection_model
        self.introspection_prob = introspection_prob
        self.introspection_decay = introspection_decay_rate

        self.fixed_advise_teacher = fixed_advise_teacher

        self.multiagent = multiagent

        self.teacher = teacher
        self.teacher_dt = teacher_dt

        if self.multiagent:
            self.feature_extractor = featureExtractors.MultiagentFeatureExtractor()
        else:
            self.feature_extractor = featureExtractors.VectorFeatureExtractor()

        self.last_state = None

    def getProgress(self):
        if self.gameOver:
            return 1.0
        else:
            return self.rules.getProgress(self)

    def _agentCrash( self, agentIndex, quiet=False):
        if not quiet: traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        self.rules.agentCrash(self, agentIndex)

    def init(self):
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]

    def step(self, raw_action):
        self.last_state = self.state.deepCopy()

        numAgents = len( self.agents )

        for agentIndex in range(numAgents):
            if not self.gameOver:
                # Fetch the next agent
                agent = self.agents[agentIndex]

                observation = self.state

                # Calculate pacman action
                if agentIndex == 0:
                    if not self.multiagent:
                        student_action = Actions.actionToDirection(raw_action)
                        teacher_action = None

                        # If we are performing action advising, calculate the teacher's action
                        if self.advice_mode is not None:
                            last_state_features = self.feature_extractor.getUnconditionedFeatures(self.last_state)
                            teacher_action, teacher_action_prob, teacher_importance = self.teacher.get_action(last_state_features)
                            teacher_action = Actions.actionToDirection(teacher_action)

                        action = student_action
                        action_source = "student"

                        # If no advising
                        if self.advice_mode == None:
                            # Take the student's action
                            should_advise = False

                        # If action advising
                        elif self.advice_mode == "aa":
                            self.advice_call_counter += 1

                            # Determine whether advice should be given with the current advice strategy
                            should_advise, self.advice_budget = determine_give_advice(
                                self.advice_budget, self.advice_strategy, teacher_importance,
                                teacher_action, student_action, self.advice_call_counter
                            )

                        # If explainable action advising
                        elif self.advice_mode == "eaa":
                            introspection_action = None
                            
                            if self.introspection_model:
                                # Get the predicted action from the student's introspection model
                                introspection_action = self.introspection_model.predict(last_state_features)

                            # If we have a valid predicted action, then follow it according to introspection probability
                            if introspection_action is not None and np.random.rand() < self.introspection_prob:
                                action = Actions.actionToDirection(introspection_action)
                                action_source = "dt"
                                self.introspection_prob *= self.introspection_decay
                                should_advise = False

                            else:
                                # Determine if we should give advice from teacher
                                should_advise, self.advice_budget = determine_give_advice(
                                    self.advice_budget, self.advice_strategy, teacher_importance,
                                    teacher_action, student_action, self.advice_call_counter
                                )

                                # If we should...
                                if should_advise:
                                    # Additionally get the teacher's action from the DT representation
                                    teacher_dt_action, teacher_dt_action_prob, _ = self.teacher_dt.get_action(last_state_features)
                                    teacher_dt_action = Actions.actionToDirection(teacher_dt_action)

                                    # If the teacher's DT action is the same as the original teacher action and the probability is sufficiently high
                                    if teacher_action == teacher_dt_action and np.max(teacher_dt_action_prob) > 0.2:
                                        # Take the explanation from the teacher's DT and add it to the introspection model
                                        explanation = self.teacher_dt.get_explanation(last_state_features, Actions._actionToInt[teacher_action])
                                        self.introspection_model.add_explanation(explanation)

                        elif self.advice_mode == "fixed":
                            should_advise = False
                            action = Actions.actionToDirection(raw_action)

                            if self.fixed_advise_teacher.advice_budget > 0:
                                last_state_features_list = self.feature_extractor.getUnconditionedFeatures(self.last_state)
                                reuse_action_int = self.fixed_advise_teacher.reuse_advice(last_state_features_list) 
                                if reuse_action_int != None:
                                    action = Actions.actionToDirection(reuse_action_int)
                                    action_source = "teacher"
                                else:
                                    self.fixed_advise_teacher.handle_no_reuse_advice(last_state_features_list)
                                
                                score = self.last_state.generateSuccessor(0, action).getScore()
                                self.fixed_advise_teacher.compute_q_change(last_state_features_list, action, score)
        

                        if should_advise:
                            action = teacher_action
                            action_source = "teacher"

                    else:
                        action = agent.getAction(observation)
                # Calculate ghost action
                else:
                    if not self.multiagent:
                        action = agent.getAction(observation)
                    else:
                        action = Actions.actionToDirection(raw_action)

                # Execute the action
                self.state = self.state.generateSuccessor( agentIndex, action )

                # Allow for game specific conditions (winning, losing, etc.)
                self.rules.process(self.state, self)

        if not self.multiagent:
            new_features = self.feature_extractor.getUnconditionedFeatures(self.state)

            # The normal reward is the change in score, including per-step penalties
            reward = self.state.getScore() - self.last_state.getScore()
        else:
            new_features = self.feature_extractor.getUnconditionedFeatures(self.state)

            # If the game is a "loss", this means the pacman lost and was eaten by the ghosts
            if self.state.isLose():
                reward = 5
            # If the game is a "win", this means the pacman either ate all the ghosts or all the food without getting eaten itself
            elif self.state.isWin():
                reward = -5
            else:
            # Otherwise, the game is in progress and there is a per-step penalty
                reward = -0.01

        info_dict = {"action_source": action_source}
        return new_features, reward, self.gameOver, info_dict
