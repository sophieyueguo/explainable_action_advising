# featureExtractors.py
# --------------------
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

import numpy as np

from .definitions import Directions, Actions
from . import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

def modifiedClosestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    backtrace = {}
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)

        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            while (pos_x, pos_y) in backtrace and backtrace[(pos_x, pos_y)] != (pos[0], pos[1]):
                pos_x, pos_y = backtrace[(pos_x, pos_y)]

            # Return the vector direction that the closest food is in
            return pos_x - pos[0], pos_y - pos[1]

        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            if (nbr_x, nbr_y) not in expanded:
                fringe.append((nbr_x, nbr_y, dist+1))
                backtrace[(nbr_x, nbr_y)] = (pos_x, pos_y)

    # no food found
    return None

class VectorFeatureExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """
    def _findGhosts(self, pos, dir_vec, walls, ghosts, ghost_radius=2):
        # Finds ghosts down a particular direction
        x, y = pos
        dx, dy = dir_vec

        next_x = x + dx
        next_y = y + dy

        if (next_x < 0 or next_x >= walls.width) or (next_y < 0 or next_y >= walls.height) or (walls[next_x][next_y]):
            return False

        fringe = [(next_x, next_y, 0)]
        expanded = set()

        # Add the original starting node to the expanded list so we don't re-explore it
        expanded.add((x, y))
        
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)

            # Skip node if we have already explored it, the depth is greater than ghost_radius
            if (pos_x, pos_y) in expanded or dist > ghost_radius:
                continue

            expanded.add((pos_x, pos_y))

            # If we find a ghost here, then return
            for ghost in ghosts:
                if ghost == (pos_x, pos_y):
                    return True

            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)

            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))

        # No ghost found
        return False

    def getUnconditionedFeatures(self, state):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        x, y = state.getPacmanPosition()

        closest_food = modifiedClosestFood((x, y), food, walls)
        food_features = [0, 0, 0, 0]
        if closest_food is not None:
            food_features[Actions._actionToInt[Actions.vectorToDirection(closest_food)]] = 1

        ghost_features = []
        wall_features = []

        # Search whether there's a neighboring wall
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            next_y = y_int + dy
            if (next_x < 0 or next_x == walls.width) or (next_y < 0 or next_y == walls.height) or (walls[next_x][next_y]):
                wall_features.append(1)
            else:
                wall_features.append(0)

        # Search whether there's a ghost within ghost_radius tiles
        ghost_radius = 3
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        for dir, vec in Actions._directionsAsList:
            if self._findGhosts((x_int, y_int), vec, walls, ghosts, ghost_radius=ghost_radius):
                ghost_features.append(1)
            else:
                ghost_features.append(0)

        ghost_mode = 0

        for index in range(1, len(state.data.agentStates)):
            if state.data.agentStates[index].scaredTimer > 0:
                ghost_mode = 1
                break

        features = np.concatenate((wall_features, ghost_features, food_features, [ghost_mode])).astype(int)

        return features

class MultiagentFeatureExtractor(FeatureExtractor):
    def getUnconditionedFeatures(self, state):
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        x, y = state.getPacmanPosition()

        wall_features = []

        # Search whether there's a neighboring wall
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            next_y = y_int + dy
            if (next_x < 0 or next_x == walls.width) or (next_y < 0 or next_y == walls.height) or (walls[next_x][next_y]):
                wall_features.append(1)
            else:
                wall_features.append(0)

        ghost_mode = 0

        for index in range(1, len(state.data.agentStates)):
            if state.data.agentStates[index].scaredTimer > 0:
                ghost_mode = 1
                break

        for ghost in range(len(ghosts)):
            pass
            features = np.concatenate((wall_features, ghost_features, food_features, [ghost_mode])).astype(int)

        return None