from __future__ import print_function
import numpy as np
import os, sys, time, datetime, json, random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import random

##############################################################
# The turn function should always return a move to indicate where to go
# The four possibilities are defined here
##############################################################

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

##############################################################
# Please put your code here (imports, variables, functions...)
##############################################################
width = None
height = None

maze_state = None
cheese_map = None

def maze_processing(maze_map, pieces_of_cheese):
    global maze_state
    global cheese_map
    maze = np.zeros(len(maze_map) * 4)
    cheese_map = np.zeros(len(maze_map) * 4)
    count = 0
    for location in maze_map:
        x = location[0]
        y = location[1]
        for i in range(0, 4):
            pos = (x + (i - 1 if i % 2 == 0 else 0), y + (i - 2 if i % 2 == 1 else 0))
            if pos in maze_map[location]:
                maze[count * 4 + i] = maze_map[location][pos]
            else:
                maze[count * 4 + i] = 100
            if pos in pieces_of_cheese:
                cheese_map[count * 4 + i] = 1
        count += 1
    return maze, cheese_map


def state_observer(player_location):
    global width
    global maze_state
    global cheese_map

    for i in range(0, 4):
        maze_state[width * player_location[0] + player_location[1] + i] -= 50

    # if(player_location):

    state_map = np.add(maze_state, -100 * cheese_map)
    print(state_map)


def completion_check(pieces_of_cheese):
    return len(pieces_of_cheese) == 0

##############################################################
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
# ------------------------------------------------------------
# maze_map : dict(pair(int, int), dict(pair(int, int), int))
# maze_width : int
# maze_height : int
# player_location : pair(int, int)
# opponent_location : pair(int,int)
# pieces_of_cheese : list(pair(int, int))
# time_allowed : float
##############################################################

def preprocessing(maze_map, maze_width, maze_height, player_location, opponent_location, pieces_of_cheese,
                  time_allowed):
    global height

    height = maze_height
    maze_processing(maze_map, pieces_of_cheese)


##############################################################
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
# ------------------------------------------------------------
# maze_map : dict(pair(int, int), dict(pair(int, int), int))
# maze_width : int
# maze_height : int
# player_location : pair(int, int)
# opponent_location : pair(int,int)
# player_score : float
# opponent_score : float
# pieces_of_cheese : list(pair(int, int))
# time_allowed : float
##############################################################

def turn(maze_map, maze_width, maze_height, player_location, opponent_location, player_score, opponent_score,
         pieces_of_cheese, time_allowed):
    global model
    all_moves = [MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP]

   # prediction = model.predict(state_observer(player_location))

    #if np.random.rand() < epsilon:
    #
    #else:
    #    action = np.argmax(experience.predict(prev_envstate))

    #action = np.argmax(prediction[0])
    action = random.choice(all_moves)

    return action