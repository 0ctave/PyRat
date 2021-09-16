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
epsilon = 0.1

width = None
height = None

maze_state = None
cheese_map = None
model = None


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


def training_processing(maze, **opt):
    global model
    model = build_model(maze)

    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()


def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    return model


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


def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


class Episode(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]  # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
