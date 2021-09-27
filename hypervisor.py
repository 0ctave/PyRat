from __future__ import print_function
import numpy as np
import os, sys, time, datetime, json, random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import random
import multiprocessing
import pyrat

epsilon = 0.1

MOVE_DOWN = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
MOVE_UP = 3

actions_dict = {
    MOVE_DOWN: 'D',
    MOVE_LEFT: 'L',
    MOVE_RIGHT: 'R',
    MOVE_UP: 'U',
}


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
            # q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * q_sa
        return inputs, targets


def build_model():
    model = Sequential()
    model.add(Dense(36, input_shape=(36,)))
    model.add(PReLU())
    model.add(Dense(36))
    model.add(PReLU())
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    return model


def pyrat_instance(child_link):
    args = ["--rat", "AIs/RNN.py", "-x", "4", "-y", "4", "-p", "2", "--start_random", "-mt", "50", "--rnn",
            "--synchronous",
            "--auto_exit",
            "--preparation_time", "0",
            "--nodrawing",
            ]

    pyrat.main_bis(child_link, args)


def maze_processing(maze_map):
    maze_state = np.zeros(len(maze_map) * 4)
    count = 0
    for location in maze_map:
        x = location[0]
        y = location[1]
        for i in range(0, 4):
            pos = (x + (i - 1 if i % 2 == 0 else 0), y + (i - 2 if i % 2 == 1 else 0))
            if pos in maze_map[location]:
                maze_state[count * 4 + i] = maze_map[location][pos]
            else:
                maze_state[count * 4 + i] = 100

        count += 1
    return maze_state


def state_observer(maze_state, width, player_location, pieces_of_cheese):
    tmp = np.copy(maze_state)
    for i in range(0, 4):
        tmp[width * player_location[0] + player_location[1] + i] -= 50

    cheese_map = np.zeros(len(maze_state))
    for location in pieces_of_cheese:
        x = location[0]
        y = location[1]
        for i in range(0, 4):
            cheese_map[(width * x + y) * 4 + i] = 1

    return np.add(tmp, -100 * cheese_map).reshape((1, -1))


def register_episode(experience, prev_maze_state, action, reward, maze_state, game_over):
    episode = [prev_maze_state, action, reward, maze_state, game_over]
    experience.remember(episode)


def maze_episodes(maze_state, width, model, experience, win_history, data_size, child):
    loss = 0.0

    # get initial envstate (1d flattened canvas)

    n_episodes = 0
    action = MOVE_DOWN
    state = state_observer(maze_state, width, [0, 0], [])

    game_over = False
    print(child.recv())
    while not game_over:

        prev_state = state

        cheese_map = child.recv()
        player_position = child.recv()
        state = state_observer(maze_state, width, player_position, cheese_map)
        # Get next action

        if np.random.rand() < epsilon:
            action = random.choice([MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP])
        else:
            action = np.argmax(experience.predict(prev_state))


        # Apply action, get reward and new envstate
        child.send(actions_dict[action])

        reward = child.recv()
        print("reward : ", reward)

        game_state = child.recv()

        if game_state == ["win"]:
            win_history.append(1)
            game_over = True
        elif game_state == ["lose"]:
            win_history.append(0)
            game_over = True


        register_episode(experience, prev_state, action, reward, state, False)
        n_episodes += 1

        # Train neural network model
        inputs, targets = experience.get_data(data_size=data_size)
        h = model.fit(
            inputs,
            targets,
            epochs=8,
            batch_size=16,
            verbose=0,
        )
        loss = model.evaluate(inputs, targets, verbose=0)


    print("FINISHED")

    return n_episodes, loss


def training_processing(**opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    model = build_model()

    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    experience = Episode(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    hsize = 1024 // 2  # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0

        parent_link, child_link = multiprocessing.Pipe()

        p = multiprocessing.Process(target=pyrat_instance, args=(child_link,))
        p.start()
        width = parent_link.recv()
        maze_map = parent_link.recv()
        maze_state = maze_processing(maze_map)
        n_episodes, loss = maze_episodes(maze_state, width, model, experience, win_history, data_size, parent_link)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9: epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize:  # and completion_check(model, qmaze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds


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


if __name__ == '__main__':
    training_processing(epochs=1, max_memory=8 * 64, data_size=64)
