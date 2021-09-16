##############################################################
# The turn function should always return a move to indicate where to go
# The four possibilities are defined here
##############################################################
import numpy as np

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'



##############################################################
# Please put your code here (imports, variables, functions...)
##############################################################

# Import of random module
import random




def random_move () :
    
    # Returns a random move
    all_moves = [MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP]
    return random.choice(all_moves)



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

def preprocessing (maze_map, maze_width, maze_height, player_location, opponent_location, pieces_of_cheese, time_allowed) :
    maze = np.zeros(len(maze_map) * 4)
    count = 0
    for location in maze_map:
        x = location[0]
        y = location[1]
        for i in range(0, 4):
            pos = (x + (i - 1 if i % 2 == 0 else 0), y + (i - 2 if i % 2 == 1 else 0))
            print(pos)
            if pos in maze_map[location]:
                maze[count * 4 + i] = maze_map[location][pos]
            else:
                maze[count * 4 + i] = 0
        count+=1
    # Nothing to do here
    pass



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

def turn (maze_map, maze_width, maze_height, player_location, opponent_location, player_score, opponent_score, pieces_of_cheese, time_allowed) :

    # Returns a random move each turn
    return random_move()



