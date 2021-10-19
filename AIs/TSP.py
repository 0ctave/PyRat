# -*- coding: utf-8 -*-
"""djikstra.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cc8QXG2Q-WqKlE-3Y91H5U09KERPlozI

<h1><b><center>How to use this file with PyRat?</center></b></h1>

Google Colab provides an efficient environment for writing codes collaboratively with your group. For us, teachers, it allows to come and see your advancement from time to time, and help you solve some bugs if needed.

The PyRat software is a complex environment that takes as an input an AI file (as this file). It requires some resources as well as a few Python libraries, so we have installed it on a virtual machine for you.

PyRat is a local program, and Google Colab is a distant tool. Therefore, we need to indicate the PyRat software where to get your code! In order to submit your program to PyRat, you should follow the following steps:

1.   In this notebook, click on *Share* (top right corner of the navigator). Then, change sharing method to *Anyone with the link*, and copy the sharing link;
2.   On the machine where the PyRat software is installed, start a terminal and navigate to your PyRat directory;
3.   Run the command `python ./pyrat.py --rat "<link>" <options>`, where `<link>` is the share link copied in step 1. (put it between quotes), and `<options>` are other PyRat options you may need.
python ./pyrat.py --rat "https://colab.research.google.com/drive/1Q55Ye33U8yhjcuekJd7_Z4zDhNfz-O2D"

<h1><b><center>Pre-defined constants</center></b></h1>

A PyRat program should -- at each turn -- decide in which direction to move. This is done in the `turn` function later in this document, which should return one of the following constants:
"""
import numpy as np

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

"""<h1><b><center>Your coding area</center></b></h1>

Please put your functions, imports, constants, etc. between this text and the PyRat functions below. You can add as many code cells as you want, we just ask that you keep things organized (one function per cell, commented, etc.), so that it is easier for the teachers to help you debug your code!
"""

# Import of random module
#
import random

moves = None
import heapq


# python pyrat.py --rat "https://colab.research.google.com/drive/1cc8QXG2Q-WqKlE-3Y91H5U09KERPlozI?usp=sharing" -p 1 -x 15 -y 15 -d 0.5 --random_seed 1

def random_move(b):
    # Return a random move
    all_moves = b
    return random.choice(all_moves)


"""<h1><b><center>PyRat functions</center></b></h1>

The `preprocessing` function is called at the very start of a game. It is attributed a longer time compared to `turn`, which allows you to perform intensive computations. If you store the results of these computations into **global** variables, you will be able to reuse them in the `turn` function.

*Input:*
*   `maze_map` -- **dict(pair(int, int), dict(pair(int, int), int))** -- The map of the maze where the game takes place. This structure associates each cell with the dictionry of its neighbors. In that dictionary of neighbors, keys are cell coordinates, and associated values the number of moves required to reach that neighbor. As an example, `list(maze_map[(0, 0)].keys())` returns the list of accessible cells from `(0, 0)`. Then, if for example `(0, 1)` belongs to that list, one can access the number of moves to go from `(0, 0)` to `(0, 1)` by the code `maze_map[(0, 0)][0, 1)]`.
*   `maze_width` -- **int** -- The width of the maze, in number of cells.
*   `maze_height` -- **int** -- The height of the maze, in number of cells.
*   `player_location` -- **pair(int, int)** -- The initial location of your character in the maze.
*   `opponent_location` -- **pair(int,int)** -- The initial location of your opponent's character in the maze.
*   `pieces_of_cheese` -- **list(pair(int, int))** -- The initial location of all pieces of cheese in the maze.
*   `time_allowed` -- **float** -- The time you can take for preprocessing before the game starts checking for moves.

*Output:*
*   This function does not output anything.
"""


def preprocessing(maze_map, maze_width, maze_height, player_location, opponent_location, pieces_of_cheese,
                  time_allowed):
    global moves

    maze_map_mirror_processing(maze_map, maze_width, maze_height)
    #meta_graph, route_meta_graph = build_meta_graph(maze_map, [player_location] + pieces_of_cheese)
    #print(meta_graph)
    #print(route_meta_graph)
    #a = pieces_of_cheese.pop(0)
    #moves = moves_from_locations(find_route(djikstra(player_location, maze_map), player_location, a))
    # We are getting all the moves here to store them in the global variable moves
    pass


"""The `turn` function is called each time the game is waiting
for the player to make a decision (*i.e.*, to return a move among those defined above).

*Input:*
*   `maze_map` -- **dict(pair(int, int), dict(pair(int, int), int))** -- The map of the maze. It is the same as in the `preprocessing` function, just given here again for convenience.
*   `maze_width` -- **int** -- The width of the maze, in number of cells.
*   `maze_height` -- **int** -- The height of the maze, in number of cells.
*   `player_location` -- **pair(int, int)** -- The current location of your character in the maze.
*   `opponent_location` -- **pair(int,int)** -- The current location of your opponent's character in the maze.
*   `player_score` -- **float** -- Your current score.
*   `opponent_score` -- **float** -- The opponent's current score.
*   `pieces_of_cheese` -- **list(pair(int, int))** -- The location of remaining pieces of cheese in the maze.
*   `time_allowed` -- **float** -- The time you can take to return a move to apply before another time starts automatically.

*Output:*
*   A chosen move among `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT` or `MOVE_RIGHT`.
"""

def build_meta_graph (maze_map, locations) :
    print(maze_map)
    meta_graph = {}
    route_meta_graph = {}
    for location in locations:
        priority_queue = []
        meta_graph[location] = {}
        visited = [location]
        route = [(location, None)]

        for neighbour in maze_map[location]:  # We are searching for his kids
            heapq.heappush(priority_queue, (maze_map[location][neighbour], (neighbour, location)))
            visited.append(neighbour)

            if neighbour in locations:
                meta_graph[location][neighbour] = maze_map[location][neighbour]

        while priority_queue:
            weight, (parent, ancestor) = heapq.heappop(priority_queue)
            route.append((parent, ancestor))
            if parent in locations:
                meta_graph[location][parent] = weight

            for child in maze_map[parent]:
                if child not in visited:
                    heapq.heappush(priority_queue, (maze_map[parent][child] + weight, (child, parent)))
                    visited.append(child)

        route_meta_graph[location] = route
    return meta_graph, route_meta_graph

def maze_map_mirror_processing(maze_map, maze_width, maze_heigh):
    mirror_maze_map = {}
    for location in maze_map:
        mirror_maze_map[location] = {}
        for neighbour in maze_map[location]:
            #if maze_map
            mirror_maze_map[location][neighbour] = maze_map[location][neighbour]

    print(maze_map)
    print(mirror_maze_map)
    pass


def build_mirror_meta_graph (maze_map, maze_width, maze_heigh, locations):

    maze_map = maze_map_mirror_processing(maze_map, maze_width, maze_heigh)

    meta_graph = {}
    route_meta_graph = {}
    for location in locations:
        priority_queue = []
        meta_graph[location] = {}
        visited = [location]
        route = [(location, None)]

        for neighbour in maze_map[location]:  # We are searching for his kids
            heapq.heappush(priority_queue, (maze_map[location][neighbour], (neighbour, location)))
            visited.append(neighbour)

            if neighbour in locations:
                meta_graph[location][neighbour] = maze_map[location][neighbour]

        while priority_queue:
            weight, (parent, ancestor) = heapq.heappop(priority_queue)
            route.append((parent, ancestor))
            if parent in locations:
                meta_graph[location][parent] = weight

            for child in maze_map[parent]:
                if child not in visited:
                    heapq.heappush(priority_queue, (maze_map[parent][child] + weight, (child, parent)))
                    visited.append(child)

        route_meta_graph[location] = route
    return meta_graph, route_meta_graph

def move_from_locations(source_location, target_location):
    difference = (target_location[0] - source_location[0], target_location[1] - source_location[1])
    if difference == (0, -1):
        return MOVE_DOWN
    elif difference == (0, 1):
        return MOVE_UP
    elif difference == (1, 0):
        return MOVE_RIGHT
    elif difference == (-1, 0):
        return MOVE_LEFT
    else:
        raise Exception("Impossible move")


def turn(maze_map, maze_width, maze_height, player_location, opponent_location, player_score, opponent_score,
         pieces_of_cheese, time_allowed):
    # We are executing the first move of the list
    global moves
    if moves:
        a = moves.pop(0)
    else:
        a = random_move([MOVE_DOWN, MOVE_UP, MOVE_RIGHT, MOVE_LEFT])
        print("RANDOM MOVES !")

    return a


def djikstra(start_vertex, graph):
    # Djikstra traversal

    route = [(start_vertex, None)]  # We are starting the route from the start (logic) with a parent classified as None
    priority_queue = []
    visited = [start_vertex]
    for neighbour in graph[start_vertex]:  # We are searching for his kids
        heapq.heappush(priority_queue, (graph[start_vertex][neighbour], (neighbour, start_vertex)))
        visited.append(neighbour)

    while priority_queue:
        weight, (parent, ancestor) = heapq.heappop(priority_queue)
        route.append((parent, ancestor))
        for child in graph[parent]:
            if child not in visited:
                heapq.heappush(priority_queue, (graph[parent][child] + weight, (child, parent)))
                visited.append(child)

    return route


# Because a parent can have multiple child but a child only have one parent we are starting from the arrival to go to the start
def find_route(routing_table, source_location, target_location):
    tmp = None
    a = None
    len(routing_table)
    route = [target_location]  # We start our route at the arrival
    while a != source_location:  # Until we dont arrive at the beggining we continue the search for a road
        i = 0  # We use this variable to explore the list by iterations
        while tmp != target_location:  # We try to reach the arrival
            tmp, a = routing_table[i]  # For this we are parcouring the list
            i += 1
        target_location = a  # Once we have found the arrival we place the arrival on the parent of the arrival
        route.append(a)  # And we add the parent to the road

    return route


def moves_from_locations(locations):
    moves = []  # We initiate the variable
    i = len(locations) - 1  # We are parcouring the list from the end
    while i > 0:  # Until we doesnt reach the end of locations we continue
        source = locations[i]
        destination = locations[i - 1]
        moves.append(move_from_locations(source, destination))  # We add to the queue the next move
        i -= 1

    return moves
