# %%
"""
<h1><b><center>How to use this file with PyRat?</center></b></h1>
"""

# %%
"""
Google Colab provides an efficient environment for writing codes collaboratively with your group. For us, teachers, it allows to come and see your advancement from time to time, and help you solve some bugs if needed.

The PyRat software is a complex environment that takes as an input an AI file (as this file). It requires some resources as well as a few Python libraries, so we have installed it on a virtual machine for you.

PyRat is a local program, and Google Colab is a distant tool. Therefore, we need to indicate the PyRat software where to get your code! In order to submit your program to PyRat, you should follow the following steps:

1.   In this notebook, click on *Share* (top right corner of the navigator). Then, change sharing method to *Anyone with the link*, and copy the sharing link;
2.   On the machine where the PyRat software is installed, start a terminal and navigate to your PyRat directory;
3.   Run the command `python ./pyrat.py --rat "<link>" <options>`, where `<link>` is the share link copied in step 1. (put it between quotes), and `<options>` are other PyRat options you may need.
python ./pyrat.py --rat "https://colab.research.google.com/drive/1Q55Ye33U8yhjcuekJd7_Z4zDhNfz-O2D"
"""

# %%
"""
<h1><b><center>Pre-defined constants</center></b></h1>
"""

# %%
"""
A PyRat program should -- at each turn -- decide in which direction to move. This is done in the `turn` function later in this document, which should return one of the following constants:
"""

# %%
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'


# %%
"""
<h1><b><center>Your coding area</center></b></h1>
"""

# %%
"""
Please put your functions, imports, constants, etc. between this text and the PyRat functions below. You can add as many code cells as you want, we just ask that you keep things organized (one function per cell, commented, etc.), so that it is easier for the teachers to help you debug your code!
"""

# %%
# Import of random module
#
import random
import numpy as np
moves=None
import heapq
#python pyrat.py --rat "https://colab.research.google.com/drive/1hMpNtO_EjwjsuyJDNSg2x4y467cMpeTW?usp=sharing" -p 5 -x 15 -y 15 -d 0.5 --random_seed 1


# %%
def random_move (b) :

    # Return a random move
    all_moves = b
    return random.choice(all_moves)

# %%
"""
<h1><b><center>PyRat functions</center></b></h1>
"""

# %%
"""
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

# %%
def preprocessing (maze_map, maze_width, maze_height, player_location, opponent_location, pieces_of_cheese, time_allowed) :
    global moves
    graph,route_meta=build_meta_graph(maze_map, pieces_of_cheese + [player_location])
    moves=moves_from_locations(tsp(graph,player_location,route_meta,pieces_of_cheese))
    pass

# %%
"""
The `turn` function is called each time the game is waiting
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

# %%
def build_meta_graph (maze_map, locations) :
    meta_graph = {} #We initialize the dictionnary where we are going to store the meta_graph vertices
    route_meta_graph = {}#We initialize the dictionnary where we are going to store the route in the graph between the vertices of the meta_graph
    for location in locations: #We parcour the list of cheeses to define them as vertices in our meta graph
        priority_queue = []#We are going to use the priority queue because we are going to use a djikstra to find the best way
        meta_graph[location] = {}#We initialize the road from the cheese location to other cheeses
        visited = [location]#We note the cheese as visited
        route = [(location, None)]#So this is going to be the start of our route

        for neighbour in maze_map[location]:  # We are searching for his kids
            heapq.heappush(priority_queue, (maze_map[location][neighbour], (neighbour, location)))#We push the neighbour in the priority queue
            visited.append(neighbour)#We note the case as visited

            if neighbour in locations:#If the neighbour is a cheese then 
                meta_graph[location][neighbour] = maze_map[location][neighbour]#we add it to the meta graph

        while priority_queue:#As long as we have vertices to go
            weight, (parent, ancestor) = heapq.heappop(priority_queue)#We get them with their weight and we remove them from the priority queue
            route.append((parent, ancestor))#We add them to the road
            if parent in locations:#If it's a cheese 
                meta_graph[location][parent] = weight#We add it to the meta graph

            for child in maze_map[parent]:#We parcour the neighbour of the vertex
                if child not in visited:#If we didnt went to the neighbour
                    heapq.heappush(priority_queue, (maze_map[parent][child] + weight, (child, parent)))#We add him to the priority queue
                    visited.append(child)#We marked him as visited

        route_meta_graph[location] = route#We add to the dictionnary the road from a cheese to another
    return meta_graph, route_meta_graph

# %%
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
    a=moves.pop(0)
    return a

def create_structure():
    # Create an empty FIFO
    structure = []
    return structure


def push_to_structure(structure, element):
    # Add an element to the FIFO
    structure.append(element)
    return structure


def pop_from_structure(structure):
    # Extract an element from the FIFO
    a = structure.pop(0)
    return structure, a


def djikstra(start_vertex, graph):
    # Djikstra traversal

    route = [(start_vertex, None)]  # We are starting the route from the start (logic) with a parent classified as None
    priority_queue = []
    visited_nodes=[start_vertex]
    for neighbour in graph[start_vertex].keys():  # We are searching for his kids
        heapq.heappush(priority_queue,(graph[start_vertex][neighbour],(neighbour,start_vertex)))
        visited_nodes.append(neighbour)
    while priority_queue:
          weight,a=heapq.heappop(priority_queue)
          current_node,parrent_node=a
          route = push_to_structure(route, (current_node, parrent_node))
          parrent_node=current_node
          for child in list(graph[current_node].keys()):  # We are searching all his kids

            if child not in visited_nodes:  # If he doesnt have a father (ie we didnt visited him earlier)
                visited_nodes.append(child)  # We are adding him to the list
                heapq.heappush(priority_queue,(graph[current_node][child]+weight,(child,current_node)))
    print(route)
    return route


# Because a parent can have multiple child but a child only have one parent we are starting from the arrival to go to the start
def find_route(routing_table, source_location, target_location):
    print("routing", routing_table)
    print("source", source_location)
    print("target", target_location)

    tmp = None
    a = None
    route = [target_location]  # We start our route at the arrival
    while a != source_location:  # Until we dont arrive at the beggining we continue the search for a road
        i = 0  # We use this variable to explore the list by iterations
        while tmp != target_location:  # We try to reach the arrival
            tmp, a = routing_table[i]  # For this we are parcouring the list
            i += 1
        target_location = a  # Once we have found the arrival we place the arrival on the parent of the arrival
        route.append(a)  # And we add the parent to the road

    return route[::-1]


def moves_from_locations(locations):
    print("locations", locations)
    moves = []  # We initiate the variable
    i = len(locations) - 1  # We are parcouring the list from the end
    while i > 0:  # Until we doesnt reach the end of locations we continue
        source = locations[i]
        destination = locations[i - 1]
        moves.append(move_from_locations(source, destination))  # We add to the queue the next move
        i -= 1

    return moves

def tsp(graph, initial_vertex,route_meta,pieces_of_cheese) :
    # Recursive implementation of a TSP
    print(graph)
    weight=100000
    best_path=[]
    def _tsp(visited_vertex, current_lenght) :
          nonlocal weight,best_path  
          if len(visited_vertex)==len(pieces_of_cheese)+1: # Condition to exit recursion
            if current_lenght<weight:
              weight=current_lenght
              best_path=visited_vertex
            return
          for neighbour in graph[visited_vertex[len(visited_vertex)-1]]: # Create the sub tree of solutions
            if neighbour not in visited_vertex :
              w=graph[visited_vertex[len(visited_vertex)-1]][neighbour]
              _tsp(visited_vertex+[neighbour],current_lenght+w) # Call _tsp on the leafs of the tree
            else :
              continue   
    _tsp([initial_vertex],0)

    print(best_path)
    path = [(0, 0)]
    for i in range(0, len(best_path) - 1):
        path = path[:-1] + find_route(route_meta[best_path[i]], best_path[i], best_path[i+1])
    
    return path[::-1]

        