import datetime
import heapq
import math

import numpy as np
from numpy import random

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

isRat = True
moves = []

def anneal(corpus, cold=1e-9, alpha=0.9995, stopper=100000):
    i = 0

    temperature = math.sqrt(len(corpus))
    #temperature = 10000
    best_solution = current_solution = nearest_neighbour(corpus)
    best_weight = current_weight = solution_weight(corpus, best_solution)

    while temperature >= cold and i < stopper:
        candidate = list(current_solution)
        l = random.randint(2, len(corpus) - 1)
        i = random.randint(0, len(corpus) - l)

        candidate[i: (i + l)] = reversed(candidate[i: (i + l)])

        candidate_weight = solution_weight(corpus, candidate)
        if candidate_weight < current_weight:
            current_solution = candidate
            current_weight = candidate_weight
            if candidate_weight < best_weight:
                best_solution = candidate
                best_weight = candidate_weight
        else:
            if random.random() < math.exp(-abs(candidate_weight - best_weight) / temperature):
                current_solution = candidate
                current_weight = candidate_weight

        temperature *= alpha
        i += 1

    return best_solution


def solution_weight(corpus, solution):
    return sum(corpus[solution[i]][solution[i + 1]] for i in range(0, len(solution) - 1))


def nearest_neighbour(input_corpus):
    corpus = input_corpus.copy()
    current_node = list(corpus.keys())[0]
    route = []
    visited_nodes = []
    while corpus:
        children = corpus.pop(current_node).copy()
        for node in visited_nodes:
            if node in children.keys():
                children.pop(node)
        visited_nodes.append(current_node)
        route.append(current_node)
        if children:
            current_node = min(children, key=children.get)

    return route


def preprocessing(maze_map, maze_width, maze_height, player_location, opponent_location, pieces_of_cheese,
                  time_allowed):
    if player_location != (0, 0):
        isRat = False

    meta_graph, route_meta_graph = build_meta_graph(maze_map, [player_location] + pieces_of_cheese)

    solution = time_keeper(anneal, meta_graph)
    print(solution_weight(meta_graph, solution))
    print(solution_weight(meta_graph, time_keeper(nearest_neighbour, meta_graph)))
    moves = moves_from_locations(find_route(route_meta_graph, player_location, solution))
    print(time_keeper(nearest_neighbour, meta_graph))

    # maze_map_mirror_processing(maze_map, maze_width, maze_height)
    # pieces_of_cheese_mirror_processing(pieces_of_cheese, maze_width, maze_height)

    return


def build_meta_graph(maze_map, locations):
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
    mirror_width = np.round(maze_width / 2) + 1
    mirror_height = np.round(maze_heigh / 2) + 1
    for location in maze_map:
        if sum(location) < mirror_width + mirror_height:
            mirror_maze_map[location] = {}
            for neighbour in maze_map[location]:
                if sum(neighbour) < mirror_width + mirror_height:
                    mirror_maze_map[location][neighbour] = maze_map[location][neighbour]
    # return mirror_maze_map


def pieces_of_cheese_mirror_processing(pieces_of_cheese, maze_width, maze_heigh):
    mirror_pieces_of_cheese = []
    mirror_width = np.round(maze_width / 2) + 1
    mirror_height = np.round(maze_heigh / 2) + 1
    for location in pieces_of_cheese:
        if sum(location) < mirror_width + mirror_height:
            mirror_pieces_of_cheese.append(location)
    return mirror_pieces_of_cheese


def turn(maze_map, maze_width, maze_height, player_location, opponent_location, player_score, opponent_score,
         pieces_of_cheese, time_allowed):

    a = moves.pop(0)

    return a


def djikstra(start_vertex, graph):
    # Djikstra traversal

    route = [(start_vertex, None)]  # We are starting the route from the start (logic) with a parent classified as None
    priority_queue = []
    visited_nodes = [start_vertex]
    for neighbour in graph[start_vertex].keys():  # We are searching for his kids
        heapq.heappush(priority_queue, (graph[start_vertex][neighbour], (neighbour, start_vertex)))
        visited_nodes.append(neighbour)
    while priority_queue:
        weight, (current_node, parrent_node) = heapq.heappop(priority_queue)
        route.append((current_node, parrent_node))
        for child in list(graph[current_node].keys()):  # We are searching all his kids
            if child not in visited_nodes:  # If he doesnt have a father (ie we didnt visited him earlier)
                visited_nodes.append(child)  # We are adding him to the list
                heapq.heappush(priority_queue, (graph[current_node][child] + weight, (child, current_node)))
    print(route)
    return route


def time_keeper(function, *args):
    start_time = datetime.datetime.now()
    result = function(*args)
    print("The function :", function.__name__, "ran in", datetime.datetime.now() - start_time)
    return result

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