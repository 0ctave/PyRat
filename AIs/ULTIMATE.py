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


def anneal(corpus, temperature):
    alpha = 0.9995
    cold_temperature = 0.00000001
    stopper = 10000000
    i = 0
    best_solution = nearest_neighbour(corpus)
    best_weight = solution_weight(corpus, best_solution)

    current_solution = nearest_neighbour(corpus)
    current_weight = best_weight

    while temperature >= cold_temperature and i < stopper:
        candidate = np.copy(best_solution)
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

def solution_weight(corpus, solution):
    return sum(corpus[solution[i]][solution[i + 1]] for i in range(0, len(solution) - 1))


def nearest_neighbour(corpus):
    current_node = list(corpus.keys())[0]
    route = []
    visited_nodes = []
    print(corpus)
    while corpus:
        children = corpus.pop(current_node)
        print(visited_nodes)
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
    print(meta_graph)
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
    return MOVE_UP


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
