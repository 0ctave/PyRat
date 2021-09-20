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
    a = moves.pop(0)
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


def traversal(start_vertex, graph):
    # BFS traversal
    route = [(start_vertex, None)]  # We are starting the route from the start (logic) with a parent classified as None
    neighbours = []  # We initialize the neighbours list
    liste_enfant = [start_vertex]  # We initialize the list where we are going to store every vertex we had explored
    parent_actuel = start_vertex  # Our parent is now the starting vertex
    for voisin_a_l_origine in graph[start_vertex].keys():  # We are searching for his kids

        neighbours.append((voisin_a_l_origine, parent_actuel))  # We add them to the neighbours list

    while len(neighbours) != 0:  # We search every accessible vertex and dont stop until we visited every vertex
        neighbours, (noeud_actuel, parent_actuel) = pop_from_structure(
            neighbours)  # We take the first child parent couple of the list
        route = push_to_structure(route, (noeud_actuel, parent_actuel))  # We add them to the route
        parent_actuel = noeud_actuel  # Our new parent is now the child

        for enfant in list(graph[noeud_actuel].keys()):  # We are searching all his kids

            if enfant not in liste_enfant:  # If he doesnt have a father (ie we didnt visited him earlier)
                liste_enfant.append(enfant)  # We are adding him to the list
                neighbours = push_to_structure(neighbours, (enfant, parent_actuel))  # And in the routing table

    return route


# Because a parent can have multiple child but a child only have one parent we are starting from the arrival to go to the start
def find_route(routing_table, source_location, target_location):
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