import numpy as np
import time
from numba import jit

def read_data(path):
    instance_author = path.split('/')[-2]
    coordinates = []
    intervals = []

    if instance_author == 'DaSilvaUrrutia':
        with open(path, 'r') as reader:
            reader.readline()
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))

    elif instance_author == 'DumasEtAl':
        with open(path, 'r') as reader:        
            [reader.readline() for i in range(6)]
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                    
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))
            
            coordinates.pop()
            intervals.pop()

    elif instance_author == 'GendreauEtAl':
        with open(path, 'r') as reader:        
            [reader.readline() for i in range(6)]
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                    
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))
            
            coordinates.pop()
            intervals.pop()

    elif instance_author == 'OhlmannThomas':
        with open(path, 'r') as reader:        
            [reader.readline() for i in range(6)]
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                    
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))
            
            coordinates.pop()
            intervals.pop()

    dimension = len(coordinates)

    distance_matrix = np.zeros((dimension, dimension))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            px = np.power(coordinates[i][0] - coordinates[j][0], 2)
            py = np.power(coordinates[i][1] - coordinates[j][1], 2)
            dist = np.floor(np.sqrt(px + py))
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    problem = {
        'coordinates': coordinates,
        'intervals': intervals,
        'distance_matrix': distance_matrix,
        'dimension': dimension
    }
    
    return problem

def pre_process(distance_matrix, intervals):
    for i in range(len(intervals)):
        for j in range(len(intervals)):
            if intervals[i][0] + distance_matrix[i, j] > intervals[j][1]:
                distance_matrix[i, j] = np.int32(999999)

def calculate_solution_cost(solution, distance_matrix):
    '''
        Calc cost of solution.
        solution: solution of traveler, ex:
        [0, 2, 4, 1, 3, 0]  <- for 5 cities and starting city is 0 
        distance_matrix: costs of arcs
        return: cost of solution
    '''
    cost = 0.
    for i in range(len(solution) - 1):
        cost += distance_matrix[solution[i]][solution[i+1]]
    return cost

def is_feasible(solution, distance_matrix, intervals):
    cost = 0
    current_node = solution[0]
    node_flags = [True] * len(solution)

    for i in solution[1:]:

        # TSP constraint
        if node_flags[i] == True:
            node_flags[i] = False
        else:
            return False
        
        # Windows Constraint
        if cost < intervals[current_node][0]:
            cost = intervals[current_node][0]
        elif cost > intervals[current_node][1]:
            return False
        cost += distance_matrix[current_node, i]
        current_node = i
    
    return True

@jit(nopython=True)
def best_step(current_state, current_time, actions, distance_matrix, intervals):
    best_step = 0
    best_distance = np.int32(999999)
    for step in actions:
        if current_time < intervals[step][0]:
            current_distance = intervals[step][0] + distance_matrix[current_state, step]
        else:
            current_distance = current_time + distance_matrix[current_state, step]
        if current_distance < best_distance:
            best_distance = current_distance
            best_step = step
    return np.int32(best_step)

@jit(nopython=True)
def one_step_viability(solution, current_time, distance_matrix, intervals):
    all_states = np.arange(distance_matrix.shape[0], dtype=np.int32)
    actions = np.zeros(1, dtype=np.int32)[:-1]

    for step in all_states:
        if step in solution:
            continue
        elif distance_matrix[solution[-1], step] == np.int32(999999):
            continue
        one_step_time = current_time + distance_matrix[solution[-1], step]
        if one_step_time < intervals[step][1]:
            actions = np.append(actions, step)
    
    return actions



@jit(nopython=True)
def backtraking_greed_policy(solution, current_time, distance_matrix, intervals):
    i = 0
    num_nodes = distance_matrix.shape[0]
    level = []
    # level = np.zeros(1, dtype=np.int32)[:-1]
    time_control = np.zeros(num_nodes, dtype=np.int32)
    time_control[i] = current_time
    # time_control = [current_time]
    
    while solution.size != num_nodes:
        # Existem ações pra seguir?
        if i >= len(level):
            actions = one_step_viability(solution, current_time, distance_matrix, intervals)
            # level.append(actions.copy())
            level.append(actions)
            if i == 0 and actions.size == 0:
                # Infeasible
                print('Infeasible')
                return solution
            continue

        if len(level[i]) != 0:
            # step = np.random.choice((level[i]))
            step = best_step(solution[-1], current_time, (level[i]), distance_matrix, intervals)
            rm = np.where(level[i] == step)[0][0]
            
            level[i] = np.delete(level[i], rm)
            solution = np.append(solution, step)

            if current_time + distance_matrix[solution[-2], solution[-1]]\
                <= intervals[solution[-1]][0]:
                current_time = intervals[solution[-1]][0]
            else:
                current_time += distance_matrix[solution[-2], solution[-1]]
            
            i += 1
            time_control[i] = current_time
        else:
            level.pop()
            solution = solution[:-1]
            i -= 1
            current_time = time_control[i]
            time_control[i + 1] = 0

    return solution

def rollout_algorithm(problem, starting_node=0):
    # Distance Matrix
    distance_matrix = problem['distance_matrix']
    # Number of cities
    num_cities = problem['dimension']
    # Initial solution
    # solution = List([starting_node])
    solution = [starting_node]

    # Rollout Algorithm run for num_cities - 1 steps
    for _ in range(num_cities - 1):
        # Initialize a copy to current solution
        current_solution = solution.copy()
        # What we want optimize, rollout cost
        best_rollout_cost = np.inf
        # Best next city!
        best_next_city = None
        
        # Run over cities not visiteds
        for j in set(range(num_cities)) - set(solution):
            # Adding candidate next city
            current_solution.append(j)

            # Run Base Policy
            nn_solution = backtraking_greed_policy(problem, current_solution.copy(), distance_matrix.copy())
            # rollout_cost = calculate_solution_cost(nn_solution, List(distance_matrix))
            rollout_cost = calculate_solution_cost(nn_solution, distance_matrix)
            # Tests to optimize costs.
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_next_city = j
            
            # Remove cadidate
            current_solution.pop()
        # Adding best next city
        solution.append(best_next_city)
    # End of algorithm with start city
    solution.append(starting_node)

    return solution

np.random.seed(13123123)
# path = '/home/rodrigomeneses/Documents/repositorios/rollout_tsptw/instances/tsptw_data/DumasEtAl/n20w20.002.txt'
path = 'instances/tsptw_data/DumasEtAl/n20w20.002.txt'
problem = read_data(path)
pre_process(problem['distance_matrix'], problem['intervals'])

# solution = np.array([1, 17, 10, 20, 18, 19, 11, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15, 1]) - 1
solution = np.array([1, 2, 15, 19, 5, 9, 8, 6, 16, 12, 17, 20, 18, 11, 13, 4, 7, 3, 21, 10, 14, 1]) - 1
# solution = np.array([1, 6, 20, 14, 2, 9, 17, 5, 10, 4, 8, 16, 15, 18, 12, 3, 11, 19, 21, 7, 13, 1]) - 1
# solution = np.array([1, 12, 4, 3, 20, 8, 16, 10, 9, 6, 7, 11, 15, 5, 13, 17, 19, 14, 21, 18, 2, 1]) - 1
# solution = np.array([1, 20, 12, 8, 19, 17, 14, 9, 4, 18, 3, 6, 11, 5, 16, 10, 15, 7, 21, 2, 13, 1]) - 1
print(calculate_solution_cost(solution, problem['distance_matrix']))
print(is_feasible(solution, problem['distance_matrix'], problem['intervals']))
print(backtraking_greed_policy(np.array([0], dtype=np.int32), 0, np.array(problem['distance_matrix'], dtype=np.int32), np.array(problem['intervals'], dtype=np.int32)))