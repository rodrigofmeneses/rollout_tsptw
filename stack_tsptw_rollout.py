#%%
import numpy as np
from numba import njit
import time
#%%
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

@njit
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
#%%

@njit
def is_feasible(solution, distance_matrix, intervals):

    num_nodes = distance_matrix.shape[0]
    cost = 0
    current_node = solution[0]
    node_flags = np.ones(num_nodes)

    for i in range(1, num_nodes + 1):
        # TSP constraint
        if node_flags[solution[i]] == True:
            node_flags[solution[i]] = False
        else:
            return False
        
        # Windows Constraint
        if cost < intervals[current_node][0]:
            cost = intervals[current_node][0]
        elif cost > intervals[current_node][1]:
            return False
        cost += distance_matrix[current_node, solution[i]]
        current_node = solution[i]
    
    return True

#%%
@njit
def backtracking(solution, distance_matrix, intervals):
    
    num_nodes = distance_matrix.shape[0]
    all_states = np.arange(num_nodes)
    solutions_stack = []
    
    solutions_stack.append(solution)

    while len(solutions_stack) != 0:

        current_solution = solutions_stack.pop()
        c_step = num_nodes - 2
        for i in range(2, num_nodes + 2):
            if current_solution[i] == 0:
                c_step = i
                break
        
        if is_feasible(current_solution[1:], distance_matrix, intervals):
            return current_solution
        
        action_list = np.zeros((num_nodes, num_nodes + 2), dtype=np.int32)
        for i in range(num_nodes):
            step = all_states[i]
            if step in current_solution[1:]:
                continue
            one_step_time = current_solution[0] + distance_matrix[current_solution[c_step - 1], step]
            if one_step_time <= intervals[step][1]:
                action_list[i] = current_solution[:]
                action_list[i][0] = max(one_step_time, intervals[step][0])
                action_list[i][c_step] = step
                
        action_list = action_list[np.argsort(action_list[:, 0])][::-1]

        for i in range(num_nodes):
            if action_list[i][0] != 0:
                solutions_stack.append(action_list[i])
        
    return current_solution

#%%
@njit
def one_step_viability(solution, current_time, distance_matrix, intervals):
    all_states = np.arange(distance_matrix.shape[0], dtype=np.int32)
    actions = np.zeros(1, dtype=np.int32)[:-1]

    for step in all_states:
        if step in solution:
            continue
        elif distance_matrix[solution[-1], step] == np.int32(999999):
            continue
        one_step_time = current_time + distance_matrix[solution[-1], step]
        if one_step_time <= intervals[step][1]:
            actions = np.append(actions, step)
    
    return actions

#%%
def rollout_algorithm(problem, starting_node=0):
    # Distance Matrix
    distance_matrix = np.array(problem['distance_matrix'], dtype=np.int32)
    intervals = np.array(problem['intervals'], dtype=np.int32)
    current_time = 0
    # Number of cities
    num_cities = np.array(problem['dimension'], dtype=np.int32) 
    
    # Initial solution
    solution = [0, starting_node]
    # solution = np.array([starting_node], dtype=np.int32)

    # Rollout Algorithm run for num_cities - 1 steps
    for _ in range(num_cities - 1):
        # Initialize a copy to current solution
        current_solution = solution.copy()
        # What we want optimize, rollout cost
        best_rollout_cost = np.int32(999999)
        # Best next city!
        best_next_city = None
        
        # Run over viable cities not visiteds
        for j in one_step_viability(np.array(current_solution[1:], dtype=np.int32), current_solution[0], distance_matrix, intervals):
            # Adding candidate next city
            current_solution.append(j)
            auxiliar_time = current_time + distance_matrix[current_solution[-2], current_solution[-1]]
            current_solution[0] = auxiliar_time

            # Run Base Policy
            aux_sol = np.zeros(num_cities + 2, dtype=np.int32)
            aux_sol[:len(current_solution)] = current_solution
            backtracking_solution = backtracking(aux_sol, distance_matrix, intervals)
            # rollout_cost = calculate_solution_cost(nn_solution, List(distance_matrix))
            if not is_feasible(backtracking_solution[1:], distance_matrix, intervals):
                rollout_cost = np.int32(999999)
            else:
                rollout_cost = calculate_solution_cost(backtracking_solution[1:], distance_matrix)
            # Tests to optimize costs.
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_next_city = j
                best_auxiliar_time = auxiliar_time
            
            # Remove cadidate
            current_solution.pop()
        # Adding best next city
        solution.append(best_next_city)
        solution[0] = max(best_auxiliar_time, intervals[best_next_city][0])
        current_time = solution[0]
    # End of algorithm with start city
    solution.append(starting_node)

    return solution

#%%

def experiments_with(problem):
    
    # execute nearest neighbor algorithm and calculate time
    start = time.perf_counter()
    backtracking_solution = backtracking(np.zeros(problem['dimension'] + 2, dtype=np.int32), \
        np.array(problem['distance_matrix'], dtype=np.int32), \
        np.array(problem['intervals'], dtype=np.int32))
    backtracking_time = time.perf_counter() - start
    backtracking_cost = calculate_solution_cost(backtracking_solution[1:], problem['distance_matrix'])
    backtracking_solution[0] = int(backtracking_cost)

    start = time.perf_counter()
    rollout_solution = rollout_algorithm(problem)
    rollout_time = time.perf_counter() - start
    rollout_cost = calculate_solution_cost(np.array(rollout_solution[1:], dtype=np.int32), problem['distance_matrix'])
    rollout_solution[0] = int(rollout_cost)

    return rollout_solution, rollout_cost, rollout_time, backtracking_solution, backtracking_cost, backtracking_time

#%%

# for i in [20, 40, 60]:
for i in [20]:
    # to cache
    path = f'instances/tsptw_data/DumasEtAl/n{i}w20.001.txt'
    problem = read_data(path)
    backtracking_solution = backtracking(np.zeros(problem['dimension'] + 2, dtype=np.int32), \
            np.array(problem['distance_matrix'], dtype=np.int32), \
            np.array(problem['intervals'], dtype=np.int32))
    for j in [20, 40, 60, 80]:
    # for j in [20]:
        # Create Results file
        results = open(f'experiments/results_{time.strftime("%d%b%Y_%H_%M_%S", time.gmtime())}.txt', 'w')
        # Write header
        results.write('instance_name,rol_cost,rol_time,backtracking_cost,backtracking_time\n')
        for k in [1, 2, 3, 4, 5]:
            # Instance Name 
            instance = f'n{i}w{j}.00{k}'

            # Start tests
            path = f'instances/tsptw_data/DumasEtAl/{instance}.txt'
            problem = read_data(path)
            pre_process(problem['distance_matrix'], problem['intervals'])
            rollout_solution, rollout_cost, rollout_time, backtracking_solution, backtracking_cost, backtracking_time = experiments_with(problem)
            results.write(f'{instance},{rollout_cost},{rollout_time},{backtracking_cost},{backtracking_time}\n')
            results.write('rollout:      ' + str(rollout_solution) +'\n')
            results.write('backtracking: ' + str(list(backtracking_solution)) +'\n')
    
    results.close()


#%%
# path = '/home/rodrigomeneses/Documents/repositorios/rollout_tsptw/instances/tsptw_data/DumasEtAl/n20w20.001.txt'
path = 'instances/tsptw_data/DumasEtAl/n40w20.001.txt'
problem = read_data(path)
pre_process(problem['distance_matrix'], problem['intervals'])

#%%

# solution = np.array([0, 16, 9, 19, 17, 18, 10, 5, 15, 1, 11, 12, 6, 13, 7, 2, 4, 8, 20, 3, 14, 0])
# solution = np.array([1, 2, 15, 19, 5, 9, 8, 6, 16, 12, 17, 20, 18, 11, 13, 4, 7, 3, 21, 10, 14, 1]) - 1
# solution = np.array([1, 6, 20, 14, 2, 9, 17, 5, 10, 4, 8, 16, 15, 18, 12, 3, 11, 19, 21, 7, 13, 1]) - 1
# solution = np.array([1, 12, 4, 3, 20, 8, 16, 10, 9, 6, 7, 11, 15, 5, 13, 17, 19, 14, 21, 18, 2, 1]) - 1
# solution = np.array([1, 20, 12, 8, 19, 17, 14, 9, 4, 18, 3, 6, 11, 5, 16, 10, 15, 7, 21, 2, 13, 1]) - 1
# solution = backtraking_greed_policy(np.array([0], dtype=np.int32), 0, np.array(problem['distance_matrix'], dtype=np.int32), np.array(problem['intervals'], dtype=np.int32))
distance_matrix = np.array(problem['distance_matrix'], dtype=np.int32)
intervals = np.array(problem['intervals'], dtype=np.int32)

# rollout_algorithm(problem)
start = time.perf_counter()
solution = backtracking(np.zeros(problem['distance_matrix'].shape[0] + 2, dtype=np.int32), distance_matrix, intervals)
print(solution)
print(is_feasible(solution[1:], distance_matrix, intervals))
print(time.perf_counter() - start)
# print(is_feasible(solution, problem['distance_matrix'], problem['intervals']))

#%%
