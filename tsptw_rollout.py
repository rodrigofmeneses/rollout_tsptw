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
    time_control = np.zeros(num_nodes, dtype=np.int32)
    time_control[i] = current_time
    
    while solution.size != num_nodes:

        if i == -1:
                # Infeasible
                # print('Infeasible')
                return solution
        # Existem ações pra seguir?
        if i >= len(level):
            actions = one_step_viability(solution, current_time, distance_matrix, intervals)
            level.append(actions)
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
    distance_matrix = np.array(problem['distance_matrix'], dtype=np.int32)
    intervals = np.array(problem['intervals'], dtype=np.int32)
    current_time = np.int32(0)
    # Number of cities
    num_cities = np.array(problem['dimension'], dtype=np.int32) 
    
    # Initial solution
    solution = [starting_node]
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
        for j in one_step_viability(np.array(current_solution, dtype=np.int32), current_time, distance_matrix, intervals):
            # Adding candidate next city
            current_solution.append(j)
            auxiliar_time = current_time + distance_matrix[current_solution[-2], current_solution[-1]]

            # Run Base Policy
            backtracking_solution = backtraking_greed_policy(np.array(current_solution, dtype=np.int32), auxiliar_time, distance_matrix, intervals)
            # rollout_cost = calculate_solution_cost(nn_solution, List(distance_matrix))
            if len(backtracking_solution) != num_cities:
                rollout_cost = np.int32(999999)
            else:
                rollout_cost = calculate_solution_cost(backtracking_solution, distance_matrix)
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

def experiments_with(problem):
    start = time.time()
    rollout_solution = rollout_algorithm(problem)
    rollout_time = time.time() - start
    rollout_cost = calculate_solution_cost(rollout_solution, problem['distance_matrix'])
    
    # execute nearest neighbor algorithm and calculate time
    start = time.time()
    backtracking_solution = backtraking_greed_policy(np.array([0], dtype=np.int32), 0, \
        np.array(problem['distance_matrix'], dtype=np.int32), \
        np.array(problem['intervals'], dtype=np.int32))
    backtracking_time = time.time() - start
    backtracking_cost = calculate_solution_cost(backtracking_solution, problem['distance_matrix'])

    return rollout_solution, rollout_cost, rollout_time, backtracking_solution, backtracking_cost, backtracking_time

np.random.seed(13123123)

# for i in [20, 40, 60]:
for i in [20]:
    # for j in [20, 40, 60, 80]:
    for j in [20]:
        # Create Results file
        results = open(f'experiments/results_{time.strftime("%d%b%Y_%H_%M_%S", time.gmtime())}.txt', 'w')
        # Write header
        results.write('instance_name,rol_cost,rol_time,nn_cost,nn_time\n')
        for k in [1, 2, 3, 4, 5]:
            # Instance Name 
            instance = f'n{i}w{j}.00{k}'

            # Start tests
            path = f'instances/tsptw_data/DumasEtAl/{instance}.txt'
            problem = read_data(path)
            pre_process(problem['distance_matrix'], problem['intervals'])
            rollout_solution, rollout_cost, rollout_time, backtracking_solution, backtracking_cost, backtracking_time = experiments_with(problem)
            results.write(f'{instance},{rollout_cost},{rollout_time},{backtracking_cost},{backtracking_time}\n')
            results.write(str(np.array(rollout_solution)) +'\n')
            results.write(str(np.array(backtracking_solution)) +'\n')
    
    results.close()

    