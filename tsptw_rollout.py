import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

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

    problem = {
        'coordinates': coordinates,
        'intervals': intervals,
        'distance_matrix': np.floor(squareform(pdist(coordinates))),
        'dimension': len(coordinates)
    }
    
    return problem


def calculate_solution_cost(solution, distance_matrix, intervals):
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


def random_policy(solution, distance_matrix, intervals):
    
    pass


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
            nn_solution = random_policy(problem, current_solution.copy(), distance_matrix.copy())
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


path = 'instances/tsptw_data/DumasEtAl/n20w20.001.txt'
problem = read_data(path)

solution = np.array([1, 17, 10, 20, 18, 19, 11, 6, 16, 2, 12, 13, 7, 14, 8, 3, 5, 9, 21, 4, 15, 1]) - 1
# solution = np.array([1, 2, 15, 19, 5, 9, 8, 6, 16, 12, 17, 20, 18, 11, 13, 4, 7, 3, 21, 10, 14, 1]) - 1
# solution = np.array([1, 6, 20, 14, 2, 9, 17, 5, 10, 4, 8, 16, 15, 18, 12, 3, 11, 19, 21, 7, 13, 1]) - 1
# solution = np.array([1, 12, 4, 3, 20, 8, 16, 10, 9, 6, 7, 11, 15, 5, 13, 17, 19, 14, 21, 18, 2, 1]) - 1
# solution = np.array([1, 20, 12, 8, 19, 17, 14, 9, 4, 18, 3, 6, 11, 5, 16, 10, 15, 7, 21, 2, 13, 1]) - 1
print(calculate_solution_cost(solution, problem['distance_matrix'], problem['intervals']))
print(is_feasible(solution, problem['distance_matrix'], problem['intervals']))