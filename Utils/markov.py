import numpy as np
import collections

def transition_matrix(list_stats):
    print(list_stats)
    # num of wolf stats
    num_stats = 3
    # init matrix
    matrix = [[0]*num_stats for _ in range(num_stats)]
    for (i, j) in zip(list_stats, list_stats[1:]):
        matrix[i-1][j-1] += 1

    # convert to probabilities:
    for row in matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
            
    return matrix

def contribution_matrix(list_transition_matrix: list)->list:
    list_contribution = []
    for matrix in list_transition_matrix:
        print(matrix)
        p0 = [0.5, 0.4, 0.1]
        for _ in range(100):
            p0 = np.mat(p0) * np.mat(matrix)
        list_contribution.append(p0.getA()[0][0])

    return list_contribution

def aggregate_markov(list_wolfs: list, list_contribution: list)->collections.OrderedDict:
    list_params = []
    for wolf in list_wolfs:
        list_params.append(wolf.params)
    global_params = list_params[0]

    for params_name in list_params[0].keys():
        list_params_value = []
        for params, contribution in zip(list_params, list_contribution):
            list_params_value.append(params[params_name] * contribution)
        global_params[params_name] = sum(list_params_value)
        del list_params_value

    return global_params
    

