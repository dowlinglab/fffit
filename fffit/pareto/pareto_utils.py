

'''
Functions to calculate pareto optimal set.

Bridgette Befort, Ryan DeFever, Alexander Dowling, Edward Maginn.

Created: 6/11/2020

Resources:
https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

http://code.activestate.com/recipes/578287-multidimensional-pareto-front/

https://oapackage.readthedocs.io/en/latest/examples/example_pareto.html
'''

import numpy as np

def compare_pareto_sets(set1,set2):
    '''
    Function which compares two pareto sets *Only works if the pareto sets have the same order
    
    Inputs:
    set1, set2 -- pareto set
    
    Output:
    similarity -- if True, pareto sets are the same
    '''
    compare = set1 == set2
    similar = compare.all()
    return similar

def is_pareto_efficient_simple(costs,max_value=False):
    '''
    Function to find the pareto-efficient points, fast for many datapoints, but slower for multiple costs. Function updates its knowledge of efficiency after evaluating each row.
    
    
    Inputs:
    costs -- An (n_points, n_costs) array
    max_value -- takes highest cost values (uses >)
    **Note**: If the goal is highest cost values, use >. If the goal is lowest cost values, use <
    
    Output:
    is_efficient -- A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    '''
    #Make an output array
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    
    #for row i and cost value c in the costs array
    for i, c in enumerate(costs):
        #look at a specific row
        if is_efficient[i]:
            #check row against all others and assign them is_efficient values (True/False) dependin on if they have better/worse values than the current row, thus is efficient is continuously updated with new information
            if max_value == True:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  
            is_efficient[i] = True  # And list index i as efficient... this will change as other row evaluations are done
            
    return is_efficient

def is_pareto_efficient(costs,max_value=False, return_mask = True):
    '''
    Function to find the pareto-efficient points, fast for many datapoints, but slower for multiple costs; faster than pareto efficient simple; constantly updates pareto information, moves fast because it removes previously dominated points from comparison
    
    Inputs:
    costs -- An (n_points, n_costs) array
    return_mask -- True to return a mask
    max_value -- takes highest cost values (uses >)
    **Note**: If the goal is highest cost values, use >. If the goal is lowest cost values, use <
    
    Output:
    is_efficient -- An array of indices of pareto-efficient points.
        *If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    '''
    #make array of the indices of the costs [0,1,2...len(cost)]
    is_efficient = np.arange(costs.shape[0])
    
    #number of points (number of rows of costs array)
    n_points = costs.shape[0]
    
    next_point_index = 0  # Initialize next point counter
    
    #do until the input costs array has been searched through
    while next_point_index<len(costs):
        if max_value==True:
            nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)#true/false array for whether the costs of the current row are greater/less than the costs in the other rows
        else:
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        #assign true value for row being examined
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Apply non_dominated points mask to is_efficient array
        #costs/input file now contains only nondominated points so far
        costs = costs[nondominated_point_mask]
        #next point to examine is the sum of the T/F values up to the previous point plus 1
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    
    #if the return maks is true, the output is a boolean array, otherwise it will just be the indices of the efficient points
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    
def find_pareto_set(data,pareto_fun,max_value=False):
    '''
    Function which runs a pareto efficiency function and returns the indices of the pareto set and the pareto and dominated set values
    
    Input:
    data -- array of costs 
    pareto_fun -- function to calculate the pareto set
    
    Output:
    result -- indices of pareto set
    paretoPoints -- array of pareto optimal points
    dominatedPoints -- array of dominated points
    '''
    
    #calculate pareto set
    result = pareto_fun(data,max_value)
    
    #Make data array a list
    data_list = data.tolist()
    
    paretoPoints=[]
    dominatedPoints=[]
    
    for i in range(len(data_list)):
        #Append point to either the pareto or dominate sets based on result
        if result[i] == True:
            paretoPoints.append(data_list[i])
        else:
            dominatedPoints.append(data_list[i])
            
    paretoPoints = np.array(paretoPoints)
    dominatedPoints = np.array(dominatedPoints)
    
    return result, paretoPoints, dominatedPoints