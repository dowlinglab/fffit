"""
Functions to calculate pareto optimal set.

Bridgette Befort, Ryan DeFever, Alexander Dowling, Edward Maginn.

Created: 6/11/2020

Resources:
https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

http://code.activestate.com/recipes/578287-multidimensional-pareto-front/

https://oapackage.readthedocs.io/en/latest/examples/example_pareto.html
"""

import numpy as np


def compare_pareto_sets(set1, set2):
    """
    Compare two pareto sets and return True if they are identical.
    Only works if the pareto sets have the same order.

    Parameters
    ----------
    set1 :
        pareto set
    set2 :
        pareto set

    Returns
    -------
    boolean
        True if the pareto sets are the same
    """
    compare = set1 == set2
    similar = compare.all()
    return similar


def is_pareto_efficient_simple(costs, max_front=False):
    """
    Find and return pareto-efficient points given costs

    Implmentation is fast for many datapoints, but slower for multiple costs.
    Function updates its knowledge of efficiency after evaluating each row.


    Parameters
    ----------
    costs : np.ndarray, shape=(n_points, n_costs)
        the costs for each point
    max_front : boolean, optional, default=False
        Find the pareto set for the highest costs

    Returns
    -------
    is_efficient : np.ndarray, shape=(n_points, ), dtype=bool
        indicates if each point is Pareto efficient
    """
    # Make an output array
    is_efficient = np.ones(costs.shape[0], dtype=bool)

    # Loop over all rows in the cost array
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Check row against all others and assign them is_efficient values
            # (True/False) depending on if they have better/worse values than
            # the current row, thus is efficient is continuously updated
            # with new information
            if max_front:
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] > c, axis=1
                )
            else:
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                )
            # And list index i as efficient...
            # this will change as other row evaluations are done
            is_efficient[i] = True

    return is_efficient


def is_pareto_efficient(costs, max_front=False):
    """
    Find and return pareto-efficient points given costs

    Method is fast for many datapoints, but slower for multiple costs;
    faster than pareto efficient simple; constantly updates pareto
    information, moves fast because it removes previously dominated
    points from comparison

    Parameters
    ----------
    costs : np.ndarray, shape=(n_points, n_costs)
        the costs for each point
    max_front : boolean, optional, default=False
        Find the pareto set for the highest costs

    Returns
    -------
    is_efficient : np.ndarray, shape=(n_points,), dtype=bool
        indicates if each point is Pareto efficient
    """
    # Make array of the indices of the costs [0,1,2...len(cost)]
    n_points = costs.shape[0]
    is_efficient_idxs = np.arange(n_points)

    next_point_index = 0  # Initialize next point counter

    # Do until the input costs array has been searched through
    # Note costs is updated each iteration so we can't use n_points
    while next_point_index < costs.shape[0]:
        if max_front == True:
            # Bool array for whether the costs of the current row
            # are greater/less than the costs in the other rows
            nondominated_point_mask = np.any(
                costs > costs[next_point_index], axis=1
            )
        else:
            nondominated_point_mask = np.any(
                costs < costs[next_point_index], axis=1
            )
        # Assign true value for row being examined
        nondominated_point_mask[next_point_index] = True
        # Apply non_dominated points mask to is_efficient array
        is_efficient_idxs = is_efficient_idxs[
            nondominated_point_mask
        ]
        # Costs/input file now contains only nondominated points so far
        costs = costs[nondominated_point_mask]
        # Next point to examine is the sum of the T/F values
        # up to the previous point plus 1
        next_point_index = (
            np.sum(nondominated_point_mask[:next_point_index]) + 1
        )

    # Convert to boolean array for return
    is_efficient = np.zeros(n_points, dtype=bool)
    is_efficient[is_efficient_idxs] = True

    return is_efficient


def find_pareto_set(data, pareto_fun, max_front=False):
    """
    Run pareto efficiency function and return pareto indices and costs

    Parameters
    ----------
    data : np.ndarray, shape=(n_points, n_costs)
        array with costs for some set of points
    pareto_fun : function
        function to use when calculating the pareto set
    max_front : boolean, optional, default=False
        Find the pareto set for the highest costs

    Returns
    -------
    result : np.ndarray, shape=(n_points,)
        indices of pareto set
    pareto_points : np.ndarray, shape=(n_pareto_points,)
        costs of pareto optimal points
    dominated_points : np.ndarray, shape=(n_dominated_points)
        costs of dominated points
    """

    # calculate pareto set
    result = pareto_fun(data, max_front)

    # Make data array a list
    data_list = data.tolist()

    pareto_points = []
    dominated_points = []

    for i in range(len(data_list)):
        # Append point to either the pareto or dominate sets based on result
        if result[i] == True:
            pareto_points.append(data_list[i])
        else:
            dominated_points.append(data_list[i])

    pareto_points = np.array(pareto_points)
    dominated_points = np.array(dominated_points)

    return result, pareto_points, dominated_points
