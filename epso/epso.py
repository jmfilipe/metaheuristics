"""
Based on:

https://pythonhosted.org/pyswarm/

"""

import multiprocessing
import math
from functools import partial

import numpy as np


def _wrap_fitness(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def update_mutation_rate(iter_, itermax, itermin=0, mmax=0.7, mmin=0.01, tau=5):
    """
    When :py:func:`epso` uses `dynamic_mutation_rate=True` this function is used to update the mutation rate proportionally
    to the number of iterations and considering user-defined max and minimum values for the mutation rate, resulting in:

    .. math::
        \\frac{1}{1-e^{\\tau}} \cdot (1-e^{(\\tau \cdot \\frac{itermax-iter}{itermax-itermin})}) \cdot mr_{max} + mr_{min}

    Args:
        iter_ (int): current iteration number
        itermax (int): maximum number of iterations
        itermin (int): minimum number of iterations, usually 0
        mmax (Optional[float]): maximum mutation rate. Defaults to 0.7
        mmin (Optional[float]): minimum mutation rate. Defaults to 0.01
        tau (Optional[float]): defines the curvature of the exponential function. Defaults to 5

    Returns:
        (float):


    """

    return 1/(1-math.exp(tau)) * (1-math.exp(tau*(itermax-iter_)/(itermax-itermin))) * mmax + mmin


def calc_velocity(position, velocity, weights, best_pos, best_global_pos, communication):
    """
    Calculates the new velocity for the particle, using the new found weights.

    Args:
        position (ndarray): current position of the particle
        velocity (ndarray): previous velocity
        weights (ndarray): wheights (inertia, memory, cooperation and target
        best_pos (ndarray): best position of each particle
        best_global_pos (ndarray): best global position
        communication (float): communication factor, so the probability of the particle using the information of
         the best global position

    Returns:
        (ndarray)

    """

    r_gbest = np.random.normal(size=position.shape)
    r_commu = np.random.uniform(size=position.shape) < communication

    # Update the particles velocities
    return weights[0]*velocity + \
           weights[1]*(best_pos - position) + \
           weights[2]*r_commu*((1+weights[3]*r_gbest)*best_global_pos - position)


def enforce_limits(position, x_min, x_max):
    """
    Checks if the particle's position is outside its boundaries and if necessary corrects it.

    Args:
        position (ndarray): current position
        x_min (ndarray): lower bound
        x_max (ndarray): upper bound

    Returns:
        (ndarray)

    """
    maskl = position < x_min
    masku = position > x_max

    return position*(~np.logical_or(maskl, masku)) + x_min*maskl + x_max*masku


def epso(pop, x_min, x_max, fitness_fun, x_init=None, maxiter=500, threads=1, communication=0.75,
         mutation_rate=0.4, dynamic_mutation_rate=False, mutation_rate_max=0.7, mutation_rate_min=0.01, tau=5,
         max_it_without_change=40, tolerance=0.1,
         *args, **kwargs):
    """

    Args:
        pop (int): swarm size, number of set of particles to be used in the metaheuristic
        x_min (ndarray): lower bound of the particle position
        x_max (ndarray): upper bound of the particle position
        fitness_fun : fitness function to be used to evaluate particle's performance
        x_init (Optional[ndarray]): starting position. Defaults to None
        maxiter (int): maximum number of iterations - stopping criteria
        threads (int): number of threads (cpu) used for parallel processing
        communication (float): epso parameter - which defines the probability of each particle using the best value found by all the swarm
        mutation_rate (float): epso parameter - defines the changing rate of the weights mutation process
        dynamic_mutation_rate (Optional[bool]): False if `mutaion_rate`is static, True otherwise. Defaults to False
        tau (Optional[float]): used in :py:func:`update_mutation_rate` and defines the curvature of the exponential function.
            Defaults to 5
        mutation_rate_min(Optional[float]): maximum mutation rate, used in :py:func:`update_mutation_rate`. Defaults to 0.7
        mutation_rate_max(Optional[float]): minimum mutation rate, used in :py:func:`update_mutation_rate`. Defaults to 0.01
        tolerance (Optional[float]): difference between the current global best fitness and the new best found, if the difference is larger than
            the tolerance --> `max_it_without_change` is reseted
        max_it_without_change (Optional[int]): defines maximum acceptable number of iterations without change in the fitness function, after
            this value the optimization stops - stopping criteria
        *args: variable length argument list, to be used in the fitness function
        **kwargs: Arbitrary keyword arguments, to be used in the fitness function

    Returns:
        (tuple): tuple containing:

            best_global_fit (ndarray): fitness function which results from the best position found

            best_global_pos (ndarray): best position found

    """

    assert len(x_min) == len(x_max), 'Lower- and upper-bounds must be the same length'
    assert np.all(x_max >= x_min), 'All upper-bound values must be greater than lower-bound values'

    fitness_function = partial(_wrap_fitness, fitness_fun, args, kwargs)

    if threads > 1:
        pool = multiprocessing.Pool(threads)
        mp_pool = pool.map
    else:
        mp_pool = map

    dim = len(x_max)

    v_max = np.abs(x_max - x_min)
    v_min = - v_max

    if x_init is None:
        # Initialize the particle's position
        position = np.random.rand(pop, dim)  # x
        position = x_min + position*(x_max - x_min)
    else:
        position = np.tile(x_init, (pop, 1))

    weights = np.array([np.random.rand(pop, dim)] * 4)

    best_pos = np.zeros_like(position)  # p
    best_fitness = np.ones(pop)*np.inf  # fp

    best_global_pos = []  # g
    best_global_fit = np.inf  # fg

    # Calculate objective and constraints for each particle
    fitness = np.array(list(mp_pool(fitness_function, position)))
    # fitness = fitness_function(position[0, :])

    i_update = np.array(fitness < best_fitness)
    best_pos[i_update, :] = position[i_update, :].copy()
    best_fitness[i_update] = fitness[i_update]

    # Update swarm's best position
    i_min = np.argmin(best_fitness)
    if best_fitness[i_min] < best_global_fit:
        best_global_fit = best_fitness[i_min]
        best_global_pos = best_pos[i_min, :].copy()
    else:
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        best_global_pos = position[0, :].copy()

    # Initialize the particle's velocity
    velocity = v_min + np.random.rand(pop, dim)*(v_max - v_min)

    # Iterate until termination criterion met ##################################
    it = 0
    it_without_change = 0

    while (it <= maxiter) and (it_without_change <= max_it_without_change):

        if dynamic_mutation_rate:
            mutation_rate = update_mutation_rate(iter_=it, itermax=maxiter, mmax=mutation_rate_max, mmin=mutation_rate_min, tau=tau)

        # weights of the replica (child) are mutated
        weights_child = enforce_limits(weights + np.random.normal(size=weights.shape)*mutation_rate, 0, 1)

        # velocity is calculated for both the parent and child positions
        velocity_child = calc_velocity(position, velocity, weights_child, best_pos, best_global_pos, communication)
        position_child = position + velocity_child

        velocity = calc_velocity(position, velocity, weights, best_pos, best_global_pos, communication)
        position = position + velocity

        # checks max and min limits
        position = enforce_limits(position, x_min, x_max)
        position_child = enforce_limits(position_child, x_min, x_max)

        # fitness evaluation
        fitness = np.array(list(mp_pool(fitness_function, position)))
        fitness_child = np.array(list(mp_pool(fitness_function, position_child)))

        # idx selects the particle whose parent achieved a better results than the child
        idx = np.array(fitness_child < fitness)
        # elitist selection
        fitness[idx] = fitness_child[idx]
        position[idx, :] = position_child[idx, :]
        weights[0, idx] = weights_child[0, idx]
        weights[1, idx] = weights_child[1, idx]
        weights[2, idx] = weights_child[2, idx]
        weights[3, idx] = weights_child[3, idx]

        # global bests are updated
        i_update = np.array(fitness < best_fitness)
        best_pos[i_update, :] = position[i_update, :].copy()
        best_fitness[i_update] = fitness[i_update]

        # Update swarm's best position
        i_min = np.argmin(best_fitness)
        if best_fitness[i_min] < best_global_fit:
            if abs(best_fitness[i_min] - best_global_fit) > tolerance:
                it_without_change = 0
            else:
                it_without_change += 1
            best_global_fit = best_fitness[i_min]
            best_global_pos = best_pos[i_min, :].copy()
        else:
            it_without_change += 1

        if it % 10 == 0:
            print('iter: {}, best fit: {}'.format(it, round(best_global_fit, 2)))
        it += 1

    if it >= maxiter:
        print('Maximum number of iterations reached! \n')
    elif it_without_change >= max_it_without_change:
        print('Maximum number of iterations without change reached! \n')

    return best_global_fit, best_global_pos

