"""
Inspired on https://gist.github.com/bellbind/741853
"""
import time
from functools import partial
import multiprocessing

import numpy as np


def _wrap_fitness(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def select_random(position, fits_populations):
    rnd = np.random.randint(0, len(fits_populations)-1)
    return position[rnd], fits_populations[rnd]


def tournament(position, fits_populations):
    """
    Randomly selects two members of the population, compares both and yields the one with better fitness.
    Args:
        position:
        fits_populations:

    Returns:

    """
    alice, alicef = select_random(position, fits_populations)
    bob, bobf = select_random(position, fits_populations)
    return alice if alicef < bobf else bob


def parents_(position, fits_populations):
    """
    Using :py:func:`tournament` a father and mother elements are selected.
    ``tournament`` randomly selects two members of the population, compares both and yields the one with better fitness.

    Args:
        position:
        fits_populations:

    Returns:

    """
    while True:
        father = tournament(position, fits_populations)
        mother = tournament(position, fits_populations)
        yield (father, mother)


def crossover(parents, dim):
    """
    Performs `Two Point Crossover`: In this crossover scheme two points are selected from the binary string. In this from
    beginning of chromosome to the first crossover point is copied from one parent, the part from the first to the second
    crossover point is copied from the second parent and the rest is copied from the first parent.

    Alternative methods on: http://www.obitko.com/tutorials/genetic-algorithms/crossover-mutation.php

    Args:
        parents:
        dim:

    Returns:
        (child1, child2)

    """
    father, mother = parents
    index1 = np.random.randint(1, dim - 2)
    index2 = np.random.randint(1, dim - 2)
    if index1 > index2:
        index1, index2 = index2, index1
    child1 = np.concatenate([father[:index1], mother[index1:index2], father[index2:]], axis=0)
    child2 = np.concatenate([mother[:index1], father[index1:index2], mother[index2:]], axis=0)
    return child1, child2


def enforce_limits(position, x_min, x_max):
    maskl = position < x_min
    masku = position > x_max
    return position*(~np.logical_or(maskl, masku)) + x_min*maskl + x_max*masku


class ga:

    def __init__(self, pop, x_min, x_max, fitness_func, x_init=None,
                 probability_crossover=0.9, probability_mutation=0.2,
                 elitism=1, nthreads=1, memory=False, *args, **kwargs):
        """
        Genetic algorithms are inspired by Darwin's theory about evolution.

        Algorithm is started with a set of solutions (represented by chromosomes) called population. Solutions from one population are taken
        and used to form a new population. This is motivated by a hope, that the new population will be better than the old one. Solutions which
        are selected to form new solutions (offspring) are selected according to their fitness - the more suitable they are the more chances they have to reproduce.
        (http://www.obitko.com/tutorials/genetic-algorithms/ga-basic-description.php)

        - [1-Start] Generate random population of n chromosomes (suitable solutions for the problem), if ``x_init`` is given it is used as warm start instead of a random population
        - [2-Fitness] Evaluate the fitness of each chromosome in the population
        - [3-New population] Create a new population by repeating following steps until the new population is complete
            - [Selection] Select two parent chromosomes from a population according to their fitness (the better fitness, the bigger chance to be selected)
            - [Crossover] With a crossover probability cross over the parents to form a new offspring (children). If no crossover was performed, offspring is an exact copy of parents.
            - [Mutation] With a mutation probability mutate new offspring at each locus (position in chromosome).
            - [Accepting] Place new offspring in a new population
            - [Elitism] The param `elitism` defines the number of candidates that will be passed, unchanged, into the next generation
        - [4-Replace] Use new generated population for a further run of algorithm
        - [Loop] Go to step [2-Fitness]

        This is repeated until the maximum number of iterations or the number of iterations without change are met.

        In Windows, it is mandatory to use the instruction ``if __name__ == "__main__":`` in the parent file in order to use parallel computation.

        Args:
            pop (int):
                population size
            x_min (ndarray):
                array with minimum value for each position
            x_max (ndarray:
                array with maximum value for each position
            fitness_func (func):
                fitness function to be used
            x_init (Optional[ndarray]):
                Starting position (warm start). Defaults to ``None``.
            probability_crossover (Optional[float]):
                probability of doing crossover (merging two elements of the population, selected according to
                the :py:func:`parents` funtion and the crossover is done using the method defined in :py:func:`crossover`.
                Defaults to 0.9
            probability_mutation (Optional[float]):
                probability of mutating a given position, according to the :py:func:`mutation` function. Defaults to 0.2
            elitism (Optional[int]):
                Indicates the number of best positions found that are kept in memory. Defaults to 1.
            nthreads (Optional[int]):
                number of threads to be used when the fitness function is evaluated using parallel computation. Defaults to 1, so
                single thread computation
            memory (Optional[bool]):
                if true, all positions and fitnesses are saved and duplicated ones ignored (fitness is not evaluated). Defaults to false.
                Use only if the fitness functions is very expensive in computational resources
            *args:
            **kwargs: this can be used to pass aditional arguments to the fitness function

        Returns:


        Example:

            To run the algorithm, first initialize the ga() object and them start the metaheuristic with ga.run() ::

                import numpy as np
                from genetic_algorithm import ga


                def sphere(x, **kwargs):
                    sum_ = kwargs.get('initial_sum', 0)
                    for i in range(len(x)-1):
                        sum_ += x[i]**2
                    return sum_


                if __name__ == "__main__":

                    dim = 20
                    x_min_ga = np.zeros(dim)
                    x_max_ga = np.ones(dim) * 5

                    obj = ga(pop=500, x_min=x_min_ga, x_max=x_max_ga, fitness_func=sphere, nthreads=3)
                    fit, pos = obj.run(maxiter=1000, max_it_without_change=25, tolerance=.001)

        """

        self.memory = memory
        assert len(x_min) == len(x_max), 'Lower- and upper-bounds must be the same length'
        assert np.all(x_max > x_min), 'All upper-bound values must be greater than lower-bound values'

        self.x_min = x_min
        self.x_max = x_max
        self.pop = pop
        self.dim = len(x_max)

        self.probability_crossover = probability_crossover
        self.probability_mutation = probability_mutation
        self.elitism = elitism

        if nthreads > 1:
            self.pool = multiprocessing.Pool(nthreads)
            self.apply_map = self.pool.map
        else:
            self.apply_map = map

        self.fitness_function = partial(_wrap_fitness, fitness_func, args, kwargs)

        self.iter_ = 0
        self.iter_wo_change = 0

        if x_init is None:
            # Initialize the particle's position
            self.position = np.random.rand(self.pop, self.dim)  # x
            self.position = self.x_min + self.position*(self.x_max - self.x_min)
            self.position = self.position.round()
        else:
            self.position = np.tile(x_init, (self.pop, 1))

        self.fitness = self.evaluate_fitness(use_memory=False)

        if self.memory:
            self.position_memory = self.position.copy()
            self.fitness_memory = self.fitness.copy()

        min_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[min_idx]
        self.best_position = self.position[min_idx, :]

        self.keep_fitness = np.ones(self.elitism) * np.Inf
        self.keep_position = self.position[:self.elitism, :]

    def update_memory(self):
        idx = []
        for i in range(self.pop):
            test = self.position[i]
            if not any(np.equal(self.position_memory, test).all(1)):
                idx.append(i)

        # self.position_memory = np.concatenate([self.position_memory, self.position[idx].reshape(1, self.dim)])
        # self.fitness_memory = np.concatenate([self.fitness_memory, self.fitness[idx].reshape(-1)])
        self.position_memory = np.concatenate([self.position_memory, self.position[idx]])
        self.fitness_memory = np.concatenate([self.fitness_memory, self.fitness[idx]])
        return idx

    def evaluate_fitness(self, use_memory=True):
        if use_memory:
            idx = self.update_memory()
            self.fitness[idx] = np.array(list(self.apply_map(self.fitness_function, self.position[idx])))
        else:
            self.fitness = np.array(list(self.apply_map(self.fitness_function, self.position)))
        return self.fitness

    def keep_elitism(self):
        fit_to_compare = np.concatenate([self.fitness, self.keep_fitness])
        pos_to_compare = np.concatenate([self.position, self.keep_position])
        idx = np.argsort(fit_to_compare)
        self.keep_fitness = fit_to_compare[idx[:self.elitism]]
        self.keep_position = pos_to_compare[idx[:self.elitism], :]

    def next_gen(self):
        new_position = self.next_pop()
        self.position = enforce_limits(new_position, self.x_min, self.x_max)

    def next_pop(self):
        position = np.concatenate([self.position, self.keep_position], axis=0)
        fits = np.concatenate([self.fitness, self.keep_fitness], axis=0)
        parents_generator = parents_(position, fits)
        nexts = np.empty_like(self.position)
        for i in range(self.pop):
            parents = next(parents_generator)
            cross = np.random.random() < self.probability_crossover
            children = crossover(parents, self.dim) if cross else parents
            for ch in children:
                mutate = np.random.random() < self.probability_mutation
                nexts[i, :] = self.mutation(ch, self.dim, self.x_min, self.x_max) if mutate else ch
        return nexts

    @staticmethod
    def mutation(chromosome, dim, x_min, x_max):
        index = np.random.randint(0, dim - 1)
        mutated = chromosome.copy()
        """ Continuous mutation, where a randomized normal distribution is applied to the element"""
        # mutated[index] = np.random.normal() * (x_max[index]-x_min[index])/2
        """ Integer mutation, where a integer value is selected from the available range (max - min), excluding the current value"""
        mutated[index] = np.random.choice(np.delete(np.array(range(int(x_min[index]), int(x_max[index])+1)), int(mutated[index].round())))
        return mutated

    def check_stop(self, start_time, maxiter, max_it_without_change, tolerance):
        self.iter_ += 1
        best = self.fitness.min()
        if best < self.best_fitness:
            if abs(self.best_fitness - best) > tolerance:
                self.iter_wo_change = 0
            else:
                self.iter_wo_change += 1
            best_idx = np.argmin(self.fitness)

            self.best_fitness = self.fitness[best_idx]
            self.best_position = self.position[best_idx, :]
        else:
            self.iter_wo_change += 1

        if self.iter_ % 5 == 0:
            worst = self.fitness.max()
            ave = self.fitness.sum() / self.fitness.shape[0]
            print(
                "[G %3d] score=(best: %4d, avg: %4d, worst: %4d)  best global: %4d" %
                (self.iter_, best, ave, worst, self.best_fitness))

        if self.iter_ >= maxiter:
            print('Reached maximum number of iterations. (in {} minutes)'.format(((time.time() - start_time)/60)))
            return True
        elif self.iter_wo_change >= max_it_without_change:
            print('Reached maximum number of iterations without change. (in {} minutes)'.format(((time.time() - start_time)/60)))
            return True
        else:
            return False

    def run(self, maxiter, max_it_without_change, tolerance):
        """
        Runs the Genetic Algorithm until one termination criteria is met.

        Args:
            maxiter (int): maximum number of iterations
            max_it_without_change (int): maximum number of iterations without change
            tolerance (float): convergence tolerance

        Returns:
            (tuple): tuple containing:

                best_global_fit (ndarray): fitness function which results from the best position found
                best_global_pos (ndarray): best position found
        """

        self.iter_ = 0
        self.iter_wo_change = 0

        start_time = time.time()
        while True:
            self.keep_elitism()
            self.next_gen()
            self.evaluate_fitness(self.memory)
            if self.check_stop(start_time, maxiter, max_it_without_change, tolerance):
                break

        return self.best_fitness, self.best_position
