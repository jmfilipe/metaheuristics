"""
This file has a set of fitness functions that could be used to attest epso performance

"""

import math


# Fitness Functions Examples:
def rosenbrock(x):
    sum_ = 0
    for i in range(len(x)-1):
        sum_ += 100*(x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return sum_


def schwefel(x):
    alpha = 418.982887
    fitness = 0
    for i in range(len(x)):
        fitness -= x[i]*math.sin(math.sqrt(math.fabs(x[i])))
    return float(fitness) + alpha*len(x)


def sphere(x):
    sum_ = 0
    for i in range(len(x)-1):
        sum_ += x[i]**2
    return sum_


def soma(x):
    return sum(x)



