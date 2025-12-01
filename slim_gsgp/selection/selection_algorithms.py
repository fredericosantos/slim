# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Selection operator implementation.
"""

import random
from typing import Tuple, Any

import numpy as np


def tournament_selection_min(pool_size, fitness_index = 0) -> callable:
    """
    Returns a function that performs tournament selection to select an individual with the lowest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.
        fitness_index : int
            The index of the fitness value to consider for selection. Defaults to 0 for single-objective fitness.
        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """

    def ts(pop):
        """
        Selects the individual with the lowest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmin([ind.fitness[fitness_index]for ind in pool])]

    return ts


def tournament_selection_max(pool_size, fitness_index = 0) -> callable:
    """
    Returns a function that performs tournament selection to select an individual with the highest fitness from a
    population.

    Parameters
    ----------
    pool_size : int
        Number of individuals participating in the tournament.
    fitness_index : int
        The index of the fitness value to consider for selection. Defaults to 0 for single-objective fitness.

    Returns
    -------
    Callable
        A function ('ts') that elects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness in the pool.
    Notes
    -----
    The returned function performs tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals.
    """
    def ts(pop):
        """
        Selects the individual with the highest fitness from a randomly chosen pool.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the highest fitness in the pool.
        """
        pool = random.choices(pop.population, k=pool_size)
        return pool[np.argmax([ind.fitness[fitness_index] for ind in pool])]

    return ts

def nested_tournament_selection(pool1: int = 4, pool2: int = 2, maximize1: bool = 2, maximize2: bool = 2):
    """
    Returns a function that performs nested tournament selection to select an individual with the best fitness from a
    population.

    Parameters
    ----------
    pool1 : int
        Number of individuals participating in the first level of tournaments.
    pool2 : int
        Number of individuals participating in the second level of tournaments.
    maximize1 : bool
        Whether to maximize fitness in the first level of tournaments.
    maximize2 : bool
        Whether to maximize fitness in the second level of tournaments.

    Returns
    -------
    Callable
        A function ('nts') that elects the individual with the best fitness from nested tournaments.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness after nested tournaments.
    Notes
    -----
    The returned function performs nested tournament selection by receiving a population and returning the best of {pool_size}
    randomly selected individuals across {n_tournaments} levels.
    """

    def nts(pop):
        """
        Selects the individual with the lowest fitness from nested tournaments.

        Parameters
        ----------
        pop : Population
            The population from which individuals are drawn.

        Returns
        -------
        Individual
            The individual with the lowest fitness after nested tournaments.
        """
        # Copy the population
        current_population = pop
        winners = []
        # For the amount of pools in the second tournament
        for i in range(pool2):
            # Select individuals from the current population
            ts = tournament_selection_min(pool1) if not maximize1 else tournament_selection_max(pool1)
            winner = ts(current_population)
            winners.append(winner)
        # From the winners, select the final winner
        final_winner = winners[np.argmin([ind.fitness[0] for ind in winners])] if not maximize2 \
            else winners[np.argmax([ind.fitness[0] for ind in winners])]
        return final_winner


    return nts
