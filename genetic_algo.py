#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""  
Genetic Algorithm
=================
Class to implement Genetic Algorithm having functions to mutate and breed.

Authored by Bhuvanesh Sridharan
Date: 21/3/20
"""


# In[2]:


import numpy as np


# In[20]:


class GeneticAlgo:
    """  
    Genetic Algorithm
    =================
    Class to implement Genetic Algorithm having functions to mutate and breed.
    
    Arguments:
    param_range : tuple
    num_params : int
    pop_size : int
    num_gen : int
    fitnessFunc : function with populating as input which return fitness for all chromosomes
    
    Authored by Bhuvanesh Sridharan
    Date: 21/3/20
    """
    
    def __init__(self, param_range, num_params, pop_size, mutation_prob, fitnessFunc):
        self.param_range = param_range
        self.num_params = num_params
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.population = None
        self.fitnessFunc = fitnessFunc
    
    def calc_fitness(self, pop):
        """
        Function to calculate the fitness of the population given as the argument
        """

        fitness = self.fitnessFunc(pop)
        return fitness
    
    def rank_selection(self, pop, fitness, num_select):
        """
        select top num_select number of chromosomes from the population
        """
        sorted_pop = pop[fitness.argsort()]
        selected_pop = sorted_pop[-num_select:, ...]
        #print("Returning the following population as selected: ")
        #print(selected_pop)
        return selected_pop
    
    def breed_parents(self, alpha, p1, p2):
        """
        crosses over two chromosomes of p1 and p2 by Whole Arithmetic Combination method
        
        Returns:
        Two chromosomes c1 and c2
        """
        c1 = alpha*p1 + (1-alpha)*p2
        c2 = alpha*p2 + (1-alpha)*p1
        
        return np.array([c1, c2])
        
    def mutate(self, pop, prob, mag):
        """
        mutates the population by magnitude mag with probability = prob
        
        returns:
        Mutated Population
        """
        for i in range(len(pop)):
            if np.random.random() < prob:
                pop[i] = pop[i] + mag * np.random.uniform(-1,1,pop[i].shape)
                for j in range(len(pop[i])):
                    if pop[i][j] > self.param_range[1]:
                        pop[i][j] = self.param_range[1]
                    if pop[i][j] < self.param_range[0]:
                        pop[i][j] = self.param_range[0]
        return pop
    
    def init_population(self, seed = None):
        """
        Initialises the population with seed if applicable or starts with a random population
        """
        if seed :
            pop = np.repeat([seed], self.pop_size, axis = 0)
            pop = self.mutate(pop, 1, 1)
            self.population = pop
        else :
            self.population = np.random.uniform(self.param_range[0],self.param_range[1], size = (self.pop_size, self.num_params))
    
    def evolve_gen(self):
        """
        evolves one generation of the population
            1. Breeds the population
            2. takes top half of the population
        """
        pop = self.population
        new_pop = np.zeros(shape=pop.shape)
        parent_ind = np.arange(self.pop_size)
        np.random.shuffle(parent_ind)
        
        for i in range(0, parent_ind.size, 2):
            pop = np.concatenate((pop, self.breed_parents(0.3, pop[i], pop[i+1])))
            
        self.mutate(pop, self.mutation_prob, 1)
        
        fitness = self.calc_fitness(pop)
        self.population = self.rank_selection(pop, fitness, self.pop_size)


# The below functions are to test the GA Algorithm written above:

# In[21]:


def tempfunc(pop):
    x = np.array([1, -1, 0])
    fitness = np.sum(pop * x, axis = 1)
    return fitness


# In[22]:


def testGA():
    ga = GeneticAlgo((-10,10), 3, 100, 6, tempfunc)
    ga.init_population()
    for i in range(10):
        ga.evolve_gen()
    print(ga.rank_selection(ga.population, ga.calc_fitness(ga.population), 1))

