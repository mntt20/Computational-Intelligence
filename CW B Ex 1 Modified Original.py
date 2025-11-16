from random import randint, random
from operator import add
import matplotlib.pyplot as plt

def individual(length, min, max):
    """ Create a member of the population."""
    return [randint(min, max) for x in range(length)]

def population(count, length, min, max):
    """
    Create a number of individuals (i.e. a population).
    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values
    """
    return [individual(length, min, max) for x in range(count)]

def fitness(individual, target):
    """
    Determine the fitness of an individual. Higher is better.
    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    sum_values = sum(individual)
    return abs(target - sum_values)

def grade(pop, target):
    """ Find average fitness for a population."""
    summed = sum(fitness(x, target) for x in pop)
    return summed / (len(pop) * 1.0)

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01, min = 0, max = 100):
    graded = [(fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    # Elite selection, top 20% of population
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # randomly add other individuals to promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            individual[pos_to_mutate] = randint(min, max)

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)
    return parents

# Main code
target = 67
p_count = 50
i_length = 1
i_min = 0
i_max = 100
generations = 100
early_stop_fraction = 0.1  # stop if this fraction of pop hits target

p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target)]

for gen in range(generations):
    # Early stop by exact hits
    zero_hits = sum(1 for ind in p if fitness(ind, target) == 0)
    if zero_hits >= int(early_stop_fraction * p_count) or zero_hits > 0:
        solution_found = True
        stop_generation = gen
        # Print one exact solution
        winner = next(ind for ind in p if fitness(ind, target) == 0)
        print(f"Exact target {target} reached at generation {gen} with {zero_hits}/{p_count} exact.")
        print(f"Example solution: {winner}")
        break
    
    # # Stop when one individual hits target exactly
    # if any(fitness(ind, target) == 0 for ind in p):
    #     print(f"Exact target {target} reached at generation {gen}")
    #     print(f"Average fitness for population: {fitness_history[-1]}")
    #     break


    # Evolve with directed mutation + elite refinement
    p = evolve(
        p, target, retain=0.2, random_select=0.02, mutate=0.02, min = i_min, max = i_max
    )
    fitness_history.append(grade(p, target))


for datum in fitness_history:
    print(datum)

plt.figure(figsize=(9, 5))
plt.plot(fitness_history, marker='o', lw=1.6)
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title(f'Modified Original GA for Target {target}, Population ={p_count}, Fitness = {fitness_history[-1]}, Generations = {len(fitness_history)-1}')
plt.grid(True, alpha=0.3)
plt.show()
