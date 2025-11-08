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

def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [(fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    # Implement elistist selection of parents, top 20% become parents for next generation
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
            # this mutation is not ideal, because it restricts the range of possible values
            individual[pos_to_mutate] = randint(min(individual), max(individual))

    # crossover parents to create children
    num_parents = len(parents)
    desired_pop_size = len(pop) - num_parents
    children = []
    # Make children until population is refilled
    while len(children) < desired_pop_size:
        # Choose two different parent indices randomly
        male = randint(0, num_parents - 1)
        female = randint(0, num_parents - 1)
        if male != female: # Ensures same parent is not used twice for one child
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            child = male[:half] + female[half:] # First half genes from male, second half from female
            children.append(child) # Add baby to next generation
    parents.extend(children) # Add children to parents for next generation
    return parents

def find_target(population, target, max_generations = 100):
    for gen in range(max_generations):
        # Evolve population
        population = evolve(population, target)
        # Check for when the sum of individuals equal the target
        for indiv in population:
            if sum(indiv) == target:
                print(f"Target {target} found at generation {gen+1}: {indiv}")
                return indiv, gen+1
    print("Target not found within max generations.")
    return None, max_generations

# Example usage
target = 173
p_count = 1000
i_length = 1
i_min = 0
i_max = 200
generations = 100

# Track if solution is found
solution_found= False
solution_generation = 0

# Initialise population
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target)]

for gen in range(generations):
    # Check if target found (fitness = 0)
    for indiv in p:
        if fitness(indiv, target) == 0:
            solution_found = True
            solution_generation = gen + 1
            print(f"Target {target} found in {solution_generation} generations!")
            print(f"Solution: {indiv}")
            print(f"Sum: {sum(indiv)}")
            break  # Exit inner loop
    if solution_found:
        break # Exit outer loop

    # Evolve to next generation
    p = evolve(p, target, retain=0.2, random_select=0.05, mutate=0.01)
    fitness_history.append(grade(p, target))

 # Report results if not found
if not solution_found:
    best_indiv = min(p, key=lambda x: fitness(x, target))
    print(f"\n Target not found after {generations} generations")
    print(f"Best solution: {best_indiv}, Sum: {sum(best_indiv)}")

# Print fitness history
print(f"\nFitness History:")
for i, datum in enumerate(fitness_history):
    print(f"Generation {i}: {datum:.2f}")

# Plot fitness over generations
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title(f'GA Optimization - Target {target} (Stopped at Gen {len(fitness_history)-1})')
plt.grid(True, alpha=0.3)
plt.show()

# for i in range(generations):
#     p = evolve(p, target)
#     fitness_history.append(grade(p, target))
#     find_target(p, target,generations)

# for datum in fitness_history:
#     print(datum)

# plt.plot(fitness_history)
# plt.show()
