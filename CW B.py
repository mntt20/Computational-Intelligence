from random import randint, random
import matplotlib.pyplot as plt

def individual(length, min_, max_):
    """ Create a member of the population (single integer as a list). """
    return [randint(min_, max_) for _ in range(length)]

def population(count, length, min_, max_):
    """ Create a population of individuals. """
    return [individual(length, min_, max_) for _ in range(count)]

def fitness(individual, target):
    """ Absolute error to target value. """
    return abs(target - sum(individual))

def grade(pop, target):
    """ Find average fitness for a population. """
    return sum(fitness(x, target) for x in pop) / len(pop)

def evolve(pop, target, retain=0.3, random_select=0.02, mutate=0.01, i_min=0, i_max=200):
    graded = [(fitness(x, target), x) for x in pop]
    graded.sort(key=lambda t: t[0])
    parents = [x[1][:] for x in graded[:int(len(graded)*retain)]]

    # Elitism: always keep best solution unchanged
    best = parents[0][:]

    # Random selection for diversity
    for individual in graded[int(len(graded)*retain):]:
        if random_select > random():
            parents.append(individual[1][:])

    # Mutation (re-generate value for higher exploration)
    for individual in parents:
        if mutate > random():
            individual[0] = randint(i_min, i_max)

    # Crossover (for single values, just duplicate/perturb)
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        idx = randint(0, parents_length - 1)
        child = parents[idx][:]
        if random() < mutate:
            child[0] = randint(i_min, i_max)
        children.append(child)

    # Keep best, then rest of parents and children
    new_pop = [best] + parents[1:] + children[:len(pop)-1-len(parents)]
    return new_pop

# --- Fast target finding demo ---

target = 173
p_count = 50   # Higher population for faster coverage
i_length = 1    # One value per individual
i_min = 0
i_max = 200
generations = 100

solution_found = False
solution_generation = 0
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target)]

for gen in range(generations):
    # Check if target is found (fitness == 0)
    for indiv in p:
        if fitness(indiv, target) == 0:
            solution_found = True
            solution_generation = gen + 1
            print(f"Target {target} found in {solution_generation} generations!")
            print(f"Solution: {indiv}  Sum: {sum(indiv)}")
            break
    if solution_found:
        break
    p = evolve(p, target, retain=0.3, random_select=0.02, mutate=0.23, i_min=i_min, i_max=i_max)
    fitness_history.append(grade(p, target))

if not solution_found:
    best_indiv = min(p, key=lambda x: fitness(x, target))
    print(f"\nTarget not found after {generations} generations")
    print(f"Best solution: {best_indiv} Sum: {sum(best_indiv)}")

print(f"\nFitness History:")
for i, datum in enumerate(fitness_history):
    print(f"Generation {i}: {datum:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(fitness_history, marker='o')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title(f'GA Optimisation - Target {target} (Stopped at Gen {len(fitness_history)-1})')
plt.grid(True, alpha=0.3)
plt.show()
