from random import randint, random
import matplotlib.pyplot as plt

# Individual creation
def individual(length, min_, max_):
    return [randint(min_, max_) for _ in range(length)]

# Population creation
def population(count, length, min_, max_):
    return [individual(length, min_, max_) for _ in range(count)]

# Fitness calculation
def fitness(indiv, target):
    return abs(target - sum(indiv))

# Calculate average fitness
def grade(pop, target):
    return sum(fitness(x, target) for x in pop) / float(len(pop))

# Linear mutation rate decay
def mutation_rate(gen, num_generations, start, end):
    frac = gen / float(num_generations-1)  # Progress from 0 to 1
    return start * (1-frac) + end * frac

# Evolution function
def evolve(pop, target, mutation, retain=0.2, random_select=0.05, min_=0, max_=100):
    graded = [(fitness(x, target), x) for x in pop]
    graded.sort(key=lambda x: x[0])  # Sort by fitness (lower is better)
    parents = [x[1] for x in graded[:int(len(graded)*retain)]]

    # Random selection to maintain diversity
    for x in graded[int(len(graded)*retain):]:
        if random() < random_select:
            parents.append(x[1])

    # Mutation (only once per parent)
    for indiv in parents:
        if random() < mutation:
            pos = randint(0, len(indiv)-1)
            indiv[pos] = randint(min_, max_)

    # Crossover
    children = []
    desired_length = len(pop) - len(parents)
    while len(children) < desired_length:
        male_idx, female_idx = randint(0, len(parents)-1), randint(0, len(parents)-1)
        if male_idx != female_idx:
            male, female = parents[male_idx], parents[female_idx]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)
    return parents + children

# Main routine
if __name__ == '__main__':
    target = 173  # Target sum
    p_count = 100  # Population size
    i_length = 5   # Individual length
    i_min = 0
    i_max = 100
    num_generations = 20
    start_mutation = 0.01   # Start high for exploration
    end_mutation = 0.001    # End low for exploitation

    p = population(p_count, i_length, i_min, i_max)
    fitness_history = [grade(p, target)]

    for gen in range(num_generations):
        mut_rate = mutation_rate(gen, num_generations, start_mutation, end_mutation)
        p = evolve(p, target, mutation=mut_rate, retain=0.2, random_select=0.05, min_=i_min, max_=i_max)
        fitness_history.append(grade(p, target))

    for datum in fitness_history:
        print(datum)

    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Genetic Algorithm Fitness Over Generations')
    plt.show()
