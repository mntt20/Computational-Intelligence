from random import randint, random
import matplotlib.pyplot as plt

# ----------------- GA primitives -----------------

def individual(length, min_, max_):
    # Always returns a list of length 'length'
    return [randint(min_, max_) for _ in range(length)]

def population(count, length, min_, max_):
    return [individual(length, min_, max_) for _ in range(count)]

def fitness(indiv, target):
    # Expect indiv like [x]; keep robust but consistent
    if isinstance(indiv, list):
        x = sum(indiv)  # length==1 -> indiv[0]
    else:
        # Fallback in case something upstream returns an int
        x = int(indiv)
    return abs(target - x)

def grade(pop, target):
    return sum(fitness(x, target) for x in pop) / float(len(pop))

def mutation_rate(gen, num_generations, start, end):
    denom = max(1, num_generations - 1)
    frac = gen / float(denom)
    return start * (1 - frac) + end * frac

# ----------------- Evolution -----------------

def evolve(pop, target, mutation, retain=0.2, random_select=0.05,
           min_=0, max_=100, elitism=1):
    """
    Population elements are ALWAYS 1-element lists: [int].
    Returns a new population with the same invariant.
    """
    # Rank by fitness (lower is better)
    graded = [(fitness(x, target), x) for x in pop]
    graded.sort(key=lambda t: t[0])

    # Select parents (copy genomes)
    retain_count = max(1, int(len(graded) * retain))
    parents = [g[1][:] for g in graded[:retain_count]]  # deep copy lists

    # Elites (keep best k untouched)
    elites = [g[1][:] for g in graded[:elitism]]

    # Diversity injection (copy genome)
    for fit, geno in graded[retain_count:]:
        if random() < random_select:
            parents.append(geno[:])

    # Ensure we have parents
    if not parents:
        parents = [graded[0][1][:]]

    # Mutate parents: for i_length=1 mutate the single gene
    for indiv in parents:
        if random() < mutation:
            indiv[0] = randint(min_, max_)

    # Children: for i_length=1, crossover is trivial; copy + optional mutate
    children = []
    desired_length = len(pop) - len(parents) - len(elites)
    desired_length = max(0, desired_length)

    while len(children) < desired_length:
        idx = randint(0, len(parents) - 1)
        child = parents[idx][:]  # copy list
        if random() < mutation:
            child[0] = randint(min_, max_)
        children.append(child)

    # Assemble next population
    next_pop = elites + parents + children

    # Pad/trim and enforce invariant
    while len(next_pop) < len(pop):
        next_pop.append([randint(min_, max_)])  # ensure [int]
    if len(next_pop) > len(pop):
        next_pop = next_pop[:len(pop)]

    # Invariant checks (enable during debugging)
    # for g in next_pop:
    #     assert isinstance(g, list) and len(g) == 1 and isinstance(g[0], int)

    return next_pop

# ----------------- Main -----------------

if __name__ == '__main__':
    target = 173          # Target value
    p_count = 50          # Population size
    i_length = 1          # One gene per individual
    i_min, i_max = 0, 200
    num_generations = 100
    start_mutation = 0.20 # Start higher for exploration (per-individual)
    end_mutation   = 0.02 # End lower for exploitation
    elitism = 2
    random_select = 0.02
    retain = 0.3

    # Init
    p = population(p_count, i_length, i_min, i_max)
    fitness_history = [grade(p, target)]

    # Early-stop if any exact or 20% exact
    early_stop_fraction = 1.0

    for gen in range(num_generations):
        # Early-stop checks on current population
        zero_hits = sum(1 for ind in p if fitness(ind, target) == 0)
        if zero_hits >= max(1, int(early_stop_fraction * p_count)):
            print(f"Early stop at generation {gen}: {zero_hits}/{p_count} exact solutions.")
            break

        mut_rate = mutation_rate(gen, num_generations, start_mutation, end_mutation)
        p = evolve(
            p, target,
            mutation=mut_rate,
            retain=retain,
            random_select=random_select,
            min_=i_min, max_=i_max,
            elitism=elitism
        )
        fitness_history.append(grade(p, target))

    # Report best
    best = min(p, key=lambda x: fitness(x, target))
    print(f"Best solution: {best}  Sum: {sum(best)}  Fitness: {fitness(best, target)}")

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(fitness_history, marker='o', lw=1.6)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title(f'Dynamic Mutation GA target={target}, Pop={p_count}, Fitness = {fitness_history[-1]}, Generations = {len(fitness_history)-1}')
    plt.grid(True, alpha=0.3)
    plt.show()
