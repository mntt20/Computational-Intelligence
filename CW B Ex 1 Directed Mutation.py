from random import randint, random
import matplotlib.pyplot as plt

# --- GA primitives ---

def individual(length, min_, max_):
    """ Create a member of the population. """
    return [randint(min_, max_) for _ in range(length)]

def population(count, length, min_, max_):
    """
    Create a number of individuals (i.e. a population).
    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values
    """
    return [individual(length, min_, max_) for _ in range(count)]

def fitness(individual, target):
    """
    Determine the fitness of an individual. Lower is better.
    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    return abs(target - sum(individual))

def grade(pop, target):
    """ Find average fitness for a population. """
    return sum(fitness(x, target) for x in pop) / len(pop)

# --- Deterministic local search to prevent stagnation ---

def local_refine(indiv, target, i_min, i_max):
    """
    1-step or 2-step tweak around current values to reduce error.
    """
    curr = indiv[0]
    best = curr
    best_fit = abs(target - curr)

    # try neighbors
    for delta in (-2, -1, 1, 2):  # small local moves
        v = curr + delta
        if i_min <= v <= i_max:
            f = abs(target - v)
            if f < best_fit:
                best_fit = f
                best = v

    indiv[0] = best
    return indiv

# --- Evolution with directed mutation and elitism ---

def evolve(pop, target, retain=0.3, random_select=0.02, mutate=0.05, i_min=0, i_max=200,
           use_directed=True, refine_elites=True, n_elites=2):
    # rank
    graded = [(fitness(x, target), x) for x in pop]
    graded.sort(key=lambda t: t[0])

    retain_count = max(1, int(len(graded) * retain))
    parents = [g[1][:] for g in graded[:retain_count]]

    # optional local refine on elites to close last gaps (e.g., 3 -> 0)
    if refine_elites:
        for k in range(min(n_elites, len(parents))):
            parents[k] = local_refine(parents[k][:], target, i_min, i_max)

    # diversity
    for fit, geno in graded[retain_count:]:
        if random() < random_select:
            parents.append(geno[:])

    # ensure parents not empty
    if not parents:
        parents = [graded[0][1][:]]

    # elitism: preserve best
    elites = [parents[0][:]]

    # mutation
    # For 1-gene genome, directed mutation pushes toward target; also keep random jumps
    new_parents = []
    for indiv in parents:
        gene = indiv[0]
        if random() < mutate:
            if use_directed and gene != target:
                # move a step toward target, clamped to bounds
                step = 1 if gene < target else -1
                gene = max(i_min, min(i_max, gene + step))
            else:
                # random re-roll
                gene = randint(i_min, i_max)
        new_parents.append([gene])

    parents = new_parents

    # crossover (for 1-gene, crossover is trivial; keep mixing via copying/mutation)
    parents_len = len(parents)
    desired_len = len(pop) - len(elites) - (parents_len - 1)
    children = []
    while len(children) < max(0, desired_len):
        idx = randint(0, parents_len - 1)
        child = parents[idx][:]  # copy
        # small chance to jump to diversify
        if random() < mutate * 0.5:
            child[0] = randint(i_min, i_max)
        children.append(child)

    # next generation
    # Keep elites, rest of parents, and fill with children
    next_pop = elites + parents[1:] + children
    # Adjust size exactly
    if len(next_pop) < len(pop):
        # pad with random individuals if needed (rare)
        next_pop += [individual(1, i_min, i_max) for _ in range(len(pop) - len(next_pop))]
    elif len(next_pop) > len(pop):
        next_pop = next_pop[:len(pop)]

    return next_pop

# --- Run ---

target = 173
p_count = 10
i_length = 1
i_min = 0
i_max = 200
generations = 100
early_stop_fraction = 1.0  # stop when 100% exact

p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target)]
solution_found = False
stop_generation = None
   
for gen in range(generations):
    # Early stop by exact hits
    zero_hits = sum(1 for ind in p if fitness(ind, target) == 0)
    if zero_hits >= int(early_stop_fraction * p_count):
        solution_found = True
        stop_generation = gen
        # Print one exact solution
        winner = next(ind for ind in p if fitness(ind, target) == 0)
        print(f"Exact target {target} reached at generation {gen} with {zero_hits}/{p_count} exact.")
        print(f"Example solution: {winner}  Sum: {sum(winner)}")
        break

    # Evolve with directed mutation + elite refinement
    p = evolve(
        p, target,
        retain=0.3, random_select=0.02, mutate=0.08,  # a bit higher mutation helps exploration
        i_min=i_min, i_max=i_max,
        use_directed=True, refine_elites=True, n_elites=3
    )
    fitness_history.append(grade(p, target))

if not solution_found:
    best = min(p, key=lambda x: fitness(x, target))
    print(f"No exact solution within {generations} generations.")
    print(f"Best: {best} Sum: {sum(best)} Fitness: {fitness(best, target)}")

# Plot
plt.figure(figsize=(9, 5))
plt.plot(fitness_history, marker='o', lw=1.6)
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title(f'Directed Mutation GA for Target {target} (pop={p_count}) fitness = {fitness_history[-1]}')
plt.grid(True, alpha=0.3)
plt.show()
