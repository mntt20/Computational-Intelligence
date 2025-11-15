from random import randint, random
import matplotlib.pyplot as plt
import numpy as np
# ---------------- GA primitives ----------------

def individual(length, min_val, max_val):
    return [randint(min_val, max_val) for _ in range(length)]

def population(count, length, min_val, max_val):
    return [individual(length, min_val, max_val) for _ in range(count)]

def fitness(individual, target_coeffs):
    # average absolute coefficient error (lower is better)
    return sum(abs(a - b) for a, b in zip(individual, target_coeffs)) / len(target_coeffs)

def grade(pop, target_coeffs):
    return sum(fitness(x, target_coeffs) for x in pop) / len(pop)

# ---------------- Selection ----------------

def tournament_select(pop, target_coeffs, k=3):
    """Pick best from k random contestants."""
    best = None
    best_fit = float('inf')
    n = len(pop)
    for _ in range(k):
        cand = pop[randint(0, n - 1)]
        f = fitness(cand, target_coeffs)
        if f < best_fit:
            best_fit = f
            best = cand
    return best[:]

# ---------------- Crossover ----------------

def two_point_crossover(p1, p2):
    L = len(p1)
    if L < 3:
        # fallback to one-point
        cut = randint(1, L - 1) if L > 1 else 0
        return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]
    a = randint(1, L - 2)
    b = randint(a + 1, L - 1)
    c1 = p1[:a] + p2[a:b] + p1[b:]
    c2 = p2[:a] + p1[a:b] + p2[b:]
    return c1, c2

# ---------------- Mutation ----------------


# ---------------- Evolution (pure GA) ----------------

def evolve(pop, target_coeffs,
           min_val=-100, max_val=100,
           elitism=2,  # keep top-k unchanged
           tournament_k=3,
           pc=0.8,     # crossover probability
           pm=0.01,    # per-gene mutation probability
           step=1):
    # Elitism
    ranked = sorted(pop, key=lambda x: fitness(x, target_coeffs))
    elites = [ind[:] for ind in ranked[:elitism]]

    # Generate rest using selection + crossover + mutation
    next_pop = elites[:]
    N = len(pop)
    while len(next_pop) < N:
        # parent selection via tournament
        p1 = tournament_select(pop, target_coeffs, k=tournament_k)
        p2 = tournament_select(pop, target_coeffs, k=tournament_k)

        # crossover
        if random() < pc:
            c1, c2 = two_point_crossover(p1, p2)
        else:
            c1, c2 = p1[:], p2[:]

        # mutation (random reset)
        for g in range(len(c1)):
            if random() < pm:
                c1[g] = randint(min_val, max_val)
        for g in range(len(c2)):
            if random() < pm:
                c2[g] = randint(min_val, max_val)

        # append
        if len(next_pop) < N:
            next_pop.append(c1)
        if len(next_pop) < N:
            next_pop.append(c2)

    # exact size
    return next_pop[:N]

# ---------------- Parameter Sweep and Plotting ----------------

def generations_to_solution(target_coeffs, N, pc, pm, max_generations,
                            elitism=4, k=3, step=1, coeff_min=-100, coeff_max=100, genome_len=6):
    pop = population(N, genome_len, coeff_min, coeff_max)
    for gen in range(max_generations):
        best = min(pop, key=lambda x: fitness(x, target_coeffs))
        if fitness(best, target_coeffs) == 0.0:
            return gen
        pop = evolve(pop, target_coeffs,
                     min_val=coeff_min, max_val=coeff_max,
                     elitism=elitism, tournament_k=k,
                     pc=pc, pm=pm, step=step)
    return max_generations  # cap if not solved

def sweep_mutation_rates(target_coeffs,
                         mutation_rates,
                         trials=10,
                         N=150, pc=0.80, max_generations=350,
                         elitism=4, k=3, step=1,
                         coeff_min=-100, coeff_max=100, genome_len=6):
    means, stds, solve_rates = [], [], []
    for pm in mutation_rates:
        gens = []
        for _ in range(trials):
            g = generations_to_solution(target_coeffs, N=N, pc=pc, pm=pm, max_generations=max_generations,
                                        elitism=elitism, k=k, step=step,
                                        coeff_min=coeff_min, coeff_max=coeff_max, genome_len=genome_len)
            gens.append(g)
        arr = np.array(gens, dtype=int)
        means.append(arr.mean())
        stds.append(arr.std(ddof=1))
        solve_rates.append((arr < max_generations).mean())
        print(f"pm={pm:.3f} | mean={arr.mean():.1f}, std={arr.std(ddof=1):.1f}, solve_rate={100*solve_rates[-1]:.0f}%")
    return np.array(means), np.array(stds), np.array(solve_rates)


def plot_mutation_sweep(mutation_rates, means, stds,
                        fixed_pc=0.80, fixed_N=150, trials=12, max_generations=350):
    plt.figure(figsize=(8.5,5))
    plt.errorbar(mutation_rates, means, yerr=stds, fmt='-o', capsize=4, linewidth=1.6,
                 label=f'pc={fixed_pc}, Population={fixed_N}, Trials={trials}, Cap={max_generations}')
    plt.xlabel('Mutation rate (pm)')
    plt.ylabel('Mean generations to exact solution')
    plt.title('Effect of Mutation Rate on GA Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def sweep_crossover_rates(target_coeffs,
                          crossover_rates,
                          trials=10,
                          N=150, pm=0.1, max_generations=350,
                          elitism=4, k=3, step=1,
                          coeff_min=-100, coeff_max=100, genome_len=6):
    means, stds, solve_rates = [], [], []
    for pc in crossover_rates:
        gens = []
        for _ in range(trials):
            g = generations_to_solution(target_coeffs, N=N, pc=pc, pm=pm, max_generations=max_generations,
                                        elitism=elitism, k=k, step=step,
                                        coeff_min=coeff_min, coeff_max=coeff_max, genome_len=genome_len)
            gens.append(g)
        arr = np.array(gens, dtype=int)
        means.append(arr.mean())
        stds.append(arr.std(ddof=1))
        solve_rates.append((arr < max_generations).mean())
        print(f"pc={pc:.2f} | mean={arr.mean():.1f}, std={arr.std(ddof=1):.1f}, solve_rate={100*solve_rates[-1]:.0f}%")
    return np.array(means), np.array(stds), np.array(solve_rates)

def plot_crossover_sweep(crossover_rates, means, stds,
                         fixed_pm=0.1, fixed_N=150, trials=12, max_generations=350):
    plt.figure(figsize=(8.5,5))
    plt.errorbar(crossover_rates, means, yerr=stds, fmt='-o', capsize=4, linewidth=1.6,
                 label=f'pm={fixed_pm}, Population={fixed_N}, Trials={trials}, Cap={max_generations}')
    plt.xlabel('Crossover probability (pc)')
    plt.ylabel('Mean generations to exact solution')
    plt.title('Effect of Crossover Probability on GA Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def sweep_population_sizes(target_coeffs,
                           pop_sizes,
                           trials=10,
                           pc=0.80, pm=0.1, max_generations=350,
                           elitism=4, k=3, step=1,
                           coeff_min=-100, coeff_max=100, genome_len=6):
    means, stds, solve_rates = [], [], []
    for N in pop_sizes:
        gens = []
        for _ in range(trials):
            g = generations_to_solution(target_coeffs, N=N, pc=pc, pm=pm, max_generations=max_generations,
                                        elitism=elitism, k=k, step=step,
                                        coeff_min=coeff_min, coeff_max=coeff_max, genome_len=genome_len)
            gens.append(g)
        arr = np.array(gens, dtype=int)
        means.append(arr.mean())
        stds.append(arr.std(ddof=1))
        solve_rates.append((arr < max_generations).mean())
        print(f"Population ={N} | Mean={arr.mean():.1f}, Standard Deviation={arr.std(ddof=1):.1f}, Solve Rate={100*solve_rates[-1]:.0f}%")
    return np.array(means), np.array(stds), np.array(solve_rates)

def plot_population_sweep(pop_sizes, means, stds,
                          fixed_pm=0.1, fixed_pc=0.80, trials=12, max_generations=350):
    plt.figure(figsize=(8.5,5))
    plt.errorbar(pop_sizes, means, yerr=stds, fmt='-o', capsize=4, linewidth=1.6,
                 label=f'pm={fixed_pm}, pc={fixed_pc}, Trials={trials}, Cap={max_generations}')
    plt.xlabel('Population size (N)')
    plt.ylabel('Mean generations to exact solution')
    plt.title('Effect of Population Size on GA Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# ---------------- Run ----------------

# if __name__ == '__main__':
#     target_coefficients = [25, 18, 31, -14, 7, -19]

#     p_count = 150          # larger population improves recombination
#     genome_len = 6
#     coeff_min, coeff_max = -100, 100
#     generations = 300

#     # GA parameters (pure GA)
#     elitism = 4
#     tournament_k = 3
#     pc = 0.9
#     pm = 0.12   # per-gene mutation prob
#     step = 1    # integer step size

#     pop = population(p_count, genome_len, coeff_min, coeff_max)
#     fitness_history = [grade(pop, target_coefficients)]
#     best_history = []

#     solved = False
#     for gen in range(generations):
#         best = min(pop, key=lambda x: fitness(x, target_coefficients))
#         best_fit = fitness(best, target_coefficients)
#         best_history.append(best_fit)

#         # early stop on exact match
#         if best_fit == 0.0:
#             solved = True
#             print(f"[SUCCESS] Found exact coefficients at gen {gen}: {best}")
#             break

#         pop = evolve(pop, target_coefficients,
#                      min_val=coeff_min, max_val=coeff_max,
#                      elitism=elitism, tournament_k=tournament_k,
#                      pc=pc, pm=pm, step=step)
#         fitness_history.append(grade(pop, target_coefficients))

#         if (gen + 1) % 20 == 0:
#             print(f"Gen {gen+1}: best={best_fit:.3f}, avg={fitness_history[-1]:.3f}")

#     if not solved:
#         best = min(pop, key=lambda x: fitness(x, target_coefficients))
#         print(f"No exact solution in {generations} generations.")
#         print(f"Best: {best}, fitness={fitness(best, target_coefficients):.3f}")

#     # Plot
#     plt.figure(figsize=(9,5))
#     plt.plot(fitness_history, label='Average Fitness', lw=1.2)
#     plt.plot(best_history, label='Best Fitness', lw=1.6)
#     plt.xlabel('Generation'); plt.ylabel('Avg Abs Error per Coefficient')
#     plt.title(f'Pure GA for Coefficients (pop={p_count})')
#     plt.grid(True, alpha=0.3); plt.legend()
#     plt.show()

if __name__ == '__main__':
    target_coefficients = [25, 18, 31, -14, 7, -19]

    # Arrays to test (reasonable ranges with more points)
    mutation_rates  = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    crossover_rates = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90]
    pop_sizes       = list(range(10, 241, 20))

    # Use your existing mutation sweep
    means_pm, std_pm, rate_pm = sweep_mutation_rates(
        target_coeffs=target_coefficients,
        mutation_rates=mutation_rates,
        trials=12, N=150, pc=0.80, max_generations=350,
        elitism=4, k=3, step=1
    )
    plot_mutation_sweep(mutation_rates, means_pm, std_pm)

    # Crossover sweep
    means_pc, std_pc, rate_pc = sweep_crossover_rates(
        target_coeffs=target_coefficients,
        crossover_rates=crossover_rates,
        trials=12, N=150, pm=0.1, max_generations=350,
        elitism=4, k=3, step=1
    )
    plot_crossover_sweep(crossover_rates, means_pc, std_pc)

    # Population size sweep
    means_N, std_N, rate_N = sweep_population_sizes(
        target_coeffs=target_coefficients,
        pop_sizes=pop_sizes,
        trials=12, pc=0.80, pm=0.1, max_generations=350,
        elitism=4, k=3, step=1
    )
    plot_population_sweep(pop_sizes, means_N, std_N)

