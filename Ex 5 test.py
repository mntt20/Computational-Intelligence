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

def mutate_individual(ind, min_val, max_val, pm=0.1, step=1):
    """Per-gene random perturbation by ±step with prob pm."""
    child = ind[:]
    for i in range(len(child)):
        if random() < pm:
            delta = randint(-step, step)
            child[i] = max(min_val, min(max_val, child[i] + delta))
    return child

# ---------------- Evolution (pure GA) ----------------

def evolve(pop, target_coeffs,
           min_val=-100, max_val=100,
           elitism=2,  # keep top-k unchanged
           tournament_k=3,
           pc=0.9,     # crossover probability
           pm=0.15,    # per-gene mutation probability
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

        # mutation
        c1 = mutate_individual(c1, min_val, max_val, pm=pm, step=step)
        if len(next_pop) < N:
            next_pop.append(c1)
        if len(next_pop) < N:
            c2 = mutate_individual(c2, min_val, max_val, pm=pm, step=step)
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
                         N=150, pc=0.90, max_generations=350,
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
                        fixed_pc=0.90, fixed_N=150, trials=12, max_generations=350):
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
                          N=150, pm=0.12, max_generations=350,
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
                         fixed_pm=0.12, fixed_N=150, trials=12, max_generations=350):
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
                           pc=0.90, pm=0.12, max_generations=350,
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
                          fixed_pm=0.12, fixed_pc=0.90, trials=12, max_generations=350):
    plt.figure(figsize=(8.5,5))
    plt.errorbar(pop_sizes, means, yerr=stds, fmt='-o', capsize=4, linewidth=1.6,
                 label=f'pm={fixed_pm}, pc={fixed_pc}, Trials={trials}, Cap={max_generations}')
    plt.xlabel('Population size (N)')
    plt.ylabel('Mean generations to exact solution')
    plt.title('Effect of Population Size on GA Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# ---------------- Bit encoding for schema analysis ----------------

def int_to_bin_fixed(n, bits):
    """Two's complement fixed-width binary string for signed int n."""
    return format(n & ((1 << bits) - 1), f'0{bits}b')

def genome_to_bitstring(individual, bits_per_coeff=16):
    """Concatenate fixed-width chunks for the 6 coefficients."""
    return ''.join(int_to_bin_fixed(v, bits_per_coeff) for v in individual)

# ---------------- Schema utilities (over {0,1,*}) ----------------

def schema_order(schema):
    return sum(1 for c in schema if c in ('0','1'))

def schema_defining_length(schema):
    idx = [i for i,c in enumerate(schema) if c in ('0','1')]
    if not idx:
        return 0
    return max(idx) - min(idx)

def schema_matches(schema, genome_bits):
    return all(s == '*' or s == g for s, g in zip(schema, genome_bits))

def schema_stats(schema, population, target_coeffs, bits_per_coeff=16):
    matches = []
    for ind in population:
        bits = genome_to_bitstring(ind, bits_per_coeff)
        if schema_matches(schema, bits):
            matches.append(ind)
    m = len(matches)
    if m == 0:
        return 0, 0.0
    fit_vals = [fitness(ind, target_coeffs) for ind in matches]
    return m, sum(fit_vals) / m

def schema_predict_next_m(mH, fH, fbar, pc, pm, deltaH, orderH, L):
    # Holland's schema theorem (lower bound)
    if mH == 0 or fbar == 0:
        return 0.0
    crossover_survival = max(0.0, 1.0 - pc * (deltaH / max(1, (L - 1))))
    mutation_survival = (1.0 - pm) ** orderH
    return mH * (fH / fbar) * crossover_survival * mutation_survival

# ---------------- Schema experiment runner ----------------

def make_schema_from_positions(L, fixed_positions):
    """fixed_positions: dict {index: '0' or '1'}"""
    s = ['*'] * L
    for i, v in fixed_positions.items():
        s[i] = v
    return ''.join(s)

def schema_experiment(target_coeffs,
                      bits_per_coeff=16,
                      pc=0.9, pm=0.12,
                      N=150, generations=200,
                      coeff_min=-100, coeff_max=100,
                      elitism=4, k=3, step=1):
    """
    Run the GA once while logging schema counts/predictions for:
    - H_short: compact (small defining length), order o
    - H_long : spread (large defining length), same order o
    """
    L = bits_per_coeff * len(target_coeffs)

    # Define two schemata with same order (e.g., o=6), different δ
    order = 6

    # Short schema: cluster 6 fixed bits in a tight block near the start
    short_idxs = list(range(8, 8+order))  # contiguous block
    short_positions = {i: ('1' if j % 2 == 0 else '0') for j, i in enumerate(short_idxs)}
    H_short = make_schema_from_positions(L, short_positions)

    # Long schema: spread the same 6 fixed bits across chromosome
    long_idxs = [1, L//5, 2*L//5, 3*L//5, 4*L//5, L-2]
    long_positions = {i: ('1' if j % 2 == 0 else '0') for j, i in enumerate(long_idxs)}
    H_long = make_schema_from_positions(L, long_positions)

    oS, oL = schema_order(H_short), schema_order(H_long)
    dS, dL = schema_defining_length(H_short), schema_defining_length(H_long)
    print(f"H_short: order={oS}, delta={dS}")
    print(f"H_long : order={oL}, delta={dL}")

    # Initialize GA population
    pop = population(N, len(target_coeffs), coeff_min, coeff_max)

    # Logs
    log = {
        'gen': [],
        'm_short': [], 'm_long': [],
        'pred_short_next': [], 'pred_long_next': [],
        'fH_short': [], 'fH_long': [], 'fbar': []
    }

    for gen in range(generations):
        # Current stats
        fbar = grade(pop, target_coeffs)
        mS, fHS = schema_stats(H_short, pop, target_coeffs, bits_per_coeff)
        mL, fHL = schema_stats(H_long,  pop, target_coeffs, bits_per_coeff)

        predS = schema_predict_next_m(mS, fHS, fbar, pc, pm, dS, oS, L)
        predL = schema_predict_next_m(mL, fHL, fbar, pc, pm, dL, oL, L)

        log['gen'].append(gen)
        log['m_short'].append(mS)
        log['m_long'].append(mL)
        log['pred_short_next'].append(predS)
        log['pred_long_next'].append(predL)
        log['fH_short'].append(fHS)
        log['fH_long'].append(fHL)
        log['fbar'].append(fbar)

        # Early stop if exact found (optional)
        best = min(pop, key=lambda x: fitness(x, target_coeffs))
        if fitness(best, target_coeffs) == 0.0:
            print(f"Exact coefficients found at gen {gen}.")
            break

        # Evolve one generation using your pure GA
        pop = evolve(pop, target_coeffs,
                     min_val=coeff_min, max_val=coeff_max,
                     elitism=elitism, tournament_k=k,
                     pc=pc, pm=pm, step=step)

    return log, {'H_short': H_short, 'H_long': H_long, 'L': L, 'oS': oS, 'oL': oL, 'dS': dS, 'dL': dL, 'pc': pc, 'pm': pm}

def plot_schema_results(log, meta):
    # Align predictions with next-generation actuals
    gens = log['gen']
    if len(gens) < 2:
        print("Not enough generations for plotting.")
        return
    g_next = gens[1:]
    mS_next = log['m_short'][1:]
    mL_next = log['m_long'][1:]
    predS   = log['pred_short_next'][:-1]
    predL   = log['pred_long_next'][:-1]

    plt.figure(figsize=(12,4.5))
    # Short schema
    plt.subplot(1,2,1)
    plt.plot(g_next, mS_next, label='H_short actual m(k+1)', color='green')
    plt.plot(g_next, predS, '--', label='H_short predicted E[m(k+1)]', color='darkgreen')
    plt.title(f"H_short (o={meta['oS']}, δ={meta['dS']})")
    plt.xlabel('Generation'); plt.ylabel('Count'); plt.grid(True, alpha=0.3); plt.legend()

    # Long schema
    plt.subplot(1,2,2)
    plt.plot(g_next, mL_next, label='H_long actual m(k+1)', color='red')
    plt.plot(g_next, predL, '--', label='H_long predicted E[m(k+1)]', color='darkred')
    plt.title(f"H_long (o={meta['oL']}, δ={meta['dL']})")
    plt.xlabel('Generation'); plt.ylabel('Count'); plt.grid(True, alpha=0.3); plt.legend()
    plt.suptitle(f"Schema survival (pc={meta['pc']}, pm={meta['pm']}, L={meta['L']})")
    plt.tight_layout()
    plt.show()

    # Optional: selection multiplier f(H)/f̄
    ratio_short = [(fh/fb) if fb>0 and fh>0 else 0 for fh,fb in zip(log['fH_short'], log['fbar'])]
    ratio_long  = [(fh/fb) if fb>0 and fh>0 else 0 for fh,fb in zip(log['fH_long'],  log['fbar'])]
    plt.figure(figsize=(9,4))
    plt.plot(log['gen'], ratio_short, label='f(H_short)/f̄', color='green')
    plt.plot(log['gen'], ratio_long, label='f(H_long)/f̄', color='red')
    plt.axhline(1.0, color='gray', linestyle=':')
    plt.xlabel('Generation'); plt.ylabel('Selection multiplier'); plt.title('Schema selection advantage')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.show()

# ---------------- Run ----------------
# Runs the GA with fixed parameters and plots fitness history
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
#     plt.plot(fitness_history, label='Average Fitness', marker='o', lw=1.2)
#     plt.plot(best_history, label='Best Fitness', lw=1.6)
#     plt.xlabel('Generation'); plt.ylabel('Avg Abs Error per Coefficient')
#     plt.title(f'Pure GA for Coefficients (pop={p_count})')
#     plt.grid(True, alpha=0.3); plt.legend()
#     plt.show()

# Plots generation vs parameter sweeps for mutation rate, crossover rate, and population size
# if __name__ == '__main__':
#     target_coefficients = [25, 18, 31, -14, 7, -19]

#     # Arrays to test (reasonable ranges with more points)
#     mutation_rates  = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
#     crossover_rates = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90]
#     pop_sizes       = list(range(10, 241, 20))

#     # Use your existing mutation sweep
#     means_pm, std_pm, rate_pm = sweep_mutation_rates(
#         target_coeffs=target_coefficients,
#         mutation_rates=mutation_rates,
#         trials=12, N=150, pc=0.90, max_generations=350,
#         elitism=4, k=3, step=1
#     )
#     plot_mutation_sweep(mutation_rates, means_pm, std_pm)

#     # Crossover sweep
#     means_pc, std_pc, rate_pc = sweep_crossover_rates(
#         target_coeffs=target_coefficients,
#         crossover_rates=crossover_rates,
#         trials=12, N=150, pm=0.12, max_generations=350,
#         elitism=4, k=3, step=1
#     )
#     plot_crossover_sweep(crossover_rates, means_pc, std_pc)

#     # Population size sweep
#     means_N, std_N, rate_N = sweep_population_sizes(
#         target_coeffs=target_coefficients,
#         pop_sizes=pop_sizes,
#         trials=12, pc=0.90, pm=0.12, max_generations=350,
#         elitism=4, k=3, step=1
#     )
#     plot_population_sweep(pop_sizes, means_N, std_N)

# Runs Schema Code
if __name__ == '__main__':
    target_coefficients = [25, 18, 31, -14, 7, -19]

    # Choose GA parameters for a clear schema effect:
    # high crossover emphasizes defining length; modest mutation to reduce noise.
    pc = 0.95
    pm = 0.08
    N  = 200
    generations = 180

    log, meta = schema_experiment(
        target_coeffs=target_coefficients,
        bits_per_coeff=16,
        pc=pc, pm=pm, N=N, generations=generations,
        coeff_min=-100, coeff_max=100,
        elitism=4, k=3, step=1
    )
    plot_schema_results(log, meta)
