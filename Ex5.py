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

# ---------------- Evolution (pure GA) ----------------

def evolve(pop, target_coeffs,
           min_val=-100, max_val=100,
           elitism=2,  # keep top-k unchanged
           tournament_k=3,
           pc=0.8,     # crossover probability
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
    crossover_effect = max(0.0, 1.0 - pc * (deltaH / max(1, (L - 1))))
    mutation_effect = (1.0 - pm) ** orderH
    return mH * (fH / fbar) * crossover_effect * mutation_effect

# ---------------- Schema experiment runner ----------------

def make_schema_from_positions(L, fixed_positions):
    """fixed_positions: dict {index: '0' or '1'}"""
    s = ['*'] * L
    for i, v in fixed_positions.items():
        s[i] = v
    return ''.join(s)

def schema_experiment(target_coeffs,
                      bits_per_coeff=6,
                      pc=0.8, pm=0.12,
                      N=150, generations=200,
                      coeff_min=-100, coeff_max=100,
                      elitism=4, k=3, step=1):
    """
    Run the GA once while logging schema counts/predictions for:
    - H_short: compact (small defining length), order o
    - H_long : spread (large defining length), same order o
    """
    L = bits_per_coeff * len(target_coeffs)

    # Define two schemata with same order
    order = 3

    # Short schema: cluster 3 fixed bits in a tight block near the start
    short_idxs = list(range(8, 8+order))  # contiguous block
    short_positions = {i: ('1' if j % 2 == 0 else '0') for j, i in enumerate(short_idxs)}
    H_short = make_schema_from_positions(L, short_positions)

    # Long schema: spread the same 3 fixed bits across chromosome
    # long_idxs = [1, L//5, 2*L//5, 3*L//5, 4*L//5, L-2]
    long_idxs = [1, 2*L//5, L-1]
    long_positions = {i: ('1' if j % 2 == 0 else '0') for j, i in enumerate(long_idxs)}
    H_long = make_schema_from_positions(L, long_positions)

    oS, oL = schema_order(H_short), schema_order(H_long)
    dS, dL = schema_defining_length(H_short), schema_defining_length(H_long)
    print(f"H_short: order={oS}, delta={dS}")
    print(f"H_long : order={oL}, delta={dL}")

    # Initialise GA population
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

        # Early stop if exact found
        best = min(pop, key=lambda x: fitness(x, target_coeffs))
        if fitness(best, target_coeffs) == 0.0:
            print(f"Exact coefficients found at gen {gen}.")
            break

        # Evolve one generation using your pure GA
        pop = evolve(pop, target_coeffs,
                     min_val=coeff_min, max_val=coeff_max,
                     elitism=elitism, tournament_k=k,
                     pc=pc, pm=pm, step=step)
        
        selection_short = (fHS / fbar) if fbar > 0 else 0
        crossover_short = 1 - pc * (dS / (L - 1))
        mutation_short = (1 - pm) ** oS
        print(f"Gen {gen} -- SHORT SCHEMA")
        print(f"  m(H,k)={mS},   selection={selection_short:.3f}, crossover={crossover_short:.3f}, mutation={mutation_short:.3f}")
        print(f"  pred_next_count={predS:.2f}")
        if gen > 0: # show actual count next gen
            print(f"  actual_next_count={log['m_short'][gen]}")

            # --- Print schema effects for H_long ---
        selection_long = (fHL / fbar) if fbar > 0 else 0
        crossover_long = 1 - pc * (dL / (L - 1))
        mutation_long = (1 - pm) ** oL
        print(f"Gen {gen} -- LONG SCHEMA")
        print(f"  m(H,k)={mL},   selection={selection_long:.3f}, crossover={crossover_long:.3f}, mutation={mutation_long:.3f}")
        print(f"  pred_next_count={predL:.2f}")
        if gen > 0:
            print(f"  actual_next_count={log['m_long'][gen]}")
        print("-"*38)

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



# Runs Schema Code
if __name__ == '__main__':
    target_coefficients = [25, 18, 31, -14, 7, -19]

    # Choose GA parameters for a clear schema effect:
    # high crossover emphasises defining length; modest mutation to reduce noise.
    pc = 0.8
    pm = 0.08
    N  = 200
    generations = 180

    log, meta = schema_experiment(
        target_coeffs=target_coefficients,
        bits_per_coeff=8,
        pc=pc, pm=pm, N=N, generations=generations,
        coeff_min=-100, coeff_max=100,
        elitism=4, k=3, step=1
    )
    plot_schema_results(log, meta)

    ratio_short = [(fh/fb) if fb>0 and fh>0 else 0 for fh,fb in zip(log['fH_short'], log['fbar'])]
    ratio_long  = [(fh/fb) if fb>0 and fh>0 else 0 for fh,fb in zip(log['fH_long'],  log['fbar'])]
    plt.figure(figsize=(9,4))
    plt.plot(log['gen'], ratio_short, label='f(H_short)/f̄', color='green')
    plt.plot(log['gen'], ratio_long, label='f(H_long)/f̄', color='red')
    plt.axhline(1.0, color='gray', linestyle=':')
    plt.xlabel('Generation'); plt.ylabel('Selection multiplier'); plt.title('Schema selection advantage')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.show()

    # Setup data
    generations = log['gen']
    mut_short = [(1 - meta['pm']) ** meta['oS']] * len(generations)
    mut_long = [(1 - meta['pm']) ** meta['oL']] * len(generations)
    cross_short = [1 - meta['pc'] * (meta['dS'] / (meta['L'] - 1))] * len(generations)
    cross_long = [1 - meta['pc'] * (meta['dL'] / (meta['L'] - 1))] * len(generations)

    # Create side-by-side plots [1][2][3][7]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Mutation Survival plot
    ax1.plot(generations, mut_short, label='H_short mutation survival', color='green')
    ax1.plot(generations, mut_long, label='H_long mutation survival', color='red')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Survival fraction')
    ax1.set_title('Mutation Survival Probability')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Crossover Survival plot
    ax2.plot(generations, cross_short, label='H_short crossover survival', color='green')
    ax2.plot(generations, cross_long, label='H_long crossover survival', color='red')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Survival fraction')
    ax2.set_title('Crossover Survival Probability')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # # Mutation effect plot
    # mut_short = [(1 - meta['pm']) ** meta['oS']] * len(log['gen'])
    # mut_long  = [(1 - meta['pm']) ** meta['oL']] * len(log['gen'])
    # plt.figure(figsize=(7,4))
    # plt.plot(log['gen'], mut_short, label='H_short mutation survival', color='green')
    # plt.plot(log['gen'], mut_long, label='H_long mutation survival', color='red')
    # plt.xlabel('Generation'); plt.ylabel('Survival fraction'); plt.title('Mutation Survival Probability')
    # plt.grid(True, alpha=0.3); plt.legend()
    # plt.show()


    # # Crossover effect plot
    # # For all generations and both H_short and H_long:
    # cross_short = [1 - meta['pc'] * (meta['dS'] / (meta['L'] - 1))] * len(log['gen'])
    # cross_long  = [1 - meta['pc'] * (meta['dL'] / (meta['L'] - 1))] * len(log['gen'])
    # plt.figure(figsize=(7,4))
    # plt.plot(log['gen'], cross_short, label='H_short crossover survival', color='green')
    # plt.plot(log['gen'], cross_long, label='H_long crossover survival', color='red')
    # plt.xlabel('Generation'); plt.ylabel('Survival fraction'); plt.title('Crossover Survival Probability')
    # plt.grid(True, alpha=0.3); plt.legend()
    # plt.show()
    # --- Print schema effects for H_short ---




    # # Optional: selection multiplier f(H)/f̄
    # ratio_short = [(fh/fb) if fb>0 and fh>0 else 0 for fh,fb in zip(log['fH_short'], log['fbar'])]
    # ratio_long  = [(fh/fb) if fb>0 and fh>0 else 0 for fh,fb in zip(log['fH_long'],  log['fbar'])]
    # plt.figure(figsize=(9,4))
    # plt.plot(log['gen'], ratio_short, label='f(H_short)/f̄', color='green')
    # plt.plot(log['gen'], ratio_long, label='f(H_long)/f̄', color='red')
    # plt.axhline(1.0, color='gray', linestyle=':')
    # plt.xlabel('Generation'); plt.ylabel('Selection multiplier'); plt.title('Schema selection advantage')
    # plt.grid(True, alpha=0.3); plt.legend()
    # plt.show()

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