from random import randint, random
import matplotlib.pyplot as plt
import numpy as np


def individual(length, min_val, max_val):
    """Create an individual with integer coefficients."""
    return [randint(min_val, max_val) for _ in range(length)]


def population(count, length, min_val, max_val):
    """Create a population of individuals."""
    return [individual(length, min_val, max_val) for _ in range(count)]


def polynomial(x, coefficients):
    """Evaluate polynomial at x with given integer coefficients."""
    return (coefficients[0] * x**5 + 
            coefficients[1] * x**4 + 
            coefficients[2] * x**3 + 
            coefficients[3] * x**2 + 
            coefficients[4] * x + 
            coefficients[5])


def fitness(individual, target_coeffs, x_values=None):
    """
    Fitness measures the average absolute error between integer coefficients.
    Lower is better.
    """
    error = sum(abs(a - b) for a, b in zip(individual, target_coeffs)) / len(target_coeffs)
    return error


def grade(pop, target_coeffs, x_values=None):
    """Find average fitness for a population."""
    summed = sum(fitness(x, target_coeffs) for x in pop)
    return summed / float(len(pop))


def evolve(pop, target_coeffs, x_values=None, retain=0.2, random_select=0.05, mutate=0.2, 
           mutation_strength=1, min_val=-50, max_val=50):
    """Evolve the population with integer arithmetic."""
    graded = [(fitness(x, target_coeffs), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    
    # Elitist selection
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    
    # Random selection for diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    
    # Mutation - mutate integers by adding or subtracting 1 (bounded)
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            change = randint(-mutation_strength, mutation_strength)
            new_val = individual[pos_to_mutate] + change
            # Clip to min/max
            individual[pos_to_mutate] = max(min_val, min(max_val, new_val))
    
    # Crossover
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    
    while len(children) < desired_length:
        male_idx = randint(0, parents_length - 1)
        female_idx = randint(0, parents_length - 1)
        if male_idx != female_idx:
            male = parents[male_idx]
            female = parents[female_idx]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)
    
    parents.extend(children)
    return parents


# Main program
if __name__ == '__main__':
    # Target polynomial coefficients (integers)
    target_coefficients = [25, 18, 31, -14, 7, -19]

    # GA Parameters
    p_count = 100          # Population size
    i_length = 6           # 6 coefficients (5 powers + 1 const)
    coeff_min = -50
    coeff_max = 50
    max_generations = 200

    # Stopping criteria
    fitness_threshold = 0.01
    stagnation_limit = 20

    # Initialize population
    p = population(p_count, i_length, coeff_min, coeff_max)
    fitness_history = []
    best_fitness_history = []

    solution_found = False
    solution_generation = 0
    best_fitness = float('inf')
    stagnation_counter = 0

    for gen in range(max_generations):
        avg_fitness = grade(p, target_coefficients)
        fitness_history.append(avg_fitness)
        
        current_best = min(p, key=lambda x: fitness(x, target_coefficients))
        current_best_fitness = fitness(current_best, target_coefficients)
        best_fitness_history.append(current_best_fitness)
        
        if current_best_fitness <= fitness_threshold:
            solution_found = True
            solution_generation = gen + 1
            print(f"\n[SUCCESS] Solution found in {solution_generation} generations!")
            print(f"Best coefficients: {current_best}")
            print(f"Target coefficients: {target_coefficients}")
            print(f"Fitness: {current_best_fitness:.6f}")
            break
        
        if abs(current_best_fitness - best_fitness) < 0.001:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_fitness = current_best_fitness
        
        if stagnation_counter >= stagnation_limit:
            solution_generation = gen + 1
            print(f"\n[STAGNATION] Stopped at generation {solution_generation}")
            print(f"Best coefficients: {current_best}")
            print(f"Target coefficients: {target_coefficients}")
            print(f"Fitness: {current_best_fitness:.6f}")
            break
        
        if (gen + 1) % 20 == 0:
            print(f"Gen {gen + 1}: Best fitness = {current_best_fitness:.6f}, Avg fitness = {avg_fitness:.6f}")
        
        p = evolve(p, target_coefficients, retain=0.2, random_select=0.05, 
                   mutate=0.2, mutation_strength=1, min_val=coeff_min, max_val=coeff_max)

    if not solution_found and stagnation_counter < stagnation_limit:
        best_indiv = min(p, key=lambda x: fitness(x, target_coefficients))
        best_fit = fitness(best_indiv, target_coefficients)
        print(f"\n[MAX GENERATIONS] Stopped at {max_generations} generations")
        print(f"Best coefficients: {best_indiv}")
        print(f"Target coefficients: {target_coefficients}")
        print(f"Fitness: {best_fit:.6f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Fitness history
    axes[0, 0].plot(fitness_history, linewidth=2, label='Average Fitness')
    axes[0, 0].plot(best_fitness_history, linewidth=2, label='Best Fitness', color='green')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness (Error)')
    axes[0, 0].set_title('Fitness Over Generations')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log scale fitness
    axes[0, 1].semilogy(fitness_history, linewidth=2, label='Average Fitness')
    axes[0, 1].semilogy(best_fitness_history, linewidth=2, label='Best Fitness', color='green')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Fitness (Error) - Log Scale')
    axes[0, 1].set_title('Fitness Over Generations (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Compare polynomials
    x_plot = np.linspace(-100, 100, 100)
    best_final = min(p, key=lambda x: fitness(x, target_coefficients))
    
    y_target = [polynomial(x, target_coefficients) for x in x_plot]
    y_optimised = [polynomial(x, best_final) for x in x_plot]
    
    axes[1, 0].plot(x_plot, y_target, linewidth=2, label='Target Polynomial', color='blue')
    axes[1, 0].plot(x_plot, y_optimised, linewidth=2, linestyle='--', label='Optimised Polynomial', color='red')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Target vs Optimised Polynomial')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Coefficient comparison
    coeffs_names = ['x^5', 'x^4', 'x^3', 'x^2', 'x', 'const']
    x_pos = np.arange(len(coeffs_names))
    
    axes[1, 1].bar(x_pos - 0.2, target_coefficients, 0.4, label='Target', color='blue', alpha=0.7)
    axes[1, 1].bar(x_pos + 0.2, best_final, 0.4, label='Optimised', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Coefficient')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Coefficient Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(coeffs_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
