def foo(x,y):
    return 25*x**5 + 18*x**4 + 31*x**3 - 14*x**2 + 7*x - 19 -y

from random import randint, random, uniform
import matplotlib.pyplot as plt
import numpy as np


def individual(length, min_val, max_val):
    """Create an individual with real-valued coefficients."""
    return [uniform(min_val, max_val) for _ in range(length)]


def population(count, length, min_val, max_val):
    """Create a population of individuals."""
    return [individual(length, min_val, max_val) for _ in range(count)]


def polynomial(x, coefficients):
    """
    Evaluate polynomial at x with given coefficients.
    coefficients = [c5, c4, c3, c2, c1, c0] for: c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
    """
    return (coefficients[0] * x**5 + 
            coefficients[1] * x**4 + 
            coefficients[2] * x**3 + 
            coefficients[3] * x**2 + 
            coefficients[4] * x + 
            coefficients[5])


def fitness(individual, target_coeffs, x_values):
    """
    Calculate fitness by comparing predicted vs actual polynomial values.
    Lower fitness is better (represents error).
    """
    total_error = 0
    for x in x_values:
        predicted = polynomial(x, individual)
        actual = polynomial(x, target_coeffs)
        total_error += abs(predicted - actual)
    return total_error / len(x_values)


def grade(pop, target_coeffs, x_values):
    """Find average fitness for a population."""
    summed = sum(fitness(x, target_coeffs, x_values) for x in pop)
    return summed / float(len(pop))


def evolve(pop, target_coeffs, x_values, retain=0.2, random_select=0.05, mutate=0.1, 
           mutation_strength=0.5, min_val=-50, max_val=50):
    """Evolve the population."""
    graded = [(fitness(x, target_coeffs, x_values), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    
    # Elitist selection
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    
    # Random selection for diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    
    # Mutation
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            # Add Gaussian noise for mutation
            individual[pos_to_mutate] += uniform(-mutation_strength, mutation_strength)
            # Clip to valid range
            individual[pos_to_mutate] = max(min_val, min(max_val, individual[pos_to_mutate]))
    
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


# EXERCISE 4: Optimize polynomial coefficients
if __name__ == '__main__':
    # Target polynomial: y = 25x^5 + 18x^4 + 31x^3 - 14x^2 + 7x - 19
    target_coefficients = [25, 18, 31, -14, 7, -19]
    
    # GA Parameters
    p_count = 100          # Population size
    i_length = 6           # 6 parameters (5 coefficients + 1 constant)
    coeff_min = -50        # Minimum coefficient value
    coeff_max = 50         # Maximum coefficient value
    max_generations = 200
    
    # Test points for fitness evaluation
    x_values = np.linspace(-2, 2, 20)  # Test polynomial at 20 points
    
    # Stopping criteria
    fitness_threshold = 0.1
    stagnation_limit = 20
    
    # Initialize
    p = population(p_count, i_length, coeff_min, coeff_max)
    fitness_history = []
    best_fitness_history = []
    
    solution_found = False
    solution_generation = 0
    best_fitness_ever = float('inf')
    stagnation_counter = 0
    
    print("EXERCISE 4: Optimizing 5th-order polynomial coefficients")
    print(f"Target: y = 25x^5 + 18x^4 + 31x^3 - 14x^2 + 7x - 19")
    print(f"Population: {p_count}, Max generations: {max_generations}\n")
    
    for gen in range(max_generations):
        # Calculate fitness
        avg_fitness = grade(p, target_coefficients, x_values)
        fitness_history.append(avg_fitness)
        
        # Track best individual
        current_best = min(p, key=lambda x: fitness(x, target_coefficients, x_values))
        current_best_fitness = fitness(current_best, target_coefficients, x_values)
        best_fitness_history.append(current_best_fitness)
        
        # Check for convergence
        if current_best_fitness <= fitness_threshold:
            solution_found = True
            solution_generation = gen + 1
            print(f"\n[SUCCESS] Solution found in {solution_generation} generations!")
            print(f"Best coefficients: {[f'{c:.4f}' for c in current_best]}")
            print(f"Target coefficients: {target_coefficients}")
            print(f"Fitness: {current_best_fitness:.6f}")
            break
        
        # Stagnation detection
        if abs(current_best_fitness - best_fitness_ever) < 0.001:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_fitness_ever = current_best_fitness
        
        if stagnation_counter >= stagnation_limit:
            solution_generation = gen + 1
            print(f"\n[STAGNATION] Stopped at generation {solution_generation}")
            print(f"Best coefficients: {[f'{c:.4f}' for c in current_best]}")
            print(f"Target coefficients: {target_coefficients}")
            print(f"Fitness: {current_best_fitness:.6f}")
            break
        
        # Progress report
        if (gen + 1) % 20 == 0:
            print(f"Gen {gen + 1}: Best fitness = {current_best_fitness:.6f}, Avg fitness = {avg_fitness:.6f}")
        
        # Evolve
        p = evolve(p, target_coefficients, x_values, retain=0.2, random_select=0.05, 
                   mutate=0.2, mutation_strength=1.0, min_val=coeff_min, max_val=coeff_max)
    
    # Final report
    if not solution_found and stagnation_counter < stagnation_limit:
        best_indiv = min(p, key=lambda x: fitness(x, target_coefficients, x_values))
        best_fit = fitness(best_indiv, target_coefficients, x_values)
        print(f"\n[MAX GENERATIONS] Stopped at {max_generations} generations")
        print(f"Best coefficients: {[f'{c:.4f}' for c in best_indiv]}")
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
    x_plot = np.linspace(-2, 2, 100)
    best_final = min(p, key=lambda x: fitness(x, target_coefficients, x_values))
    
    y_target = [polynomial(x, target_coefficients) for x in x_plot]
    y_optimized = [polynomial(x, best_final) for x in x_plot]
    
    axes[1, 0].plot(x_plot, y_target, linewidth=2, label='Target Polynomial', color='blue')
    axes[1, 0].plot(x_plot, y_optimized, linewidth=2, linestyle='--', label='Optimized Polynomial', color='red')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('Target vs Optimized Polynomial')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Coefficient comparison
    coeffs_names = ['x^5', 'x^4', 'x^3', 'x^2', 'x', 'const']
    x_pos = np.arange(len(coeffs_names))
    
    axes[1, 1].bar(x_pos - 0.2, target_coefficients, 0.4, label='Target', color='blue', alpha=0.7)
    axes[1, 1].bar(x_pos + 0.2, best_final, 0.4, label='Optimized', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Coefficient')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Coefficient Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(coeffs_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
