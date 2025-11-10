from random import randint, random
import matplotlib.pyplot as plt
import numpy as np


# Helper binary conversion functions

def int_to_bin(n, length):
    """Convert integer n to binary string of fixed length."""
    return format(n if n >= 0 else (1 << length) + n, f'0{length}b')

def bin_to_int(b):
    """Convert binary string b to signed integer."""
    n = int(b, 2)
    if b[0] == '1':  # if sign bit set
        n -= (1 << len(b))
    return n


# Genetic Algorithm functions with binary encoding

def individual_binary(length, min_val, max_val, bits=16):
    """Create an individual as a binary string representing integer coefficients."""
    res = ""
    for _ in range(length):
        val = randint(min_val, max_val)
        res += int_to_bin(val, bits)
    return res

def population_binary(count, length, min_val, max_val, bits=16):
    """Create a population of binary string individuals."""
    return [individual_binary(length, min_val, max_val, bits) for _ in range(count)]

def decode_individual_binary(binary_str, length, bits=16):
    """Decode a binary string into a list of integers."""
    result = []
    for i in range(length):
        start = i * bits
        end = start + bits
        chunk = binary_str[start:end]
        result.append(bin_to_int(chunk))
    return result

def fitness_binary(individual_bin, target_coeffs, length, bits=16):
    """Fitness as average absolute difference of integer coefficients."""
    individual_ints = decode_individual_binary(individual_bin, length, bits)
    error = sum(abs(a - b) for a, b in zip(individual_ints, target_coeffs)) / len(target_coeffs)
    return error

def grade_binary(pop, target_coeffs, length, bits=16):
    """Average fitness of population."""
    total = sum(fitness_binary(x, target_coeffs, length, bits) for x in pop)
    return total / len(pop)


def two_point_crossover_binary(parent1, parent2):
    """Two-point crossover on binary strings."""
    length = len(parent1)
    point1 = randint(1, length - 2)
    point2 = randint(point1 + 1, length - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def polynomial(x, coefficients):
    return (coefficients[0] * x**5 + 
            coefficients[1] * x**4 + 
            coefficients[2] * x**3 + 
            coefficients[3] * x**2 + 
            coefficients[4] * x + 
            coefficients[5])


def bit_flip_mutation_binary(bin_string, mutation_rate):
    """Bit-flip mutation on binary string."""
    bits = list(bin_string)
    for i in range(len(bits)):
        if random() < mutation_rate:
            bits[i] = '1' if bits[i] == '0' else '0'
    return ''.join(bits)

def two_point_crossover_binary(parent1, parent2):
    """Two-point crossover for bitstrings."""
    l = len(parent1)
    point1 = randint(1, l - 2)
    point2 = randint(point1 + 1, l - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2



def evolve_binary(pop, target_coeffs, length, bits=16, retain=0.2, random_select=0.05, mutation_rate=0.01):
    graded = [(fitness_binary(ind, target_coeffs, length, bits), ind) for ind in pop]
    graded.sort(key=lambda x: x[0])
    parents = [x[1] for x in graded[:int(len(graded)*retain)]]

    # Random selection for diversity
    for _, individual in graded[int(len(graded)*retain):]:
        if random_select > random():
            parents.append(individual)

    desired_length = len(pop) - len(parents)
    children = []

    while len(children) < desired_length:
        male_idx, female_idx = randint(0, len(parents)-1), randint(0, len(parents)-1)
        if male_idx != female_idx:
            male = parents[male_idx]
            female = parents[female_idx]
            child1, child2 = two_point_crossover_binary(male, female)
            children.append(child1)
            if len(children) < desired_length:
                children.append(child2)

    all_individuals = parents + children

    # Mutation (bit-flip)
    mutated_population = [
        bit_flip_mutation_binary(ind, mutation_rate)
        for ind in all_individuals
    ]

    return mutated_population



# Main program

if __name__ == '__main__':
    target_coefficients = [25, 18, 31, -14, 7, -19]
    p_count = 100
    i_length = 6
    bits_per_coeff = 16
    coeff_min = -50
    coeff_max = 50
    max_generations = 200
    fitness_threshold = 0.01
    stagnation_limit = 20

    population_bin = population_binary(p_count, i_length, coeff_min, coeff_max, bits_per_coeff)
    fitness_history = []
    best_fitness_history = []

    solution_found = False
    best_fitness = float('inf')
    stagnation_counter = 0

    for gen in range(max_generations):
        avg_fitness = grade_binary(population_bin, target_coefficients, i_length, bits_per_coeff)
        fitness_history.append(avg_fitness)

        current_best = min(population_bin, key=lambda ind: fitness_binary(ind, target_coefficients, i_length, bits_per_coeff))
        current_best_fitness = fitness_binary(current_best, target_coefficients, i_length, bits_per_coeff)
        best_fitness_history.append(current_best_fitness)

        if current_best_fitness <= fitness_threshold:
            solution_found = True
            print(f"\n[SUCCESS] Solution found in {gen + 1} generations!")
            best_coeffs = decode_individual_binary(current_best, i_length, bits_per_coeff)
            print(f"Best coefficients: {best_coeffs}")
            print(f"Target coefficients: {target_coefficients}")
            print(f"Fitness: {current_best_fitness:.6f}")
            break

        if abs(current_best_fitness - best_fitness) < 0.001:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            best_fitness = current_best_fitness

        if stagnation_counter >= stagnation_limit:
            print(f"\n[STAGNATION] Stopped at generation {gen + 1}")
            best_coeffs = decode_individual_binary(current_best, i_length, bits_per_coeff)
            print(f"Best coefficients: {best_coeffs}")
            print(f"Target coefficients: {target_coefficients}")
            print(f"Fitness: {current_best_fitness:.6f}")
            break

        if (gen + 1) % 20 == 0:
            print(f"Gen {gen + 1}: Best fitness = {current_best_fitness:.6f}, Avg fitness = {avg_fitness:.6f}")

        population_bin = evolve_binary(population_bin, target_coefficients, i_length, bits_per_coeff, 
                                       retain=0.2, random_select=0.05, mutation_rate=0.01)

    if not solution_found and stagnation_counter < stagnation_limit:
        best_ind = min(population_bin, key=lambda ind: fitness_binary(ind, target_coefficients, i_length, bits_per_coeff))
        best_fit = fitness_binary(best_ind, target_coefficients, i_length, bits_per_coeff)
        best_coeffs = decode_individual_binary(best_ind, i_length, bits_per_coeff)
        print(f"\n[MAX GENERATIONS] Stopped at {max_generations} generations")
        print(f"Best coefficients: {best_coeffs}")
        print(f"Target coefficients: {target_coefficients}")
        print(f"Fitness: {best_fit:.6f}")

    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(fitness_history, label='Avg Fitness')
    axes[0, 0].plot(best_fitness_history, label='Best Fitness', color='green')
    axes[0, 0].set_title('Fitness Over Generations (Binary Encoding)')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(fitness_history, label='Avg Fitness')
    axes[0, 1].semilogy(best_fitness_history, label='Best Fitness', color='green')
    axes[0, 1].set_title('Fitness Over Generations (Log Scale)')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Fitness Error (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    x_plot = np.linspace(-100, 100, 100)
    best_final = decode_individual_binary(current_best, i_length, bits_per_coeff)
    y_target = [polynomial(x, target_coefficients) for x in x_plot]
    y_optimised = [polynomial(x, best_final) for x in x_plot]

    axes[1, 0].plot(x_plot, y_target, label='Target Polynomial', color='blue')
    axes[1, 0].plot(x_plot, y_optimised, label='Optimised Polynomial', linestyle='--', color='red')
    axes[1, 0].set_title('Target vs Optimised Polynomial')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    coeff_names = ['x^5', 'x^4', 'x^3', 'x^2', 'x^1', 'const']
    x_pos = np.arange(len(coeff_names))
    axes[1, 1].bar(x_pos - 0.2, target_coefficients, 0.4, label='Target', color='blue', alpha=0.7)
    axes[1, 1].bar(x_pos + 0.2, best_final, 0.4, label='Optimised', color='red', alpha=0.7)
    axes[1, 1].set_title('Coefficient Comparison')
    axes[1, 1].set_xlabel('Coefficient')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(coeff_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
