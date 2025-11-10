import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppose you have multiple runs' results:
# Each entry: (generations taken to solution, success flag True/False)
run_data = [
    (38, True),
    (42, True),
    (50, True),
    (100, False),  # Timeout, no solution found
    (45, True),
]

# Extract data
generations = [x[0] for x in run_data if x[1]]
success_rate = sum(x[1] for x in run_data) / len(run_data) * 100

# --- Graph 1: Fitness History (from a single run) ---
def plot_fitness_history(fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Fitness Over Generations (Single Run)')
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage:
# plot_fitness_history(fitness_history)  # fitness_history from your GA run

# --- Graph 2: Histogram of Generations to Solution ---
plt.figure(figsize=(8, 5))
plt.hist(generations, bins=10, range=(min(generations), max(generations)), alpha=0.7, color='b')
plt.xlabel('Generations to Find Solution')
plt.ylabel('Frequency')
plt.title('Distribution of Generations to Solution over Multiple Runs')
plt.grid(True, alpha=0.3)
plt.show()

# --- Table: Summary Statistics ---
df = pd.DataFrame(run_data, columns=['Generations', 'Success'])
summary_table = pd.DataFrame({
    'Metric': ['Mean Generations', 'Median Generations', 'Success Rate (%)'],
    'Value': [np.mean(generations), np.median(generations), success_rate]
})
print(summary_table)

# --- Graph 3: Mutation Rate Impact ---
mutation_rates1 = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
avg_generations1 = [2, 1, 0, 0, 0, 0]
mutation_rates2 = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
avg_generations2 = [20, 37, 13, 8, 35, 44]
mutation_rates3 = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
avg_generations3 = [10, 18, 2, 5, 9, 18] 

best_retain_rate = [0, 0, 0, 0, 0, 0]
worst_retain_rate =[15, 17, 20, 18, 14, 22]
avg_retain_rate= [6, 4.8, 7.6, 6, 4.6, 7.8]
retain_rate = [0, 0.01, 0.05, 0.1, 0.2, 0.3]

best_pop = [100, 0, 0, 0, 0]
worst_pop = [100, 100, 17, 53, 130]
avg_pop =[100, 80, 3.6, 17.8, 26]
population = [10,30,50,100,200]
plt.figure(figsize=(8, 5))
plt.plot(population, best_pop, marker='o', color='g', label= "Best Performance")
plt.plot(population, worst_pop, marker='o', color='b', label= "Worst Performance")
plt.plot(population, avg_pop, marker='o', color='r', label= "Average Performance")
plt.xlabel('Population Size')
plt.ylabel('Average Generations to Solution')
plt.title('Effect of Population Size on GA Convergence Speed')
plt.legend(loc= "upper left")
plt.grid(True, alpha=0.3)
plt.show()
