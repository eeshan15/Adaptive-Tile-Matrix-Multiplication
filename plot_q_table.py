import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Q-table
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Convert state-action Q-values into usable form
states = list(q_table.keys())
matrix_sizes = sorted(set([s[0] for s in states]))
tile_sizes = sorted(set([s[1] for s in states]))

# Create action and reward matrices
action_matrix = np.zeros((len(matrix_sizes), len(tile_sizes)))
reward_matrix = np.zeros_like(action_matrix)

for i, m in enumerate(matrix_sizes):
    for j, t in enumerate(tile_sizes):
        q_vals = q_table.get((m, t), [0, 0])
        best_action = np.argmax(q_vals)
        best_reward = q_vals[best_action]
        action_matrix[i, j] = best_action
        reward_matrix[i, j] = -best_reward  # negate because reward = -time

# Plot 1: Heatmap of best actions
plt.figure(figsize=(10, 6))
sns.heatmap(action_matrix, annot=True, cmap='coolwarm', xticklabels=tile_sizes, yticklabels=matrix_sizes)
plt.title("Best Action (0=Static, 1=Dynamic)")
plt.xlabel("Tile Size")
plt.ylabel("Matrix Size")
plt.tight_layout()
plt.show()

# Plot 2: Heatmap of Execution Time
plt.figure(figsize=(10, 6))
sns.heatmap(reward_matrix, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=tile_sizes, yticklabels=matrix_sizes)
plt.title("Best Time (ms) for State")
plt.xlabel("Tile Size")
plt.ylabel("Matrix Size")
plt.tight_layout()
plt.show()
