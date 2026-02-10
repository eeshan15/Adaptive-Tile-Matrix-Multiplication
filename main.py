# main.py
import numpy as np
import pickle
from env import MatMulEnv

env = MatMulEnv()

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.1    # exploration rate
episodes = 5000

# Discretization
def get_state(matrix_size, tile_size):
    return (matrix_size, tile_size)

# Initialize Q-table
Q = {}

for ep in range(episodes):
    state = env.reset()
    s = get_state(*state)

    if s not in Q:
        Q[s] = [0.0, 0.0]  # Two actions: [static, dynamic]

    # ε-greedy action selection
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1])
    else:
        action = np.argmax(Q[s])

    next_state, reward, done, info = env.step(action)

    if done:
        # Q-learning update
        Q[s][action] = Q[s][action] + alpha * (reward - Q[s][action])

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}: State={s}, Action={'Static' if action==0 else 'Dynamic'}, Reward={reward:.4f}, Time={info['time_ms']:.4f}ms")

# Save Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

print("Training done ✅. Q-table saved as q_table.pkl.")
