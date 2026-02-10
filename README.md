
# Reinforcement Learning-Based Adaptive Tile Size Selection for Matrix Multiplication Optimization on CUDA

## Overview

Matrix multiplication is a critical operation in high-performance computing and deep learning. This project implements a Reinforcement Learning (RL) framework to dynamically optimize tile sizes for CUDA kernels. By leveraging Q-learning, the system adaptively chooses between static and dynamic tiled kernel implementations based on observed performance metrics, allowing for automated performance tuning without extensive manual intervention.
![Tiled_matrix_multiplication](https://github.com/user-attachments/assets/1076d1fc-09bd-4050-a810-1b3cc1a835d6)

---

## Methodology

### State Representation

The state  at any given time step  is defined as a tuple of the input matrix size and the selected tile size:


* **Matrix Dimension ():** Dimensions range from 128 to 1024.
* **Tile Size ():** Supported sizes include .

### Action Space

The agent selects from a discrete action space :

* **Action 0 (Static):** Invokes a CUDA kernel with hardcoded tile sizes for optimized performance on specific scales.
* **Action 1 (Dynamic):** Invokes a templated CUDA kernel where the tile size is passed at runtime, offering higher flexibility.



### Reward Function

The reward  is defined as the negative of the execution time in milliseconds to incentivize latency minimization:


### Q-Learning Framework

The agent utilizes a tabular Q-learning update rule to optimize the policy over time:


Training is conducted for 5,000 episodes using an -greedy policy.

---

## System Requirements

* **Hardware:** NVIDIA GPU (e.g., RTX 3050 4GB).
* **Software:** CUDA Toolkit v12.6, Python 3.13.
* **Python Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`.

---

## How to Run

### 1. Kernel Compilation

Compile the CUDA kernels into a shared library to be interfaced via `ctypes`.

```bash
nvcc -shared -o libmatmul.dll wrapper.cu static_tile_matmul.cu dynamic_tile_matmul.cu

```

### 2. Training the Agent

Execute the main script to initiate the 5000-episode training sequence. This will generate the `q_table.pkl` and `rl_training_log.csv`.

```bash
python main.py

```

### 3. Analysis

Generate performance heatmaps and visualize decision patterns using the plotting utility.

```bash
python plot_q_table.py

```

---

## Results

Experimental data confirms that the RL agent successfully learns optimal tiling strategies.

* **Learning Progression:** Episode rewards improved from approximately **-5.6750 ms** in early stages to **-1.0021 ms** by episode 5000.
* **Decision Trends:** The agent favors static kernels for small matrices () and transitions to dynamic tiling for medium-to-large workloads to maximize throughput.



---

## Repository Maintenance (.gitignore)

To keep the repository clean, it is recommended to ignore compiled binaries and temporary files:

```text
# Compiled Binaries
*.dll
*.exe
*.exp
*.lib
*.o
*.obj

# Python
__pycache__/
*.pyc

# Logs and Data
*.csv
*.pkl

# IDE
.vscode/

```

---

## License

This project is licensed under the **MIT License**.

---

## Citation

If you use this research or code, please cite the following publication:

**IEEE Xplore Link:** [https://ieeexplore.ieee.org/document/11371069](https://ieeexplore.ieee.org/document/11371069)

```bibtex
@INPROCEEDINGS{11371069,
  author={Bablani, Eeshan and Verma, Chahat and Basu, Shatabdi},
  booktitle={2025 2nd International Conference on Integration of Computational Intelligent System (ICICIS)}, 
  title={Reinforcement Learning-Based Adaptive Tile Size Selection for Matrix Multiplication Optimization on CUDA}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Visualization;Q-learning;Runtime;Graphics processing units;Throughput;Performance metrics;Computational efficiency;Kernel;Optimization;Tuning;Matrix multiplication;Tile size selection;Reinforcement learning;Tiling optimization;Q-Learning;CUDA},
  doi={10.1109/ICICIS65613.2025.11371069}}

```

---
