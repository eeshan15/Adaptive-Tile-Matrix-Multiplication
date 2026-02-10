# env.py
import ctypes
import numpy as np
import os


class MatMulEnv:
    def __init__(self, matrix_size_range=(128, 1024), tile_sizes=[8, 16, 32]):
        self.matrix_size_range = matrix_size_range
        self.tile_sizes = tile_sizes
        self.reset()

        # Load DLL
        dll_name = r"C:\\Users\\ASUS\\OneDrive\\Desktop\\adaptive-matmul\\libmatmul.dll"
        self.lib = ctypes.CDLL(dll_name)
        self.lib.run_matmul.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # A (float pointer)
            ctypes.POINTER(ctypes.c_float),   # B (float pointer)
            ctypes.POINTER(ctypes.c_float),   # C (float pointer)
            ctypes.c_int,                     # wA (int)
            ctypes.c_int,                     # wB (int)
            ctypes.c_int,                     # tileSize (int)
            ctypes.POINTER(ctypes.c_float)    # time_ms (float pointer)
        ]

        # Correct return type
        self.lib.run_matmul.restype = None
        
        print(self.lib)

    def reset(self):
        self.matrix_size = np.random.choice(np.arange(*self.matrix_size_range, 64))
        self.tile_size = np.random.choice(self.tile_sizes)
        self.A = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
        self.B = np.random.rand(self.matrix_size, self.matrix_size).astype(np.float32)
        self.C = np.zeros((self.matrix_size, self.matrix_size), dtype=np.float32)
        return (self.matrix_size, self.tile_size)

    def step(self, action):
        assert action in [0, 1], "Action must be 0 (static) or 1 (dynamic)"
        m, t = self.matrix_size, self.tile_size

        # Flatten arrays
        A_flat = self.A.flatten()
        B_flat = self.B.flatten()
        C_flat = self.C.flatten()

        # Convert to ctypes
        A_ct = A_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        B_ct = B_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        C_ct = C_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        time_ct = ctypes.c_float()

        # Call DLL
        self.lib.run_matmul(A_ct, B_ct, C_ct, m, m, t, ctypes.byref(time_ct))

        reward = -float(time_ct.value)  # Lower time = higher reward
        obs = (m, t)
        done = True  # One-step episode
        info = {"time_ms": float(time_ct.value), "action": action}
        return obs, reward, done, info
