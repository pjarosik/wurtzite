import cupy as cp

NU = 0.35
BETA_ONES = cp.eye(3)  # Jedynka 3x3
DIMS = 3  # Liczba wymiarow (x, y, z)
DIS_TOLERANCE = 1e-1
N_GLIDE_PLANE_POINTS = 10000