import numpy as np

v1 = np.array([1, 0])
v2 = np.array([0, 1])
v3 = np.array([np.sqrt(2), np.sqrt(2)])

# Dimension
v1.shape


# Magnitude
np.sqrt(np.sum(v1**2))

np.linalg.norm(v1)


np.linalg.norm(v3)


# Dot product
np.sum(v1 * v2)


v1 @ v3
