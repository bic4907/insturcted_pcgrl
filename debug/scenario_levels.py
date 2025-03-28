import numpy as np


level_0 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 4, 1, 7, 1, 7, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 7, 1, 2],
    [2, 6, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 2],
    [2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 7, 1, 1, 2],
    [2, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 2, 2, 7, 2, 2, 2, 1, 2, 2, 1, 2],
    [2, 2, 2, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2],
    [2, 1, 1, 1, 1, 2, 2, 1, 1, 7, 2, 2, 2, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 7, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])


level_1 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2],
    [2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2],
    [2, 1, 1, 7, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2],
    [2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 2, 4, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 6, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2],
    [2, 2, 1, 7, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2],
    [2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])


level_2 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2],
    [2, 1, 1, 7, 1, 2, 2, 1, 4, 1, 2, 2, 1, 7, 1, 2],
    [2, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2],
    [2, 2, 2, 1, 6, 1, 2, 2, 2, 1, 1, 1, 1, 1, 7, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 7, 1, 2, 2, 2, 2],
    [2, 1, 7, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2],
    [2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 7, 1, 2],
    [2, 1, 7, 1, 1, 1, 2, 1, 7, 1, 2, 2, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2],
    [2, 1, 6, 1, 1, 1, 7, 1, 2, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])

level_3 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 7, 7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 7, 7, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 6, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 5, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2],
    [2, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 1, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])


level_4 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2],
    [2, 1, 4, 1, 5, 1, 6, 1, 2, 2, 2, 2, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2],
    [2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 4, 2],
    [2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 2, 1, 1, 4, 1, 1, 1, 2, 6, 1, 1, 2],
    [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2],
    [2, 1, 1, 1, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 1, 2],
    [2, 1, 6, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 2],
    [2, 2, 2, 2, 2, 1, 1, 6, 1, 1, 1, 1, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 6, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])


level_5 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2],
    [2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2],
    [2, 1, 1, 2, 7, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2],
    [2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2],
    [2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 2],
    [2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2],
    [2, 1, 2, 5, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2],
    [2, 1, 6, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2],
    [2, 2, 2, 7, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])

level_6 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 7, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2],
    [2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 7, 1, 2],
    [2, 1, 1, 7, 1, 1, 2, 1, 1, 1, 2, 6, 1, 2, 1, 2],
    [2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2],
    [2, 1, 2, 4, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2],
    [2, 1, 1, 2, 1, 1, 1, 7, 1, 1, 2, 2, 2, 2, 1, 2],
    [2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 5, 1, 1, 1, 2],
    [2, 2, 1, 6, 1, 2, 2, 5, 1, 1, 1, 1, 2, 2, 1, 2],
    [2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2],
    [2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2],
    [2, 2, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 2, 1, 1, 2],
    [2, 2, 1, 1, 1, 1, 1, 1, 6, 1, 2, 1, 1, 1, 1, 2],
    [2, 2, 1, 7, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 8],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])

level_7 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2],
    [2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2],
    [2, 1, 1, 7, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2],
    [2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2],
    [2, 1, 2, 4, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2],
    [2, 1, 1, 2, 1, 1, 1, 7, 1, 1, 2, 2, 2, 2, 1, 2],
    [2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 7, 1, 2],
    [2, 2, 1, 6, 1, 2, 2, 5, 1, 1, 1, 1, 2, 2, 1, 2],
    [2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2],
    [2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2],
    [2, 2, 1, 1, 1, 1, 1, 1, 6, 1, 2, 1, 1, 1, 1, 2],
    [2, 2, 1, 7, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 8],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])

level_8 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2],
    [2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 7, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2],
    [2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 2, 1, 2, 6, 1, 1, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])

level_9 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 2, 2],
    [2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 7, 1, 1, 2, 2],
    [2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2],
    [2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2],
    [2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2],
    [2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2],
    [2, 2, 5, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2],
    [2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2],
    [2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2],
    [2, 2, 2, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 8, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])

level_10 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2],
    [2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2],
    [2, 1, 1, 1, 1, 2, 1, 6, 1, 1, 2, 2, 1, 2, 2, 2],
    [2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 7, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 8, 2],
    [2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])
level_11 = np.array([
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2],
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 1, 1, 2, 1, 1, 6, 2, 2, 2, 2, 2, 2],
    [2, 1, 5, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2],
    [2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2, 4, 1, 1, 2, 2, 2, 2, 2],
    [2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
    [2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8, 2],
    [2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
])
AllLevels = [level_0, level_1, level_2, level_3, level_4, level_5, level_6, level_7, level_8, level_9, level_10, level_11]
# AllLevels = [level_10]
