import numpy as np
from typing import Tuple


def unpairing_maps(pairs: np.ndarray, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits horizontally concatenated previous-current map pairs into two independent maps.

    This function takes a set of horizontally combined map pairs (previous and current states)
    and separates them into their original independent maps. The split is determined by the 
    `cols` parameter.

    Parameters:
        pairs (np.ndarray): A 3D array where previous and current maps are concatenated horizontally.
                            Shape: (batch_size, num_columns, num_channels).
        cols (int): The column index used as the split point.
                    Specifies the number of columns for the first map (previous map).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the separated maps.
            - First element: Previous state map (`prev_env_map`).
            - Second element: Current state map (`curr_env_map`).

    Example:
        >>> pairs = np.array([[[1, 2], [5, 6]], [[3, 4], [7, 8]]])
        >>> cols = 1  
        >>> prev_map, curr_map = unpairing_maps(pairs, cols)
        >>> print(prev_map)  
        [[[1 2]]
         [[3 4]]]
        >>> print(curr_map)  
        [[[5 6]]
         [[7 8]]]
    """
    prev_env_map = pairs[:, 0:cols, :]
    curr_env_map = pairs[:, cols:, :]
    return prev_env_map, curr_env_map
