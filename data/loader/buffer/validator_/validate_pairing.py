import numpy as np


def validate_pairing(
    prev_env_map: np.ndarray,
    curr_env_map: np.ndarray,
    pairs: np.ndarray,
):
    """
    Validates whether the given pairs match the previous and current environment maps.

    Parameters:
    prev_env_map (np.ndarray): A 2D array representing the previous environment map.
    curr_env_map (np.ndarray): A 2D array representing the current environment map.
    pairs (np.ndarray): A 1D array containing the pairs to be validated.

    Returns:
    bool: True if all pairs match, otherwise False.

    Example:
    >>> prev = np.array([[1, 2], [3, 4]])
    >>> curr = np.array([[1, 2], [5, 6]])
    >>> pairs = np.array([[1, 2, 1, 2], [3, 4, 5, 6]])
    >>> result = validate_pairing(prev, curr, pairs)
    >>> print(result)  
    """
    return np.all(
        np.array(
            [
                np.all(np.concatenate([prev_map, curr_map]) == pairs[i])
                for i, (prev_map, curr_map) in enumerate(
                    zip(prev_env_map, curr_env_map)
                )
            ]
        )
    )
