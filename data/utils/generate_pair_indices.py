import numpy as np


def generate_pair_indices(done_indices: np.ndarray):
    """
    Extracts previous and current indices of unfinished steps from the given `done_indices` array.

    Args:
        done_indices (np.ndarray): A 1D boolean array indicating completion status.
                                   `True` represents a completed index, while `False` represents an unfinished index.

    Returns:
        tuple:
            - prev_indices (np.ndarray): An array of previous indices for unfinished steps.
            - curr_indices (np.ndarray): An array of current indices for unfinished steps.

    Notes:
        - A patch is needed to exclude cases where the step difference is 2 or more.
        - Previous indices are extracted within a valid range from `undone_indices`.

    Raises:
        AssertionError: If `done_indices` is not a 1D array.
    """


    if done_indices.ndim != 1:
        raise TypeError(
            f"done_indices is expected to be a 1D array. Current dimension: {done_indices.ndim},"
        )
    undone_indices = np.argwhere(done_indices != True).ravel()
    done_indices = np.argwhere(done_indices == True).ravel()
    prev_indices = np.argwhere(undone_indices - 1 >= 0).squeeze()
    curr_indices = (prev_indices + 1).squeeze()

    return prev_indices, curr_indices


if __name__ == "__main__":
    sample_indices = np.array([False, False, False, False, True, False, False, False])
    print(generate_pair_indices(sample_indices))
