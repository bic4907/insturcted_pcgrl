import numpy as np


def generate_pair_indices(done_indices: np.ndarray):
    """
    Extracts previous/current index pairs for unfinished steps.

    This function identifies index pairs that need processing from a boolean array
    indicating completion status. For each unfinished index, it returns the index
    along with the immediately preceding index.

    Parameters
    ----------
    done_indices : np.ndarray
        A 1D boolean array indicating completion status.
        - True: The step at this index is completed.
        - False: The step at this index is unfinished.

    Returns
    -------
    tuple
        - prev_indices (np.ndarray): Array of previous indices for unfinished steps.
        - curr_indices (np.ndarray): Array of current indices for unfinished steps.

    Notes
    -----
    - This function only handles consecutive step indices. Gaps of more than one step
      require additional handling.
    - Since there are no steps before index 0, an unfinished first step is excluded
      from the results.

    Exceptions
    ----------
    TypeError
        Raised if `done_indices` is not a 1D array.
    """

    if done_indices.ndim != 1:
        raise TypeError(
            f" Done indices is expected to be a 1D array. Current dimension: {done_indices.ndim},"
        )
    undone_indices = np.argwhere(done_indices != True).ravel()
    done_indices = np.argwhere(done_indices == True).ravel()
    prev_indices = np.argwhere(undone_indices - 1 >= 0).squeeze()
    curr_indices = (prev_indices + 1).squeeze()

    return prev_indices, curr_indices


if __name__ == "__main__":
    sample_indices = np.array([False, False, False, False, True, False, False, False])
    print(generate_pair_indices(sample_indices))
