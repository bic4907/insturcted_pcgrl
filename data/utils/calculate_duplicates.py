import numpy as np
from flax.struct import dataclass


@dataclass
class Duplicates:
    """
    A data storage class for results related to duplicate pair checks.

    Attributes:
        unique_pairs (np.ndarray): A 2D array containing unique pairs, 
                                   where each row represents a non-duplicate pair.
        unique_counts (np.ndarray): A 1D array containing the count of occurrences for each unique pair.
        duplicates_ratio (float): The ratio of unique pairs, calculated as the number of unique pairs 
                                  divided by the total number of pairs.
        num_unique_value (int): The number of unique pairs.
        num_total_value (int): The total number of pairs.
        min_count (int): The minimum occurrence count among unique pairs.
        max_count (int): The maximum occurrence count among unique pairs.
    """


    unique_pairs: np.ndarray
    unique_counts: np.ndarray
    duplicates_ratio: float
    num_unique_value: int
    num_total_value: int
    min_count: int
    max_count: int


def calculate_duplicates(pairs: np.ndarray) -> Duplicates:
    """
    Calculates duplicate pairs from the given array and returns statistical information about them.

    Parameters:
        pairs (np.ndarray): A 2D array containing pairs for duplication analysis. 
                            Each row represents a single pair.

    Returns:
        Duplicates: A Duplicates object containing statistical data related to duplicate pair checks.

    Example:
        >>> pairs = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        >>> result = calculate_duplicates(pairs)
        >>> print(result.unique_counts)
        [2 1 1]
        >>> print(result.unique_pairs)
        [[1 2]
         [3 4]
         [5 6]]
        >>> print(result.duplicates_ratio)
        0.75
    """

    unique_pairs, unique_counts = np.unique(pairs, axis=0, return_counts=True)
    num_total_value = pairs.shape[0]
    num_unique_value = unique_pairs.shape[0]
    duplicates_ratio = (num_total_value - num_unique_value) / num_total_value
    min_count, max_count = np.min(unique_counts).item(), np.max(unique_counts).item()
    return Duplicates(
        unique_pairs=unique_pairs,
        unique_counts=unique_counts,
        duplicates_ratio=duplicates_ratio,
        num_unique_value=num_unique_value,
        num_total_value=num_total_value,
        min_count=min_count,
        max_count=max_count,
    )
