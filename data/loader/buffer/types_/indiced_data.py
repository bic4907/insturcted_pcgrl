import numpy as np
from flax.struct import dataclass


@dataclass
class IndicedData:
    """
    A data structure for storing map data and indices.

    This class contains arrays for map observations and environment maps,
    as well as index arrays for previous and current states.
    Unlike traditional data types, it avoids data duplication, improving memory efficiency.

    Attributes:
        map_obs (np.ndarray):
            A list of arrays representing map observations.
        env_map (np.ndarray):
            A list of arrays representing environment maps.
        prev_indices (np.ndarray):
            An array of indices representing previous states.
        curr_indices (np.ndarray):
            An array of indices representing current states.
    """

    map_obs: np.ndarray
    env_map: np.ndarray
    prev_indices: np.ndarray
    curr_indices: np.ndarray
