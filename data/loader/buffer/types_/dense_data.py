from flax.struct import dataclass
import numpy as np


@dataclass
class DenseData:
    """
    The DenseData class is a data structure that stores previous and current map observations
    along with environment maps.

    Attributes:
        prev_map_obs (np.ndarray): Array of previous map observations.
        curr_map_obs (np.ndarray): Array of current map observations.
        prev_env_map (np.ndarray): Array of previous environment maps.
        curr_env_map (np.ndarray): Array of current environment maps.
    """
    prev_map_obs: np.ndarray
    curr_map_obs: np.ndarray
    prev_env_map: np.ndarray
    curr_env_map: np.ndarray
