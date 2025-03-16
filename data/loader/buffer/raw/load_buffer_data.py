import numpy as np
from ..types_ import IndicedData

from .load_each_buffer_data import load_each_buffer_data
from ..utils_ import generate_pair_indices


def load_buffer_data(
    buffer_dir: str,
    file_limit: int = 0,
    n_jobs: int = 4,
):
    """
    Loads data from the buffer directory and processes it into previous/current state pairs.

    After loading map observations and environment maps from multiple data files,
    the data is processed into previous/current state pairs for consecutive time steps.

    Parameters
    ----------
    buffer_dir : str
        Path to the directory where the data files are stored.
    file_limit : int, optional
        Limit on the number of files to load. If 0 or less, all files are loaded (default: 0).
    n_jobs : int, optional
        Number of parallel jobs to use for processing (default: 4).

    Returns
    -------
    IndicedData
        A data object containing the following fields:
        - map_obs (np.ndarray): Array containing all map observations.
        - env_map (np.ndarray): Array containing all environment maps.
        - prev_indices (np.ndarray): Indices for previous states.
        - curr_indices (np.ndarray): Indices for current states.

        prev_indices and curr_indices are used together to identify state pairs:
        - Previous state: map_obs[prev_indices], env_map[prev_indices]
        - Current state: map_obs[curr_indices], env_map[curr_indices]

    Raises
    ------
    ValueError
        If the buffer directory does not exist or file loading fails.

    Example
    -------
    ```python
    loaded_data = load_buffer_data(
        buffer_dir="/path/to/buffer",
        file_limit=10,
        n_jobs=4
    )
    prev_states = loaded_data.map_obs[loaded_data.prev_indices]
    curr_states = loaded_data.map_obs[loaded_data.curr_indices]
    ```
    """
    data = load_each_buffer_data(buffer_dir, file_limit, n_jobs)
    done_indices = np.concat(data.done_indices).ravel()
    map_obs = data.map_obs
    env_map = data.env_map
    prev_indices, curr_indices = generate_pair_indices(done_indices)
    done_indices = np.concat(done_indices, axis=None)
    map_obs = np.concat(map_obs, axis=0)
    env_map = np.concat(env_map, axis=0)
    map_obs = map_obs.reshape((-1, *map_obs.shape[-3:]))
    env_map = env_map.reshape((-1, *env_map.shape[-2:]))
    return IndicedData(
        map_obs=map_obs,
        env_map=env_map,
        prev_indices=prev_indices,
        curr_indices=curr_indices,
    )
