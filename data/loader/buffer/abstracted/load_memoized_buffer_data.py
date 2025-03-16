from os.path import isfile
import numpy as np

from .load_indiced_data import load_indiced_data
from ..raw import load_buffer_data
from ..types_ import IndicedData


def load_memoized_buffer_data(
    buffer_dir: str,
    target_path: str,
    file_limit: int = 0,
    n_jobs: int = 4,
    ignore_file: bool = False,
) -> IndicedData:
    """
    Load data from buffer using memoization (caching).

    This function loads memoized data from the given directory and saves the merged data
    to the specified target path. If a saved file already exists and `ignore_file` is False,
    the data is loaded from the existing file.

    Args:
        buffer_dir (str):
            Path to the directory containing data files.
        target_path (str):
            Path to save the merged data file.
        file_limit (int, optional):
            Limit on the number of files to use. Default is 0, which indicates using all available files.
        n_jobs (int, optional):
            Number of parallel jobs to use for loading data. Default is 4.
        ignore_file (bool, optional):
            Whether to ignore the existing cached file and reload data. Default is False.

    Returns:
        IndicedData:
            An IndicedData object containing the loaded data. This includes map observations,
            environment map arrays, and previous and current state indices.

    Raises:
        FileNotFoundError:
            If the file does not exist at the specified path.

    Example:
    ```
        data = load_memoized_data(
            buffer_dir="/path/to/buffer",
            target_path="/path/to/target_file.npz",
            file_limit=10,
            n_jobs=4,
            ignore_file=False
        )
    ```
    """
    if not ignore_file and file_limit != 0:
        print("Warning: The file_limit setting may be ignored when using cache.")
    is_cached = isfile(target_path) and not ignore_file

    if not is_cached:
        data = load_buffer_data(
            buffer_dir,
            file_limit=file_limit,
            n_jobs=n_jobs,
        )

        dataset = {
            "map_obs": data.map_obs,
            "env_map": data.env_map,
            "prev_indices": data.prev_indices,
            "curr_indices": data.curr_indices,
        }

        np.savez(target_path, **dataset)
    else:
        data = load_indiced_data(target_path)

    return data
