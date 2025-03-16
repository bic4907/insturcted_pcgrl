import numpy as np

from ..types_ import IndicedData


def load_indiced_data(data_path: str) -> IndicedData:
    """
    Load dense data from the specified path and return a DenseData object.

    Args:
        data_path (str): Path to the data file to load.

    Returns:
        DenseData: DenseData object containing the loaded dense data.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        KeyError: If required keys are missing from the data file.
    """

    data = np.load(data_path, allow_pickle=True)

    map_obs = data.get("map_obs")
    env_map = data.get("env_map")

    prev_indices = data.get("prev_indices")
    curr_indices = data.get("curr_indices")

    return IndicedData(
        map_obs=map_obs,
        env_map=env_map,
        prev_indices=prev_indices,
        curr_indices=curr_indices,
    )
