import numpy as np

from ..types_ import DenseData


def load_dense_data(data_path: str) -> DenseData:
    """
        Loads dense data from the specified path and returns a DenseData object.

        Args:
            data_path (str): Path to the data file to be loaded.

        Returns:
            DenseData: A DenseData object containing the loaded dense data.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            KeyError: If the required keys are missing in the data file.
    """

    data = np.load(data_path, allow_pickle=True)

    prev_map_obs = data.get("prev_map_obs")
    curr_map_obs = data.get("curr_map_obs")

    prev_env_map = data.get("prev_env_map")
    curr_env_map = data.get("curr_env_map")

    return DenseData(
        prev_map_obs=prev_map_obs,
        curr_map_obs=curr_map_obs,
        prev_env_map=prev_env_map,
        curr_env_map=curr_env_map,
    )
