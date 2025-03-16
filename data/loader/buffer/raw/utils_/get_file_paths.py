from glob import glob
from os.path import join
import numpy as np
from pipe import Pipe

from typing import Optional
import logging


def get_file_paths(buffer_dir: str, limited_to: Optional[int] = None):
    file_list = glob(join(buffer_dir, "**", "*.npz"), recursive=True) | Pipe(np.array)
    n_sample = limited_to if isinstance(limited_to, int) else len(file_list)
    np.random.shuffle(file_list)
    return file_list[:n_sample]


if __name__ == "__main__":
    TEST_DIR = "./pcgrl_buffer"
    N_RANDOMIZE_TEST = 3
    N_FILE_LIMIT = 10

    path_for_all_samples = np.array(
        [get_file_paths(TEST_DIR) for _ in range(N_RANDOMIZE_TEST)]
    )
    path_for_limited_samples = np.array(
        [
            get_file_paths(TEST_DIR, limited_to=N_FILE_LIMIT)
            for _ in range(N_RANDOMIZE_TEST)
        ]
    )
    all_samples_unique = (
        len(np.unique(path_for_all_samples, axis=0)) == N_RANDOMIZE_TEST
    )
    limited_samples_unique = (
        len(np.unique(path_for_limited_samples, axis=0)) == N_RANDOMIZE_TEST
    )

    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"RANDOMIZE TEST (No file num limit): {'Pass' if all_samples_unique else 'Fail'}"
    )
    logging.info(
        f"RANDOMIZE TEST (File num limit {N_FILE_LIMIT}): {'Pass' if limited_samples_unique else 'Fail'}"
    )
