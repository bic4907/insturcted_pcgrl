
import logging
import os
from os.path import basename, join
from typing import List, Tuple

from functools import lru_cache

import numpy as np
from tqdm import tqdm

from data import generate_pair_indices, load_each_data
from data.utils import calculate_duplicates, pairing_maps
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

n_jobs = 32


@lru_cache
def check_sample_duplicates(
    buffer_dir: str,
    n_jobs: int = 4,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Loads sample data from the given buffer directory and calculates the duplication ratio for each sample.

    Args:
        buffer_dir (str): Path to the directory where sample data is stored.
        n_jobs (int): Number of parallel jobs for processing. Default is 4.

    Returns:
        tuple:
            - file_list (List[str]): A list of processed files.
            - duplicate_ratios (np.ndarray): An array containing the duplication ratio for each sample.
            - num_of_samples (np.ndarray): An array containing the number of data points considered in each sample.
    """

    data = load_each_data(buffer_dir, n_jobs=n_jobs)
    done_indices = data.done_indices
    env_map = data.env_map
    file_list = data.file_list

    duplicate_ratio_list: List[float] = []
    num_of_samples: List[int] = []

    for map_, indices_ in tqdm(
        zip(env_map, done_indices),
        total=len(env_map),
        desc="Calculating duplicate sample ratio per sample",
    ):
        prev_indices, curr_indices = generate_pair_indices(indices_.ravel())
        map_ = map_.reshape((-1, *map_.shape[-2:]))
        prev_map, curr_map = map_[prev_indices], map_[curr_indices]

        duplicates = calculate_duplicates(pairing_maps(prev_map, curr_map))
        duplicates_ratio = duplicates.duplicates_ratio

        duplicate_ratio_list.append(duplicates_ratio)
        num_of_samples.append(prev_map.shape[0])

    return file_list, np.array(duplicate_ratio_list), np.array(num_of_samples)


def check_each_sample_duplicates(
    buffer_dir: str,
    e: float = 0.5,
    by_dir: bool = False,
    n_jobs: int = 4,
) -> Tuple[List[str], List[List[str]], List[float]]:
    """
    Examines the duplication ratio of individual samples within the buffer directory 
    and identifies files exceeding the threshold `e`.

    Parameters
    ----------
    buffer_dir : str
        Path to the buffer directory containing subdirectories with sample data.
    e : float, optional (default=0.5)
        Duplication ratio threshold. Files or directories exceeding this threshold 
        are marked for removal.
    by_dir : bool, optional (default=False)
        If True, calculates the average duplication ratio per directory and 
        identifies entire directories for removal.
        If False, calculates duplication ratios per file and identifies only 
        individual files exceeding the threshold.
    n_jobs : int, optional (default=4)
        Number of CPU cores to use for parallel processing during duplication checks.

    Returns
    -------
    tuple
        - buffer_paths : List[str]
            List of all subdirectory paths within the buffer directory.
        - files_to_remove : List[List[str]]
            List of file paths to be removed for each subdirectory.
            - If `by_dir=True`, all files in a directory are included if the 
              average duplication ratio exceeds `e`.
            - If `by_dir=False`, only individual files exceeding the duplication 
              ratio threshold are included.
        - avg_duplicate_ratios : List[float]
            List of average duplication ratios for each directory.
    """
    buffer_paths = [join(buffer_dir, dir_) for dir_ in os.listdir(buffer_dir)]
    files_to_remove: List[List[str]] = []
    avg_duplicate_ratios: List[float] = []

    for buffer_path in buffer_paths:
        file_list, duplicate_ratios, _ = check_sample_duplicates(buffer_path, n_jobs)
        avg_ratio = np.mean(duplicate_ratios) if len(duplicate_ratios) > 0 else 0.0
        avg_duplicate_ratios.append(avg_ratio)

        if by_dir:
            if avg_ratio > e:
                files_to_remove.append(file_list)
            else:
                files_to_remove.append([])
        else:
            indices_above_e = np.where(duplicate_ratios > e)[0]
            files_above_e = [file_list[idx] for idx in indices_above_e]
            files_to_remove.append(files_above_e)

    return buffer_paths, files_to_remove, avg_duplicate_ratios


def remove_duplicate_files(
    buffer_dir: str,
    e: float = 0.5,
    by_dir: bool = False,
    n_jobs: int = 4,
) -> None:
    """
    Prompts the user for confirmation before removing files or directories 
    that exceed the duplication ratio threshold `e`.

    Parameters
    ----------
    buffer_dir : str
        Path to the buffer directory containing subdirectories with sample data.
    e : float, optional (default=0.5)
        Duplication ratio threshold. Files or directories exceeding this threshold 
        are marked for removal.
    by_dir : bool, optional (default=False)
        If True, removal is based on the average duplication ratio per directory.
        If False, removal is based on the duplication ratio of individual files.
    n_jobs : int, optional (default=4)
        Number of CPU cores to use for parallel processing during duplication checks.

    Returns
    -------
    None
        Performs file or directory removal operations and does not return a value.
    """

    buffer_paths, files_to_remove, avg_duplicate_ratios = check_each_sample_duplicates(
        buffer_dir, e, by_dir, n_jobs
    )

    for buffer_path, files, avg_ratio in zip(
        buffer_paths, files_to_remove, avg_duplicate_ratios
    ):
        if len(files) == 0:
            logger.info(f"No files to remove in {basename(buffer_path)}.")
            continue

        if by_dir:
            print(
                f"\nAverage duplication ratio in {basename(buffer_path)}: {avg_ratio:.2f} (exceeds threshold {e})"
            )
            print(f"Number of files to remove: {len(files)}")
        else:
            print(f"\nFiles in {basename(buffer_path)} exceeding the duplication ratio threshold {e}:")
            for file in files:
                print(f"  - {file}")
            print(f"A total of {len(files)} files exceed the threshold.")

        while True:
            response = input("\nDo you want to remove these items? (yes/no): ").strip().lower()
            if response in ["yes", "no"]:
                break
            print("Invalid input. Please enter 'yes' or 'no'.")

        if response == "yes":
            for file in files:
                try:
                    os.remove(file)
                    logger.info(f"Removed: {file}")
                except OSError as err:
                    logger.error(f"Failed to remove file: {file} - {err}")
            logger.info(f"Successfully removed {len(files)} files from {basename(buffer_path)}.")
        else:
            logger.info(f"Removal canceled for {basename(buffer_path)}.")



def main() -> None:
    """
    Main function that executes the duplication ratio check and file removal process.
    """

    buffer_dir = "./pcgrl_buffer"
    print("Checking and removing duplicate files on a per-file basis:")
    remove_duplicate_files(
        buffer_dir=buffer_dir,
        e=0.5,
        by_dir=False,
        n_jobs=n_jobs,
    )

    print("\nChecking and removing duplicate files on a per-directory basis:")
    remove_duplicate_files(
        buffer_dir=buffer_dir,
        e=0.5,
        by_dir=True,
        n_jobs=n_jobs,
    )



if __name__ == "__main__":
    main()
