
import logging
import os
import shutil
from os.path import abspath, basename, dirname, join
from typing import List, Tuple

from functools import lru_cache

import hydra
import jax
import matplotlib.pyplot as plt  
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tenumerate

# from conf.config import BertTrainConfig
from data import generate_pair_indices, load_each_data
from data.utils import calculate_duplicates, pairing_maps
from LLM.path_utils import init_config

log_level = os.getenv(
    "LOG_LEVEL", "INFO"
).upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
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
            - file_list (List[str]): List of processed files.
            - duplicate_ratios (np.ndarray): Array containing the duplication ratio for each sample.
            - num_of_samples (np.ndarray): Array containing the number of data points considered in each sample.

    Notes:
        - The duplication ratio is calculated based on the previous and current indices of each sample.
        - Uses tqdm to visually display the progress.
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
    is_sorted: bool = False,
    n_jobs: int = 4,
):
    """
    Examines the duplication ratio of individual samples within the buffer directory and returns the results.

    This function iterates through the subdirectories of the specified buffer directory,
    checks for duplicate samples using the `check_sample_duplicates` function, and collects
    duplication ratios. It can optionally sort duplication ratios in descending order and tracks
    the maximum sample count among all subdirectories. Additionally, it returns a list of processed
    subdirectory paths.

    Parameters
    ----------
    buffer_dir : str
        Path to the buffer directory containing subdirectories with sample data.
    is_sorted : bool, optional (default=False)
        If True, sorts the duplication ratios in descending order within each subdirectory.
        If False, returns the ratios in their original order.
    n_jobs : int, optional (default=4)
        Number of CPU cores to use for parallel processing during duplication checks.

    Returns
    -------
    tuple
        A tuple containing the following:
        - buffer_paths : List[str]
            A list of all subdirectory paths within the buffer directory.
            Can be used in conjunction with duplication ratios.
        - all_ratios : List[np.ndarray]
            A list of NumPy arrays containing the duplication ratios for each subdirectory.
            If `is_sorted` is True, the ratios are sorted; otherwise, they remain unsorted.
        - num_of_sample : int
            The maximum number of samples across all subdirectories.

    Examples
    --------
    >>> ratios, paths, max_samples = check_each_sample_duplicates("/path/to/buffer", is_sorted=True, n_jobs=2)
    >>> print(ratios[0])  
    >>> print(paths[0])   
    >>> print(max_samples)  
    """

    buffer_paths = [join(buffer_dir, dir_) for dir_ in os.listdir(buffer_dir)]

    all_ratios: List[np.ndarray] = []
    num_of_sample = 0

    for buffer_path in buffer_paths:
        file_list, duplicate_ratios, _ = check_sample_duplicates(buffer_path, n_jobs)
        indices_ = np.argsort(duplicate_ratios)[::-1]
        sorted_duplicates_ratios = (
            duplicate_ratios[indices_] if is_sorted else duplicate_ratios
        )
        all_ratios.append(sorted_duplicates_ratios)
        num_of_sample = max(np.max(len(file_list)).item(), num_of_sample)

    return buffer_paths, all_ratios, num_of_sample


def plot_each_profile(
    buffer_dir: str,
    plot_dir: str,
    e: float = 0.5,
    n_jobs: int = 4,
):
    """
    Visualizes the duplication ratio of each sample within the buffer directory and saves the plots.

    This function calls `check_each_sample_duplicates` to retrieve duplication ratio data, 
    then generates line plots for each subdirectory. It annotates the maximum duplication ratio,
    displays the 95% confidence interval as a shaded region, and marks the threshold `e` as a 
    horizontal line. The number of samples exceeding `e` is included in the legend.
    The resulting plots are saved as PNG files in the specified `plot_dir`.

    Parameters
    ----------
    buffer_dir : str
        Path to the buffer directory containing subdirectories with sample data.
    plot_dir : str
        Path to the directory where the generated plots will be saved.
        If the directory does not exist, it will be created automatically.
    e : float, optional (default=0.5)
        Duplication ratio threshold, marked as a red dashed line in the plot.
        The number of samples exceeding this threshold is also computed.
    n_jobs : int, optional (default=4)
        Number of CPU cores to use for parallel processing when calling `check_each_sample_duplicates`.

    Returns
    -------
    None
        This function does not return any values but saves the plots as files in `plot_dir`.

    Examples
    --------
    >>> plot_each_profile("/path/to/buffer", "/path/to/plots", e=0.7, n_jobs=2)
    """

    buffer_paths, all_ratios, num_of_sample = check_each_sample_duplicates(
        buffer_dir,
        n_jobs=n_jobs,
    )

    for i, buffer_path in tenumerate(buffer_paths, desc="Generating duplication ratio plot"):
        duplicates_ratios = all_ratios[i]

        plt.figure(figsize=(32, 8))
        ax = plt.gca()
        ax.plot(
            range(num_of_sample),
            duplicates_ratios,
            marker="o",  
            linestyle="-",  
            linewidth=2,  
            label=f"Duplicate Ratios - {basename(buffer_path)}",
            alpha=0.6,  
        )
        max_ratio = np.max(duplicates_ratios)
        max_index = np.argmax(duplicates_ratios)
        plt.annotate(
            f"Maximum Duplicate Ratio: {max_ratio:.2f}",
            xy=(max_index, max_ratio),
            xytext=(max_index, max_ratio + 0.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=10,
            color="black",
        )
        mean = np.mean(all_ratios)
        std_err = np.std(all_ratios) / np.sqrt(len(all_ratios))
        margin_of_error = std_err * 1.96  
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        x_range = range(num_of_sample)
        ax.fill_between(
            x_range,
            [lower_bound] * num_of_sample,
            [upper_bound] * num_of_sample,
            color="gray",
            alpha=0.2,
            label="Overall Confidence Interval (95%)",
        )

        count_above_threshold = np.sum(duplicates_ratios > e)
        ax.axhline(
            y=e,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold e = {e} (Count: {count_above_threshold} of {len(duplicates_ratios)})",
        )
        plt.title("File-wise Duplicate Sample Ratios", fontsize=14, pad=20)
        plt.xlabel("File Index", fontsize=12)
        plt.ylabel("Duplicate Ratio", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            borderaxespad=0.0,
            framealpha=0.8,
            labelspacing=0.5,
            ncol=2,  
        )
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        filename = f"{basename(buffer_path)}.png"
        filepath = abspath(join(plot_dir, filename))
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        logger.info(f"Duplication ratio plot has been saved at {filepath}.")


def plot_average(
    buffer_dir: str,
    plot_path: str,
    e: float = 0.5,
    n_jobs: int = 4,
):
    """
    Visualizes the average duplication ratio of all samples within the buffer directory and saves it as a single plot.

    Parameters
    ----------
    buffer_dir : str
        Path to the buffer directory containing sample data.
    plot_path : str
        File path where the generated plot will be saved. Example: "/path/to/average_plot.png"
    e : float, optional (default=0.5)
        Duplication ratio threshold, marked as a red dashed line in the plot.
    n_jobs : int, optional (default=4)
        Number of CPU cores to use for parallel processing.

    Returns
    -------
    None
        Saves the plot as a PNG file at `plot_path`.
    """
    buffer_paths, all_ratios, num_of_sample = check_each_sample_duplicates(
        buffer_dir,
        n_jobs=n_jobs,
    )
    average_ratios = np.mean(all_ratios, axis=0)
    plt.figure(figsize=(10, 6))  
    ax = plt.gca()
    ax.plot(
        range(num_of_sample),
        average_ratios,
        marker="o",  
        linestyle="-",  
        linewidth=2,
        label="Average Duplicate Ratio",
    )
    max_ratio = np.max(average_ratios)  
    max_index = np.argmax(average_ratios)  
    plt.annotate(
        f"Maximum Duplicate Ratio: {max_ratio:.2f}",
        xy=(max_index, max_ratio),
        xytext=(max_index, max_ratio + 0.05),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        fontsize=10,
        color="black",
    )
    ax.axhline(
        y=e,
        color="red",
        linestyle="--",
        label=f"Threshold e = {e}",
    )
    plt.title("Average Duplicate Sample Ratios")
    plt.xlabel("File Index")
    plt.ylabel("Average Duplicate Ratio")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    filepath = abspath(plot_path)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    logger.info(f"Average duplication ratio plot has been saved at {filepath}.")
    plt.close()  


# @hydra.main(
#     version_base=None,
#     config_path=abspath(join(dirname(__file__), "..", "conf")),
#     config_name="train_bert",
# )
def main(
    # config: BertTrainConfig,
) -> None:
    # config = init_config(config)
    # np.random.seed(config.seed)
    # exp_dir = config.exp_dir
    # logger.info(f"jax devices: {jax.devices()}")
    # logger.info(f"running experiment at {exp_dir}")

    # if config.overwrite and os.path.exists(exp_dir):
    #     shutil.rmtree(exp_dir)

    buffer_dir = "./pcgrl_buffer"
    plot_dir = join(dirname(__file__), "plots")

    os.makedirs(plot_dir, exist_ok=True)
    plot_each_profile(
        buffer_dir=buffer_dir,
        plot_dir=plot_dir,
        e=0.5,
        n_jobs=n_jobs,
    )
    plot_average(
        buffer_dir=buffer_dir,
        plot_path=join(plot_dir, "average_duplicate_ratios.png"),
        e=0.5,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    main()
