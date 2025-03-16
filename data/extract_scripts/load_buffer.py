from data.loader.buffer import load_memoized_buffer_data
from data.loader.buffer.utils_ import pairing_maps, extract_unique_pair_indices


def load_buffer(
    source_dir: str,  
    cache_path: str,  
    total_files: int,  
    n_jobs: int = 4,  
    ignore_file: bool = False,  
):
    """
    Load data from an external source and store it in the processing buffer.

    This is an impure function with side effects, as it reads external data
    and can modify internal states.

    Args:
        source_dir (str): Directory path containing the source data.
        cache_path (str): File path to cache the processed data.
        total_files (int): Total number of files to process.
        n_jobs (int, optional): Number of parallel workers to use (default: 4).
        ignore_file (bool, optional): Whether to ignore existing cache files (default: False).

    Returns:
        tuple: (Processed data object, selected pair indices)
    """

    buffer_data = load_memoized_buffer_data(
        source_dir,
        cache_path,
        total_files,
        n_jobs,
        ignore_file,
    )
    environment_pairs = pairing_maps(
        buffer_data.env_map[buffer_data.prev_indices],
        buffer_data.env_map[buffer_data.curr_indices],
    )
    index_pairs = pairing_maps(
        buffer_data.prev_indices[..., None],
        buffer_data.curr_indices[..., None],
    )
    selected_pair_indices = extract_unique_pair_indices(environment_pairs)
    return buffer_data, index_pairs[selected_pair_indices]
