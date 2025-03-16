import numpy as np  

from data.loader.instruction.load_instruction import Instruction
from data.loader.instruction import (
    load_instruction as load_instruction_,
)  


def load_instruction(
    csv_paths: list[str], n_job: int
) -> tuple[Instruction, np.ndarray]:
    """
    Load instruction data from a list of CSV file paths and return the corresponding indices.

    Args:
        csv_paths (list[str]): List of CSV file paths to load.
        n_job (int): Number of jobs to use for parallel processing.

    Returns:
        tuple[Instruction, np.ndarray]:
            - data: Loaded instruction data object.
            - indices: Index array of the loaded data.
    """

    data = load_instruction_(
        csv_paths,  
        n_job,  
    )
    indices = np.arange(data.instructions.shape[0])
    return data, indices
