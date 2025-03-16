from os.path import abspath

import numpy as np
from itertools import combinations
from os import path

def hamming_distance(grid1, grid2):
    """Compute the Hamming Distance between two 2D grid maps using NumPy."""

    return np.sum(grid1 != grid2)

def compute_hamming_distance(grids):
    """
    Compute Diversity based on Hamming Distance for multiple 2D grid maps.
    
    - Generates all index combinations and computes the Hamming Distance.
    - Implemented using NumPy.
    """
    num_grids = grids.shape[0]
    distances = []
    for idx1, idx2 in combinations(range(num_grids), 2):
        dist = hamming_distance(grids[idx1], grids[idx2])
        distances.append(dist)
    avg_distance = np.mean(distances)
    max_possible_distance = grids.shape[1] * grids.shape[2]  # (m × n)
    normalized_diversity = avg_distance / max_possible_distance

    return normalized_diversity


if __name__ == '__main__':
    import pandas as pd

    csv_path = path.abspath(path.join(path.dirname(__file__), "..", "..", "instruct", "test", "bert", "scn-1_se-1_1.csv"))

    instruct_df = pd.read_csv(csv_path)


    eval_dir = path.abspath(path.join(path.dirname(__file__), "..", "..", "saves",
                                      "embed-bert_inst-scn-1_se-1_model-rand_exp-def_s-0",
                                      "eval_embed-bert_inst-scn-1_se-1_1"))

    scores = list()

    for row_i, row in instruct_df.iterrows():
        states = list()
        for seed_i in range(1, 7):
            state = np.load(f"{eval_dir}/reward_{row_i}/seed_{seed_i}/state_0.npy")
            states.append(state)
        states = np.array(states)
        score = compute_hamming_distance(states)
        scores.append(score)

    diversity_df = instruct_df.copy()
    diversity_df = diversity_df.loc[:, ~diversity_df.columns.str.startswith('embed')]
    diversity_df['diversity'] = scores

    print(diversity_df)
