#!/bin/bash

SEEDS=(1 2 3)
INSTRUCTION_SETS=("scn-1_se-1" "scn-1_se-2" "scn-1_se-3" "scn-1_se-4" "scn-1_se-5")

BUFFER_PATH="./pcgrl_buffer"

for INSTRUCTION in "${INSTRUCTION_SETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Running Offline Data Collection for Instruction Set: ${INSTRUCTION}, Seed: ${SEED}..."
        python ./collect_traj.py overwrite=True n_envs=600 traj_path=$BUFFER_PATH traj_freq=1 seed=$SEED traj_max_envs=30 instruct=$INSTRUCTION

    done
done
