import chex
import jax.numpy as jnp
import jax

from functools import partial

from envs.probs.dungeon3 import Dungeon3Tiles
from instruct_rl.dataclass import NormalizationWeights


def evaluate_amount_llm(
    curr_env_map: chex.Array,
    cond: chex.Array,
    weights: chex.Array = NormalizationWeights,
    tile_type: chex.Array = Dungeon3Tiles,
) -> chex.Array:
    """
    Returns the reward value based on the improvement in loss for previous-current steps 
    of bats, scorpions, or spiders.

    Args:
        curr_env_map (chex.Array): Current map configuration.
        cond (chex.Array): Desired number of tiles specified by the user 
                           (e.g., [3] -> "Summon 3 of the specified tile type").
        tile_type (Dungeon3Tiles): Type of tile to be aggregated.
        weight (float, optional): Weight applied to the reward (recommended to use with `partial`, default: 0.05).

    Returns:
        chex.Array: Reward value (a 1D vector containing a single value, e.g., [2]).
    """

    curr_amount = jnp.sum(curr_env_map == tile_type)
    curr_loss = jnp.subtract(curr_amount, cond)
    reward = curr_loss
    reward = reward.astype(float)

    # normalize the amount reward
    reward = jnp.divide(reward, weights)

    return reward

