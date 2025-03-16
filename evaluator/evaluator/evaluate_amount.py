import chex
import jax.numpy as jnp
import jax

from functools import partial

from envs.probs.dungeon3 import Dungeon3Tiles


@partial(jax.jit, static_argnames=("weight",))
def evaluate_amount(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_type: Dungeon3Tiles,
    weight: float = 1.5,
) -> chex.Array:
    """
    Returns the reward value based on the improvement in loss for previous-current steps 
    of bats, scorpions, or spiders.

    Args:
        prev_env_map (chex.Array): Previous map configuration.
        curr_env_map (chex.Array): Current map configuration.
        cond (chex.Array): Desired number of tiles specified by the user 
                           (e.g., [3] -> "Summon 3 of the specified tile type").
        tile_type (Dungeon3Tiles): Type of tile to be aggregated.
        weight (float, optional): Weight applied to the reward (recommended to use with `partial`, default: 0.05).

    Returns:
        chex.Array: Reward value (a 1D vector containing a single value, e.g., [2]).
    """

    prev_amount = jnp.sum(prev_env_map == tile_type)
    curr_amount = jnp.sum(curr_env_map == tile_type)
    prev_loss = jnp.abs(jnp.subtract(prev_amount, cond))
    curr_loss = jnp.abs(jnp.subtract(curr_amount, cond))
    reward = prev_loss - curr_loss
    reward = reward.astype(float)
    reward = reward * weight

    return reward
