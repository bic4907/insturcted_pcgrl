import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from ..aggregator import aggregate_region


def compute_region_loss(
    env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
) -> chex.Array:
    """
    Loss function for the 'region' metric.

    Counts and returns the number of rooms or empty spaces in the map.

    Args:
        env_map (chex.Array): The map to be evaluated.
        cond (chex.Array): User-defined intent (number of independent spaces).
        passable_tiles (chex.Array): Types of tiles that can be ignored.

    Returns:
        chex.Array: Loss value for the region metric.
    """

    n_regions = aggregate_region(env_map, passable_tiles).astype(float)

    loss = jnp.subtract(n_regions, cond)

    return loss
