import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles

from ..aggregator import aggregate_amount


def compute_amount_loss(
    env_map: chex.Array,
    tile_type: Dungeon3Tiles,
    cond: chex.Array,
) -> chex.Array:
    """
    Counts and returns the number of bats, scorpions, and spiders in the map.

    Args:
        env_map (chex.Array): Current map state.
        tile_type (Dungeon3Tiles): Type of tile to be counted.
        cond (chex.Array): User-defined intent (number of tiles).

    Returns:
        chex.Array: A single-value vector containing the entity count (e.g., [1]).
    """


    loss = jnp.abs(jnp.subtract(aggregate_amount(env_map, tile_type), cond)).astype(
        float
    )

    return loss
