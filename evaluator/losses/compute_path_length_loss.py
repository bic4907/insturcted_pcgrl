import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from ..aggregator import aggregate_path_length


def compute_path_length_loss(
    env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
):
    """
    Path length loss function.

    Args:
        env_map (chex.Array): The map to be evaluated.
        cond (chex.Array): User-defined intent (path length).
        passable_tiles (chex.Array): Types of tiles that can be ignored.
    """

    path_length = aggregate_path_length(env_map, passable_tiles).astype(float)

    loss = jnp.subtract(path_length, cond)

    return loss
