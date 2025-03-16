import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible
from instruct_rl.dataclass import NormalizationWeights

from ..losses import compute_path_length_loss, compute_region_loss


def evaluate_path_length_llm(
    curr_env_map: chex.Array,
    cond: chex.Array,
    weights: chex.Array = NormalizationWeights,
    passable_tiles: chex.Array = Dungeon3Passible,
):
    """
    Path length evaluation function.

    Args:
        prev_env_map (chex.Array): Previous map state.
        curr_env_map (chex.Array): Current map state.
        cond (chex.Array): User-specified intended value (path length).
        passable_tiles (chex.Array): Types of tiles that can be ignored.

    Returns:
        chex.Array: Path value based on path length.
    """

    curr_loss = compute_path_length_loss(curr_env_map, cond, passable_tiles)

    curr_r_loss = compute_region_loss(curr_env_map, 1, passable_tiles)
    reward = curr_loss
    reward += (curr_r_loss) * 0.5
    reward = reward.astype(float)

    # normalize the path length reward
    reward = jnp.divide(reward, weights)

    return reward