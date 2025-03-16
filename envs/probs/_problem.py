from enum import IntEnum
from functools import partial

import math
import os

from typing import Optional, Tuple, Iterable

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from PIL import Image, ImageDraw, ImageFont

from envs.probs.dungeon3 import Dungeon3Tiles

from .utils import contains_names, randomize_map


class Stats(IntEnum):
    pass


@dataclass
class State:
    stats: Optional[chex.Array] = None
    region_features: Optional[chex.Array] = None
    # FIXME: A bit weird how we handle this, setting as None all over the place in problem classes...
    ctrl_trgs: Optional[chex.Array] = None


@dataclass
class Map:
    env_map: chex.Array
    map_shape: chex.Array
    static_map: chex.Array


@partial(
    jax.jit,
    static_argnums=(
        "tile_enum",
        "map_shape",
        "tile_probs",
        "randomize_map_shape",
        "empty_start",
        "tile_nums",
        "pinpoints",
    ),
)
def generate_init_map(
    rng: chex.Array,
    tile_enum: IntEnum,
    map_shape: Tuple[int, ...],
    tile_probs: chex.Array,
    randomize_map_shape: bool = False,
    empty_start: bool = False,
    tile_nums: Iterable[int] = None,
    pinpoints=False,
):
    """
    Generates a randomized tile map based on the given parameters.

    Args:
        rng (chex.Array): Random number generator state.
        tile_enum (IntEnum): Enumeration representing different tile types.
        map_shape (Tuple[int, ...]): Shape of the generated map.
        tile_probs (chex.Array): Probability distribution for each tile type.
        randomize_map_shape (bool, optional): If True, the map shape is randomized. Defaults to False.
        empty_start (bool, optional): If True, generates an empty map. 
                                      If False, tiles are randomly placed based on specified probabilities. Defaults to False.
        tile_nums (Iterable[int], optional): Set of tile numbers. Defaults to None.
        pinpoints (bool, optional): If True, enables pinpoint placement logic. Defaults to False.

    Raises:
        TypeError: If an argument has an invalid type.
        AttributeError: If a required attribute is missing or incorrectly accessed.
    """

    contains_names(tile_enum, ["EMPTY", "BORDER"])

    tile_probs = jnp.array(tile_probs)
    if empty_start:
        init_map = jnp.full(map_shape, dtype=jnp.int32, fill_value=tile_enum.EMPTY)
    else:
        init_map = jax.random.choice(rng, len(tile_enum), shape=map_shape, p=tile_probs)
    init_map, actual_map_shape = (
        randomize_map(rng, init_map, map_shape, tile_enum)
        if randomize_map_shape
        else (init_map, jnp.array(map_shape))
    )
    if tile_nums is not None and pinpoints:
        invalid_tiles = jnp.array(tile_nums)
        tile_counts = jnp.prod(actual_map_shape)

        def add_amount_of_tiles(
            carry: Tuple[chex.Array, chex.Array],
            tile_indice: int,
        ):
            rng, init_map = carry
            n_tiles_to_add = tile_nums[tile_indice]

            modifable_map = (
                #
                jnp.isin(init_map, invalid_tiles) & init_map != tile_enum.BORDER
            ).ravel()
            probs = modifable_map / jnp.sum(modifable_map)

        tile_indices = jnp.arange(len(tile_enum))
        rng, init_map = jax.lax.reduce(
            tile_indices,
        )

    return Map(
        init_map,
        actual_map_shape,
        init_map == tile_enum.BORDER,
    )
