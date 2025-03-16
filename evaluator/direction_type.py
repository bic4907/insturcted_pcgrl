from enum import IntEnum


class Direction(IntEnum):
    """
    Direction definition type (Enum):

    Defines four cardinal directions: 
    - West as 0, 
    - North as 1, 
    - East as 2, 
    - South as 3.

    It can be conveniently understood as assigning 0, 1, 2, and 3 in a clockwise order starting from the west (left).
    """


    west = 0
    north = 1
    east = 2
    south = 3
