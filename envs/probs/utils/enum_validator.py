from enum import Enum
from typing import Union, Iterable


def validate_enum_class(enum_class: Enum) -> None:
    """
    Validates whether the given class is of Enum type.

    Args:
        enum_class (Enum): The class to validate.

    Raises:
        TypeError: Raised if the given class is not an Enum.
    """

    if not isinstance(enum_class, type) or not issubclass(enum_class, Enum):
        raise TypeError(f"{enum_class} is not a valid Enum class.")


def contains_name(
    enum_class: Enum,
    name: Union[str, int],
) -> bool:
    """
    Checks whether a given name is present in the specified Enum class.

    Args:
        enum_class (Enum): The Enum class to validate.
        name (Union[str, int]): The string or integer to check for inclusion.

    Returns:
        bool: True if the name is present, otherwise False.

    Raises:
        TypeError: Raised if `enum_class` is not a valid Enum class.
        AttributeError: Raised if the given name does not exist in the Enum class.
    """

    validate_enum_class(enum_class)

    if any(name in member.name for member in enum_class):
        return True
    else:
        raise AttributeError(
            f"Attribute Error: The item '{name}' does not exist in the {enum_class.__name__} Enum. Please check the valid entries."

        )


def contains_names(
    enum_class: Enum,
    names: Iterable[Union[str, int]],
) -> bool:
    """
    Checks whether multiple names are present in the specified Enum class.

    Args:
        enum_class (Enum): The Enum class to validate.
        names (Iterable[Union[str, int]]): An iterable of strings or integers to check for inclusion.

    Returns:
        bool: True if all names are present, otherwise False.

    Raises:
        TypeError: Raised if `enum_class` is not a valid Enum class.
        AttributeError: Raised if at least one of the given names does not exist in the Enum class.
    """

    validate_enum_class(enum_class)

    for name in names:
        if not any(name in member.name for member in enum_class):
            raise AttributeError(
                f"Attribute Error: The item '{name}' does not exist in the {enum_class.__name__} Enum. Please check the valid entries."

            )
    return True


def contains_one_of(
    enum_class: Enum,
    names: Iterable[Union[str, int]],
) -> bool:
    """
    Checks whether at least one of the given names is present in the specified Enum class.

    Args:
        enum_class (Enum): The Enum class to validate.
        names (Iterable[Union[str, int]]): An iterable of strings or integers to check for inclusion.

    Returns:
        bool: True if at least one name is present, otherwise False.

    Raises:
        TypeError: Raised if `enum_class` is not a valid Enum class.
        AttributeError: Raised if at least one of the given names does not exist in the Enum class.
    """

    validate_enum_class(enum_class)

    for name in names:
        if any(name in member.name for member in enum_class):
            return True

    raise AttributeError(
        f"Attribute Error: None of the given names exist in the {enum_class.__name__} Enum. Please check the valid entries."
    )
