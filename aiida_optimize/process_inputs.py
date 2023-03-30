# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Contains default keyword arguments to pass classes as input to workchains.
"""

from functools import singledispatch

from aiida.engine.persistence import ObjectLoader
from aiida.orm import Str
import yaml

__all__ = ["PROCESS_INPUT_KWARGS", "get_fullname", "load_object"]

_YAML_IDENTIFIER = "!!YAML!!"


@singledispatch
def get_fullname(cls_obj):
    """
    Serializes an AiiDA process class / function to an AiiDA String.

    :param cls_obj: Object to be serialized
    :type cls_obj: Process
    """
    try:
        return Str(ObjectLoader().identify_object(cls_obj))
    except ValueError:
        return Str(_YAML_IDENTIFIER + yaml.dump(cls_obj))


@get_fullname.register(str)
def _(cls_name):
    return Str(cls_name)


#: Keyword arguments to be passed to ``spec.input`` for serializing an input which is a class / process into a string.
PROCESS_INPUT_KWARGS = {
    "valid_type": Str,
    "serializer": get_fullname,
}


def load_object(cls_name):
    """
    Loads the process from the serialized string.
    """
    if isinstance(cls_name, Str):
        cls_name_str = cls_name.value
    else:
        cls_name_str = str(cls_name)
    try:
        return ObjectLoader().load_object(cls_name_str)
    except ValueError as err:
        if cls_name_str.startswith(_YAML_IDENTIFIER):
            return yaml.load(cls_name_str[len(_YAML_IDENTIFIER) :], Loader=yaml.UnsafeLoader)
        raise ValueError(f"Could not load class name '{cls_name_str}'.") from err
