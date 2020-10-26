# -*- coding: utf-8 -*-
"""
Defines common helper functions.
"""
import typing as ty

from aiida import orm
from aiida.common.links import LinkType
from aiida.orm.nodes.data.base import to_aiida_type


def from_aiida_type(value):
    """
    Convert an AiiDA data object to the equivalent Python object
    """
    if isinstance(value, (orm.BaseType)):
        return value.value
    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, orm.List):
        return value.get_list()
    raise TypeError(f'value of type {type(value)} is not supported')


def get_outputs_dict(process: orm.ProcessNode) -> ty.Dict[str, orm.Node]:
    """
    Helper function to mimic the behaviour of the old AiiDA .get_outputs_dict() method.
    """
    return {
        link_triplet.link_label: link_triplet.node
        for link_triplet in process.get_outgoing(link_type=(LinkType.RETURN, LinkType.CREATE))
    }


def get_nested_result(output: ty.Dict[str, orm.Node], key: str) -> ty.Any:
    """
    Helper function to retrieve nested outputs as Python data types
    """
    nesting_kind = None

    if ':' in key:
        node_label, output_label = key.split(':')
        nesting_kind = 'Dict'
    else:
        node_label = key

    node_label = node_label.replace('.', '__')
    node = output[node_label]

    if nesting_kind == 'Dict':
        if not isinstance(node, orm.Dict):
            raise TypeError(f'{node} was expected to be orm.Dict, is {type(node)}')
        keys = output_label.split('.')
        result = node
        for key_part in keys:
            result = result[key_part]
        result = to_aiida_type(result)
    else:
        result = node

    return result
