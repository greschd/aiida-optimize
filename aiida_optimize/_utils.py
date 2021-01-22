# -*- coding: utf-8 -*-
"""
Defines common helper functions.
"""
import typing as ty
from collections import defaultdict

from plumpy.utils import AttributesFrozendict

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
from aiida.common.links import LinkType


def _get_outputs_dict(process: orm.ProcessNode, wrap_nested=False) -> ty.Dict[str, orm.Node]:
    """
    Helper function to mimic the behaviour of the old AiiDA .get_outputs_dict() method.
    """
    res = {
        link_triplet.link_label: link_triplet.node
        for link_triplet in process.get_outgoing(link_type=(LinkType.RETURN, LinkType.CREATE))
    }
    if wrap_nested:
        return _wrap_nested_links(res)  # type: ignore
    return res


def _wrap_nested_links(output_dict):
    """Wrap links containing `__` into nested dicts.
    """
    if not isinstance(output_dict, dict):
        return output_dict
    res = {}
    direct_links = set(link for link in output_dict if '__' not in link)
    nested_links = set(link for link in output_dict if '__' in link)
    nested_links_by_head = defaultdict(list)
    for link in nested_links:
        head, tail = link.split('__', 1)
        nested_links_by_head[head].append((tail, link))

    invalid_links = direct_links.intersection(nested_links_by_head.keys())
    if invalid_links:
        raise ValueError(
            f"Links cannot be both direct and nested, offending links are '{invalid_links}'."
        )

    for link in direct_links:
        res[link] = output_dict[link]

    for link_head, link_list in nested_links_by_head.items():
        res[link_head] = {tail: _wrap_nested_links(output_dict[link]) for tail, link in link_list}
    return res


def _merge_nested_keys(nested_key_inputs, target_inputs):
    """
    Maps nested_key_inputs onto target_inputs with support for nested keys:
        x.y:a.b -> x.y['a']['b']
    Note: keys will be python str; values will be AiiDA data types
    """
    def _get_nested_dict(in_dict, split_path):
        res_dict = in_dict
        for path_part in split_path:
            res_dict = res_dict.setdefault(path_part, {})
        return res_dict

    destination = _copy_nested_dict(target_inputs)

    for key, value in nested_key_inputs.items():
        full_port_path, *full_attr_path = key.split(':')
        *port_path, port_name = full_port_path.split('.')
        namespace = _get_nested_dict(in_dict=destination, split_path=port_path)

        if not full_attr_path:
            if not isinstance(value, orm.Node):
                value = to_aiida_type(value).store()
            res_value = value
        else:
            if len(full_attr_path) != 1:
                raise ValueError(f"Nested key syntax can contain at most one ':'. Got '{key}'")

            # Get or create the top-level dictionary.
            try:
                res_dict = namespace[port_name].get_dict()
            except KeyError:
                res_dict = {}

            *sub_dict_path, attr_name = full_attr_path[0].split('.')
            sub_dict = _get_nested_dict(in_dict=res_dict, split_path=sub_dict_path)
            sub_dict[attr_name] = _from_aiida_type(value)
            res_value = orm.Dict(dict=res_dict).store()

        namespace[port_name] = res_value
    return destination


def _copy_nested_dict(value):
    """
    Copy nested dictionaries. `AttributesFrozendict` is converted into
    a (mutable) plain Python `dict`.

    This is needed because `copy.deepcopy` would create new AiiDA nodes.
    """
    if isinstance(value, (dict, AttributesFrozendict)):
        return {k: _copy_nested_dict(v) for k, v in value.items()}
    return value


def _from_aiida_type(value):
    """
    Convert an AiiDA data object to the equivalent Python object
    """
    if not isinstance(value, orm.Node):
        return value
    if isinstance(value, orm.BaseType):
        return value.value
    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, orm.List):
        return value.get_list()
    raise TypeError(f'value of type {type(value)} is not supported')
