# -*- coding: utf-8 -*-
"""Defines helper functions for using `aiida-optimize`.
"""

import typing as ty

from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type

__all__ = ("get_nested_result", )


def get_nested_result(output: ty.Dict[str, orm.Node], key: str) -> orm.Node:
    """Helper function to retrieve nested outputs from AiiDA processes.

    This function supports the nested key syntax:

    - Namespaces are separated by a period
    - A colon ``:`` indicates accessing inside an AiiDA ``Dict``
    - Nested access inside the `Dict` is again separated by a period

    Examples:

    - ``'x.y'``: retrieve output ``y`` in the ``x`` namespace
    - ``'x.y:a.b'``: ``x.y`` is a ``Dict``, and we retrieve its content
      at ``['a']['b']``.

    Parameters
    ----------
    output :
        The outputs of a process, given as a dictionary mapping output
        labels to values.
    key :
        The key for which the output should be retrieved.

    Returns
    -------
    orm.Node :
        The desired result, as an AiiDA node.
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
        # TODO: do we need to `.store()` here?
        result = to_aiida_type(result)
    else:
        result = node

    return result
