# -*- coding: utf-8 -*-
"""
Defines common helper functions.
"""

import typing as ty

from aiida import orm
from aiida.common.links import LinkType


def get_outputs_dict(process: orm.ProcessNode) -> ty.Dict[str, orm.Node]:
    """
    Helper function to mimic the behaviour of the old AiiDA .get_outputs_dict() method.
    """
    return {
        link_triplet.link_label: link_triplet.node
        for link_triplet in process.get_outgoing(link_type=(LinkType.RETURN, LinkType.CREATE))
    }
