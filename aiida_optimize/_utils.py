# -*- coding: utf-8 -*-
"""
Defines common helper functions.
"""

from aiida.common.links import LinkType


def get_outputs_dict(process):
    """
    Helper function to mimic the behaviour of the old AiiDA .get_outputs_dict() method.
    """
    return {
        link_triplet.link_label: link_triplet.node
        for link_triplet in process.get_outgoing(link_type=(LinkType.RETURN, LinkType.CREATE))
    }
