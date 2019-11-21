# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
A plugin for AiiDA which defines a generic optimization WorkChain.
"""

__version__ = '0.3.0'

from . import engines, workchain

__all__ = ['workchain', 'engines']
