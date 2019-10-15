# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
A plugin for AiiDA which defines a generic optimization WorkChain.
"""

__version__ = '0.2.0'

from . import workchain, engines

__all__ = ['workchain', 'engines']
