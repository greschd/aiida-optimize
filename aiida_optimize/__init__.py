"""
A plugin for AiiDA which defines a generic optimization WorkChain.
"""

__version__ = '0.1.0b1'

from . import workchain, engines

__all__ = ['workchain', 'engines']
