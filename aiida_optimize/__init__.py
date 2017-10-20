"""
A plugin for AiiDA which defines a generic optimization WorkChain.
"""

__version__ = '0.0.0a1'

from . import workchain, engines

__all__ = ['workchain', 'engines']
