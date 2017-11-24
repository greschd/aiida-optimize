"""
Configuration file for pytest tests of aiida-optimize.
"""

import os
os.environ['PYTHONPATH'] = os.environ.get(
    'PYTHONPATH', ''
) + ':' + os.path.dirname(os.path.abspath(__file__))

from aiida_pytest import *  # pylint: disable=unused-wildcard-import
