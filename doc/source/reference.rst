.. © 2017-2019, ETH Zurich, Institut für Theoretische Physik
.. Author: Dominik Gresch <greschd@gmx.ch>

Reference
=========

.. _workchain_reference:

Optimization workchain
----------------------

.. aiida-workchain:: OptimizationWorkChain
    :module: aiida_optimize

.. _engine_reference:

Engines
-------

.. automodule:: aiida_optimize.engines
    :members:

Engine base classes
'''''''''''''''''''

.. automodule:: aiida_optimize.engines.base
    :members:
    :private-members:

Wrapper workchains
------------------

.. aiida-workchain:: AddInputsWorkChain
    :module: aiida_optimize.wrappers

.. aiida-workchain:: CreateEvaluateWorkChain
    :module: aiida_optimize.wrappers

Helper functions
----------------

.. automodule:: aiida_optimize.helpers
    :members:

Internals
---------

This section describes internal modules. Their interface may change without warning.


Result mapping
''''''''''''''

.. automodule:: aiida_optimize.engines._result_mapping
    :members:


Internal utilities
''''''''''''''''''

.. automodule:: aiida_optimize._utils
    :members:
