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

This section describes internal classes. They are useful for developing custom optimization engines.

Engine base classes
'''''''''''''''''''

.. automodule:: aiida_optimize.engines._base
    :members:


Result mapping
''''''''''''''

.. automodule:: aiida_optimize.engines._result_mapping
    :members:
