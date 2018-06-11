Tutorial
========

This tutorial will take you through a basic optimization using ``aiida-optimize``. It assumes that you are already familiar with using AiiDA.

Motivation
----------

First of all, why do we need a special optimization framework for AiiDA workflows? Couldn't we use an existing library like ``scipy.optimize`` to do that?

Imagine you have a complex function that you want to optimize. Evaluating that function involves several steps, and calls to different codes. As such, it's a perfect fit to be implemented in an AiiDA workflow. Now you could just pass this to an optimization function like ``scipy.optimize`` by executing the workflow with the ``run`` method. However, this creates a problem: Because ``run`` is a blocking call, the Python interpreter which executes this function needs to stay alive during the entire time that the optimization is running. If there's a problem anywhere in the process, the results are essentially lost.

It would be much nicer then to create the optimization process in such a way that it can be shut down at any point. In essence, we want to create a _new_ AiiDA workflow that simply wraps the one evaluating the function. As a consequence, the optimization logic cannot be written in the usual, procedural way. Instead, it needs to be encoded in a stateful "optimization engine" that can be stopped, persisted and restarted. Because doing this involves a lot of boilerplate code, ``aiida-optimize`` takes away some of that complexity and provides some built-in optimization engines.

A simple bisection
------------------

Now, we will see how to perform an optimization with ``aiida-optimize``. First, we need an AiiDA WorkChain or workfunction to optimize. As a simple example, we create a workfunction that evaluates the sine:

.. include:: ../../examples/sin_wf.py
    :code: python

Equivalently, we could also write a workchain that does the same:

.. include:: ../../examples/sin_wc.py
    :code: python

Now we can use ``aiida-optimize`` with the :class:`.Bisection` engine to find a nodal point. To do this, we run the :class:`.OptimizationWorkChain`, with the following inputs:

* ``engine`` is the optimization engine that we use. In this case, we pass the :class:`.Bisection` class.
* ``engine_kwargs`` are parameters that will be passed to the optimization engine. In the case of bisection, we pass the upper and lower boundaries of the bisection interval, and the target tolerance. Also, we need to pass the ``result_key``, which is the name of the output argument of the workfunction or workchain that we are optimizing. For workfunctions, this is always ``result``.
* ``calculation_workchain`` is the workchain function that we want to optimize. In our case, that's the ``sin`` workfunction or ``Sin`` workchain.

.. include:: ../../examples/bisection.py
    :code: python

The :class:`.OptimizationWorkChain` returns two outputs: The optimized value of the function, and the uuid of the optimal function workchain. This can be used to retrieve the exact inputs and outputs of the best run of the evaluated function.

The other optimization engines which are included in ``aiida-optimize`` are described in the `reference section <engine_reference>`_.

Developing an optimization engine
---------------------------------

In this section, we give a rough description of how the optimization engines itself are structured. If you wish to develop your own optimization engine, we also highly recommend looking at the code of the existing engines for inspiration.

The optimization engines are usually split into two parts: The implementation, and a small wrapper class. These classes have corresponding base classes, :class:`.OptimizationEngineImpl` and :class:`.OptimizationEngineWrapper`. While the implementation contains the logic of the optimization engine itself, the wrapper is a factory class which is exposed to the user, used only to instantiate an instance of the implementation.

The reason for this split is that the engine itself needs to be serializable into a "state" which can be stored between steps of the AiiDA workchain, and then re-created from that state. Since the state usually contains more parameters than what needs to be exposed when the engine is first instantiated, the wrapper is added to hide away these parameters from the end user.

The :class:`.OptimizationEngineImpl` describes the methods which need to be implemented by an optimization engine. In particular, methods for creating new inputs, updating the engine from calculation outputs, and serializing it to its state need to be provided. The base class itself keeps track of which calculations have been launched. This is done using the :class:`.ResultMapping` class, which contains a dictionary that maps a key to a :class:`.Result` containing the calculations inputs and outputs. The :class:`.OptimizationWorkChain` uses these same keys to identify the corresponding processes.
