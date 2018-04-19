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
