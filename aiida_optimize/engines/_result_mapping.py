# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the datastructures used by optimization engines to keep track of results.
"""

from fsc.export import export


@export
class Result(object):
    """
    Data object for storing the input created by the optimization engine, and the output from the calculation workchain corresponding to that input.
    """
    def __init__(self, input_, output=None):
        self.input = input_
        self.output = output


@export
class ResultMapping(object):
    """
    Maps the keys used to identify calculations to their inputs / outputs.
    """
    def __init__(self):
        self._results = {}

    @property
    def state(self):
        """
        Uniquely defines the state of the object. This can be used to create an identical copy.
        """
        return self._results

    @classmethod
    def from_state(cls, state):
        """
        Create a :class:`ResultMapping` instance from a state.
        """
        instance = cls()
        if state is not None:
            instance._results = state  # pylint: disable=protected-access
        return instance

    def add_inputs(self, inputs_list):
        """
        Adds a list of inputs to the mapping, generating new keys. Returns a dict mapping the keys to the inputs.
        """
        keys = []
        for input_value in inputs_list:
            key = self._get_new_key()
            keys.append(key)
            self._results[key] = Result(input_=input_value)

        return {k: self._results[k].input for k in keys}

    def _get_new_key(self):
        try:
            return max(self._results.keys()) + 1
        except ValueError:
            return 0

    def add_outputs(self, outputs):
        for key, out in outputs.items():
            self._results[key].output = out

    def __getattr__(self, key):
        return getattr(self._results, key)

    def __getitem__(self, key):
        return self._results[key]

    def __len__(self):
        return len(self._results)
