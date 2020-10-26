# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the datastructures used by optimization engines to keep track of results.
"""

from __future__ import annotations

import typing as ty

__all__ = ['Result', 'ResultMapping']


class Result:
    """
    Data object for storing the input created by the optimization engine, and the output from the evaluation process corresponding to that input.
    """
    def __init__(self, input_: ty.Any, output: ty.Any = None) -> None:
        self.input = input_
        self.output = output


class ResultMapping:
    """
    Maps the keys used to identify evaluations to their inputs / outputs.
    """
    def __init__(self) -> None:
        self._results: ty.Dict[int, Result] = {}

    @property
    def state(self) -> ty.Dict[int, Result]:
        """
        Uniquely defines the state of the object. This can be used to create an identical copy.
        """
        return self._results

    @classmethod
    def from_state(cls, state: ty.Optional[ty.Dict[int, Result]]) -> ResultMapping:
        """
        Create a :class:`ResultMapping` instance from a state.
        """
        instance = cls()
        if state is not None:
            instance._results = state  # pylint: disable=protected-access
        return instance

    def add_inputs(self, inputs_list: ty.List[ty.Any]) -> ty.Dict[int, Result]:
        """
        Adds a list of inputs to the mapping, generating new keys. Returns a dict mapping the keys to the inputs.
        """
        keys = []
        for input_value in inputs_list:
            for value in input_value.values():
                if not value.is_stored:
                    value.store()
            key = self._get_new_key()
            keys.append(key)
            self._results[key] = Result(input_=input_value)

        return {k: self._results[k].input for k in keys}

    def _get_new_key(self) -> int:
        try:
            return max(self._results.keys()) + 1
        except ValueError:
            return 0

    def add_outputs(self, outputs: ty.Dict[int, ty.Any]) -> None:
        for key, out in outputs.items():
            self._results[key].output = out

    def __getattr__(self, key: str) -> ty.Any:
        return getattr(self._results, key)

    def __getitem__(self, key: int) -> Result:
        return self._results[key]

    def __len__(self) -> int:
        return len(self._results)
