from fsc.export import export


class Result(object):
    __slots__ = ['input', 'output']

    def __init__(self, input_, output=None):
        self.input = input_
        self.output = output


@export
class ResultMapping(object):
    def __init__(self):
        self._results = {}

    @property
    def state(self):
        return self._results

    @classmethod
    def from_state(cls, state):
        instance = cls()
        if state is not None:
            instance._results = state
        return instance

    def add_inputs(self, inputs_list):
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
