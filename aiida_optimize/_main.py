from contextlib import contextmanager

from fsc.export import export

from aiida.orm import DataFactory
from aiida.orm.data.base import Float
from aiida.work.workchain import WorkChain, ToContext, while_
from aiida.work.run import submit

from ._result_mapping import ResultMapping


class Bisection(object):
    def __init__(self, lower, upper, tol=1e-6, result_state=None):
        self._result_mapping = ResultMapping.from_state(result_state)
        self.lower, self.upper = sorted([lower, upper])
        self.tol = tol

    @classmethod
    def from_state(cls, state):
        return cls(**state.get_dict())

    @property
    def state(self):
        return DataFactory('parameter')(
            dict=dict(
                result_state=self._result_mapping.state,
                **{
                    k: v
                    for k, v in self.__dict__.items()
                    if k not in ['_result_mapping']
                }
            )
        )

    @property
    def is_finished(self):
        return abs(self.upper - self.lower) < self.tol

    @property
    def average(self):
        return (self.upper + self.lower) / 2.

    def create_inputs(self):
        return self._result_mapping.add_inputs(self._create_inputs())

    def _create_inputs(self):
        return [{'x': Float(self.average)}]

    def update(self, outputs):
        self._result_mapping.add_outputs(outputs)
        res = outputs.values()[0]['result']
        print(res)
        if res > 0:
            self.upper = self.average
        else:
            self.lower = self.average

    @property
    def result(self):
        return Float(self.average)


@export
class TestWorkChain(WorkChain):
    _CALC_PREFIX = 'calc_'

    @classmethod
    def define(cls, spec):
        super(cls, TestWorkChain).define(spec)

        spec.outline(
            cls.create_optimizer,
            while_(cls.not_finished)(cls.launch_calcs, cls.get_results),
            cls.finalize
        )
        spec.output('result', valid_type=Float)

    @contextmanager
    def optimizer(self):
        optimizer = Bisection.from_state(self.ctx.optimizer_state)
        yield optimizer
        self.ctx.optimizer_state = optimizer.state

    def create_optimizer(self):
        optimizer = Bisection(lower=-1., upper=1., tol=1e-3)
        self.ctx.optimizer_state = optimizer.state

    def not_finished(self):
        with self.optimizer() as opt:
            return not opt.is_finished

    def launch_calcs(self):
        self.report('Launching pending calculations.')
        with self.optimizer() as opt:
            calcs = {}
            for idx, inputs in opt.create_inputs().items():
                calcs[self._CALC_PREFIX
                      + str(idx)] = submit(CalculateWorkChain, **inputs)
            self.report('Launching calculation {}'.format(idx))
        return ToContext(**calcs)

    def get_results(self):
        self.report('Checking finished calculations.')
        with self.optimizer() as opt:
            context_keys = dir(self.ctx)
            step_keys = [
                key for key in context_keys
                if key.startswith(self._CALC_PREFIX)
            ]
            outputs = {}
            for key in step_keys:
                idx = int(key.split(self._CALC_PREFIX)[1])
                self.report('Retrieving output for calculation {}'.format(idx))
                outputs[idx] = self.ctx[key].get_outputs_dict()
                delattr(self.ctx, key)
            opt.update(outputs)

    def finalize(self):
        self.report('Finalizing optimization procedure.')
        with self.optimizer() as opt:
            self.out('result', opt.result)


@export
class CalculateWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(cls, CalculateWorkChain).define(spec)

        spec.input('x', valid_type=Float)
        spec.output('result', valid_type=Float)
        spec.outline(cls.echo)

    def echo(self):
        self.report('Starting echo')
        self.out('result', self.inputs.x)
