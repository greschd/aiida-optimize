from aiida.orm import DataFactory
from aiida.orm.data.base import Float
from aiida.work.workchain import WorkChain, ToContext, while_
from aiida.work.run import submit

class Bisection:
    def __init__(self, lower, upper, tol=1e-6):
        self.lower, self.upper = sorted([lower, upper])
        self.tol = tol

    @classmethod
    def from_state(cls, state):
        return cls(**state.get_dict())

    @property
    def state(self):
        return DataFactory('parameter')(dict=self.__dict__)

    @property
    def is_finished(self):
        return abs(self.upper - self.lower) < self.tol

    @property
    def average(self):
        return (self.upper + self.lower) / 2.

    def create_inputs(self):
        return {'x': Float(self.average)}

    def update(self, results):
        assert len(results) == 1
        res = results[0]
        if res > 0:
            self.upper = self.average
        else:
            self.lower = self.average

    @property
    def result(self):
        return Float(self.average)

class TestWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(cls, TestWorkChain).define(spec)

        spec.outline(
            cls.create_optimizer,
            while_(cls.not_finished)(
                cls.step
            ),
            cls.finalize
        )
        spec.output('result', valid_type=Float)

    def create_optimizer(self):
        optimizer = Bisection(lower=-1., upper=1., tol=1e-2)
        self.ctx.optimizer_state = optimizer.state

    def not_finished(self):
        return not self.optimizer.is_finished

    @property
    def optimizer(self):
        return Bisection.from_state(self.ctx.optimizer_state)

    def step(self):
        optimizer = self.optimizer
        if hasattr(self.ctx, 'step'):
            optimizer.update([self.ctx.step.out.result])
        step = submit(CalculateWorkChain, **optimizer.create_inputs())
        self.ctx.optimizer_state = optimizer.state
        return ToContext(step=step)

    def finalize(self):
        self.out('result', self.optimizer.result)

class CalculateWorkChain(WorkChain):
    @classmethod
    def define(cls, spec):
        super(cls, CalculateWorkChain).define(spec)

        spec.input('x', valid_type=Float)
        spec.output('result', valid_type=Float)
        spec.outline(cls.echo)

    def echo(self):
        self.out('result', self.inputs.x)

if __name__ == '__main__':
    TestWorkChain.run()
