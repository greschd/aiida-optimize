"""
Defines a simple calculation workchain which just returns the input.
"""

from aiida.orm.data.base import Float
from aiida.work.workchain import WorkChain

from aiida_tools import check_workchain_step


class Echo(WorkChain):  # pylint: disable=abstract-method
    """
    WorkChain which returns the input.
    """

    @classmethod
    def define(cls, spec):
        super(Echo, cls).define(spec)

        spec.input('x', valid_type=Float)
        spec.output('result', valid_type=Float)
        spec.outline(cls.echo)

    @check_workchain_step
    def echo(self):
        self.report('Starting echo')
        self.out('result', self.inputs.x)
