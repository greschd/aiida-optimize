# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the WorkChain which runs the optimization procedure.
"""

from contextlib import contextmanager

from aiida import orm
from aiida.engine import WorkChain, while_
from aiida.engine.launch import run_get_node
from aiida.engine.utils import is_process_function

from ._utils import _get_outputs_dict, _merge_nested_keys
from .process_inputs import PROCESS_INPUT_KWARGS, load_object

__all__ = ["OptimizationWorkChain"]


class OptimizationWorkChain(WorkChain):
    """
    Runs an optimization procedure, given an optimization engine that defines the optimization
    algorithm, and a process which evaluates the function to be optimized.
    """

    _EVAL_PREFIX = "eval_"

    @classmethod
    def define(cls, spec):
        super(cls, OptimizationWorkChain).define(spec)

        spec.input("engine", help="Engine that runs the optimization.", **PROCESS_INPUT_KWARGS)
        spec.input(
            "engine_kwargs",
            valid_type=orm.Dict,
            help="Keyword arguments passed to the optimization engine.",
        )
        spec.input(
            "evaluate_process",
            help="Process which produces the result to be optimized.",
            **PROCESS_INPUT_KWARGS,
        )
        spec.input_namespace(
            "evaluate",
            required=False,
            help="Inputs that are passed to all evaluation processes.",
            dynamic=True,
        )

        spec.exit_code(
            201,
            "ERROR_EVALUATE_PROCESS_FAILED",
            message="Optimization failed because one of the evaluate processes did not finish ok.",
        )
        spec.exit_code(
            202,
            "ERROR_ENGINE_FAILED",
            message="Optimization failed because the engine did not finish ok.",
        )

        spec.outline(
            cls.create_optimizer,
            while_(cls.not_finished)(cls.launch_evaluations, cls.get_results),
            cls.finalize,
        )
        spec.output(
            "optimal_process_input",
            help="Input value of the optimal evaluation process.",
            required=False,
        )
        spec.output(
            "optimal_process_output", help="Output value of the optimal evaluation process."
        )
        spec.output("optimal_process_uuid", help="UUID of the optimal evaluation process.")
        spec.output_namespace("engine_outputs", required=False, dynamic=True)

    @contextmanager
    def optimizer(self):
        optimizer = self.engine.from_state(state=self.ctx.optimizer_state, logger=self)
        yield optimizer
        self.ctx.optimizer_state = optimizer.state

    @property
    def engine(self):
        return load_object(self.inputs.engine.value)

    @property
    def indices_to_retrieve(self):
        return self.ctx.setdefault("indices_to_retrieve", [])

    @indices_to_retrieve.setter
    def indices_to_retrieve(self, value):
        self.ctx.indices_to_retrieve = value

    def create_optimizer(self):  # pylint: disable=missing-docstring
        self.report("Creating optimizer instance.")
        optimizer = self.engine(  # pylint: disable=not-callable
            logger=self, **self.inputs.engine_kwargs.get_dict()
        )
        self.ctx.optimizer_state = optimizer.state

    def not_finished(self):
        """
        Check if the optimization needs to continue.
        """
        self.report("Checking if optimization is finished.")
        with self.optimizer() as opt:
            return not opt.is_finished

    def launch_evaluations(self):
        """
        Create evaluations for the current iteration step.
        """
        self.report("Launching pending evaluations.")
        with self.optimizer() as opt:
            evals = {}
            evaluate_process = load_object(self.inputs.evaluate_process.value)
            for idx, inputs in opt.create_inputs().items():
                self.report(f"Launching evaluation {idx}")
                inputs_merged = _merge_nested_keys(inputs, self.inputs.get("evaluate", {}))
                if is_process_function(evaluate_process):
                    _, node = run_get_node(evaluate_process, **inputs_merged)
                else:
                    node = self.submit(evaluate_process, **inputs_merged)
                evals[self.eval_key(idx)] = node
                self.indices_to_retrieve.append(idx)
        return self.to_context(**evals)

    def get_results(self):  # pylint: disable=inconsistent-return-statements
        """
        Retrieve results of the current iteration step's evaluations.
        """
        self.report("Checking finished evaluations.")
        outputs = {}
        while self.indices_to_retrieve:
            idx = self.indices_to_retrieve.pop(0)
            key = self.eval_key(idx)
            self.report(f"Retrieving output for evaluation {idx}")
            eval_proc = self.ctx[key]
            if not eval_proc.is_finished_ok:
                return self.exit_codes.ERROR_EVALUATE_PROCESS_FAILED
            outputs[idx] = _get_outputs_dict(eval_proc)

        with self.optimizer() as opt:
            opt.update(outputs)

    def finalize(self):  # pylint: disable=inconsistent-return-statements
        """
        Return the output after the optimization procedure has finished.
        """
        self.report("Finalizing optimization procedure.")
        with self.optimizer() as opt:
            if hasattr(opt, "get_engine_outputs"):
                self.out("engine_outputs", opt.get_engine_outputs())
            if not opt.is_finished_ok:
                return self.exit_codes.ERROR_ENGINE_FAILED
            optimal_process_input = opt.result_input_value
            if optimal_process_input is not None:
                assert optimal_process_input.is_stored
                self.out("optimal_process_input", optimal_process_input)
            optimal_process_output = opt.result_output_value
            optimal_process_output.store()
            self.out("optimal_process_output", optimal_process_output)
            result_index = opt.result_index
            optimal_process = self.ctx[self.eval_key(result_index)]
            self.out("optimal_process_uuid", orm.Str(optimal_process.uuid).store())

    def eval_key(self, index):
        """
        Returns the evaluation key corresponding to a given index.
        """
        return self._EVAL_PREFIX + str(index)
