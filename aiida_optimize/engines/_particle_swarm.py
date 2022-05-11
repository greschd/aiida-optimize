# -*- coding: utf-8 -*-

# pylint: disable=invalid-name
# Author: Emanuele Bosoni <bosoe4@gmail.com>
"""
Defines a Particle-Swarm optimization engine.
"""

import typing as ty
from numpy.random import uniform, get_state, set_state
import numpy as np
from decorator import decorator
from copy import deepcopy
from aiida import orm

from ..helpers import get_nested_result
from .base import OptimizationEngineImpl, OptimizationEngineWrapper

__all__ = ["ParticleSwarm"]

# Parameters choosen according to http://dx.doi.org/10.1109/CEC.2003.1299391
C1 = 1.49445
C2 = 1.49445
OMEGA = 0.5


def update_method(next_submit=None):
    """
    Decorator for methods which update the results.
    """
    @decorator
    def inner(func, self, outputs):
        self.next_submit = next_submit
        self.next_update = None
        func(self, outputs)

    return inner


def submit_method(next_update=None):
    """
    Decorator for methods which submit new evaluations.
    """
    @decorator
    def inner(func, self):
        self.next_submit = None
        self.next_update = next_update
        return func(self)

    return inner


class _ParticleSwarmImpl(OptimizationEngineImpl):
    """
    Implementation class for the Particle-Swarm optimization engine.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        particles: ty.List[float],  # ty.Optional[ty.List[float]],
        max_iter: int,
        input_key: str,
        result_key: str,
        logger,
        num_iter=0,
        next_submit="submit_initialize",
        next_update=None,
        finished=False,
        exceeded_max_iters=False,
        result_state=None,
        global_best=None,
        fun_global_best=None,
        local_best=None,
        fun_local_best=None,
        velocities=None,
        rand_state=None,
    ):
        super().__init__(logger=logger, result_state=result_state)

        self.particles = np.array(particles)
        n_vars = len(self.particles[0])
        assert all(len(part) == n_vars for part in self.particles)
        self.velocities = velocities

        self.global_best = global_best
        self.fun_global_best = fun_global_best
        self.local_best = local_best
        self.fun_local_best = fun_local_best

        self.max_iter = max_iter
        self.num_iter = num_iter

        self.input_key = input_key
        self.result_key = result_key

        self.next_submit = next_submit
        self.next_update = next_update

        self.finished = finished
        self.exceeded_max_iters = exceeded_max_iters

        self.rand_state = rand_state

    @submit_method(next_update="update_general")
    def submit_initialize(self):
        n_parts = len(self.particles)
        n_vars = len(self.particles[0])
        self.local_best = self.particles
        self.fun_local_best = np.full(n_parts, np.inf)
        self.fun_global_best = np.inf
        # Initialize the velocities to random number in [-1,1]
        self.velocities = np.zeros((n_parts, n_vars))
        for line, _ in enumerate(self.velocities):
            for col, _ in enumerate(self.velocities[line]):
                self.velocities[line][col] = uniform(-1, 1)
        self.rand_state = get_state()
        self._logger.report("Submitting first step.")
        return [self._to_input_list(x) for x in self.particles]

    def _to_input_list(self, x):
        input_list = orm.List()
        input_list.extend(x)
        return {self.input_key: input_list}

    @update_method(next_submit="new_iter")
    def update_general(self, outputs):  # pylint: disable=missing-function-docstring
        fun_particles = np.array(self._get_values(outputs))
        for index, val in enumerate(fun_particles):
            if val < self.fun_local_best[index]:
                self.fun_local_best[index] = val
                self.local_best[index] = self.particles[index]
        for index, val in enumerate(self.fun_local_best):
            if val < self.fun_global_best:
                self.fun_global_best = val
                self.global_best = self.local_best[index]

    def _get_values(self, outputs):
        return [get_nested_result(res, self.result_key).value for _, res in sorted(outputs.items())]

    def create_particle(self):  # pylint: disable=missing-function-docstring
        n_var = len(self.particles[0])
        new_vel = deepcopy(self.velocities)
        for idx, val in enumerate(self.particles):
            new_vel[idx] = [
                self.update_vel(
                    OMEGA,
                    self.velocities[idx][i],
                    C1,
                    C2,
                    val[i],
                    self.local_best[idx][i],
                    self.global_best[i],
                ) for i in range(n_var)
            ]

        new_parts = deepcopy(self.particles)
        for fd, _ in enumerate(self.particles):
            new_parts[fd] = np.array(new_vel[fd]) + self.particles[fd]

        return np.array(new_parts), np.array(new_vel)

    @staticmethod
    def update_vel(omega, v, c1, c2, x, pi, pg):  # pylint: disable=too-many-arguments
        return omega * v + c1 * uniform(0, 1) * (pi - x) + c2 * uniform(0, 1) * (pg - x)

    @submit_method()
    def new_iter(self):  # pylint: disable=missing-function-docstring
        self.check_finished()
        if self.finished:
            self.next_update = "finalize"
            return []
        self.num_iter += 1
        self._logger.report(
            f"Start of Particle-Swarm iteration {self.num_iter}, max number of iterations: {self.max_iter}."
        )
        self.next_update = "update_general"
        set_state(self.rand_state)
        self.particles, self.velocities = self.create_particle()
        self.rand_state = get_state()

        return [self._to_input_list(x) for x in self.particles]

    @update_method()
    def finalize(self, outputs):
        pass

    def check_finished(self):
        """
        Updates the 'finished' attribute.
        """
        self._logger.report(
            f"End of Particle-Swarm iteration {self.num_iter}, max number of iterations: {self.max_iter}."
        )
        self._logger.report(f"Function value at global best {self.fun_global_best}")
        self._logger.report(f"Variables at global best {self.global_best}")
        if not self.finished:
            if self.num_iter >= self.max_iter:
                self._logger.report("Number of iterations exceeded the maximum. Stop.")
                self.exceeded_max_iters = True
                self.finished = True

    @property
    def _state(self):
        state_dict = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_result_mapping", "_logger", "xtol", "ftol"]
        }
        return state_dict

    @property
    def is_finished(self):
        return self.finished

    @property
    def is_finished_ok(self):
        # If reeintroduce the tollerance values, also need to modify this!!!
        # return self.is_finished and not self.exceeded_max_iters
        return self.is_finished

    def _create_inputs(self):
        return getattr(self, self.next_submit)()

    def _update(self, outputs):
        getattr(self, self.next_update)(outputs)

    @property
    def result_value(self):
        value = super().result_value  # pylint: disable=no-member
        return value

    def _get_optimal_result(self):
        """
        Return the index and optimization value of the best evaluation process.
        """
        cost_values = {
            k: get_nested_result(v.output, self.result_key)
            for k, v in self._result_mapping.items()
        }
        opt_index, opt_output = min(cost_values.items(), key=lambda item: item[1].value)
        opt_input = self._result_mapping[opt_index].input[self.input_key]

        return (opt_index, opt_input, opt_output)

    def get_engine_outputs(self):
        return {"last_particles": orm.List(list=self.local_best.tolist()).store()}


class ParticleSwarm(OptimizationEngineWrapper):
    """
    Engine to perform the Particle-Swarm optimization (http://dx.doi.org/10.1109/CEC.2003.1299391).

    :param particles: The current / initial set of particles. Must be a list of shape (M, N), where N is the dimension
        of the problem and M is free to choose, it will be the number of particles!
    :type particles: array

    :param max_iter: Maximum number of iteration steps.
    :type max_iter: int

    :param input_key: Name of the input argument in the evaluation process.
    :type input_key: str

    :param result_key: Name of the output argument in the evaluation process.
    :type result_key: str
    """

    _IMPL_CLASS = _ParticleSwarmImpl

    def __new__(  # pylint: disable=arguments-differ,too-many-arguments
        cls,
        particles,
        max_iter=20,
        input_key="x",
        result_key="result",
        logger=None,
    ):
        return cls._IMPL_CLASS(  # pylint: disable=no-member
            particles=particles,
            max_iter=max_iter,
            input_key=input_key,
            result_key=result_key,
            logger=logger,
        )
