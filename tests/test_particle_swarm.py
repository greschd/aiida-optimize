# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the OptimizationWorkChain.
"""

import pytest

from aiida_optimize.engines import ParticleSwarm


@pytest.mark.parametrize(
    ["func_workchain_name", "particles", "x_exact", "f_exact"],
    (
        ["X2Y2", [[1.2, 0.9], [1.0, 2.0], [2.0, 1.0], [0.1, 0.1]], [0.0, 0.0], 0.0],
    ),  # pylint: disable=too-many-arguments
)
def test_particle_swarm(
    check_optimization,
    func_workchain_name,
    particles,
    x_exact,
    f_exact,
):
    """
    Simple test of the OptimizationWorkChain, with the Nelder-Mead engine.
    """

    check_optimization(
        engine=ParticleSwarm,
        engine_kwargs=dict(particles=particles, max_iter=25),
        func_workchain_name=func_workchain_name,
        xtol=[0.1, 0.1],
        ftol=0.01,
        x_exact=x_exact,
        f_exact=f_exact,
        output_port_names=[
            "engine_outputs__last_particles",
        ],
    )
