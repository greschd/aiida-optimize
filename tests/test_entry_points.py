# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests that the entrypoints are loadable through their respective factory.
"""
from aiida.plugins.factories import (
    CalculationFactory,
    DataFactory,
    ParserFactory,
    TransportFactory,
    WorkflowFactory,
)
from importlib_metadata import entry_points
import pytest


@pytest.fixture
@pytest.mark.usefixtures("aiida_profile_clean")
def check_entrypoints():
    """
    Fixture to check that loading of all the workflow, calculation and parser
    entrypoints through the corresponding factory works for the given (base)
    module name.
    """

    def inner(module_name):  # pylint: disable=missing-docstring
        for group, factory in [
            ("aiida.workflows", WorkflowFactory),
            ("aiida.calculations", CalculationFactory),
            ("aiida.parsers", ParserFactory),
            ("aiida.data", DataFactory),
            ("aiida.transports", TransportFactory),
        ]:
            for entry_point in entry_points().select(group=group):
                if entry_point.value.split(".")[0] == module_name:
                    factory(entry_point.name)

    return inner


def test_entrypoints(check_entrypoints):  # pylint: disable=redefined-outer-name
    """
    Check that the entrypoints are valid.
    """
    check_entrypoints("aiida_optimize")
