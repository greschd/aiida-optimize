# -*- coding: utf-8 -*-
"""
Tests for the AddInputsWorkChain.
"""
# pylint: disable=unused-argument,redefined-outer-name
import pytest

from aiida import orm
from aiida.engine import run_get_node

from aiida_optimize.wrappers import AddInputsWorkChain

from sample_processes import EchoDictValue, EchoNestedValues  # pylint: disable=import-error,useless-suppression, unused-import


@pytest.mark.usefixtures('aiida_profile_clean')
def test_basic(echo_process):
    """
    Basic test, adding a single input in a List.
    """
    res, node = run_get_node(
        AddInputsWorkChain,
        sub_process=echo_process,
        added_input_values=orm.List(list=[1.]),
        added_input_keys=orm.List(list=['x']),
    )
    assert node.is_finished_ok
    assert 'result' in res
    assert res['result'].value == 1


@pytest.mark.usefixtures('aiida_profile_clean')
def test_basic_as_single_input(echo_process):
    """
    Basic test for a single input, as "bare" input.
    """
    res, node = run_get_node(
        AddInputsWorkChain,
        sub_process=echo_process,
        added_input_values=orm.Float(1),
        added_input_keys=orm.Str('x'),
    )
    assert node.is_finished_ok
    assert 'result' in res
    assert res['result'].value == 1


@pytest.mark.usefixtures('aiida_profile_clean')
def test_dict():
    """
    Test setting an attribute of a nested Dict.
    """
    res, node = run_get_node(
        AddInputsWorkChain,
        sub_process=EchoDictValue,
        inputs={'x': orm.Float(1)},
        added_input_values=orm.List(list=[2.]),
        added_input_keys=orm.List(list=['a:b.c'])
    )
    assert node.is_finished
    assert 'x' in res
    assert 'c' in res
    assert res['x'].value == 1
    assert res['c'].value == 2


@pytest.mark.usefixtures('aiida_profile_clean')
def test_dict_as_single_input():
    """
    Test setting an attribute of a nested Dict, as "bare" input.
    """
    res, node = run_get_node(
        AddInputsWorkChain,
        sub_process=EchoDictValue,
        inputs={'x': orm.Float(1)},
        added_input_values=orm.Float(2),
        added_input_keys=orm.Str('a:b.c')
    )
    assert node.is_finished
    assert 'x' in res
    assert 'c' in res
    assert res['x'].value == 1
    assert res['c'].value == 2


@pytest.mark.usefixtures('aiida_profile_clean')
def test_both():
    """
    Test setting both the attribute of a Dict and a plain input.
    """
    res, node = run_get_node(
        AddInputsWorkChain,
        sub_process=EchoDictValue,
        added_input_values=orm.List(list=[1., 2.]),
        added_input_keys=orm.List(list=['x', 'a:b.c'])
    )
    assert node.is_finished
    assert 'x' in res
    assert 'c' in res
    assert res['x'].value == 1
    assert res['c'].value == 2


@pytest.mark.usefixtures('aiida_profile_clean')
def test_nested():
    """
    Test setting more complicated nested inputs.
    """
    res, node = run_get_node(
        AddInputsWorkChain,
        sub_process=EchoNestedValues,
        added_input_values=orm.List(list=[1., 2.]),
        added_input_keys=orm.List(list=['x.y', 'a.b.c.d:e.f'])
    )
    assert node.is_finished
    assert 'y' in res
    assert 'f' in res
    assert res['y'].value == 1
    assert res['f'].value == 2
