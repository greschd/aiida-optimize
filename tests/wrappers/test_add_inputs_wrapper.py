# -*- coding: utf-8 -*-
"""
Tests for the AddInputsWorkChain.
"""

from aiida import orm
from aiida.engine import run_get_node

from aiida_optimize.wrappers import AddInputsWorkChain

from sample_processes import echo_process, EchoDictValue, EchoNestedValues  # pylint: disable=unused-import


def test_basic(
    configure_with_daemon,  # pylint: disable=unused-argument
    echo_process  # pylint: disable=redefined-outer-name
):
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


def test_basic_as_single_input(
    configure_with_daemon,  # pylint: disable=unused-argument
    echo_process  # pylint: disable=redefined-outer-name
):
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


def test_dict(configure_with_daemon):  # pylint: disable=unused-argument
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


def test_dict_as_single_input(configure_with_daemon):  # pylint: disable=unused-argument
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


def test_both(configure_with_daemon):  # pylint: disable=unused-argument
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


def test_nested(configure_with_daemon):  # pylint: disable=unused-argument
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
