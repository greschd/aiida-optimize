# -*- coding: utf-8 -*-
"""
Tests for the CreateEvaluateWorkChain.
"""

# pylint: disable=unused-argument,redefined-outer-name
import pytest

from aiida import orm
from aiida.plugins import WorkflowFactory
from aiida.engine.launch import run_get_node

from sample_processes import echo_process  # pylint: disable=import-error,useless-suppression, unused-import


@pytest.mark.usefixtures('aiida_profile_clean')
def test_create_evaluate_basic(echo_process):
    """
    Test the CreateEvaluateWorkChain by chaining two basic processes.
    """

    CreateEvaluateWorkChain = WorkflowFactory('optimize.wrappers.create_evaluate')  # pylint: disable=invalid-name

    res, node = run_get_node(
        CreateEvaluateWorkChain,
        create_process=echo_process,
        evaluate_process=echo_process,
        create={'x': orm.Float(1)},
        output_input_mapping=orm.Dict(dict={'result': 'x'})
    )
    assert node.is_finished_ok
    assert 'create' in res
    assert 'evaluate' in res
    assert 'result' in res['create']
    assert 'result' in res['evaluate']
    assert res['create']['result'].value == 1
    assert res['evaluate']['result'].value == 1
