# -*- coding: utf-8 -*-
"""
Tests for the ConcatenateWorkChain.
"""

from aiida import orm
from aiida.engine.launch import run_get_node
from aiida.plugins import WorkflowFactory

# pylint: disable=unused-argument,redefined-outer-name,invalid-name
import pytest

from aiida_optimize.process_inputs import get_fullname
from sample_processes import (  # pylint: disable=import-error,useless-suppression, unused-import
    Echo,
    EchoDictValue,
    EchoNestedValues,
)


@pytest.mark.usefixtures("aiida_profile_clean")
def test_concatenate_basic(echo_process):
    """
    Test the ConcatenateWorkChain by chaining three basic processes.
    """

    ConcatenateWorkChain = WorkflowFactory(  # pylint: disable=invalid-name
        "optimize.wrappers.concatenate"
    )

    res, node = run_get_node(
        ConcatenateWorkChain,
        process_labels=orm.List(
            list=[
                ("one", get_fullname(echo_process).value),
                ("two", get_fullname(echo_process).value),
                ("three", get_fullname(echo_process).value),
            ]
        ),
        process_inputs={"one": {"x": orm.Float(1)}},
        output_input_mappings=orm.List(
            list=[(("one", "two"), {"result": "x"}), (("two", "three"), {"result": "x"})]
        ),
    )
    assert node.is_finished_ok

    assert "one" in res["process_outputs"]
    assert "two" in res["process_outputs"]
    assert "three" in res["process_outputs"]

    assert "result" in res["process_outputs"]["one"]
    assert "result" in res["process_outputs"]["two"]
    assert "result" in res["process_outputs"]["three"]

    assert res["process_outputs"]["one"]["result"].value == 1
    assert res["process_outputs"]["two"]["result"].value == 1
    assert res["process_outputs"]["three"]["result"].value == 1


@pytest.mark.usefixtures("aiida_profile_clean")
def test_concatenate_wrong_label_order(echo_process):
    """
    The 'output_input_mapping' has labels in the wrong order.
    """

    ConcatenateWorkChain = WorkflowFactory(  # pylint: disable=invalid-name
        "optimize.wrappers.concatenate"
    )

    with pytest.raises(ValueError) as exc:
        run_get_node(
            ConcatenateWorkChain,
            process_labels=orm.List(
                list=[
                    ("one", get_fullname(echo_process).value),
                    ("two", get_fullname(echo_process).value),
                    ("three", get_fullname(echo_process).value),
                ]
            ),
            process_inputs={"one": {"x": orm.Float(1)}},
            output_input_mappings=orm.List(
                list=[(("two", "two"), {"result": "x"}), (("two", "three"), {"result": "x"})]
            ),
        )
    assert "cannot pass outputs" in str(exc.value).lower()


@pytest.mark.usefixtures("aiida_profile_clean")
def test_concatenate_duplicate_label(echo_process):
    """
    The 'process_labels' has a duplicate entry.
    """

    ConcatenateWorkChain = WorkflowFactory(  # pylint: disable=invalid-name
        "optimize.wrappers.concatenate"
    )

    with pytest.raises(ValueError) as exc:
        run_get_node(
            ConcatenateWorkChain,
            process_labels=orm.List(
                list=[
                    ("one", get_fullname(echo_process).value),
                    ("one", get_fullname(echo_process).value),
                    ("three", get_fullname(echo_process).value),
                ]
            ),
            process_inputs={"one": {"x": orm.Float(1)}},
            output_input_mappings=orm.List(
                list=[(("one", "two"), {"result": "x"}), (("two", "three"), {"result": "x"})]
            ),
        )
    assert "duplicate" in str(exc.value).lower()
    assert "process_labels" in str(exc.value).lower()


@pytest.mark.usefixtures("aiida_profile_clean")
def test_concatenate_invalid_input_label():
    """
    The 'process_inputs' contains an invalid process label.
    """

    ConcatenateWorkChain = WorkflowFactory(  # pylint: disable=invalid-name
        "optimize.wrappers.concatenate"
    )

    with pytest.raises(ValueError) as exc:
        run_get_node(
            ConcatenateWorkChain,
            process_labels=orm.List(
                list=[
                    ("one", get_fullname(Echo).value),
                    ("two", get_fullname(Echo).value),
                    ("three", get_fullname(Echo).value),
                ]
            ),
            process_inputs={"one": {"x": orm.Float(1)}, "invalid_label": {"x": orm.Float(2.0)}},
            output_input_mappings=orm.List(
                list=[(("one", "two"), {"result": "x"}), (("two", "three"), {"result": "x"})]
            ),
        )
    assert "does not match any of the 'process_labels'" in str(exc.value)


@pytest.mark.usefixtures("aiida_profile_clean")
def test_concatenate_invalid_mapping_label():
    """
    The 'output_input_mapping' contains an invalid process label.
    """

    ConcatenateWorkChain = WorkflowFactory(  # pylint: disable=invalid-name
        "optimize.wrappers.concatenate"
    )

    with pytest.raises(ValueError) as exc:
        run_get_node(
            ConcatenateWorkChain,
            process_labels=orm.List(
                list=[
                    ("one", get_fullname(Echo).value),
                    ("two", get_fullname(Echo).value),
                    ("three", get_fullname(Echo).value),
                ]
            ),
            process_inputs={"one": {"x": orm.Float(1)}, "two": {"x": orm.Float(2.0)}},
            output_input_mappings=orm.List(
                list=[
                    (("one", "two"), {"result": "x"}),
                    (("two", "invalid_label"), {"result": "x"}),
                ]
            ),
        )
    assert "process labels" in str(exc.value)
    assert "do not exist" in str(exc.value)


@pytest.mark.usefixtures("aiida_profile_clean")
def test_concatenate_nested_keys():
    """Concatenate processes with nested input and output keys."""
    ConcatenateWorkChain = WorkflowFactory(  # pylint: disable=invalid-name
        "optimize.wrappers.concatenate"
    )

    res, node = run_get_node(
        ConcatenateWorkChain,
        process_labels=orm.List(
            list=[
                ("one", get_fullname(EchoNestedValues).value),
                ("two", get_fullname(EchoNestedValues).value),
                ("three", get_fullname(EchoDictValue).value),
            ]
        ),
        process_inputs={
            "one": {
                "x": {"y": orm.Float(1)},
                "a": {"b": {"c": {"d": orm.Dict(dict=dict({"e": {"f": 2}}))}}},
            },
            "three": {"a": orm.Dict(dict={"b": {"c": 3}})},
        },
        output_input_mappings=orm.List(
            list=[
                (
                    ("one", "two"),
                    {
                        "y": "a.b.c.d:e.f",
                        "f": "x.y",
                    },
                ),
                (
                    ("two", "three"),
                    {
                        "y": "x",
                        "f": "f.g",
                    },
                ),
            ]
        ),
    )
    assert node.is_finished_ok

    assert "one" in res["process_outputs"]
    assert "two" in res["process_outputs"]
    assert "three" in res["process_outputs"]

    assert res["process_outputs"]["one"]["y"].value == 1
    assert res["process_outputs"]["one"]["f"].value == 2

    assert res["process_outputs"]["two"]["y"].value == 2
    assert res["process_outputs"]["two"]["f"].value == 1

    assert res["process_outputs"]["three"]["x"].value == 2
    assert res["process_outputs"]["three"]["c"].value == 3
    assert res["process_outputs"]["three"]["d"]["e"].get_dict() == {"f": {"g": 1}}


@pytest.mark.usefixtures("aiida_profile_clean")
def test_double_passing():
    """Pass inputs from two preceding processes to the last one."""
    ConcatenateWorkChain = WorkflowFactory(  # pylint: disable=invalid-name
        "optimize.wrappers.concatenate"
    )

    res, node = run_get_node(
        ConcatenateWorkChain,
        process_labels=orm.List(
            list=[
                ("one", get_fullname(EchoNestedValues).value),
                ("two", get_fullname(EchoNestedValues).value),
                ("three", get_fullname(EchoDictValue).value),
            ]
        ),
        process_inputs={
            "one": {
                "x": {"y": orm.Float(1)},
                "a": {"b": {"c": {"d": orm.Dict(dict=dict({"e": {"f": 2}}))}}},
            },
        },
        output_input_mappings=orm.List(
            list=[
                (
                    ("one", "two"),
                    {
                        "y": "a.b.c.d:e.f",
                        "f": "x.y",
                    },
                ),
                (
                    ("two", "three"),
                    {
                        "y": "x",
                        "f": "f.g",
                    },
                ),
                (("one", "three"), {"y": "a:b.c"}),
            ]
        ),
    )
    assert node.is_finished_ok

    assert "one" in res["process_outputs"]
    assert "two" in res["process_outputs"]
    assert "three" in res["process_outputs"]

    assert res["process_outputs"]["one"]["y"].value == 1
    assert res["process_outputs"]["one"]["f"].value == 2

    assert res["process_outputs"]["two"]["y"].value == 2
    assert res["process_outputs"]["two"]["f"].value == 1

    assert res["process_outputs"]["three"]["x"].value == 2
    assert res["process_outputs"]["three"]["c"].value == 1
    assert res["process_outputs"]["three"]["d"]["e"].get_dict() == {"f": {"g": 1}}
