#!/usr/bin/env bash
set -ev

# Make sure the folder containing the workchains is in the python path before the daemon is started
SYSTEM_TESTS="${GITHUB_WORKSPACE}/.github/system_tests"

export PYTHONPATH="${PYTHONPATH}:${SYSTEM_TESTS}"

verdi daemon start 4
verdi -p test_aiida run ${SYSTEM_TESTS}/test_daemon.py
verdi daemon stop