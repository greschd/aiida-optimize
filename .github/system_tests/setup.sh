#!/usr/bin/env bash
set -ev

# Replace the placeholders in configuration files with actual values
CONFIG="${GITHUB_WORKSPACE}/.github/system_tests"
sed -i "s|PLACEHOLDER_WORK_DIR|${GITHUB_WORKSPACE}|" "${CONFIG}/localhost.yml"

verdi setup --non-interactive --config "${CONFIG}/profile.yml"

# set up localhost computer
verdi computer setup --non-interactive --config "${CONFIG}/localhost.yml"
verdi computer configure core.local localhost --config "${CONFIG}/localhost-config.yml"
verdi computer test localhost

verdi profile setdefault test_aiida
verdi config set runner.poll.interval 0