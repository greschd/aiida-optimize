#!/bin/bash

# Author: Dominik Gresch <greschd@gmx.ch>

# Be verbose, and stop with error as soon there's one
set -ev

pip install -U pip wheel setuptools
pip install 'aiida-core<1.6'

case "$INSTALL_TYPE" in
    testing)
        pip install .[testing]
        ;;
    testing_sdist)
        python setup.py sdist
        ls -1 dist/ | xargs -I % pip install dist/%[testing]
        ;;
    dev_precommit)
        pip install .[dev]
        ;;
    docs)
        pip install .[docs]
        ;;
esac

reentry scan
