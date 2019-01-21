# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
usage: pip install .[dev]
"""

import re
from setuptools import setup, find_packages

# Get the version number
with open('./aiida_optimize/__init__.py') as f:
    MATCH_EXPR = "__version__[^'\"]+['\"]([^'\"]+)"
    VERSION = re.search(MATCH_EXPR, f.read()).group(1).strip()

if __name__ == '__main__':
    setup(
        name='aiida-optimize',
        url='http://z2pack.ethz.ch/aiida-plugins/aiida-optimize',
        version=VERSION,
        description='AiiDA Plugin for running optimization algorithms.',
        author='Dominik Gresch',
        author_email='greschd@gmx.ch',
        license='MIT',
        classifiers=[
            'Development Status :: 3 - Alpha', 'Environment :: Plugins',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 2.7', 'Topic :: Scientific/Engineering :: Physics'
        ],
        keywords=['AiiDA', 'workflows', 'optimization'],
        packages=find_packages(),
        include_package_data=True,
        setup_requires=['reentry'],
        reentry_register=True,
        install_requires=[
            'aiida-core', 'fsc.export', 'aiida-tools', 'future', 'numpy', 'scipy', 'decorator',
            'pyyaml'
        ],
        extras_require={
            ':python_version < "3"': ['chainmap'],
            'dev': [
                'pytest>=3.6', 'pytest-cov', 'aiida-pytest', 'yapf==0.25', 'pre-commit',
                'sphinx-rtd-theme', 'prospector==0.12.11', 'pylint==1.9.3'
            ]
        },
        entry_points={
            'aiida.workflows':
            ['optimize.optimize = aiida_optimize.workchain:OptimizationWorkChain']
        },
    )
