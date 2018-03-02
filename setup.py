"""
usage: pip install .[dev]
"""

import re
from setuptools import setup, find_packages

# Get the version number
with open('./aiida_optimize/__init__.py') as f:
    MATCH_EXPR = "__version__[^'\"]+(['\"])([^'\"]+)"
    VERSION = re.search(MATCH_EXPR, f.read()).group(2).strip()

if __name__ == '__main__':
    setup(
        name='aiida-optimize',
        version=VERSION,
        description='AiiDA Plugin for running optimization algorithms.',
        author='Dominik Gresch',
        author_email='greschd@gmx.ch',
        license='MIT',
        classifiers=[
            'Development Status :: 3 - Alpha', 'Environment :: Plugins',
            'Framework :: AiiDA', 'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering :: Physics'
        ],
        keywords='aiida workflows optimization',
        packages=find_packages(),
        include_package_data=True,
        setup_requires=['reentry'],
        reentry_register=True,
        install_requires=[
            'aiida-core', 'fsc.export', 'aiida-tools', 'future', 'numpy',
            'scipy', 'decorator', 'pyyaml'
        ],
        extras_require={
            ':python_version < "3"': ['chainmap'],
            'dev': [
                'pytest', 'pytest-cov', 'aiida-pytest', 'yapf==0.20',
                'pre-commit', 'sphinx-rtd-theme', 'prospector'
            ]
        },
        entry_points={
            'aiida.workflows': [
                'optimize.optimize = aiida_optimize.workchain:OptimizationWorkChain'
            ]
        },
    )
