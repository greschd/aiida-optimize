import re
from setuptools import setup, find_packages

# Get the version number
with open('./aiida_optimize/__init__.py') as f:
    match_expr = "__version__[^'\"]+(['\"])([^'\"]+)"
    version = re.search(match_expr, f.read()).group(2).strip()

if __name__ == '__main__':
    setup(
        name='aiida-optimize',
        version=version,
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
        packages=find_packages(exclude=['aiida']),
        include_package_data=True,
        setup_requires=['reentry'],
        reentry_register=True,
        install_requires=['aiida-core', 'fsc.export'],
        extras_require={
            ':python_version < "3"': [],
            'dev': ['aiida-pytest', 'yapf', 'pre-commit', 'sphinx-rtd-theme']
        },
        entry_points={},
    )