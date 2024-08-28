# -*- coding: utf-8 -*-

from setuptools import setup
from codecs import open
from os import path
import re

package_name = "pyqlaw"

requires = [
    'sympy>=1.12',
    'numpy>=1.20.4',
    'numba>=0.54.0',
    'matplotlib>=3.8.4',
    'scipy>=1.13.0',
    'tqdm>=4.66.4',
    'spiceypy>=5.1.0',
]

root_dir = path.abspath(path.dirname(__file__))

# def _requirements():
#     return [name.rstrip() for name in open(path.join(root_dir, 'environment.yaml')).readlines()]

# def _test_requirements():
#     return [name.rstrip() for name in open(path.join(root_dir, 'test-requirements.txt')).readlines()]

with open(path.join(root_dir, "pyqlaw", '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    license = re.search(r'__license__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author = re.search(r'__author__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author_email = re.search(r'__author_email__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    url = re.search(r'__url__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

assert version
assert license
assert author
assert author_email
assert url

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    packages=["pyqlaw"],
    version=version,
    license=license,
    install_requires=requires,
    author=author,
    author_email=author_email,
    url=url,
    description='high-fidelity spaceraft dynamics propagator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='astrodynamics',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
)
