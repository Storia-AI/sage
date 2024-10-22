# Work-around the fact that `pip install -e .` doesn't work with `pyproject.toml` files.
from setuptools import setup

setup()