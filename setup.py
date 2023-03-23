from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="attention-jax",
    packages=find_packages(exclude=["notebooks"]),
    version="0.0.1",
    license="MIT",
    description="Attention Transformer - Jax",
    author="Daniel T. Plop",
    url="https://github.com/plopd/attention-is-all-you-need-jax",
    keywords=["attention mechanism"],
    install_requires=required,
)
