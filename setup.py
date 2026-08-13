from setuptools import find_packages, setup

setup(
    name="neuro_rnn",
    version="1.0.0",
    description=(
        "Spatially embedded, biologically constrained RNNs for studying how "
        "brain geometry and cognitive inputs shape emergent dynamics and topology"
    ),
    url="https://github.com/abeyh/neuro_rnn",
    license="BSD-3-Clause",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.12",
)
