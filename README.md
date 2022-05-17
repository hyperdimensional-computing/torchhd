<p align="center">
    <a href="https://github.com/hyperdimensional-computing/torchhd/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-orange.svg?style=flat" /></a>
    <a href="https://pypi.org/project/torch-hd/"><img alt="pypi version" src="https://img.shields.io/pypi/v/torch-hd.svg?style=flat&color=orange" /></a>
    <a href="https://anaconda.org/torchhd/torchhd"><img alt="conda version" src="https://img.shields.io/conda/v/torchhd/torchhd?label=conda&style=flat&color=orange" /></a>
    <a href="https://github.com/hyperdimensional-computing/torchhd/actions/workflows/test.yml?query=branch%3Amain"><img alt="tests status" src="https://img.shields.io/github/workflow/status/hyperdimensional-computing/torchhd/Test/main?label=tests&style=flat" />
    </a><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat" />
</p>

<div align="center">
    <a href="https://github.com/hyperdimensional-computing/torchhd">
        <img width="380px"  alt="Torchhd logo" src="https://raw.githubusercontent.com/hyperdimensional-computing/torchhd/main/docs/images/torchhd-logo.svg" />
    </a>
</div>

# Torchhd

Torchhd is a Python library for Hyperdimensional Computing.

* **Easy-to-use:** Torchhd makes it painless to develop a wide range of Hyperdimensional Computing (HDC) applications and algorithms. For someone new to the field we provide Pythonic abstractions and examples to get you started fast. For the experienced researchers we made the library modular by design, giving you endless flexibility to prototype new ideas in no-time.
* **Performant:** The library is build on top of the high-performance PyTorch library, giving you optimized tensor execution without the headaches. Moreover, PyTorch makes it effortless to accelerate your code on a GPU.

## Installation

Torchhd is hosted on PyPi, use the following command to install:

```bash
pip install torch-hd
```

<!-- ```bash
conda install -c torchhd torchhd
``` -->

## Documentation

You can find documentation for Torchhd [on the website](https://torchhd.readthedocs.io).

## Examples

We have several examples [in the repository](https://github.com/hyperdimensional-computing/torchhd/tree/main/examples). Here is a simple one to get you started:

```python
import torch
import torchhd

d = 10000  # number of dimensions

### create a hypervector for each symbol
# keys for each field of the dictionary: fruit, weight, and season
keys = torchhd.functional.random_hv(3, d)
# fruits are: apple, lemon, mango
fruits = torchhd.functional.random_hv(3, d)
# there are 10 weight levels
weights = torchhd.functional.level_hv(10, d)
# the for seasons: winter, spring, summer, fall
seasons = torchhd.functional.circular_hv(4, d)

# map a float between min, max to an index of size 10
# we map the 10 weight levels between 0 to 200 grams
weight_index = torchhd.functional.value_to_index(149.0, 0, 200, 10)

values = torch.stack([
    fruits[0],
    weights[weight_index],
    seasons[3],
])
# creates a dictionary: 
# record = key[0] * value[0] + key[1] * value[1] + key[2] * value[2]
record = torchhd.functional.struct(keys, values)

#### Similar Python code
# 
# record = dict(
#     fruit="apple", 
#     weight=149.0,
#     season="fall"
# )
# 
```

This example creates a hypervector that represents the record of a fruit, storing its species, weight, and growing season as one hypervector. This is achieved by combining the atomic information units into a structure (similar to a Python dictionary).

You will notice that we first declare all the symbols which are used to represent information. Note the type of hypervector used for each type of information, the fruits and keys use random hypervectors as they represent unrelated information whereas the weights and seasons use level and circular-hypervectors because they have linear and circular-correlations, respectively.

## About

Initial development of Torchhd was performed by Mike Heddes and Igor Nunes as part of their research in Hyperdimensional Computing at the University of California, Irvine. The library was extended with significant contributions from Pere Vergés and Dheyay Desai. Torchhd later merged with a project by Rishikanth Chandrasekaran who worked on similar problems as part of his research at the University of California, San Diego.

## Contributing

### Creating a New Release

- A GitHub release triggers a GitHub action that builds the library and publishes it to PyPi and Conda in addition to the documentation website.
- Before creating a new GitHub release, increment the version number in [setup.py](https://github.com/hyperdimensional-computing/torchhd/blob/main/setup.py) using [semantic versioning](https://semver.org).
- When creating a new GitHub release, set the tag according to [PEP 440](https://peps.python.org/pep-0440/), e.g., 1.5.2, and provide a clear description of the changes. You can use GitHub's "auto-generate release notes" button. Look at previous releases for examples.

### License

This library is [MIT licensed](https://github.com/hyperdimensional-computing/torchhd/blob/main/LICENSE).
