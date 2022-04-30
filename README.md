<h1 align="center">
    <a href="https://mikeheddes.github.io/hdc-lib">Hyperdimensional Computing Library</a><br/>

</h1>
<p align="center">
    <a href="https://github.com/mikeheddes/hdc-lib/blob/main/LICENSE">
    <img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-orange.svg?style=flat" /></a>
    <a href="https://pypi.org/project/hdc/"><img alt="pypi version" src="https://img.shields.io/pypi/v/hdc.svg?style=flat&color=orange" /></a>
    <a href="https://github.com/mikeheddes/hdc-lib/actions/workflows/test.yml?query=branch%3Amain"><img alt="tests status" src="https://img.shields.io/github/workflow/status/mikeheddes/hdc-lib/Testing/main?label=tests&style=flat" /></a>
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat" />
</p>

This is a Python library for Hyperdimensional Computing.

* **Easy-to-use:** This library makes it painless to develop a wide range of Hyperdimensional Computing (HDC) applications and algorithms. For someone new to the field we provide Pythonic abstractions and examples to get you started fast. For the experienced researchers we made the library modular by design, giving you endless flexibility to prototype new ideas in no-time.
* **Performant:** The library is build on top of the high-performance PyTorch library, giving you optimized tensor execution without the headaches. Moreover, PyTorch makes it effortless to accelerate your code on a GPU.

## Installation

The library is hosted on PyPi and Conda, use one of the following commands to install:

```bash
pip install hdc
```

```bash
conda install -c conda-forge hdc
```

## Documentation

You can find the library's documentation [on the website](https://mikeheddes.github.io/hdc-lib).

## Examples

We have several examples [in the repository](/examples/). Here is a simple one to get you started:

```python
import hdc
import torch

d = 10000  # number of dimensions

### create a hypervector for each symbol
# keys for each field of the dictionary: fruit, weight, and season
keys = hdc.functional.random_hv(3, d)
# fruits are: apple, lemon, mango
fruits = hdc.functional.random_hv(3, d)
# there are 10 weight levels
weights = hdc.functional.level_hv(10, d)
# the for seasons: winter, spring, summer, fall
seasons = hdc.functional.circular_hv(4, d)

# map a float between min, max to an index of size 10
# we map the 10 weight levels between 0 to 200 grams
weight_index = hdc.functional.value_to_index(149.0, 0, 200, 10)

values = torch.stack([
    fruits[0],
    weights[weight_index],
    seasons[3],
])
# creates a dictionary: 
# record = key[0] * value[0] + key[1] * value[1] + key[2] * value[2]
record = hdc.functional.struct(keys, values)

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

## Contributing

### Creating a New Release

- A GitHub release triggers a GitHub action that builds the library and publishes it to PyPi and Conda in addition to the documentation website.
- Before creating a new GitHub release, increment the version number in [setup.py](/setup.py) using [semantic versioning](https://semver.org).
- When creating a new GitHub release, set the tag to be "v{version number}", e.g., v1.5.2, and provide a clear description of the changes.

### License

This library is [MIT licensed](./LICENSE).
