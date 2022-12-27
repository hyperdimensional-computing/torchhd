<p align="center">
    <a href="https://github.com/hyperdimensional-computing/torchhd/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-orange.svg?style=flat" /></a>
    <a href="https://pypi.org/project/torch-hd/"><img alt="pypi version" src="https://img.shields.io/pypi/v/torch-hd.svg?style=flat&color=orange" /></a>
    <a href="https://anaconda.org/torchhd/torchhd"><img alt="conda version" src="https://img.shields.io/conda/v/torchhd/torchhd?label=conda&style=flat&color=orange" /></a>
    <a href="https://github.com/hyperdimensional-computing/torchhd/actions/workflows/test.yml?query=branch%3Amain"><img alt="tests status" src="https://img.shields.io/github/actions/workflow/status/hyperdimensional-computing/torchhd/test.yml?branch=main&label=tests&style=flat" />
    </a><img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat" />
</p>

<div align="center">
    <a href="https://github.com/hyperdimensional-computing/torchhd">
        <img width="380px"  alt="Torchhd logo" src="https://raw.githubusercontent.com/hyperdimensional-computing/torchhd/main/docs/images/torchhd-logo.svg" />
    </a>
</div>

# Torchhd

Torchhd is a Python library for *Hyperdimensional Computing* (also known as *Vector Symbolic Architectures*).

* **Easy-to-use:** Torchhd makes it painless to develop a wide range of Hyperdimensional Computing (HDC) applications and algorithms. For someone new to the field, we provide Pythonic abstractions and examples to get you started fast. For the experienced researchers, we made the library modular by design, giving you endless flexibility to prototype new ideas in no-time.
* **Performant:** The library is build on top of the high-performance [PyTorch](https://pytorch.org/) library, giving you optimized tensor execution without the headaches. Moreover, PyTorch makes it effortless to accelerate your code on a GPU.

## Installation

Torchhd is hosted on [PyPi](https://pypi.org/project/torch-hd/) and [Anaconda](https://anaconda.org/torchhd/torchhd). Use one of the following commands to install:

```bash
pip install torch-hd
```

```bash
conda install -c torchhd torchhd
```

## Documentation

You can find documentation for Torchhd [on the website](https://torchhd.readthedocs.io).

Check out the [Getting Started](https://torchhd.readthedocs.io/en/stable/getting_started.html) page for a quick overview.

The API documentation is divided into several sections:

- [`torchhd`](https://torchhd.readthedocs.io/en/stable/torchhd.html)
- [`torchhd.embeddings`](https://torchhd.readthedocs.io/en/stable/embeddings.html)
- [`torchhd.structures`](https://torchhd.readthedocs.io/en/stable/structures.html)
- [`torchhd.datasets`](https://torchhd.readthedocs.io/en/stable/datasets.html)

You can improve the documentation by sending pull requests to this repository.

## Examples

We have several examples [in the repository](https://github.com/hyperdimensional-computing/torchhd/tree/main/examples). Here is a simple one to get you started:

```python
import torch, torchhd

d = 10000  # number of dimensions

# create the hypervectors for each symbol
keys = torchhd.random_hv(3, d)
country, capital, currency = keys

usa, mex = torchhd.random_hv(2, d)  # United States and Mexico
wdc, mxc = torchhd.random_hv(2, d)  # Washington D.C. and Mexico City
usd, mxn = torchhd.random_hv(2, d)  # US Dollar and Mexican Peso

# create country representations
us_values = torch.stack([usa, wdc, usd])
us = torchhd.hash_table(keys, us_values)

mx_values = torch.stack([mex, mxc, mxn])
mx = torchhd.hash_table(keys, mx_values)

# combine all the associated information
mx_us = torchhd.bind(torchhd.inverse(us), mx)

# query for the dollar of mexico
usd_of_mex = torchhd.bind(mx_us, usd)

memory = torch.cat([keys, us_values, mx_values], dim=0)
torchhd.cos_similarity(usd_of_mex, memory)
# tensor([-0.0062,  0.0123, -0.0057, -0.0019, -0.0084, -0.0078,  0.0102,  0.0057,  0.3292])
# The hypervector for the Mexican Peso is the most similar.
```

This example is from the paper [What We Mean When We Say "What's the Dollar of Mexico?": Prototypes and Mapping in Concept Space](https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf) by Kanerva. It first creates hypervectors for all the symbols that are used in the computation, i.e., the variables for `country`, `capital`, and `currency` and their values for both countries. These hypervectors are then combined to make a single hypervector for each country using a hash table structure. A hash table encodes key-value pairs as: `k1 * v1 + k2 * v2 + ... + kn * vn`. The hash tables are then bound together to form their combined representation which is finally queried by binding with the Dollar hypervector to obtain the approximate Mexican Peso hypervector. The similarity output shows that the Mexican Peso hypervector is indeed the most similar one.


## About

Initial development of Torchhd was performed by [Mike Heddes](https://www.mikeheddes.nl/) and [Igor Nunes](https://sites.uci.edu/inunes/) as part of their research in Hyperdimensional Computing at the University of California, Irvine. The library was extended with significant contributions from Pere Vergés and Dheyay Desai. Torchhd later merged with a project by Rishikanth Chandrasekaran, who worked on similar problems as part of his research at the University of California, San Diego.

## Contributing

### Documentation

To build the documentation locally, use `pip install -r docs/requirements.txt` to install the required packages. Then, with `sphinx-build -b html docs build` you can generate the html documentation in the `/build` directory. To create a clean build, remove the `/build` and `/docs/generated` directories.

### Creating a New Release

- A GitHub release triggers a GitHub action that builds the library and publishes it to PyPi and Conda in addition to the documentation website.
- Before creating a new GitHub release, increment the version number in [version.py](https://github.com/hyperdimensional-computing/torchhd/blob/main/torchhd/version.py) using [semantic versioning](https://semver.org).
- When creating a new GitHub release, set the tag according to [PEP 440](https://peps.python.org/pep-0440/), e.g., v1.5.2, and provide a clear description of the changes. You can use GitHub's "auto-generate release notes" button. Look at previous releases for examples.

### License

This library is [MIT licensed](https://github.com/hyperdimensional-computing/torchhd/blob/main/LICENSE).


## Cite

Consider citing [our paper](https://arxiv.org/abs/2205.09208) if you use Torchhd in your work:

```
@article{heddes2022torchhd,
  title={Torchhd: An Open-Source Python Library to Support Hyperdimensional Computing Research},
  author={Heddes, Mike and Nunes, Igor and Vergés, Pere and Desai, Dheyay and Givargis, Tony and Nicolau, Alexandru},
  journal={arXiv preprint arXiv:2205.09208},
  year={2022}
}
```
