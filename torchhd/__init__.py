#
# MIT License
#
# Copyright (c) 2023 Mike Heddes, Igor Nunes, Pere Vergés, Denis Kleyko, and Danny Abraham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import torchhd.functional as functional
import torchhd.embeddings as embeddings
import torchhd.structures as structures
import torchhd.models as models
import torchhd.memory as memory
import torchhd.datasets as datasets
import torchhd.utils as utils

from torchhd.tensors.base import VSATensor
from torchhd.tensors.bsc import BSCTensor
from torchhd.tensors.map import MAPTensor
from torchhd.tensors.hrr import HRRTensor
from torchhd.tensors.fhrr import FHRRTensor

from torchhd.functional import (
    ensure_vsa_tensor,
    empty,
    identity,
    random,
    level,
    thermometer,
    flocet,
    circular,
    bind,
    bundle,
    permute,
    inverse,
    negative,
    cleanup,
    create_random_permute,
    randsel,
    multirandsel,
    soft_quantize,
    hard_quantize,
    cosine_similarity,
    cos,
    dot_similarity,
    dot,
    hamming_similarity,
    multiset,
    multibundle,
    multibind,
    bundle_sequence,
    bind_sequence,
    hash_table,
    cross_product,
    ngrams,
    graph,
    resonator,
    ridge_regression,
    map_range,
    value_to_index,
    index_to_value,
)

from torchhd.version import __version__

__all__ = [
    "__version__",
    "VSATensor",
    "BSCTensor",
    "MAPTensor",
    "HRRTensor",
    "FHRRTensor",
    "functional",
    "embeddings",
    "structures",
    "models",
    "memory",
    "datasets",
    "utils",
    "ensure_vsa_tensor",
    "empty",
    "identity",
    "random",
    "level",
    "thermometer",
    "circular",
    "bind",
    "bundle",
    "permute",
    "inverse",
    "negative",
    "cleanup",
    "create_random_permute",
    "randsel",
    "multirandsel",
    "soft_quantize",
    "hard_quantize",
    "cosine_similarity",
    "cos",
    "dot_similarity",
    "dot",
    "hamming_similarity",
    "multiset",
    "multibundle",
    "multibind",
    "bundle_sequence",
    "bind_sequence",
    "hash_table",
    "cross_product",
    "ngrams",
    "graph",
    "resonator",
    "ridge_regression",
    "map_range",
    "value_to_index",
    "index_to_value",
]
