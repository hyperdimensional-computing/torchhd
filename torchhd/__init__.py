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
