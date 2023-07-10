.. _functional:

torchhd
=========================

.. currentmodule:: torchhd

This module consists of the basic hypervector generation functions and operations used on hypervectors.

Basis-hypervector sets
----------------------------------

.. autosummary:: 
    :toctree: generated/
    :template: function.rst

    empty
    identity
    random
    level
    thermometer
    circular
    

Operations
--------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    bind
    bundle
    permute
    inverse
    negative
    cleanup
    randsel
    multirandsel
    create_random_permute
    resonator
    ridge_regression
    soft_quantize
    hard_quantize


Similarities
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    cosine_similarity
    dot_similarity
    hamming_similarity


Encodings
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    multiset
    multibind
    bundle_sequence
    bind_sequence
    hash_table
    cross_product
    ngrams
    graph


VSA Models
------------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    VSATensor
    BSCTensor
    MAPTensor
    HRRTensor
    FHRRTensor
    VTBTensor


Utilities
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    ensure_vsa_tensor
    map_range
    value_to_index
    index_to_value
