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

    empty_hv
    identity_hv
    random_hv
    level_hv
    thermometer_hv
    circular_hv
    

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
    clipping
    soft_quantize
    hard_quantize


Similarities
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    cos_similarity
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

    VSA_Model
    BSC
    MAP
    HRR
    FHRR


Utilities
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    as_vsa_model
    map_range
    value_to_index
    index_to_value