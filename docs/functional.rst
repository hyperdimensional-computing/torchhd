.. _functional:

torchhd.functional
=========================

.. currentmodule:: torchhd.functional 

This module consists of the basic hypervector generation functions and operations used on hypervectors.

Basis-hypervector sets
----------------------------------

.. autosummary:: 
    :toctree: generated/
    :template: function.rst

    identity_hv
    random_hv
    level_hv
    circular_hv
    

Operations
--------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    bind
    unbind
    bundle
    permute
    cleanup
    randsel
    multirandsel
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


Utilities
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    
    map_range
    value_to_index
    index_to_value