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
    bundle
    permute
    cleanup
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
    sequence
    distinct_sequence
    hash_table
    ngrams


Utilities
------------------------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    
    map_range
    value_to_index
    index_to_value