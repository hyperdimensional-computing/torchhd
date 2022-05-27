import torchhd.functional as functional

def plot_basis_set(A, ax=None, **kwargs):
    """Displays a similarity map for the provided set of hypervectors as a 
    matrix map

    Args:
        A: 2D array-like
            The set of hypervectors whose similarity map is to be displayed
        
        ax: matplotlib Axes, optional
        Axes in which to draw the plot
        
    Other Parameters:
        **kwargs: `~matplotlib.pyplot.matshow` arguments

    Returns:
        ax: matplotlib.axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to use plotting functionality. \
        See https://matplotlib.org/stable/users/installing/index.html for more information.")

    sim = []
    for vector in A:
        sim.append(functional.cosine_similarity(vector, A).tolist())
    if ax is None:
        ax = plt.gca()
    axes = ax.matshow(sim, **kwargs)
    return axes

def plot_vector_similarity(A, vec, ax=None, **kwargs):
    """Displays a stem graph showing the similarity of input vector vec
    with hypervectors in input set A

    Args:
        A:  2D array-like
            Set of Hypervectors
        
        vec:    1D array-like

        ax: matplotlib Axes, optional
        Axes in which to draw the plot
    
    Other Parameters:
        **kwargs: `~matplotlib.pyplot.stem` arguments

    Returns:
        axes: matplotlib.axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to use plotting functionality. \
        See https://matplotlib.org/stable/users/installing/index.html for more information.")
    
    final = functional.cosine_similarity(vec, A).tolist()
    if ax is None:
        ax  = plt.gca()
    axes = ax.stem(final, **kwargs)
    return axes