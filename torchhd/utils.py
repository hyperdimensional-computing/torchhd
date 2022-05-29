import torchhd.functional as functional

def plot_basis_set(memory, ax=None, **kwargs):
    """Displays a similarity map for the provided set of hypervectors as a 
    matrix map

    Args:
        memory: 2D array-like
            The set of hypervectors whose similarity map is to be displayed
        
        ax: matplotlib Axes, optional
        Axes in which to draw the plot
        
    Other Parameters:
        **kwargs: `~matplotlib.pyplot.matshow` arguments

    Returns:
        ax: matplotlib.axes

    Examples::
        
        >>>  import matplotlib as plt
        >>>  memory = torchhd.random_hvs(10, 10000)
        >>>  utils.plot_basis_set(memory)
        >>>  plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to use plotting functionality. \
        See https://matplotlib.org/stable/users/installing/index.html for more information.")

    sim = []
    for vector in memory:
        sim.append(functional.cosine_similarity(vector, memory).tolist())
    if ax is None:
        ax = plt.gca()
    axes = ax.matshow(sim, **kwargs)
    return axes

def plot_similarity(vector, memory, ax=None, **kwargs):
    """Displays a stem graph showing the similarity of input vector vec
    with hypervectors in input set A

    Args:
        vector:    1D array-like

        memory:  2D array-like
                Set of Hypervectors
        
        ax: matplotlib Axes, optional
        Axes in which to draw the plot
    
    Other Parameters:
        **kwargs: `~matplotlib.pyplot.stem` arguments

    Returns:
        axes: matplotlib.axes

    Examples::
        
        >>>  import matplotlib as plt
        >>>  memory = torchhd.level_hvs(10, 10000)
        >>>  vector = torchhd.random_hv(1, 10000)
        >>>  utils.plot_similarity(memory, vector)
        >>>  plt.show()

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Install matplotlib to use plotting functionality. \
        See https://matplotlib.org/stable/users/installing/index.html for more information.")
    
    final = functional.cosine_similarity(vector, memory).tolist()
    if ax is None:
        ax  = plt.gca()
    axes = ax.stem(final, **kwargs)
    return axes