from torch import Tensor
import torchhd.functional as functional

def plot_basis_set(memory:Tensor, ax=None, **kwargs):
    """Displays a similarity map for the provided set of hypervectors as a 
    matrix map

    Args:
        memory (torch.Tensor): 2D array-like
            The set of hypervectors whose similarity map is to be displayed
        
        ax (matplotlib.axes, optional):
            Axes in which to draw the plot
        
    Other Parameters:
        **kwargs: `~matplotlib.axes.Axes.matshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.matshow.html>` arguments

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

def plot_similarity(vector:Tensor, memory:Tensor, ax=None, **kwargs):
    """Displays a stem graph showing the similarity of input vector vec
    with hypervectors in input set A

    Args:
        vector (torch.Tensor):  1D array-like

        memory (torch.Tensor):  2D array-like
                                Set of Hypervectors
        
        ax (matplotlib.axes, optional):
            Axes in which to draw the plot
    
    Other Parameters:
        **kwargs: `~matplotlib.pyplot.stem <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stem.html?highlight=stem#matplotlib.axes.Axes.stem>` arguments

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