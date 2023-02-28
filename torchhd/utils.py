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
import torch
from torch import Tensor

import torchhd.functional as functional


def plot_pair_similarity(memory: Tensor, ax=None, **kwargs):
    """Plots the pair-wise similarity of a hypervector set.

    Args:
        memory (Tensor): The set of :math:`n` hypervectors whose pair-wise similarity is to be displayed.
        ax (matplotlib.axes, optional): Axes in which to draw the plot.

    Other Parameters:
        **kwargs: `matplotlib.axes.Axes.pcolormesh <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.pcolormesh.html>`_ arguments.

    Returns:
        matplotlib.collections.QuadMesh: `matplotlib.collections.QuadMesh <https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.QuadMesh>`_.

    Shapes:
        - Memory: :math:`(n, d)`

    Examples::

        >>>  import matplotlib.pyplot as plt
        >>>  hv = torchhd.level(10, 10000)
        >>>  utils.plot_pair_similarity(hv)
        >>>  plt.show()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Install matplotlib to use plotting functionality. \
        See https://matplotlib.org/stable/users/installing/index.html for more information."
        )

    similarity = functional.cosine_similarity(memory, memory).tolist()

    if ax is None:
        ax = plt.gca()

    xy = torch.arange(memory.size(-2))
    x, y = torch.meshgrid(xy, xy)

    ax.set_aspect("equal", adjustable="box")
    return ax.pcolormesh(x, y, similarity, **kwargs)


def plot_similarity(input: Tensor, memory: Tensor, ax=None, **kwargs):
    """Plots the similarity of an one hypervector with a set of hypervectors.

    Args:
        input (torch.Tensor): Hypervector to compare against the memory.
        memory (torch.Tensor): Set of :math:`n` hypervectors.
        ax (matplotlib.axes, optional): Axes in which to draw the plot.

    Other Parameters:
        **kwargs: `matplotlib.axes.Axes.stem <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stem.html?highlight=stem#matplotlib.axes.Axes.stem>`_ arguments.

    Returns:
        StemContainer: `matplotlib.container.StemContainer <https://matplotlib.org/stable/api/container_api.html#matplotlib.container.StemContainer>`_.

    Shapes:
        - Input: :math:`(d)`
        - Memory: :math:`(n, d)`

    Examples::

        >>>  import matplotlib.pyplot as plt
        >>>  hv = torchhd.level(10, 10000)
        >>>  utils.plot_similarity(hv[4], hv)
        >>>  plt.show()

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Install matplotlib to use plotting functionality. \
        See https://matplotlib.org/stable/users/installing/index.html for more information."
        )

    similarity = functional.cosine_similarity(input, memory).tolist()

    if ax is None:
        ax = plt.gca()

    return ax.stem(similarity, **kwargs)
