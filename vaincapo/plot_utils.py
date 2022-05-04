"""Module containing plotting tools."""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cmx
from matplotlib.patches import Circle

from vaincapo.utils import quat_to_hopf


def plot_posterior(
    quat_samples: np.ndarray, quat_gt: Optional[np.ndarray] = None
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        quat_samples: rotation samples in quaternion [w, x, y, z], shape (N, 4)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)

    Returns:
        figure instance
    """
    rainbow = plt.get_cmap("gist_rainbow")
    s1_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=0, vmax=2 * np.pi), cmap=rainbow
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True)
    if quat_gt is not None:
        hopf_gt = quat_to_hopf(quat_gt[None, :])[0]
        ax.add_patch(
            Circle(
                xy=hopf_gt[:2],
                radius=0.1,
                linewidth=3,
                facecolor="none",
                edgecolor=s1_cm.to_rgba(hopf_gt[2]),
            )
        )
    hopf_samples = quat_to_hopf(quat_samples)
    ax.scatter(*hopf_samples.T[:2], s=2, color=s1_cm.to_rgba(hopf_samples.T[2]))
    plt.show()

    return fig
