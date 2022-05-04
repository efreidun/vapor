"""Module containing plotting tools."""

from typing import Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.cm as cmx
from matplotlib.patches import Circle

from vaincapo.utils import quat_to_hopf


def plot_posterior(
    tra_samples: np.ndarray,
    quat_samples: np.ndarray,
    tra_mins: Iterable[float],
    tra_maxs: Iterable[float],
    tra_gt: Optional[np.ndarray] = None,
    quat_gt: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plot many samples drawn from a posterior distribution.

    Args:
        tra_samples: translation samples , shape (N, 3)
        quat_samples: rotation samples in quaternion [w, x, y, z], shape (N, 4)
        tra_mins: minimum boundaries of translations, shape (3,)
        tra_maxs: maximum boundaries of translations, shape (3,)
        tra_gt: groundtruth translation, shape (3,)
        quat_gt: groundtruth rotation in quaternion [w, x, y, z], shape (4,)

    Returns:
        figure instance
    """
    s1_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=0, vmax=2 * np.pi), cmap=plt.get_cmap("hsv")
    )
    z_cm = cmx.ScalarMappable(
        norm=clrs.Normalize(vmin=tra_mins[2], vmax=tra_maxs[2]),
        cmap=plt.get_cmap("plasma"),
    )

    fig = plt.figure()
    tra_ax = fig.add_subplot(121)
    tra_ax.set_xlim([tra_mins[0], tra_maxs[0]])
    tra_ax.set_ylim([tra_mins[1], tra_maxs[1]])
    tra_ax.set_title("translation posterior")
    if tra_gt is not None:
        tra_ax.add_patch(
            Circle(
                xy=tra_gt[:2],
                radius=0.075,
                linewidth=3,
                facecolor="none",
                edgecolor=z_cm.to_rgba(tra_gt[2]),
            )
        )
    tra_ax.scatter(*tra_samples.T[:2], s=2, color=z_cm.to_rgba(tra_samples.T[2]))

    rot_ax = fig.add_subplot(122, projection="mollweide")
    rot_ax.grid(True)
    rot_ax.set_title("rotation posterior")
    if quat_gt is not None:
        hopf_gt = quat_to_hopf(quat_gt[None, :])[0]
        rot_ax.add_patch(
            Circle(
                xy=hopf_gt[:2],
                radius=0.2,
                linewidth=2,
                facecolor="none",
                edgecolor=s1_cm.to_rgba(hopf_gt[2]),
            )
        )
    hopf_samples = quat_to_hopf(quat_samples)
    rot_ax.scatter(*hopf_samples.T[:2], s=2, color=s1_cm.to_rgba(hopf_samples.T[2]))
    plt.show()

    return fig
