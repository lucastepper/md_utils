import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.constants as consts


def minimum_image_distance(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> float:
    """ Compute minimum image distance between two points.
    Args:
        pos1 (np.ndarray(ndim)): Position of first point.
        pos2 (np.ndarray(ndim)): Position of second point.
        box (np.ndarray(ndim)): Box vectors.
    Returns:
        float: Minimum image distance between the two points.
    """
    delta = pos1 - pos2
    delta -= np.rint(delta / box) * box
    return np.linalg.norm(delta)


def get_kbt(temp: float) -> float:
    """ Computes kbt in kJ/mol from temperature in K. """
    kbt_joule = temp * consts.k * consts.N_A
    return kbt_joule / 1000


def save_ndx(save_dir, name, idxs, len_line=80):
    """Save a list of indexes (1-based) to file in gmx ndx format."""
    idxs = list(idxs)
    assert all(isinstance(idx, int) for idx in idxs), "Indexes need to be integers"
    save_name = os.path.join(save_dir, name + ".ndx")
    # pylint: disable=unused-variable
    len_per_idx = max(len(str(idx)) for idx in idxs)
    lines = [f"[ {name} ]\n"]
    line = ""
    while len(idxs) > 0:
        item_to_add = f"{str(idxs.pop(0)).rjust(len_per_idx)} "
        if len(line + item_to_add) < len_line:
            line += item_to_add
        else:
            lines.append(line.rstrip(" ") + "\n")
            line = item_to_add
    lines.append(line.rstrip(" ") + "\n")
    with open(save_name, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def get_fe(
    traj: Union[list, np.ndarray],
    bins: int = 100,
    kbt: float = 1.0,
    plot: bool = False,
    axes: plt.Axes = None,
):
    """Computes the free energy from a trajectory. Optionally plots it.
    Arguments:
        traj (list or np.ndarray): trajectories, can be single or multiple.
        bins (int): number of bins for the histogram, default: 100.
        plot (bool): whether to plot the free energy, default: False.
        axes (matplotlib.axes.Axes): axes to plot on, default: None -> makes new plot.
    Returns:
        fe (np.ndarray[2, nbins - 1]): free energy evaluated at the bin centers
            stacked with the bin centers [binc, fe].
    """
    hist, vals = np.histogram(traj, bins=bins)
    vals = (vals[1:] + vals[:-1]) / 2
    fe = np.zeros_like(hist) * np.nan
    fe[hist != 0] -= kbt * np.log(hist[hist != 0])
    fe -= np.min(fe)
    # Plot
    if plot:
        if axes is None:
            _fig, axes = plt.subplots(1, 1)
        axes.plot(vals, fe)
        axes.set_xlabel(r"$x$ (nm)")
        axes.set_ylabel(r"$F / k_{\mathrm{B}}T$")
        axes.set_ylim(-0.4, 4)
        plt.show()
    return np.stack([vals, fe], axis=0)


def plot_vel(
    vel: Union[list, np.ndarray], mass: float, kbt: float = 2.494, bins: int = 100, axes=None
):
    """Plot velocity distribution and compare to reference
    Arguments:
        vel (list or np.ndarray): velocities, can be single or multiple trajs.
        mass (float): mass of the particle.
        kbt (float): kbt, default: 2.494 kJ/mol.
        bins (int): number of bins for the histogram, default: 100.
        axes (matplotlib.axes.Axes): axes to plot on, default: None -> makes new plot.
    Returns:
        axes (matplotlib.axes.Axes) The axes passed in or the new axes.
    """
    if not isinstance(vel, np.ndarray):
        vel = np.array(vel)
    hist, values = np.histogram(vel, bins=bins, density=True)
    values = (values[1:] + values[:-1]) / 2
    if not axes:
        _fig, axes = plt.subplots(1, 1)
    axes.plot(values, hist, label="histogram")
    axes.plot(values, norm.pdf(values, loc=0.0, scale=np.sqrt(kbt / mass)), label="ref")
    return axes
