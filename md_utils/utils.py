import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import scipy.constants as consts


def minimum_image_distance(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> float:
    """Compute minimum image distance between two points.
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
    """Computes kbt in kJ/mol from temperature in K."""
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
        kbt (float): thermal energy, default 1.0.
        plot (bool): whether to plot the free energy, default: False.
        axes (matplotlib.axes.Axes): axes to plot on, default: None -> makes new plot.
    Returns:
        fe (np.ndarray[2, nbins - 1]): free energy evaluated at the bin centers
            stacked with the bin centers [binc, fe].
    """
    hist, vals = np.histogram(traj, bins=bins)
    vals = (vals[1:] + vals[:-1]) / 2
    with np.errstate(divide="ignore"):
        fe = -kbt * np.log(hist)
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


def plot_fe_convergence(
    traj: np.ndarray,
    bins: int = 100,
    kbt: float = 1.0,
    n_fes: int = 10,
    label: str = r"$n_{\mathrm{data}}$",
    factor: float = 1.0,
    axes: plt.Axes = None,
):
    """Computes the free energy from a trajectory, using range(n_fes)/n_fes * len(data)
    data points. This helps to check if the free energy is converged.
    Arguments:
        traj (list or np.ndarray): trajectories, can be single or multiple.
        bins (int): number of bins for the histogram, default: 100.
        kbt (float): thermal energy, default 1.0.
        n_fes (int): number of free energies to plot.
        axes (matplotlib.axes.Axes): axes to plot on, default: None -> makes new plot.
        label (str): Label for colormap, default n_data
        factor (float): Conversion factor for colorbar numbers, could be time step
            default colorbar numbering are number of steps
    Returns:
        axes
    """
    if not isinstance(traj, np.ndarray):
        raise TypeError
    traj = traj.flatten()
    if axes is None:
        _fig, axes = plt.subplots(1, 1)
    len_chunk = int(np.ceil(len(traj) / n_fes))
    cm_base = plt.cm.nipy_spectral
    norm = plt.Normalize(vmin=0, vmax=len(traj) * factor)
    sm = mpl.cm.ScalarMappable(cmap=cm_base, norm=norm)
    cm = lambda x: sm.to_rgba(x)

    for i in range(1, n_fes + 1):
        fe = get_fe(traj[: i * len_chunk], bins=bins, kbt=kbt)
        axes.plot(*fe, color=cm(i * len_chunk * factor))
    cb = plt.colorbar(sm, ax=axes, label=label)
    cb.formatter.set_powerlimits((0, 0))
    return axes


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


import numpy as np
import h5py
from typing import Optional, Union


def get_fe_from_h5(
    file: Union[str, Path],
    dset_name: str,
    bins: int = 100,
    kbt: float = 1.0,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """Compute the free energy from a 1d trajectory in a  dataset in a h5py file.
    Never load more than chunk_size data at a ttime
    Arguments:
        file (str or Path): Path to the h5py file.
        dset_name (str): Name of the dataset to load traj from.
        bins (int): number of bins for the histogram, default: 100.
        kbt (float): thermal energy, default 1.0.
        chunk_size (int): Maximum number of data points to load at a time.
            default: None -> load all data at once.
    Returns:
        fe (np.ndarray[2, nbins - 1]): free energy evaluated at the bin centers
            stacked with the bin centers [bins, fe].
    """
    with h5py.File(file, "r") as fh:
        if chunk_size is None:
            chunk_size = len(fh[dset_name])
        n_chunks = int(np.ceil((len(fh[dset_name]) / chunk_size)))
        idxs_chunks = np.linspace(0, len(fh[dset_name]), n_chunks + 1, dtype=int)
        # Iterate over chunks to find min and max
        traj_min, traj_max = np.inf, -np.inf
        for idx_1, idx_2 in zip(idxs_chunks[:-1], idxs_chunks[1:]):
            chunk = fh[dset_name][idx_1:idx_2]
            traj_min = min(traj_min, np.min(chunk))
            traj_max = max(traj_max, np.max(chunk))
        # Compute histogram by iterating over chunks
        hist_sum = np.zeros(bins)
        for idx_1, idx_2 in zip(idxs_chunks[:-1], idxs_chunks[1:]):
            chunk = fh[dset_name][idx_1:idx_2]
            hist, values = np.histogram(chunk, bins=bins, range=(traj_min, traj_max))
            hist_sum += hist
    values = (values[1:] + values[:-1]) / 2
    with np.errstate(divide="ignore"):
        fe = -kbt * np.log(hist_sum)
    fe -= np.min(fe)
    return np.stack([values, fe], axis=0)
