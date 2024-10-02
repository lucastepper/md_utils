import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
import scipy.constants as consts
import h5py


def minimum_image_distance(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray):
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


def clean_fe(fe: np.ndarray, verbose=True) -> np.ndarray:
    """Clean up the free energy by removing any nans or infs and the bins to the left or right
    of that (depending on the position of the nan or inf). Also check that the gradient of the
    free energy has the right sign at the boundaries. If not, remove the boundary bins until
    it has.
    Arguments:
        fe (np.ndarray, 2d): free energy, stacked along dim 0 (values, fe).
        verbose (bool): whether to print out the number of bins removed, default: True.
    Returns:
        fe (np.ndarray, 2d): cleaned free energy, stacked along dim 0 (values, fe)
    """
    bins_start = fe.shape[1]
    to_remove = np.where(~np.isfinite(fe[1]))[0]
    fe_indexes = [0, fe.shape[1]]
    # now order the indexes by if they are closer to the left or right boundary
    for idx in to_remove:
        if idx < (fe.shape[1] / 2):
            fe_indexes[0] = max(fe_indexes[0], idx + 1)
        else:
            fe_indexes[1] = min(fe_indexes[1], idx)
    fe = fe[:, fe_indexes[0] : fe_indexes[1]]
    # check boundaries, left boundary sign should be negative, right positive such that a particle
    # rolls back into the well when outside the boundaries
    while np.sign(np.diff(fe[1])[0]) != -1:
        fe = fe[:, 1:]
    while np.sign(np.diff(fe[1])[-1]) != +1:
        fe = fe[:, :-1]
    if verbose:
        print(f"Removed {bins_start - fe.shape[1]} bins, of a total of {bins_start=}.")
    return fe


def to_list_ndarrays(traj: Union[list, np.ndarray]) -> list:
    """We assume that the input is either an np.ndarray or a list of np.ndarrays
    or a list of numbers. We want to return a list of np.ndarrays.
    """
    # check if and what kind of list we have
    if isinstance(traj, list):
        # assume we either have a list of numbers or list of np.ndarrays
        if not isinstance(traj[0], np.ndarray):
            traj = np.array(traj)
    # now, if we have a single ndarray wrap it in a list
    if isinstance(traj, np.ndarray):
        traj = [traj]
    return traj


def get_traj_range(traj: Union[list, np.ndarray]) -> tuple:
    """Get the range of the trajectory."""
    traj = to_list_ndarrays(traj)
    traj_min, traj_max = np.inf, -np.inf
    for t in traj:
        traj_min = min(traj_min, np.min(t))
        traj_max = max(traj_max, np.max(t))
    return traj_min, traj_max


def get_mass(trajs: Union[np.ndarray, list], dt: float, kbt: float):
    """Compute the mass for a given trajectory using the equipartition theorem <v^2> = kbt / m.
    Arguments:
        trajs: trajectories, uses half-step gradient to compute velocities
        dt: time step
        kbt: thermal energy
    Returns:
        mass
    """
    trajs = to_list_ndarrays(trajs)
    vels = [np.diff(x) / dt for x in trajs]
    sum_vsqrd = sum(np.sum(v**2) for v in vels)
    mean_vsqrd = sum_vsqrd / sum(len(v) for v in vels)
    return kbt / mean_vsqrd


def get_conditional_mass(
    trajs: Union[np.ndarray, list], dt: float, kbt: float, intervals: np.ndarray
):
    """Compute the conditional mass for a given trajectory using the equipartition theorem
    <v^2>_x = kbt / m. Return one mass for each region between intervals[i] and intervals[i + 1],
    resulting in len(intervals) - 1 masses.
    Arguments:
        trajs: trajectories, uses half-step gradient to compute velocities. Condition on the
            position one half step before.
        dt: time step
        kbt: thermal energy
        intervals: borders of regions in which to compute the mass.
            Assumes that intervals are ordered
    Returns:
        conditional mass. Entries for empty bins are nans
    """
    trajs = to_list_ndarrays(trajs)
    vel_squared = [(np.diff(x) / dt) ** 2 for x in trajs]
    sum_vsqrds = np.zeros(len(intervals) - 1)
    count_vsqrds = np.zeros(len(intervals) - 1)
    # iter through bins
    for ibin, (start, end) in enumerate(zip(intervals[:-1], intervals[1:])):
        for i, trj in enumerate(trajs):
            mask = (trj >= start) & (trj < end)
            sum_vsqrds[ibin] += np.sum(vel_squared[i][mask[:-1]])
            count_vsqrds[ibin] += np.sum(mask[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_vsqrd = sum_vsqrds / count_vsqrds
        return kbt / mean_vsqrd


def get_fe(
    traj: Union[list, np.ndarray],
    bins: int = 100,
    kbt: float = 1.0,
    plot: bool = False,
    axes=None,
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
    traj = to_list_ndarrays(traj)
    traj_range = get_traj_range(traj)
    hist_sum = np.zeros(bins)
    vals = np.linspace(*traj_range, bins + 1)  # type: ignore
    for trj in traj:
        hist, vls = np.histogram(trj, bins=bins, range=traj_range)
        np.testing.assert_allclose(vls, vals)
        hist_sum += hist
    vals = (vals[1:] + vals[:-1]) / 2
    with np.errstate(divide="ignore"):
        fe = -kbt * np.log(hist_sum)
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
    axes=None,
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
    cm_base = plt.cm.nipy_spectral  # type: ignore
    norm = plt.Normalize(vmin=0, vmax=len(traj) * factor)  # type: ignore
    sm = mpl.cm.ScalarMappable(cmap=cm_base, norm=norm)  # type: ignore
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


def get_fe_from_h5(
    file: Union[str, Path],
    dset_name: str,
    bins: int = 100,
    kbt: float = 1.0,
    chunk_size: Optional[int] = None,
    frames_equil: int = 0,
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
        frames_equil (int): Number of frames to discard at the beginning of the trajectory.
    Returns:
        fe (np.ndarray[2, nbins - 1]): free energy evaluated at the bin centers
            stacked with the bin centers [bins, fe].
    """
    with h5py.File(file, "r") as fh:
        if chunk_size is None:
            chunk_size = len(fh[dset_name])  # type: ignore
        n_chunks = int(np.ceil((len(fh[dset_name]) / chunk_size)))  # type: ignore
        idxs_chunks = np.linspace(frames_equil, len(fh[dset_name]), n_chunks + 1, dtype=int)  # type: ignore
        # Iterate over chunks to find min and max
        traj_min, traj_max = np.inf, -np.inf
        for idx_1, idx_2 in zip(idxs_chunks[:-1], idxs_chunks[1:]):
            chunk = fh[dset_name][idx_1:idx_2]  # type: ignore
            traj_min = min(traj_min, np.min(chunk))  # type: ignore
            traj_max = max(traj_max, np.max(chunk))  # type: ignore
        # Compute histogram by iterating over chunks
        hist_sum = np.zeros(bins)
        for idx_1, idx_2 in zip(idxs_chunks[:-1], idxs_chunks[1:]):
            chunk = fh[dset_name][idx_1:idx_2]  # type: ignore
            hist, values = np.histogram(chunk, bins=bins, range=(traj_min, traj_max))  # type: ignore
            hist_sum += hist
    values = (values[1:] + values[:-1]) / 2  # type: ignore
    with np.errstate(divide="ignore"):
        fe = -kbt * np.log(hist_sum)
    fe -= np.min(fe)
    return np.stack([values, fe], axis=0)


def plot_histograms(
    traj: np.ndarray,
    dt: float,
    mass: float,
    kbt: float = 2.494,
    kharm: Optional[float] = None,
    fe: Optional[np.ndarray] = None,
    axes=None,
):
    """Plot velocity distribution and compare to reference gaussian.
    Plot the histogram of the trajectory and compare it either to a gaussian (kharm is not None)
        or to the distribution of the free energy (fe is not None).
    Arguments:
        traj (np.ndarray): trajectory to plot.
        dt (float): time step of the trajectory.
        mass (float): mass of the particle.
        kbt (float): kbt, default: 2.494 kJ/mol.
        kharm (float): harmonic constant, default: None.
        fe (np.ndarray): free energy to plot, default: None.
        axes (matplotlib.axes.Axes): axes to plot on, default: None -> makes new plot.
    """
    if axes is None:
        _fig, axes = plt.subplots(1, 2)
    # define functions to plot the ref, called given the hist bins, returns the ref disttribution
    ref_dist_funcs = [lambda x: norm.pdf(x, loc=0.0, scale=np.sqrt(kbt / mass))]
    if kharm is not None:
        ref_dist_funcs.append(lambda x: norm.pdf(x, loc=0.0, scale=np.sqrt(kharm / mass)))
    elif fe is not None:
        ref_dist = np.exp(-fe[1] / np.trapz(fe[1], fe[0]))
        ref_dist_funcs.append(lambda x: np.interp(x, fe[0], ref_dist))
    else:
        ref_dist_funcs.append(lambda x: np.nan * np.ones_like(x))
    # plot the histogram with the reference distributions
    for data, ref_dist_func, ax in zip([traj, np.diff(traj) / dt], ref_dist_funcs, axes):
        hist, values = np.histogram(traj, bins=200, density=True)
        values = (values[1:] + values[:-1]) / 2
        ax.plot(values, hist, label="histogram")
        ax.plot(values, ref_dist_func(values), label="ref")
    return axes


def plot_cond_mass(
    trajs: Union[np.ndarray, list[np.ndarray]],
    bins: list[float],
    cond_mass_ref: list[float],
    dt: float,
    kbt: float,
    width: float,
    vels: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    axes: Optional[plt.Axes] = None,
    kwargs_hist: Optional[dict] = {},
    kwargs_ref: Optional[dict] = {},
):
    """Compute the conditional mass via conditional mean vsqrd and compare to expectation
    Arguments
        trajs: trajs to compute, 1 array list of arrays, 2darray
        bins: positions where to compute cond_mass, computes conditioned on
            (x > bin_i - width / 2) & (x <= bin_i + width / 2)
        cond_mass_ref: reference values for mass, computes the conditional mass around
        dt: time step for gradient of trajs, uses central differences
        kbt: thermal energy
        width: width of bons for conitional means
        vels: if given, computes the masses from these velocities and not the gradient of trajs
            needs to have the same shape as trajs
        axes: if given, plots onto this axes. needs to have the same shape as cond_mass_ref.
            if not given, makes new fig, axes and returns them
        kwargs_hist: passed to plt.plot for histogram
        kwargs_ref: passed to plt.plot for reference
    Returns
        fig, axes
    """
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, len(bins), figsize=(20, 5), sharey=True)
        fig.subplots_adjust(wspace=0)
    if isinstance(trajs, np.ndarray) and trajs.ndim == 1:
        trajs = trajs[None, :]
    elif not isinstance(trajs, list):
        trajs = [trajs]
    if vels is None:
        vels = [np.gradient(traj, dt) for traj in trajs]
    elif isinstance(vels, np.ndarray) and vels.ndim == 1:
        vels = vels[None, :]
    elif not isinstance(vels, list):
        vels = [vels]
    count_by_bin = []
    cond_masses_calc = []
    if not isinstance(width, list):
        width = len(bins) * [width]
    assert len(width) == len(bins)
    for i in range(len(bins)):
        sum_vsqrd = 0
        count_vsqrd = 0
        vels_masked = []
        for vel, traj in zip(vels, trajs):
            mask = (traj > (bins[i] - width[i] / 2.0)) & (traj <= (bins[i] + width[i] / 2.0))
            sum_vsqrd += np.sum(vel[mask] ** 2)
            count_vsqrd += np.sum(mask)
            vels_masked.append(vel[mask])
        cond_masses_calc.append(kbt / (sum_vsqrd / count_vsqrd))
        count_by_bin.append(count_vsqrd)
        kwargs_hist = {"bins": 100} | kwargs_hist
        hist, vals = np.histogram(
            np.concatenate(vels_masked), bins=kwargs_hist.get("bins", None), density=True
        )
        vals = 0.5 * (vals[1:] + vals[:-1])
        kwargs_hist.pop("bins")
        axes[i].plot(vals, hist, **kwargs_hist)
        axes[i].plot(vals, norm.pdf(vals, scale=np.sqrt(kbt / cond_mass_ref[i])), **kwargs_ref)
        axes[i].set_title(f"x={bins[i]:.2f}")
    print(f"Num per bin: " + ", ".join(f"{n:.2e}" for n in count_by_bin))
    print(f"Ref  masses: " + ", ".join(f"{m:.2e}" for m in cond_mass_ref))
    print(f"Calc masses: " + ", ".join(f"{m:.2e}" for m in cond_masses_calc))
    return fig, axes
