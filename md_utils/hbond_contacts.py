from collections import defaultdict
from pathlib import Path
import re
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np


def get_hbonds(
    u_sim: mda.Universe,
    d_cutoff: float = 3.5,
    residue_cutoff: int = 1,
    ignore_same_chain: bool = False,
    save_file: str = "hbonds.npz",
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Save all contacts between N and O atoms residues that are n residues
    apart or on different chains. Only consider N-O pairs that are within d_cutoff.
    Arguments:
        u_sim: MDAnalysis Universe object
        d_cutoff: float, cutoff distance for N-O pairs
        residue_cutoff: int, minimum difference in residue index for N-O pairs
            default 1, i.e. only consider N-O pairs that are not in the same residue
        ignore_same_chain: bool, ignore N-O pairs that are in the same chain
            default False, i.e. consider all N-O pairs
        save_file: str, file to save results to; default hbonds.npz
        overwrite: bool, overwrite save_file if it exists; default False
    Returns:
        (hbond_names, hbond_idxs): tuple of np.ndarrays, indices and names of hbonds
    """
    nitrogens = u_sim.select_atoms("element N")
    oxygens = u_sim.select_atoms("element O")
    distances = distance_array(nitrogens.positions, oxygens.positions, box=u_sim.dimensions)
    contacts = []
    contact_idxs = []
    for i in range(distances.shape[0]):
        for j in range(i + 1, distances.shape[1]):
            if distances[i][j] <= d_cutoff:
                atom_n = nitrogens[i]
                atom_o = oxygens[j]
                if atom_n.segid == atom_o.segid:
                    if ignore_same_chain:
                        continue
                    if abs(atom_n.resid - atom_o.resid) < residue_cutoff:
                        continue
                ids = ["name", "resname", "resid", "segid"]
                contact_ident = " and ".join([f"{i} {getattr(atom_n, i)}" for i in ids])
                contact_ident += " && " + " and ".join([f"{i} {getattr(atom_o, i)}" for i in ids])
                contacts.append(contact_ident)
                contact_idxs.append([atom_n.index, atom_o.index])
    contacts = np.array(contacts)
    contact_idxs = np.array(contact_idxs)
    if save_file:
        if Path(save_file).exists() and not overwrite:
            raise FileExistsError(f"{save_file} already exists. Set overwrite=True to overwrite.")
        np.savez(save_file, contacts=contacts, contact_idxs=contact_idxs)
    return contacts, contact_idxs


def get_interchain_contact_motives(
    contacts: np.ndarray, print_results: bool = False
) -> dict[str, list[tuple[str, str]]]:
    """Look at all contacts and see which residue pairs connect
    the most chains.
    Arguments:
        contacts: np.ndarray, array of contact identifiers
        print_results: bool, print the results in abbreviated form
    Returns:
        interchain_contact_motives: dict, keys are residue pairs,
            values are list of tuple of chains
    """
    # find all residue pairs that connect by replacint segid with XX
    contact_motives = defaultdict(list)
    for contact in contacts:
        segments = re.findall(r"segid (\w+)", contact)
        if not len(segments) == 2:
            raise ValueError(f"Found {len(segments)} segments in {contact}")
        contact_no_seg = re.sub(r"segid \w+", "segid XX", contact)
        contact_motives[contact_no_seg].append(tuple(segments))
    # print the results in the form of resname1resid1-resname2resid2: n
    if print_results:
        to_print_names, to_print_lens = [], []
        for key, item in contact_motives.items():
            resnames = re.findall(r"resname (\w+)", key)
            resids = re.findall(r"resid (\d+)", key)
            to_print_names.append(f"{resnames[0]}{resids[0]}-{resnames[1]}{resids[1]}")
            to_print_lens.append(len(item))
        idxs = np.argsort(to_print_lens)[::-1]
        for name, len_item in zip(np.array(to_print_names)[idxs], np.array(to_print_lens)[idxs]):
            print(f"{name}: {len_item}")
    return contact_motives
