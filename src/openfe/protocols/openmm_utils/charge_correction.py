# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Shared utilities for explicit charge correction in alchemical protocols.

These helpers select water molecules suitable for alchemical transformation
into counterions and retrieve the necessary force field parameters. They are
protocol-independent and can be used by the hybrid topology RFE protocol,
the separated topologies (SepTop) protocol, and the absolute free energy
(AFE) protocols.

Sign convention for RFE / SepTop (hybrid topology)
---------------------------------------------------
``charge_difference`` is defined as ``formal_charge(stateA) - formal_charge(stateB)``.

* charge_difference > 0: stateB is more negative → couple a **negative** ion
  (e.g. Cl-) alongside stateB to maintain neutrality.
* charge_difference < 0: stateB is more positive → couple a **positive** ion
  (e.g. Na+) alongside stateB to maintain neutrality.

The mapping to ion type is:
    {+1: negative_ion, -1: positive_ion}[charge_difference]

Sign convention for AFE (absolute binding)
------------------------------------------
For a disappearing ligand with formal charge ``q``, the correction ion must
carry charge ``q`` to replace the charge being annihilated:

* q > 0 → grow a **positive** ion (e.g. Na+)
* q < 0 → grow a **negative** ion (e.g. Cl-)

This keeps the total system charge constant: as the ligand electrostatics
are turned off, the correction species gains the same charge.
"""

import logging

import mdtraj as mdt
import numpy as np
import numpy.typing as npt
from mdtraj.core.residue_names import _SOLVENT_TYPES
from openff.units import Quantity, unit
from openmm import NonbondedForce, System, app

logger = logging.getLogger(__name__)


def get_alchemical_waters(
    topology: app.Topology,
    positions: npt.NDArray,
    charge_difference: int,
    distance_cutoff: Quantity = 0.8 * unit.nanometer,
) -> list[int]:
    """
    Pick water residue(s) to be used for alchemical charge correction.

    Selects water molecules that are further than ``distance_cutoff`` from all
    non-solvent atoms. Returns one water residue index per unit of absolute
    charge difference.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology to search for an alchemical water.
    positions : npt.NDArray
        The coordinates of the atoms associated with ``topology``, in
        nanometers, shape (n_atoms, 3).
    charge_difference : int
        The charge difference between the two end states,
        calculated as formal_charge(stateA) - formal_charge(stateB).
    distance_cutoff : openff.units.Quantity
        The minimum distance away from solutes from which an alchemical
        water can be chosen. Default 0.8 nm.

    Returns
    -------
    chosen_residues : list[int]
        Residue indices for each chosen alchemical water.

    Raises
    ------
    ValueError
        If there are no qualifying water molecules outside the cutoff distance.

    Notes
    -----
    Based on perses.utils.charge_changing.get_water_indices.
    The first qualifying water (by residue index) is chosen for determinism.
    """
    if charge_difference == 0:
        return []

    traj = mdt.Trajectory(
        positions[np.newaxis, ...],
        mdt.Topology.from_openmm(topology),
    )

    water_atoms = traj.topology.select("water")
    solvent_residue_names = list(_SOLVENT_TYPES)
    solute_atoms = [
        atom.index
        for atom in traj.topology.atoms
        if atom.residue.name not in solvent_residue_names
    ]

    excluded_waters = mdt.compute_neighbors(
        traj,
        distance_cutoff.to(unit.nanometer).m,
        solute_atoms,
        haystack_indices=water_atoms,
        periodic=True,
    )[0]

    solvent_indices = sorted(
        {
            atom.residue.index
            for atom in traj.topology.atoms
            if (atom.index in water_atoms) and (atom.index not in excluded_waters)
        }
    )

    if len(solvent_indices) < 1:
        errmsg = (
            "There are no waters outside of a "
            f"{distance_cutoff.to(unit.nanometer)} nanometer distance "
            "of the system solutes to be used as alchemical waters"
        )
        raise ValueError(errmsg)

    chosen_residues = solvent_indices[: abs(charge_difference)]
    return chosen_residues


def get_ion_and_water_parameters(
    topology: app.Topology,
    system: System,
    ion_resname: str,
    water_resname: str = "HOH",
):
    """
    Get ion and water (oxygen/hydrogen) NonbondedForce parameters.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology containing at least one ion and one water residue.
    system : openmm.System
        The system associated with the topology.
    ion_resname : str
        The residue name of the ion (e.g. 'NA', 'CL').
    water_resname : str
        The residue name of the water. Default 'HOH'.

    Returns
    -------
    ion_charge : openmm charge quantity
    ion_sigma : openmm length quantity
    ion_epsilon : openmm energy quantity
    o_charge : openmm charge quantity
    h_charge : openmm charge quantity

    Raises
    ------
    ValueError
        If no residue matching the requested names is found.
    """

    def _find_atom(topology, resname, elementname):
        for atom in topology.atoms():
            if atom.residue.name == resname:
                if elementname is None or atom.element.symbol == elementname:
                    return atom.index
        errmsg = (
            "Error encountered when attempting to explicitly handle "
            "charge changes using an alchemical water. No residue "
            f"named: {resname} found, with element {elementname}"
        )
        raise ValueError(errmsg)

    ion_index = _find_atom(topology, ion_resname, None)
    oxygen_index = _find_atom(topology, water_resname, "O")
    hydrogen_index = _find_atom(topology, water_resname, "H")

    nbf = [i for i in system.getForces() if isinstance(i, NonbondedForce)][0]

    ion_charge, ion_sigma, ion_epsilon = nbf.getParticleParameters(ion_index)
    o_charge, _, _ = nbf.getParticleParameters(oxygen_index)
    h_charge, _, _ = nbf.getParticleParameters(hydrogen_index)

    return ion_charge, ion_sigma, ion_epsilon, o_charge, h_charge


def get_water_atom_indices(
    topology: app.Topology,
    water_resids: list[int],
) -> list[int]:
    """
    Get atom indices for the given water residue indices.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology to search.
    water_resids : list[int]
        Residue indices of waters.

    Returns
    -------
    atom_indices : list[int]
        All atom indices belonging to the specified water residues.

    Raises
    ------
    ValueError
        If any water has more than 3 atoms (virtual sites not supported).
    """
    atom_indices = []
    for res in topology.residues():
        if res.index in water_resids:
            atoms = list(res.atoms())
            if len(atoms) > 3:
                raise ValueError(
                    "Non 3-site waters (i.e. waters with virtual sites) "
                    "are not currently supported as alchemical waters"
                )
            atom_indices.extend(at.index for at in atoms)
    return atom_indices


def transform_waters_to_ions_inplace(
    topology: app.Topology,
    system: System,
    water_resids: list[int],
    charge_difference: int,
    solvent_component,
) -> None:
    """
    In-place modify NonbondedForce parameters of water atoms to ion parameters.

    For positive charge_difference (stateB more negative), the water oxygen gets
    negative ion (e.g. Cl-) parameters. For negative charge_difference (stateB
    more positive), the water oxygen gets positive ion (e.g. Na+) parameters.
    Hydrogen atoms get their charges zeroed.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology containing the waters to transform.
    system : openmm.System
        The system whose NonbondedForce will be modified in place.
    water_resids : list[int]
        Residue indices of waters to transform.
    charge_difference : int
        formal_charge(stateA) - formal_charge(stateB).
    solvent_component : openfe.SolventComponent
        Provides ion residue names (positive_ion, negative_ion).

    Raises
    ------
    ValueError
        If number of water residues doesn't match abs(charge_difference).
        If a water has virtual sites (>3 atoms).
    """
    if charge_difference == 0:
        return

    if abs(charge_difference) != len(water_resids):
        raise ValueError(
            f"Expected {abs(charge_difference)} alchemical water residue(s), "
            f"got {len(water_resids)}"
        )

    if charge_difference > 0:
        ion_resname = solvent_component.negative_ion.strip("-+").upper()
    else:
        ion_resname = solvent_component.positive_ion.strip("-+").upper()

    ion_charge, ion_sigma, ion_epsilon, o_charge, h_charge = (
        get_ion_and_water_parameters(topology, system, ion_resname, "HOH")
    )

    nbfrcs = [i for i in system.getForces() if isinstance(i, NonbondedForce)]
    if len(nbfrcs) != 1:
        raise ValueError(
            f"Expected exactly 1 NonbondedForce, found {len(nbfrcs)}"
        )
    nbf = nbfrcs[0]

    for res in topology.residues():
        if res.index in water_resids:
            atoms = list(res.atoms())
            if len(atoms) > 3:
                raise ValueError(
                    "Non 3-site waters (i.e. waters with virtual sites) "
                    "are not currently supported as alchemical waters"
                )
            for at in atoms:
                idx = at.index
                charge, sigma, epsilon = nbf.getParticleParameters(idx)
                if charge == o_charge:
                    nbf.setParticleParameters(idx, ion_charge, ion_sigma, ion_epsilon)
                else:
                    if charge != h_charge:
                        raise ValueError(
                            "Modifying an atom that doesn't match known water parameters"
                        )
                    nbf.setParticleParameters(idx, 0.0, sigma, epsilon)


def apply_afe_charge_correction_offsets(
    system: System,
    topology: app.Topology,
    water_resids: list[int],
    ligand_charge: int,
    solvent_component,
    parameter_name: str = "lambda_charge_correction",
) -> dict:
    """
    Add ``NonbondedForce`` particle parameter offsets that smoothly
    transform selected water molecule(s) into counterion(s), controlled
    by a global ``lambda`` parameter.

    This is designed for absolute free energy (AFE) protocols where a
    charged ligand is annihilated.  The correction ion carries the **same
    sign** as the disappearing ligand charge so that the total system
    charge remains constant across the lambda schedule.

    At ``parameter_name = 0`` the correction atoms behave as normal water
    (state A, ligand fully interacting).
    At ``parameter_name = 1`` the oxygen has ion parameters and hydrogen
    charges are zero (state B, ligand electrostatics annihilated).

    Parameters
    ----------
    system : openmm.System
        The (alchemical) system to modify **in place**.
    topology : openmm.app.Topology
        Topology matching the system.
    water_resids : list[int]
        Residue indices of waters to transform (from
        :func:`get_alchemical_waters`).
    ligand_charge : int
        The formal charge of the disappearing ligand.  ``+1`` grows a
        cation; ``-1`` grows an anion.
    solvent_component
        ``SolventComponent`` providing ``positive_ion`` /
        ``negative_ion`` residue names.
    parameter_name : str
        Name of the global parameter added to the ``NonbondedForce``.

    Returns
    -------
    metadata : dict
        ``ion_resname``, ``correction_atom_indices``, and
        ``water_residue_indices`` for downstream bookkeeping.

    Raises
    ------
    ValueError
        If ``ligand_charge`` is zero (no correction needed), if the
        number of waters does not match ``abs(ligand_charge)``, if
        there is not exactly one ``NonbondedForce``, or if a water
        atom has unexpected parameters.
    """
    if ligand_charge == 0:
        raise ValueError(
            "No charge correction needed for a neutral ligand"
        )

    if abs(ligand_charge) != len(water_resids):
        raise ValueError(
            f"Expected {abs(ligand_charge)} alchemical water residue(s), "
            f"got {len(water_resids)}"
        )

    if ligand_charge > 0:
        ion_resname = solvent_component.positive_ion.strip("-+").upper()
    else:
        ion_resname = solvent_component.negative_ion.strip("-+").upper()

    ion_charge, ion_sigma, ion_epsilon, o_charge, h_charge = (
        get_ion_and_water_parameters(topology, system, ion_resname, "HOH")
    )

    nbfrcs = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    if len(nbfrcs) != 1:
        raise ValueError(
            f"Expected exactly 1 NonbondedForce, found {len(nbfrcs)}"
        )
    nbf = nbfrcs[0]

    nbf.addGlobalParameter(parameter_name, 0.0)

    correction_atom_indices = []
    for res in topology.residues():
        if res.index in water_resids:
            atoms = list(res.atoms())
            if len(atoms) > 3:
                raise ValueError(
                    "Non 3-site waters (i.e. waters with virtual sites) "
                    "are not currently supported as alchemical waters"
                )
            for at in atoms:
                idx = at.index
                correction_atom_indices.append(idx)
                charge, sigma, epsilon = nbf.getParticleParameters(idx)
                if charge == o_charge:
                    nbf.addParticleParameterOffset(
                        parameter_name,
                        idx,
                        ion_charge - charge,
                        ion_sigma - sigma,
                        ion_epsilon - epsilon,
                    )
                elif charge == h_charge:
                    zero_charge = 0.0 * charge.unit
                    zero_len = 0.0 * sigma.unit
                    zero_energy = 0.0 * epsilon.unit
                    nbf.addParticleParameterOffset(
                        parameter_name,
                        idx,
                        zero_charge - charge,
                        zero_len,
                        zero_energy,
                    )
                else:
                    raise ValueError(
                        "Modifying an atom that doesn't match known "
                        "water parameters"
                    )

    return {
        "ion_resname": ion_resname,
        "correction_atom_indices": correction_atom_indices,
        "water_residue_indices": water_resids,
    }
