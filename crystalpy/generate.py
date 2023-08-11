"""
A module with utilities to generate crystalpy Crystals.
"""
import dataclasses

import numpy as np

from crystalpy.model import Crystal, Molecule
from crystalpy.definitions import UnitCellDef, get_cell_by_name
from crystalpy.io import convert_from_openbabel, convert_to_openbabel
from typing import Tuple, Union


def create_lattice(
        dimensions: Tuple[int, int, int],
        cell: Union[UnitCellDef, str],
        set_bonds: bool = True
) -> Crystal:
    """
    Creates a crystal with atoms located on a lattice.

    :param dimensions: number of cells in (OX, OY, OZ)
    :param cell: unit cell definition (if str, definition look up table
        will be used)
    :param set_bonds: should be set to True, if bonds should be generated
        for the lattice
    """
    if isinstance(cell, str):
        cell = get_cell_by_name(cell)

    nx, ny, nz = dimensions

    # Generate lattice in the cell coordinates
    p = np.array(np.meshgrid(*[range(d) for d in dimensions], indexing="ij"))
    p = p.T
    p = p.reshape((-1, 1, 3))
    c = cell.coordinates.reshape((1, -1, 3))
    atom_coords_cell = p + c  # (n cells, n atoms per cell, 3)
    atom_coords_cell = atom_coords_cell.reshape(-1, 3)  # (n atoms, 3)
    # Get cartesian coordinates
    coordinates = cell.miller_to_cartesian.dot(atom_coords_cell.T).T
    symbol = cell.atoms
    symbol = symbol*(nx*ny*nz)
    crystal = Crystal.create(
        symbol=symbol,
        bonds=np.asarray([]),
        coordinates=coordinates,
        cell=cell
    )
    if set_bonds:
        bonds = create_bonds(crystal)
        crystal = dataclasses.replace(
            crystal,
            bonds=bonds
        )
    return crystal


def create_bonds(input_molecule: Molecule) -> np.ndarray:
    molecule = convert_to_openbabel(input_molecule)
    molecule.ConnectTheDots()
    molecule.PerceiveBondOrders()
    output_molecule = convert_from_openbabel(molecule)
    molecule.Clear()
    return output_molecule.bonds

