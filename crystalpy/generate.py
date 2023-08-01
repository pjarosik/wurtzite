"""
A module with utilities to generate crystalpy Crystals.
"""
import numpy as np

from crystalpy.model import Crystal
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
    coordinates = cell.cartesian_coordinates
    symbol = cell.atoms
    #
    nx, ny, nz = dimensions
    cx, cy, cz = cell.dimensions  # dimensions of a single cell
    coordinates = np.repeat(coordinates[np.newaxis, ...], nz, axis=0)
    coordinates = np.repeat(coordinates[np.newaxis, ...], ny, axis=0)
    coordinates = np.repeat(coordinates[np.newaxis, ...], nx, axis=0)
    # (nx, ny, nz, natoms, 3)
    # Coordinate (0,0,0) of each cell.
    x_cell_coords = (np.arange(nx)-nx//2)*cx
    y_cell_coords = (np.arange(ny)-ny//2)*cy
    z_cell_coords = (np.arange(nz)-nz//2)*cz
    x_cell_coords = x_cell_coords.reshape(-1, 1, 1, 1)  # (nx, ny, nz, natoms)
    y_cell_coords = y_cell_coords.reshape(1, -1, 1, 1)  # (nx, ny, nz, natoms)
    z_cell_coords = z_cell_coords.reshape(1, 1, -1, 1)  # (nx, ny, nz, natoms)
    # Move coordinates
    coordinates[:, :, :, :, 0] += x_cell_coords
    coordinates[:, :, :, :, 1] += y_cell_coords
    coordinates[:, :, :, :, 2] += z_cell_coords
    coordinates = coordinates.reshape(-1, 3)
    # Symbols:
    symbol = symbol*(nx*ny*nz)
    crystal = Crystal.create(
        symbol=symbol,
        bonds=np.asarray([]),
        coordinates=coordinates
    )
    if set_bonds:
        crystal = create_bonds(crystal)
    return crystal


def create_bonds(input_crystal: Crystal) -> Crystal:
    molecule = convert_to_openbabel(input_crystal)
    molecule.ConnectTheDots()
    molecule.PerceiveBondOrders()
    crystal = convert_from_openbabel(molecule)
    molecule.Clear()
    return crystal

