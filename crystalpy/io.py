"""
I/O functions, to support the most popular chemical formats.
"""
import numpy as np
from crystalpy.model import Crystal
from openbabel import openbabel


def convert_from_openbabel(molecule: openbabel.OBMol) -> Crystal:
    atomic_numbers = []
    coordinates = []
    bonds = []
    for atom in openbabel.OBAtomAtomIter(molecule):
        atomic_numbers.append(atom.GetAtomicNum())
        vec = atom.GetVector()
        coordinates.append(
            [vec.GetX(), vec.GetY(), vec.GetZ()]
        )

    for bond in openbabel.OBMolBondIter(molecule):
        bonds.append(
            [bond.GetBeginAtom().GetId(), bond.GetEndAtom().GetId()]
        )
    return Crystal.create(
        atomic_number=np.asarray(atomic_numbers),
        bonds=np.asarray(bonds),
        coordinates=np.asarray(coordinates)
    )


def convert_to_openbabel(crystal: Crystal) -> openbabel.OBMol:
    molecule = openbabel.OBMol()
    for atom in crystal.get_atoms():
        ob_atom = molecule.NewAtom()
        ob_atom.SetAtomicNumber(atom.atomic_number)
        x, y, z = atom.coordinates
        ob_atom.SetVector(x, y, z)
    for bond in crystal.get_bonds():
        a, b = bond.a_id, bond.b_id
        molecule.AddBond(a+1, b+1, 1)  # NOTE: atoms indexed from 1
    return molecule


def save(file, crystal: Crystal):
    # TODO
    pass


def load(file) -> Crystal:
    pass
