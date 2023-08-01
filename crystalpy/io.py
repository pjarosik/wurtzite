"""
I/O functions, to support the most popular chemical formats.
"""
import os.path

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


def save(file: str, crystal: Crystal):
    """
    Saves crystal to the given file.
    The output file format will be determined based on the file extensions.
    """
    molecule: openbabel.OBMol = convert_to_openbabel(crystal)
    conversion = openbabel.OBConversion()
    output_format = os.path.splitext(file)[1]
    conversion.SetOutFormat(output_format)
    return conversion.WriteFile(molecule, file)


def load(file: str) -> Crystal:
    """
    Reads crystal from the given input file.
    The input file format will be automatically determined based on the
    file extensions.
    """
    input_format = os.path.splitext(file)[1]
    conversion = openbabel.OBConversion()
    conversion.SetInFormat(input_format)
    molecule = openbabel.OBMol()
    conversion.ReadFile(molecule, file)
    return convert_from_openbabel(molecule)
