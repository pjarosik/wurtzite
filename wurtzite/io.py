"""
I/O functions, to support the most popular chemical formats.
"""
import os.path

import numpy as np
import sys
from wurtzite.model import Molecule
from openbabel import openbabel


def convert_from_openbabel(molecule: openbabel.OBMol) -> Molecule:
    atomic_numbers = []
    coordinates = []
    bonds = []
    n_atoms = molecule.NumAtoms()
    for i, atom in enumerate(openbabel.OBMolAtomIter(molecule)):
        sys.stdout.write(f"\rReading atom: {(i+1)}/{n_atoms}")
        atomic_numbers.append(atom.GetAtomicNum())
        vec = atom.GetVector()
        coordinates.append(
            [vec.GetX(), vec.GetY(), vec.GetZ()]
        )

    n_bonds = molecule.NumBonds()
    for i, bond in enumerate(openbabel.OBMolBondIter(molecule)):
        sys.stdout.write(f"\rReading bond: {(i+1)}/{n_bonds}")
        bonds.append(
            [bond.GetBeginAtom().GetId(), bond.GetEndAtom().GetId()]
        )
    return Molecule.create(
        atomic_number=np.asarray(atomic_numbers),
        bonds=np.asarray(bonds),
        coordinates=np.asarray(coordinates)
    )


def convert_to_openbabel(crystal: Molecule) -> openbabel.OBMol:
    molecule = openbabel.OBMol()
    for atom in crystal.get_atoms():
        ob_atom = molecule.NewAtom()
        ob_atom.SetAtomicNum(int(atom.atomic_number))
        x, y, z = atom.coordinates
        ob_atom.SetVector(x, y, z)
    for bond in crystal.get_bonds():
        a, b = bond.a_id, bond.b_id
        molecule.AddBond(int(a)+1, int(b)+1, 1)  # NOTE: atoms indexed from 1
    return molecule


def save(file: str, crystal: Molecule):
    """
    Saves crystal to the given file.
    The output file format will be determined based on the file extensions.
    """
    molecule: openbabel.OBMol = convert_to_openbabel(crystal)
    conversion = openbabel.OBConversion()
    output_format = os.path.splitext(file)[1]
    conversion.SetOutFormat(output_format)
    return conversion.WriteFile(molecule, file)


def load(file: str) -> Molecule:
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
