from dataclasses import dataclass
from typing import Tuple, Union, List

import numpy as np
from crystalpy.dictionary import get_atoms_by_symbols


@dataclass(frozen=True)
class Atom:
    id: int
    atomic_number: int
    coordinates: Union[Tuple[float, float, float], np.ndarray]


@dataclass(frozen=True)
class Bond:
    """
    Undirected edge between two atoms.

    id is the id of the bond.

    a_id and b_id are the atom indices.
    """
    id: int
    a_id: int
    b_id: int


@dataclass(frozen=True)
class Crystal:
    """
    An instance of crystal structure.
    The crystal structure contains information about the physical arrangement
    of atoms, types of atoms (currently by atomic number) and bonds (the
    complete graph).

    The distances are assumed to be in Angstroms [A].

    :param atomic_number: a list of atomic numbers  (number of atoms, )
    :param coordinates: a nd-array with dimensions: (number of atoms, 3), where
        3 are the coordinates (x, y, z) of i-th atom. [A]
    :param bonds: a list of connect atom pairs (a, b), where
        a and b are atom indices array (number of bonds, 2)
    """
    atomic_number: np.ndarray
    bonds: np.ndarray
    coordinates: np.ndarray

    def __post_init__(self):
        if not (self.atomic_number.shape[0] == self.coordinates.shape[0]):
            raise ValueError("Atomic number and coordinate arrays "
                             "should have the same size for "
                             "for the first dimension.")

    @property
    def n_atoms(self):
        return len(self.atomic_number)

    @property
    def n_bonds(self):
        return self.bonds.shape[0]

    def get_atoms(self):
        for i in range(self.n_atoms):
            yield Atom(
                id=i,
                atomic_number=self.atomic_number[i],
                coordinates=self.coordinates[i]
            )

    def get_bonds(self):
        for i in range(self.n_bonds):
            yield Bond(id=i, a_id=self.bonds[i, 0], b_id=self.bonds[i, 1])


def create_crystal(**kwargs):
    """
    Factory function for creating crystal.

    :param: symbols: a list of symbols to use
    """
    if "atomic_number" in kwargs and "symbol" in kwargs:
        raise ValueError("Exactly one of the following should be provided: "
                         "atomic_number, symbol.")
    if "atomic_number" in kwargs:
        return Crystal(**kwargs)
    elif "symbol" in kwargs:
        symbol = kwargs["symbol"]
        kwargs.pop("symbol")
        atoms = get_atoms_by_symbols(symbol)
        numbers = [a.atomic_number for a in atoms]
        numbers = np.array(numbers)
        return Crystal(atomic_number=numbers, **kwargs)
    else:
        raise ValueError("atomic_number or symbol is missing.")


@dataclass(frozen=True)
class UnitCellDef:
    """
    Definition of unit cell.
    """
    pass
