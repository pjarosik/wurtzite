from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Crystal:
    """
    An instance of crystal structure.
    The crystal structure contains information about the physical arrangement
    of atoms, types of atoms (currently by atomic number) and bonds (the
    complete graph).

    :param atomic_number: a list of atomic numbers  (number of atoms, )
    :param coordinates: a nd-array with dimensions: (number of atoms, 3), where
        3 are the coordinates (x, y, z) of i-th atom.
    :param bonds: a list of connect atom pairs (a, b), where
        a and b are indices in the atomic_number array (number of atoms, 2)
    """
    atomic_number: np.ndarray
    bonds: np.ndarray
    coordinates: np.ndarray

    def __post_init__(self):
        if not (self.atomic_number.shape[0] == self.bonds.shape[0]
                and self.bonds.shape[0] == self.coordinates.shape[0]):
            raise ValueError("All input arrays should have the same size for "
                             "for the first dimension")


@dataclass(frozen=True)
class CrystalCellDef:
    """
    Definition of crystal cell.
    """
    pass
