from dataclasses import dataclass
from typing import Iterable, List
import numpy as np
import math
from typing import Tuple, Union, List, Sequence


@dataclass(frozen=True)
class AtomDef:
    atomic_number: int
    symbol: str
    name_en: str
    covalent_radius: float


@dataclass(frozen=True)
class UnitCellDef:
    """
    Definition of unit cell.

    :param name: name of the unit
    :param dimensions: cell dimensions, (x, y, z)  # [A]
    :param angles: alpha, beta gamma [rad]
    :param atoms: list of the atom symbols
    :param coordinates: coordinates of the atoms, should have the same length
        as `atoms`; a fraction of the unit cell size (a, b, c); dimensions:
        (n atoms, 3)
    """
    name: str
    dimensions: Union[Tuple[float, float, float], np.ndarray]
    angles: Union[Tuple[float, float, float], np.ndarray]
    atoms: List[str]
    coordinates: np.ndarray

    @staticmethod
    def create(**kwargs):
        if "atoms_with_coords" in kwargs:
            if "atoms" in kwargs or "coordinates" in kwargs:
                raise ValueError("atoms and coordinates properties should not "
                                 "be provided when atoms_with_coords is "
                                 "provided.")
            awc: List[Tuple[str, Tuple[float, float, float]]] = kwargs[
                "atoms_with_coords"]
            kwargs.pop("atoms_with_coords")
            atoms, coords = zip(*awc)
            kwargs["atoms"] = list(atoms)
            kwargs["coordinates"] = np.asarray(coords)
        return UnitCellDef(**kwargs)

    @property
    def miller_to_cartesian(self) -> np.ndarray:
        """
        Returns from miller to cartesian coordinates transform (matrix).

        The returned transform allows to convert indices exposed in the
        miller indices for this cell to the cartesian coordinates.
        """
        a, b, c = self.dimensions
        alpha, beta, gamma = self.angles
        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        cos_beta, sin_beta = np.cos(beta), np.sin(beta)
        cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

        projection_matrix = np.array([
            [a, b * cos_gamma, c * cos_beta],
            [0, b * sin_gamma,
             c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
            [0, 0, c * math.sqrt(1.0 + 2.0 * cos_alpha * cos_beta * cos_gamma - cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2) / sin_gamma]
        ])
        return projection_matrix

    @property
    def cartesian_to_miller(self) -> np.ndarray:
        return np.linalg.inv(self.miller_to_cartesian)

    @property
    def cartesian_coordinates(self) -> np.ndarray:
        projection_matrix = self.miller_to_cartesian
        result = projection_matrix.dot(self.coordinates.T)  # (3, n_atoms)
        return result.T  # (n_atoms, 3)

    def to_cartesian_indices(self, miller_indices: np.ndarray) -> np.ndarray:
        """
        Converts input miller indices to cartesian indices.

        :param miller_indices: miller indices to convert from; an array
            with dimensions (n_vectors, 3)
        :return: the input indices in the cartesian system (n_vectors, 3)
        """
        miller_indices = np.asarray(miller_indices)
        return self.miller_to_cartesian.dot(miller_indices.T).T



ATOMS = {
    AtomDef(atomic_number=1, symbol="H", name_en="Hydrogen", covalent_radius=0.31),
    AtomDef(atomic_number=2, symbol="He", name_en="Helium", covalent_radius=0.28),
    AtomDef(atomic_number=3, symbol="Li", name_en="Lithium", covalent_radius=1.28),
    AtomDef(atomic_number=4, symbol="Be", name_en="Beryllium", covalent_radius=0.96),
    AtomDef(atomic_number=5, symbol="B", name_en="Boron", covalent_radius=0.84),
    AtomDef(atomic_number=6, symbol="C", name_en="Carbon", covalent_radius=0.76),
    AtomDef(atomic_number=7, symbol="N", name_en="Nitrogen", covalent_radius=0.71),
    AtomDef(atomic_number=8, symbol="O", name_en="Oxygen", covalent_radius=0.66),
    AtomDef(atomic_number=9, symbol="F", name_en="Fluorine", covalent_radius=0.57),
    AtomDef(atomic_number=10, symbol="Ne", name_en="Neon", covalent_radius=0.58),
    AtomDef(atomic_number=11, symbol="Na", name_en="Sodium", covalent_radius=1.66),
    AtomDef(atomic_number=12, symbol="Mg", name_en="Magnesium", covalent_radius=1.41),
    AtomDef(atomic_number=13, symbol="Al", name_en="Aluminium", covalent_radius=1.21),
    AtomDef(atomic_number=14, symbol="Si", name_en="Silicon", covalent_radius=1.11),
    AtomDef(atomic_number=15, symbol="P", name_en="Phosphorus", covalent_radius=1.07),
    AtomDef(atomic_number=16, symbol="S", name_en="Sulfur", covalent_radius=1.05),
    AtomDef(atomic_number=17, symbol="Cl", name_en="Chlorine", covalent_radius=1.02),
    AtomDef(atomic_number=18, symbol="Ar", name_en="Argon", covalent_radius=1.06),
    AtomDef(atomic_number=19, symbol="K", name_en="Potassium", covalent_radius=2.03),
    AtomDef(atomic_number=20, symbol="Ca", name_en="Calcium", covalent_radius=1.76),
    AtomDef(atomic_number=21, symbol="Sc", name_en="Scandium", covalent_radius=1.70),
    AtomDef(atomic_number=22, symbol="Ti", name_en="Titanium", covalent_radius=1.60),
    AtomDef(atomic_number=23, symbol="V", name_en="Vanadium", covalent_radius=1.53),
    AtomDef(atomic_number=24, symbol="Cr", name_en="Chromium", covalent_radius=1.39),
    AtomDef(atomic_number=25, symbol="Mn", name_en="Manganese", covalent_radius=1.39),
    AtomDef(atomic_number=26, symbol="Fe", name_en="Iron", covalent_radius=1.32),
    AtomDef(atomic_number=27, symbol="Co", name_en="Cobalt", covalent_radius=1.26),
    AtomDef(atomic_number=28, symbol="Ni", name_en="Nickel", covalent_radius=1.24),
    AtomDef(atomic_number=29, symbol="Cu", name_en="Copper", covalent_radius=1.32),
    AtomDef(atomic_number=30, symbol="Zn", name_en="Zinc", covalent_radius=1.22),
    AtomDef(atomic_number=31, symbol="Ga", name_en="Gallium", covalent_radius=1.22),
    AtomDef(atomic_number=32, symbol="Ge", name_en="Germanium", covalent_radius=1.20),
    AtomDef(atomic_number=33, symbol="As", name_en="Arsenic", covalent_radius=1.19),
    AtomDef(atomic_number=34, symbol="Se", name_en="Selenium", covalent_radius=1.20),
    AtomDef(atomic_number=35, symbol="Br", name_en="Bromine", covalent_radius=1.20),
    AtomDef(atomic_number=36, symbol="Kr", name_en="Krypton", covalent_radius=1.16),
    AtomDef(atomic_number=37, symbol="Rb", name_en="Rubidium", covalent_radius=2.20),
    AtomDef(atomic_number=38, symbol="Sr", name_en="Strontium", covalent_radius=1.95),
    AtomDef(atomic_number=39, symbol="Y", name_en="Yttrium", covalent_radius=1.90),
    AtomDef(atomic_number=40, symbol="Zr", name_en="Zirconium", covalent_radius=1.75),
    AtomDef(atomic_number=41, symbol="Nb", name_en="Niobium", covalent_radius=1.64),
    AtomDef(atomic_number=42, symbol="Mo", name_en="Molybdenum", covalent_radius=1.54),
    AtomDef(atomic_number=43, symbol="Tc", name_en="Technetium", covalent_radius=1.47),
    AtomDef(atomic_number=44, symbol="Ru", name_en="Ruthenium", covalent_radius=1.46),
    AtomDef(atomic_number=45, symbol="Rh", name_en="Rhodium", covalent_radius=1.42),
    AtomDef(atomic_number=46, symbol="Pd", name_en="Palladium", covalent_radius=1.39),
    AtomDef(atomic_number=47, symbol="Ag", name_en="Silver", covalent_radius=1.45),
    AtomDef(atomic_number=48, symbol="Cd", name_en="Cadmium", covalent_radius=1.44),
    AtomDef(atomic_number=49, symbol="In", name_en="Indium", covalent_radius=1.42),
    AtomDef(atomic_number=50, symbol="Sn", name_en="Tin", covalent_radius=1.39),
    AtomDef(atomic_number=51, symbol="Sb", name_en="Antimony", covalent_radius=1.39),
    AtomDef(atomic_number=52, symbol="Te", name_en="Tellurium", covalent_radius=1.38),
    AtomDef(atomic_number=53, symbol="I", name_en="Iodine", covalent_radius=1.39),
    AtomDef(atomic_number=54, symbol="Xe", name_en="Xenon", covalent_radius=1.40),
    AtomDef(atomic_number=55, symbol="Cs", name_en="Caesium", covalent_radius=2.44),
    AtomDef(atomic_number=56, symbol="Ba", name_en="Barium", covalent_radius=2.15),
    AtomDef(atomic_number=57, symbol="La", name_en="Lanthanum", covalent_radius=2.07),
    AtomDef(atomic_number=58, symbol="Ce", name_en="Cerium", covalent_radius=2.04),
    AtomDef(atomic_number=59, symbol="Pr", name_en="Praseodymium", covalent_radius=2.03),
    AtomDef(atomic_number=60, symbol="Nd", name_en="Neodymium", covalent_radius=2.01),
    AtomDef(atomic_number=61, symbol="Pm", name_en="Promethium", covalent_radius=1.99),
    AtomDef(atomic_number=62, symbol="Sm", name_en="Samarium", covalent_radius=1.98),
    AtomDef(atomic_number=63, symbol="Eu", name_en="Europium", covalent_radius=1.98),
    AtomDef(atomic_number=64, symbol="Gd", name_en="Gadolinium", covalent_radius=1.96),
    AtomDef(atomic_number=65, symbol="Tb", name_en="Terbium", covalent_radius=1.94),
    AtomDef(atomic_number=66, symbol="Dy", name_en="Dysprosium", covalent_radius=1.92),
    AtomDef(atomic_number=67, symbol="Ho", name_en="Holmium", covalent_radius=1.92),
    AtomDef(atomic_number=68, symbol="Er", name_en="Erbium", covalent_radius=1.89),
    AtomDef(atomic_number=69, symbol="Tm", name_en="Thulium", covalent_radius=1.90),
    AtomDef(atomic_number=70, symbol="Yb", name_en="Ytterbium", covalent_radius=1.87),
    AtomDef(atomic_number=71, symbol="Lu", name_en="Lutetium", covalent_radius=1.87),
    AtomDef(atomic_number=72, symbol="Hf", name_en="Hafnium", covalent_radius=1.75),
    AtomDef(atomic_number=73, symbol="Ta", name_en="Tantalum", covalent_radius=1.70),
    AtomDef(atomic_number=74, symbol="W", name_en="Tungsten", covalent_radius=1.62),
    AtomDef(atomic_number=75, symbol="Re", name_en="Rhenium", covalent_radius=1.51),
    AtomDef(atomic_number=76, symbol="Os", name_en="Osmium", covalent_radius=1.44),
    AtomDef(atomic_number=77, symbol="Ir", name_en="Iridium", covalent_radius=1.41),
    AtomDef(atomic_number=78, symbol="Pt", name_en="Platinum", covalent_radius=1.36),
    AtomDef(atomic_number=79, symbol="Au", name_en="Gold", covalent_radius=1.36),
    AtomDef(atomic_number=80, symbol="Hg", name_en="Mercury", covalent_radius=1.32),
    AtomDef(atomic_number=81, symbol="Tl", name_en="Thallium", covalent_radius=1.45),
    AtomDef(atomic_number=82, symbol="Pb", name_en="Lead", covalent_radius=1.46),
    AtomDef(atomic_number=83, symbol="Bi", name_en="Bismuth", covalent_radius=1.48),
    AtomDef(atomic_number=84, symbol="Po", name_en="Polonium", covalent_radius=1.40),
    AtomDef(atomic_number=85, symbol="At", name_en="Astatine", covalent_radius=1.50),
    AtomDef(atomic_number=86, symbol="Rn", name_en="Radon", covalent_radius=1.50),
    AtomDef(atomic_number=87, symbol="Fr", name_en="Francium", covalent_radius=2.60),
    AtomDef(atomic_number=88, symbol="Ra", name_en="Radium", covalent_radius=2.21),
    AtomDef(atomic_number=89, symbol="Ac", name_en="Actinium", covalent_radius=2.15),
    AtomDef(atomic_number=90, symbol="Th", name_en="Thorium", covalent_radius=2.06),
    AtomDef(atomic_number=91, symbol="Pa", name_en="Protactinium", covalent_radius=2.00),
    AtomDef(atomic_number=92, symbol="U", name_en="Uranium", covalent_radius=1.96),
    AtomDef(atomic_number=93, symbol="Np", name_en="Neptunium", covalent_radius=1.90),
    AtomDef(atomic_number=94, symbol="Pu", name_en="Plutonium", covalent_radius=1.87),
    AtomDef(atomic_number=95, symbol="Am", name_en="Americium", covalent_radius=1.80),
    AtomDef(atomic_number=96, symbol="Cm", name_en="Curium", covalent_radius=1.69),
    AtomDef(atomic_number=97, symbol="Bk", name_en="Berkelium", covalent_radius=1.68),
    AtomDef(atomic_number=98, symbol="Cf", name_en="Californium", covalent_radius=1.68),
    AtomDef(atomic_number=99, symbol="Es", name_en="Einsteinium", covalent_radius=None),
    AtomDef(atomic_number=100, symbol="Fm", name_en="Fermium", covalent_radius=None),
    AtomDef(atomic_number=101, symbol="Md", name_en="Mendelevium", covalent_radius=None),
    AtomDef(atomic_number=102, symbol="No", name_en="Nobelium", covalent_radius=None),
    AtomDef(atomic_number=103, symbol="Lr", name_en="Lawrencium", covalent_radius=None),
    AtomDef(atomic_number=104, symbol="Rf", name_en="Rutherfordium", covalent_radius=None),
    AtomDef(atomic_number=105, symbol="Db", name_en="Dubnium", covalent_radius=None),
    AtomDef(atomic_number=106, symbol="Sg", name_en="Seaborgium", covalent_radius=None),
    AtomDef(atomic_number=107, symbol="Bh", name_en="Bohrium", covalent_radius=None),
    AtomDef(atomic_number=108, symbol="Hs", name_en="Hassium", covalent_radius=None),
    AtomDef(atomic_number=109, symbol="Mt", name_en="Meitnerium", covalent_radius=None),
    AtomDef(atomic_number=110, symbol="Ds", name_en="Darmstadtium", covalent_radius=None),
    AtomDef(atomic_number=111, symbol="Rg", name_en="Roentgenium", covalent_radius=None),
    AtomDef(atomic_number=112, symbol="Cn", name_en="Copernicium", covalent_radius=1.36),
    AtomDef(atomic_number=113, symbol="Nh", name_en="Nihonium", covalent_radius=None),
    AtomDef(atomic_number=114, symbol="Fl", name_en="Flerovium", covalent_radius=None),
    AtomDef(atomic_number=115, symbol="Mc", name_en="Moscovium", covalent_radius=None),
    AtomDef(atomic_number=116, symbol="Lv", name_en="Livermorium", covalent_radius=None),
    AtomDef(atomic_number=117, symbol="Ts", name_en="Tennessine", covalent_radius=None),
    AtomDef(atomic_number=118, symbol="Og", name_en="Oganesson", covalent_radius=None)
}
_ATOMS_BY_NUMBER = sorted(list(ATOMS), key=lambda a: a.atomic_number)
_ATOMS_BY_SYMBOL = dict(((a.symbol, a) for a in ATOMS))

CELLS = [
    UnitCellDef.create(
        name="3C_SiC",
        dimensions=(4.3596, 4.3596, 4.3596),
        angles=np.array((90., 90., 90.))*np.pi/180,
        atoms_with_coords=[
            ("Si", (0.0, 0.0, 0.0)),
            ("C",  (1/4, 1/4, 1/4)),
            ("Si", (1/2, 1/2, 0.0)),
            ("C",  (3/4, 3/4, 1/4)),
            ("Si", (1/2, 0.0, 1/2)),
            ("C",  (3/4, 1/4, 3/4)),
            ("Si", (0.0, 1/2, 1/2)),
            ("C",  (1/4, 3/4, 3/4)),
        ],
    ),
    UnitCellDef.create(
        name="4H_SiC",
        dimensions=(3.073, 3.073, 10.053),
        angles=np.array((90.0, 90.0, 120.0))*np.pi/180,
        atoms_with_coords=[
            ("Si", (0.0, 0.0, 0.0)),
            ("C",  (0.0, 0.0, 3/16)),
            ("Si", (2/3, 1/3, 1/4)),
            ("C",  (2/3, 1/3, 7/16)),
            ("Si", (1/3, 2/3, 1/2)),
            ("C",  (1/3, 2/3, 11/16)),
            ("Si", (2/3, 1/3, 3/4)),
            ("C",  (2/3, 1/3, 15/16)),
        ]
    ),
    UnitCellDef.create(
        name="B4_GaN",
        dimensions=(3.180, 3.180, 5.166),
        angles=np.array((90.0, 90.0, 120.0))*np.pi/180,
        atoms_with_coords=[
            ("N",  (0.0, 0.0, 0.0)),
            ("Ga", (0.0, 0.0, 3/8)),
            ("N",  (2/3, 1/3, 1/2)),
            ("Ga", (2/3, 1/3, 7/8))
        ]
    ),
    UnitCellDef.create(
        name="B4_ZnS",
        dimensions=(3.811, 3.811, 6.234),
        angles=np.array((90.0, 90.0, 120.0))*np.pi/180,
        atoms_with_coords=[
            ("S",  (0.0, 0.0, 0.0)),
            ("Zn", (0.0, 0.0, 3/8)),
            ("S",  (2/3, 1/3, 1/2)),
            ("Zn", (2/3, 1/3, 7/8))
        ]
    ),
    # See Bojarski, et al.
    UnitCellDef.create(
        name="B4_AlN",
        dimensions=(3.11, 3.11, 4.98),
        angles=np.array((90.0, 90.0, 120.0))*np.pi/180,
        atoms_with_coords=[
            ("N",  (0.0, 0.0, 0.0)),
            ("Al", (0.0, 0.0, 3/8)),
            ("N",  (2/3, 1/3, 1/2)),
            ("Al", (2/3, 1/3, 7/8))
        ]
    ),
    UnitCellDef.create(
        name="B4_InN",
        dimensions=(3.533, 3.533, 5.692),
        angles=np.array((90.0, 90.0, 120.0))*np.pi/180,
        atoms_with_coords=[
            ("N",  (0.0, 0.0, 0.0)),
            ("In", (0.0, 0.0, 3/8)),
            ("N",  (2/3, 1/3, 1/2)),
            ("In", (2/3, 1/3, 7/8))
        ]
    ),
    UnitCellDef.create(
        name="B4_SiC",
        dimensions=(1.1*3.076, 1.1*3.076, 1.1*5.048),
        angles=np.array((90.0, 90.0, 120.0))*np.pi/180,
        atoms_with_coords=[
            ("C",  (0.0, 0.0, 0.0)),
            ("Si", (0.0, 0.0, 3/8)),
            ("C",  (2/3, 1/3, 1/2)),
            ("Si", (2/3, 1/3, 7/8))
        ]
    )
]

_CELLS_BY_NAME = dict(((c.name, c) for c in CELLS))


def get_atom_by_number(atomic_number: int) -> AtomDef:
    return _ATOMS_BY_NUMBER[atomic_number - 1]


def get_atoms_by_numbers(numbers: Iterable[int]):
    return [get_atom_by_number(n) for n in numbers]


def get_atom_by_symbol(symbol: str) -> AtomDef:
    return _ATOMS_BY_SYMBOL[symbol.strip()]


def get_atoms_by_symbols(symbols: Iterable[str]) -> List[AtomDef]:
    return [get_atom_by_symbol(s) for s in symbols]


def get_cell_by_name(name: str) -> UnitCellDef:
    return _CELLS_BY_NAME[name.strip()]
