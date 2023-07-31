from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class AtomDef:
    atomic_number: int
    symbol: str
    name_en: str


ATOMS = {
    AtomDef(atomic_number=1, symbol="H", name_en="Hydrogen"),
    AtomDef(atomic_number=2, symbol="He", name_en="Helium"),
    AtomDef(atomic_number=3, symbol="Li", name_en="Lithium"),
    AtomDef(atomic_number=4, symbol="Be", name_en="Beryllium"),
    AtomDef(atomic_number=5, symbol="B", name_en="Boron"),
    AtomDef(atomic_number=6, symbol="C", name_en="Carbon"),
    AtomDef(atomic_number=7, symbol="N", name_en="Nitrogen"),
    AtomDef(atomic_number=8, symbol="O", name_en="Oxygen"),
    AtomDef(atomic_number=9, symbol="F", name_en="Fluorine"),
    AtomDef(atomic_number=10, symbol="Ne", name_en="Neon"),
    AtomDef(atomic_number=11, symbol="Na", name_en="Sodium"),
    AtomDef(atomic_number=12, symbol="Mg", name_en="Magnesium"),
    AtomDef(atomic_number=13, symbol="Al", name_en="Aluminium"),
    AtomDef(atomic_number=14, symbol="Si", name_en="Silicon"),
    AtomDef(atomic_number=15, symbol="P", name_en="Phosphorus"),
    AtomDef(atomic_number=16, symbol="S", name_en="Sulfur"),
    AtomDef(atomic_number=17, symbol="Cl", name_en="Chlorine"),
    AtomDef(atomic_number=18, symbol="Ar", name_en="Argon"),
    AtomDef(atomic_number=19, symbol="K", name_en="Potassium"),
    AtomDef(atomic_number=20, symbol="Ca", name_en="Calcium"),
    AtomDef(atomic_number=21, symbol="Sc", name_en="Scandium"),
    AtomDef(atomic_number=22, symbol="Ti", name_en="Titanium"),
    AtomDef(atomic_number=23, symbol="V", name_en="Vanadium"),
    AtomDef(atomic_number=24, symbol="Cr", name_en="Chromium"),
    AtomDef(atomic_number=25, symbol="Mn", name_en="Manganese"),
    AtomDef(atomic_number=26, symbol="Fe", name_en="Iron"),
    AtomDef(atomic_number=27, symbol="Co", name_en="Cobalt"),
    AtomDef(atomic_number=28, symbol="Ni", name_en="Nickel"),
    AtomDef(atomic_number=29, symbol="Cu", name_en="Copper"),
    AtomDef(atomic_number=30, symbol="Zn", name_en="Zinc"),
    AtomDef(atomic_number=31, symbol="Ga", name_en="Gallium"),
    AtomDef(atomic_number=32, symbol="Ge", name_en="Germanium"),
    AtomDef(atomic_number=33, symbol="As", name_en="Arsenic"),
    AtomDef(atomic_number=34, symbol="Se", name_en="Selenium"),
    AtomDef(atomic_number=35, symbol="Br", name_en="Bromine"),
    AtomDef(atomic_number=36, symbol="Kr", name_en="Krypton"),
    AtomDef(atomic_number=37, symbol="Rb", name_en="Rubidium"),
    AtomDef(atomic_number=38, symbol="Sr", name_en="Strontium"),
    AtomDef(atomic_number=39, symbol="Y", name_en="Yttrium"),
    AtomDef(atomic_number=40, symbol="Zr", name_en="Zirconium"),
    AtomDef(atomic_number=41, symbol="Nb", name_en="Niobium"),
    AtomDef(atomic_number=42, symbol="Mo", name_en="Molybdenum"),
    AtomDef(atomic_number=43, symbol="Tc", name_en="Technetium"),
    AtomDef(atomic_number=44, symbol="Ru", name_en="Ruthenium"),
    AtomDef(atomic_number=45, symbol="Rh", name_en="Rhodium"),
    AtomDef(atomic_number=46, symbol="Pd", name_en="Palladium"),
    AtomDef(atomic_number=47, symbol="Ag", name_en="Silver"),
    AtomDef(atomic_number=48, symbol="Cd", name_en="Cadmium"),
    AtomDef(atomic_number=49, symbol="In", name_en="Indium"),
    AtomDef(atomic_number=50, symbol="Sn", name_en="Tin"),
    AtomDef(atomic_number=51, symbol="Sb", name_en="Antimony"),
    AtomDef(atomic_number=52, symbol="Te", name_en="Tellurium"),
    AtomDef(atomic_number=53, symbol="I", name_en="Iodine"),
    AtomDef(atomic_number=54, symbol="Xe", name_en="Xenon"),
    AtomDef(atomic_number=55, symbol="Cs", name_en="Caesium"),
    AtomDef(atomic_number=56, symbol="Ba", name_en="Barium"),
    AtomDef(atomic_number=57, symbol="La", name_en="Lanthanum"),
    AtomDef(atomic_number=58, symbol="Ce", name_en="Cerium"),
    AtomDef(atomic_number=59, symbol="Pr", name_en="Praseodymium"),
    AtomDef(atomic_number=60, symbol="Nd", name_en="Neodymium"),
    AtomDef(atomic_number=61, symbol="Pm", name_en="Promethium"),
    AtomDef(atomic_number=62, symbol="Sm", name_en="Samarium"),
    AtomDef(atomic_number=63, symbol="Eu", name_en="Europium"),
    AtomDef(atomic_number=64, symbol="Gd", name_en="Gadolinium"),
    AtomDef(atomic_number=65, symbol="Tb", name_en="Terbium"),
    AtomDef(atomic_number=66, symbol="Dy", name_en="Dysprosium"),
    AtomDef(atomic_number=67, symbol="Ho", name_en="Holmium"),
    AtomDef(atomic_number=68, symbol="Er", name_en="Erbium"),
    AtomDef(atomic_number=69, symbol="Tm", name_en="Thulium"),
    AtomDef(atomic_number=70, symbol="Yb", name_en="Ytterbium"),
    AtomDef(atomic_number=71, symbol="Lu", name_en="Lutetium"),
    AtomDef(atomic_number=72, symbol="Hf", name_en="Hafnium"),
    AtomDef(atomic_number=73, symbol="Ta", name_en="Tantalum"),
    AtomDef(atomic_number=74, symbol="W", name_en="Tungsten"),
    AtomDef(atomic_number=75, symbol="Re", name_en="Rhenium"),
    AtomDef(atomic_number=76, symbol="Os", name_en="Osmium"),
    AtomDef(atomic_number=77, symbol="Ir", name_en="Iridium"),
    AtomDef(atomic_number=78, symbol="Pt", name_en="Platinum"),
    AtomDef(atomic_number=79, symbol="Au", name_en="Gold"),
    AtomDef(atomic_number=80, symbol="Hg", name_en="Mercury"),
    AtomDef(atomic_number=81, symbol="Tl", name_en="Thallium"),
    AtomDef(atomic_number=82, symbol="Pb", name_en="Lead"),
    AtomDef(atomic_number=83, symbol="Bi", name_en="Bismuth"),
    AtomDef(atomic_number=84, symbol="Po", name_en="Polonium"),
    AtomDef(atomic_number=85, symbol="At", name_en="Astatine"),
    AtomDef(atomic_number=86, symbol="Rn", name_en="Radon"),
    AtomDef(atomic_number=87, symbol="Fr", name_en="Francium"),
    AtomDef(atomic_number=88, symbol="Ra", name_en="Radium"),
    AtomDef(atomic_number=89, symbol="Ac", name_en="Actinium"),
    AtomDef(atomic_number=90, symbol="Th", name_en="Thorium"),
    AtomDef(atomic_number=91, symbol="Pa", name_en="Protactinium"),
    AtomDef(atomic_number=92, symbol="U", name_en="Uranium"),
    AtomDef(atomic_number=93, symbol="Np", name_en="Neptunium"),
    AtomDef(atomic_number=94, symbol="Pu", name_en="Plutonium"),
    AtomDef(atomic_number=95, symbol="Am", name_en="Americium"),
    AtomDef(atomic_number=96, symbol="Cm", name_en="Curium"),
    AtomDef(atomic_number=97, symbol="Bk", name_en="Berkelium"),
    AtomDef(atomic_number=98, symbol="Cf", name_en="Californium"),
    AtomDef(atomic_number=99, symbol="Es", name_en="Einsteinium"),
    AtomDef(atomic_number=100, symbol="Fm", name_en="Fermium"),
    AtomDef(atomic_number=101, symbol="Md", name_en="Mendelevium"),
    AtomDef(atomic_number=102, symbol="No", name_en="Nobelium"),
    AtomDef(atomic_number=103, symbol="Lr", name_en="Lawrencium"),
    AtomDef(atomic_number=104, symbol="Rf", name_en="Rutherfordium"),
    AtomDef(atomic_number=105, symbol="Db", name_en="Dubnium"),
    AtomDef(atomic_number=106, symbol="Sg", name_en="Seaborgium"),
    AtomDef(atomic_number=107, symbol="Bh", name_en="Bohrium"),
    AtomDef(atomic_number=108, symbol="Hs", name_en="Hassium"),
    AtomDef(atomic_number=109, symbol="Mt", name_en="Meitnerium"),
    AtomDef(atomic_number=110, symbol="Ds", name_en="Darmstadtium"),
    AtomDef(atomic_number=111, symbol="Rg", name_en="Roentgenium"),
    AtomDef(atomic_number=112, symbol="Cn", name_en="Copernicium"),
    AtomDef(atomic_number=113, symbol="Nh", name_en="Nihonium"),
    AtomDef(atomic_number=114, symbol="Fl", name_en="Flerovium"),
    AtomDef(atomic_number=115, symbol="Mc", name_en="Moscovium"),
    AtomDef(atomic_number=116, symbol="Lv", name_en="Livermorium"),
    AtomDef(atomic_number=117, symbol="Ts", name_en="Tennessine"),
    AtomDef(atomic_number=118, symbol="Og", name_en="Oganesson")
}
_ATOMS_BY_NUMBER = sorted(list(ATOMS), key=lambda a: a.atomic_number)
_ATOMS_BY_SYMBOL = dict(((a.symbol, a) for a in ATOMS))


def get_atom_by_number(atomic_number: int) -> AtomDef:
    return _ATOMS_BY_NUMBER[atomic_number-1]


def get_atoms_by_numbers(numbers: Iterable[int]):
    return [get_atom_by_number(n) for n in numbers]


def get_atom_by_symbol(symbol: str) -> AtomDef:
    return _ATOMS_BY_SYMBOL[symbol.strip()]


def get_atoms_by_symbols(symbols: Iterable[str]) -> List[AtomDef]:
    return [get_atom_by_symbol(s) for s in symbols]



