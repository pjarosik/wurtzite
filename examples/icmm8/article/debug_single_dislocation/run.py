import sys
import pickle
from pathlib import Path

import wurtzite as wzt
from wurtzite.model import DislocationDef
from utils_1st import *
from utils_2nd import *
from utilsv2 import *
from visualization import animate_all
from d1 import displace_love3


if __name__ == "__main__":
    REGENERATE = False
    output_dir = sys.argv[1]

    # Check if the lattice file is already available, if not, generate the new
    # lattice, with the new bonds, and save it to the lattice.pkl file.
    if not Path("lattice.pkl").exists() or REGENERATE:
        l0 = wzt.generate.create_lattice(
            dimensions=(7, 3, 1),
            cell="B4_AlN",
        )
        pickle.dump(l0, open("lattice.pkl", "wb"))
    else:
        l0 = pickle.load(open("lattice.pkl", "rb"))


    # First dislocation.
    dis_1 = DislocationDef(
        b=[1, 0, 0],
        position=[3.890, 4.03+0.35, 7.5],
        plane=(0, 0, 1),
        label="$d_1$",
        color="brown"
    )

    # Estimate the displacement for the FIRST DISLOCATION.
    u0, all_us = displace_love3(
        crystal=l0,
        position=dis_1.position,
        burgers_vector=dis_1.b,
        plane=dis_1.plane,
        bv_fraction=1.0,
    )
    l1 = l0.translate(u0)
    l1 = wzt.generate.update_bonds(l1)

    for i, u in enumerate(all_us):
        fig, ax = plt.subplots()

        li = l0.translate(u)
        li = wzt.generate.update_bonds(li)
        wzt.visualization.plot_atoms_2d(
            li, fig=fig, ax=ax, alpha=0.5
        )
        fig.savefig(f"frame_{i:04d}.png")





