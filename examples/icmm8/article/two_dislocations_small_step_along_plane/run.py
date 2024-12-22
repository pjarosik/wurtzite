import sys
import pickle
from pathlib import Path

import wurtzite as wzt
from wurtzite.model import DislocationDef
from utils_1st import *
from utils_2nd import *
from utilsv2 import *
from visualization import animate_all


if __name__ == "__main__":
    HIGHLIGHTED_ATOM = 93
    REGENERATE = False
    output_dir = sys.argv[1]

    # Check if the lattice file is already available, if not, generate the new
    # lattice, with the new bonds, and save it to the lattice.pkl file.
    if not Path("lattice.pkl").exists() or REGENERATE:
        l0 = wzt.generate.create_lattice(
            dimensions=(10, 5, 2),
            cell="B4_AlN",
        )
        pickle.dump(l0, open("lattice.pkl", "wb"))
    else:
        l0 = pickle.load(open("lattice.pkl", "rb"))


    # First dislocation.
    dis_1 = DislocationDef(
        b=[1, 0, 0],
        position=[3.890+1.0*l0.cell.dimensions[0], 4.03+0.35, 7.5],
        plane=(0, 0, 1),
        label="$d_1$",
        color="brown"
    )

    # Second dislocation.
    dis_2 = DislocationDef(
        b=[1, 0, 0],
        position=np.asarray(dis_1.position)+np.array([2*l0.cell.dimensions[0], 0.0, 0.0]),
        plane=(0, 0, 1),
        label="$d_2$",
        color="brown"
    )

    # Estimate the displacement for the FIRST DISLOCATION.
    u0, all_us = displace_love2(
        crystal=l0,
        position=dis_1.position,
        burgers_vector=dis_1.b,
        plane=dis_1.plane,
        bv_fraction=1.0,
    )

    # Apply the displacements -> the lattice after the FIRST DISLOCATION (l1).
    # d1
    l1 = l0.translate(u0)
    l1 = wzt.generate.update_bonds(l1)

    # Estimate the displacements for the SECOND DISLOCATION (u_atoms).
    # The below function considers
    d1s, d2s, u_atoms, u_crystal_plane, u_points, initial_crystal_plane, cp_atoms, u_cp_atoms = displace_all(
        # The lattice to be modified.
        crystal=l1,
        # d1 and d2
        d1=dis_1, d2=dis_2,
        # The number of iterations
        n_iter=20,
        # Displacement field scaling factor. The values < 1 allows to "slow down"
        # the algorithm.
        lr=.2,
        regenerate=REGENERATE
    )

    # The crystal plane that includes the highlighted atom.
    cp_atoms = cp_atoms[HIGHLIGHTED_ATOM]

    # Visualize the steps.
    anim = animate_all(
        output_dir=output_dir,
        l=l1,
        crystal_plane=cp_atoms,
        d1s=d1s, d2s=d2s,
        u_atoms=u_atoms,
        u_crystal_planes=u_cp_atoms,
        # Atom diplay (alpha channel)
        # xlimits, ylimits -- physical limits of the display (A)
        alpha=0.5, xlimits=(-2, 22), ylimits=(0, 12),
        highlighted_atom=HIGHLIGHTED_ATOM,
        output_format="svg"
    )

