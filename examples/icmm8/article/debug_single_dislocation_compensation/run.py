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


def calculate_b_distance(line, a, b, d1, cell):
    p_x, p_y = line
    # find the start point on the line
    d = np.hypot(p_x - a[0], p_y - a[1])
    start = np.argmin(d)
    d = np.hypot(p_x - b[0], p_y - b[1])
    end = np.argmin(d)
    bv_angstrom = cell.to_cartesian_indices(np.asarray([1, 0, 0]))
    be = np.sqrt(bv_angstrom[0] ** 2 + bv_angstrom[1] ** 2)
    bz = bv_angstrom[2]

    print(f"Atom: {b}")
    print((np.asarray(b)-np.asarray(d1.position)))
    print(f"BE: {be}, BZ: {bz}")
    print(beta((np.asarray(b)-np.asarray(d1.position)).reshape(1, -1), be=be, bz=bz))

    result = np.zeros(3)
    for i in range(start, end):
        # p = np.mean([p_x[i+1], p_x[i]]), np.mean([p_y[i+1], p_y[i]])
        p = p_x[i], p_y[i]
        p = np.asarray(p)
        # Move the d1 to the center
        p_beta = p - np.asarray(d1.position[:2])
        p_beta = np.asarray([p_beta[0], p_beta[1], 0])
        bet = np.eye(3) - beta(p_beta.reshape(1, -1), be=be, bz=bz)
        # b = np.eye(3)
        v = np.asarray([p_x[i+1] - p_x[i], p_y[i+1] - p_y[i], 0])
        result += bet.dot(v).squeeze()
    return result, np.linalg.norm(result)


def f_points(crystal, u, x, crystal_plane, d, cp_atoms=None, debug=False):
    current_x = x + u
    current_x = current_x.reshape(-1, 3)
    us = []
    be, bz = get_be_bz(crystal.cell, d.b)
    for i, p in enumerate(current_x):
        rd, delta_d = get_love_compensation(crystal_plane, point=p, debug=debug)
        new_u = love_polar((rd, delta_d), be=be, bz=bz)
        us.append(new_u)
    new_u = np.stack(us)
    result = u - new_u
    return result


if __name__ == "__main__":
    HIGHLIGHTED_ATOM = 68
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

    # DETERMINE CRYSTAL PLANES
    print("DETERMINING CRYSTAL PLANES")
    if not os.path.exists("cps.pkl"):
        crystal_plane_atoms = []
        for i, c in enumerate(l1.coordinates[HIGHLIGHTED_ATOM:(HIGHLIGHTED_ATOM+1)]):
            crystal_plane_atoms.append(get_cp(l1, d1=dis_1, position=c))
        pickle.dump(crystal_plane_atoms, open("cps.pkl", "wb"))
    else:
        crystal_plane_atoms = pickle.load(open("cps.pkl", "rb"))

    print("DETERMINING THE FINAL DISPLACEMENTS")

    # Determine the displacements again, this time using the cp compensation
    # starting from the second iteration



    cp = crystal_plane_atoms[0]

    crystal_plane_x, crystal_plane_y = cp[:, 0], cp[:, 1]

    for k, u in enumerate(all_us):
        fig, ax = plt.subplots()
        fig.set_size_inches((20, 20))
        li = l0.translate(u)
        li = wzt.generate.update_bonds(li)
        wzt.visualization.plot_atoms_2d(
            li, fig=fig, ax=ax, alpha=0.5
        )

        xlimits = (-3, 18)
        ylimits = (0, 7)
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.set_aspect("equal")


        # Find atoms, that are close to the crystal_plane
        if k > 0:
            ax.plot(crystal_plane_x, crystal_plane_y, color="plum")
            plane_atoms = []
            for i, c in enumerate(li.coordinates):
                d = np.hypot(crystal_plane_x-c[0], crystal_plane_y-c[1])
                d = np.min(d)
                if d < 0.5 and c[2] > 0.0:
                    plane_atoms.append((i, c))

            plane_atoms = sorted(plane_atoms, key=lambda v: v[1][0])

            for i in range(1, len(plane_atoms)):
                ia, a = plane_atoms[i-1]
                ib, b = plane_atoms[i]
                vec, distance = calculate_b_distance((crystal_plane_x, crystal_plane_y), a, b, d1=dis_1, cell=li.cell)
                be = 3.11
                px, py = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                ax.text(px - 0.6, py + 0.0, f"{distance / be:0.4f}")

        fig.savefig(f"frame_{k:04d}.png")





