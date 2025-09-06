import json
import pickle
from pathlib import Path
import glob
import visualization
import sys

import wurtzite as wzt
from wurtzite.model import DislocationDef
from utils_1st import *
from utils_2nd import *
from utilsv2 import *
from visualization import animate_all


output_dir = sys.argv[1] # "two_dislocations_debug"


if not Path("lattice.pkl").exists():
    l0 = wzt.generate.create_lattice(
        dimensions=(10, 5, 2),
        cell="B4_GaN",
    )
    pickle.dump(l0, open("lattice.pkl", "wb"))
else:
    l0 = pickle.load(open("lattice.pkl", "rb"))


current_points = None
def plot_function2(data, l, fig, ax, alpha, points, xlim, ylim):
    global current_points
    if len(data) == 6:
        frame, d1, d2, u_atoms, crystal_plane, u_points = data
    else:
        frame, d1, d2, u_atoms, crystal_plane = data
    print(f"Frame: {frame}")
    li = l.translate(u_atoms)
    li = wzt.generate.update_bonds(li)
    wzt.visualization.plot_atoms_2d(li, fig=fig, ax=ax, alpha=alpha, start_z=1, end_z=4)
    wzt.visualization.display_tee_2d(ax, d=d1, scale=0.4, fontsize=10)
    wzt.visualization.display_tee_2d(ax, d=d2, scale=0.4, fontsize=10)

    u_atoms[np.isclose(u_atoms, 0.0)] = 1e-5
    crystal_plane_x, crystal_plane_y = crystal_plane[:, 0], crystal_plane[:, 1]
    # CRYSTAL PLANE DISPLACEMENT
    # plt.quiver(crystal_plane_x, crystal_plane_y,
    #            u_crystal_plane[..., 0], u_crystal_plane[..., 1],
    #            color=wurtzite.visualization.vectors_to_rgb(u_crystal_plane[..., (0, 1)]))
    ax.plot(crystal_plane_x, crystal_plane_y, color="orange", linestyle="--")

    ax.set_title(f"Iteration: {frame}")
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def animate_all(l, d1s, d2s, u_atoms, crystal_planes,
                u_points=None, alpha=1.0, points=None, xlimits=None, ylimits=None,
                frames=False):
    frames = np.arange(len(d1s))
    if u_points is not None:
        data = list(zip(frames, d1s, d2s, u_atoms, crystal_planes, u_points))
    else:
        data = list(zip(frames, d1s, d2s, u_atoms, crystal_planes))
    return wzt.visualization.create_animation_frames(
        data,
        lambda data, fig, ax: plot_function2(
            data=data, fig=fig, ax=ax, l=l,
            alpha=alpha, points=points, xlim=xlimits, ylim=ylimits),
        figsize=2*np.asarray((abs(xlimits[0]-xlimits[1]), abs(ylimits[0]-ylimits[1]))),
        output_dir=output_dir,
    )



dis_1 = DislocationDef(
    b=[1, 0, 0],
    position=[3.960+1.0*l0.cell.dimensions[0], 4.03+0.35, 7.5],
    plane=(0, 0, 1),
    label="$d_1$",
    color="brown"
)

dis_2 = DislocationDef(
    b=[1, 0, 0],
    position= np.asarray(dis_1.position)+np.array([4.0*l0.cell.dimensions[0], 0.0, 0.0]),  # [23.88-2*3.811, 5.13, 7.5],
    plane=(0, 0, 1),
    label="$d_2$",
    color="brown"
)

u0, all_us = displace_love2(
    crystal=l0,
    position=dis_1.position,
    burgers_vector=dis_1.b,
    plane=dis_1.plane,
    bv_fraction=1.0,
)

# d1
l1 = l0.translate(u0)

# Run the displacement on the second dislocation
d1s, d2s, u_atoms, crystal_planes = displace_all(
    crystal=l1,
    d1=dis_1, d2=dis_2,
    n_iter=2,
)

# animate it
anim = animate_all(
    l1, d1s, d2s, u_atoms, crystal_planes, alpha=1.0,
    xlimits=(0, 25), ylimits=(0, 12),
)

