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
        dimensions=(10, 12, 2),
        cell="B4_AlN",
    )
    pickle.dump(l0, open("lattice.pkl", "wb"))
else:
    l0 = pickle.load(open("lattice.pkl", "rb"))


current_points = None
def plot_function2(data, l, fig, ax, alpha, points, xlim, ylim):
    print(f"Frame: {frame}")
    li = l.translate(u_atoms)
    li = wzt.generate.update_bonds(li)
    wzt.visualization.plot_atoms_2d(li, fig=fig, ax=ax, alpha=alpha, start_z=1, end_z=4)
    wzt.visualization.display_tee_2d(ax, d=d1, scale=0.6, fontsize=10)
    wzt.visualization.display_tee_2d(ax, d=d2, scale=0.6, fontsize=10)

    u_atoms[np.isclose(u_atoms, 0.0)] = 1e-5
    # if frame > 0:
    #     ax.quiver(
    #         l.coordinates[..., 0], l.coordinates[..., 1],
    #         u_atoms[..., 0], u_atoms[..., 1],
    #         color=wurtzite.visualization.vectors_to_rgb(u_atoms[..., (0, 1)]),
    #         angles="xy"
    #     )

    if frame > 0:
        # DO NOT display points in the initial configuration
        # Remove nans
        # u_points[np.isclose(u_points, 0.0)] = 1e-5
        # ax.quiver(
        #     points[..., 0], points[..., 1],
        #     u_points[..., 0], u_points[..., 1],
        #     color=wurtzite.visualization.vectors_to_rgb(u_points[..., (0, 1)]),
        #     scale=50
        # )
        # current_points += u_points
        pass

    # # vector lengths
    # for i, p in enumerate(points):
    #     u = u_points[i]
    #     length = np.hypot(u[0], u[1])
    #     ax.text(p[0], p[1], f"{length:.2f}")

    crystal_plane_x, crystal_plane_y = crystal_plane[:, 0], crystal_plane[:, 1]
    # CRYSTAL PLANE DISPLACEMENT
    # plt.quiver(crystal_plane_x, crystal_plane_y,
    #            u_crystal_plane[..., 0], u_crystal_plane[..., 1],
    #            color=wurtzite.visualization.vectors_to_rgb(u_crystal_plane[..., (0, 1)]))
    ax.plot(crystal_plane_x, crystal_plane_y)

    ax.set_title(f"Iteration: {frame}")
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    # plt.show()


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
        output_format="svg"
    )



dis_1 = DislocationDef(
    b=[1, 0, 0],
    position=[3.890+1.0*l0.cell.dimensions[0], 4.03+0.35, 7.5],
    plane=(0, 0, 1),
    label="$d_1$",
    color="brown"
)

dis_2 = DislocationDef(
    b=[1, 0, 0],
    position= np.asarray(dis_1.position)+np.array([2.0*l0.cell.dimensions[0], 0.0, 0.0]),  # [23.88-2*3.811, 5.13, 7.5],
    plane=(0, 0, 1),
    label="$d_2$",
    color="brown"
)

dis_3 = DislocationDef(
    b=[-1, 0, 0],
    position= np.asarray(dis_1.position)+np.array([1.0*l0.cell.dimensions[0], 1.8*l0.cell.dimensions[0], 0.0]),  # [23.88-2*3.811, 5.13, 7.5],
    plane=(0, 0, 1),
    label="$d_3$",
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
    n_iter=4,
)

l2 = l1.translate(u_atoms[-1])

u2, all_us = displace_love2(
    crystal=l2,
    position=dis_3.position,
    burgers_vector=dis_3.b,
    plane=dis_3.plane,
    bv_fraction=1.0,
)

l3 = l2.translate(u2)

fig, ax = plt.subplots()

wzt.visualization.plot_atoms_2d(l3, fig=fig, ax=ax, alpha=1.0, start_z=1, end_z=4)
wzt.visualization.display_tee_2d(ax, d=dis_1, scale=0.6, fontsize=10)
wzt.visualization.display_tee_2d(ax, d=dis_2, scale=0.6, fontsize=10)
wzt.visualization.display_tee_2d(ax, d=dis_3, scale=0.6, fontsize=10)
ax.set_aspect("equal")
xlimits = (-1, 20)
ylimits = (0, 20)
ax.set_xlim(*xlimits)
ax.set_ylim(*ylimits)
fig.set_size_inches((0.85*7.5, 0.85*7.25))
fig.savefig("three_dislocations.svg", dpi=300, bbox_inches="tight")
# plt.show()



