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
        cell="B4_ZnS",
    )
    pickle.dump(l0, open("lattice.pkl", "wb"))
else:
    l0 = pickle.load(open("lattice.pkl", "rb"))


def plot_function2(data, l, crystal_plane, fig, ax, alpha, points, xlim, ylim):
    if len(data) == 6:
        frame, d1, d2, u_atoms, u_crystal_plane, u_points = data
    else:
        frame, d1, d2, u_atoms, u_crystal_plane = data
    print(f"Frame: {frame}")
    li = l.translate(u_atoms)
    li = wzt.generate.update_bonds(li)
    wzt.visualization.plot_atoms_2d(li, fig=fig, ax=ax, alpha=alpha)
    wzt.visualization.display_tee_2d(ax, d=d1, scale=0.6, fontsize=10)
    wzt.visualization.display_tee_2d(ax, d=d2, scale=0.6, fontsize=10)

    if frame > 0:
        # DO NOT display points in the initial configuration
        # Remove nans
        # u_points[np.isclose(u_points, 0.0)] = 1e-5
        # ax.quiver(
        #     points[..., 0], points[..., 1],
        #     u_points[..., 0], u_points[..., 1],
        #     color=wurtzite.visualization.vectors_to_rgb(u_points[..., (0, 1)]))
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
    crystal_plane_x = crystal_plane_x + u_crystal_plane[:, 0]
    crystal_plane_y = crystal_plane_y + u_crystal_plane[:, 1]

    ax.plot(crystal_plane_x, crystal_plane_y)
    ax.set_title(f"Iteration: {frame}")
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    fig.set_size_inches((20, 10))


def animate_all(l, crystal_plane, d1s, d2s, u_atoms, u_crystal_planes,
                u_points=None, alpha=1.0, points=None, xlimits=None, ylimits=None,
                frames=False):
    frames = np.arange(len(d1s))
    if u_points is not None:
        data = list(zip(frames, d1s, d2s, u_atoms, u_crystal_planes, u_points))
    else:
        data = list(zip(frames, d1s, d2s, u_atoms, u_crystal_planes))
    return wzt.visualization.create_animation_frames(
        data,
        lambda data, fig, ax: plot_function2(
            data=data, fig=fig, ax=ax, l=l, crystal_plane=crystal_plane,
            alpha=alpha, points=points, xlim=xlimits, ylim=ylimits),
        figsize=2*np.asarray((10, 10)),
        output_dir=output_dir,
        output_format="png"
    )



dis_1 = DislocationDef(
    b=[1, 0, 0],
    position=[4.765, 5.43, 7.5],
    plane=(0, 0, 1),
    label="$d_1$",
    color="brown"
)

dis_2 = DislocationDef(
    b=[1, 0, 0],
    position=[23.88-2*3.811, 5.43, 7.5],
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
l1 = wzt.generate.update_bonds(l1)
# fig, ax = wzt.visualization.plot_displacement(l0, u0)
# for i, p in enumerate(l0.coordinates):
#     u = u0[i]
#     length = np.hypot(u[0], u[1])
#     ax.text(p[0], p[1], f"{length:.2f}")
# fig.savefig(f"{output_dir}/d1.svg")


# d2
# wyznacz plaszczyzne krystalograficzna wynikajaca z d1, przechodzaca przez d2
# Wspolrzedne sa w globalnym ukladzie wspolrzednych (0, 0, 0)
# OPEARCJA CZASOCHLONNA


# Displacement field sampling points
n_points = 20
x = np.linspace(0, 8, n_points)
y = np.linspace(2.5, 7.5, n_points)
xv, yv = np.meshgrid(x, y)
points = np.vstack([xv.ravel(), yv.ravel()]).T
zeros = np.zeros((points.shape[0], 1))
points = np.hstack([points, zeros])

# Run the displacement on the second dislocation
d1s, d2s, u_atoms, u_crystal_plane, u_points, initial_crystal_plane = displace_all(
    crystal=l1,
    d1=dis_1, d2=dis_2,
    points=points,
    n_iter=4,
)

# animate it
anim = animate_all(
    l1, initial_crystal_plane, d1s, d2s, u_atoms, u_crystal_plane, u_points,
    points=points, alpha=0.5,
    xlimits=(-8, 26), ylimits=(0, 20),
)

