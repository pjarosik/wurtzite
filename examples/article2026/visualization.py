import dataclasses

import numpy as np
import wurtzite as wzt
from utils_2nd import get_rotation_matrix, update_dislocation
import matplotlib.pyplot as plt
import wurtzite


def plot_displacement(
    lattice, u, 
    xlabel="OX ($\AA$)", ylabel="OY ($\AA$)", 
    title="", 
    dislocation_core_position=None, 
    dislocation_core_u=None, 
    fig=None, ax=None,
    show_labels=True, label_precision=2, div=1.0
):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    y_dim = 1
    x_dim = 0

    # Rysowanie wektorów
    if u is not None:
        q = ax.quiver(
            lattice.coordinates[..., x_dim], lattice.coordinates[..., y_dim], 
            u[..., x_dim], u[..., y_dim], 
            color=wzt.visualization.vectors_to_rgb(u[..., (x_dim, y_dim)])
        )

    # Opcjonalnie dodanie długości wektora
    if show_labels:
        lengths = np.linalg.norm(u[..., (x_dim, y_dim)], axis=-1)/div
        for (x, y, dx, dy, l) in zip(
            lattice.coordinates[..., x_dim].flatten(),
            lattice.coordinates[..., y_dim].flatten(),
            u[..., x_dim].flatten(),
            u[..., y_dim].flatten(),
            lengths.flatten()
        ):
            if l > 1e-10:  # unikamy zerowych wektorów
                ax.text(
                    x + dx, y + dy + 0.05, 
                    f"{l:.{label_precision}f}b", 
                    fontsize=8, ha="center", va="bottom", color="black"
                )

    # Pozycja rdzenia dyslokacji
    if dislocation_core_position is not None:
        pos_x, pos_y, _ = dislocation_core_position
        u_x, u_y, _ = dislocation_core_u
        l = np.hypot(u_x, u_y)/div
        q = ax.quiver(
            pos_x, pos_y, 
            u_x, u_y, 
            color=wzt.visualization.vectors_to_rgb(np.asarray([u_x, u_y, 0]).reshape(1, -1))
        )
        if show_labels:
            ax.text(
                pos_x + u_x, pos_y + u_y + 0.05, 
                    f"{l:.{label_precision}f}b", 
                    fontsize=8, ha="center", va="bottom", color="black"
                    )

    fig.tight_layout()
    return fig, ax


def plot_distances(ax, d, a, b):
    a_dist = np.hypot(d.position[0] - a[0], d.position[1] - a[1])
    b_dist = np.hypot(d.position[0] - b[0], d.position[1] - b[1])
    ax.plot([d.position[0], a[0]], [d.position[1], a[1]], zorder=100000, color="black", ls="--")  # , marker="_", lw=2, markersize=10)
    ax.plot([d.position[0], b[0]], [d.position[1], b[1]], zorder=100000, color="black", ls="--")  # , marker="_", lw=2, markersize=10)
    ax.text(a[0] - 2.0, a[1] - 2.0, f"${a_dist:2.2f} \\AA$", zorder=100000, fontsize=8)
    ax.text(b[0] + 0.0, b[1] - 2.0, f"${b_dist:2.2f} \\AA$", zorder=100000, fontsize=8)


def plot_function(data, fig, ax, d1ab, d2ab, alpha, display_tees):
    l0, d1, d2, u, frame_nr = data
    print(f"{frame_nr}")
    li = l0.translate(u)
    li = wzt.generate.update_bonds(li)
    wzt.visualization.plot_atoms_2d(li, fig=fig, ax=ax, alpha=alpha) # , plot_atom_nr=True)
    ax.set_xlim(-20, 37)
    ax.set_ylim(-2, 32)
    if display_tees:
        wzt.visualization.display_tee_2d(ax, d=d1, scale=0.6, fontsize=10)
        wzt.visualization.display_tee_2d(ax, d=d2, scale=0.6, fontsize=10)
        a, b = li.coordinates[d1ab[0]], li.coordinates[d1ab[1]]
        plot_distances(ax, d1, a, b)
        a, b = li.coordinates[d2ab[0]], li.coordinates[d2ab[1]]
        plot_distances(ax, d2, a, b)
    ax.set_title(f"Iteration: {frame_nr}")


def animate(l0, d1, d2, us, new_d1s, d1ab, d2ab, d2_xoffset=None,
            alpha=1.0, display_tees=False):

    if d2_xoffset is not None:
        d2.position[0] += d2_xoffset

    l0s = [l0]*len(us)
    frame_nrs = list(range(len(us)))

    new_d1s = new_d1s.squeeze()
    d1s = []
    d2s = []
    for i in range(len(us)):
        new_d1, _ = update_dislocation(l0, ref_d=d2, d=d1, new_pos=d1.position+new_d1s[i])
        new_d2, _ = update_dislocation(l0, ref_d=d1, d=d2)
        d1s.append(new_d1)
        d2s.append(new_d2)

    data = list(zip(l0s, d1s, d2s, us, frame_nrs))
    return wzt.visualization.create_animation_2d(
        data,
        lambda *args, **kwargs: plot_function(d1ab=d1ab, d2ab=d2ab, alpha=alpha, display_tees=display_tees, *args, **kwargs),
        figsize=1.0*np.asarray((10, 6.5))
    )



def plot_function2(data, l, fig, ax, alpha, points, xlim, ylim, crystal_plane):
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
    plt.quiver(points[..., 0], points[..., 1], 
               u_points[..., 0], u_points[..., 1], 
               color=wurtzite.visualization.vectors_to_rgb(u_points[..., (0, 1)]), scale=20)

    crystal_plane_x, crystal_plane_y = crystal_plane[:, 0], crystal_plane[:, 1]
    

    # CRYSTAL PLANE DISPLACEMENT
    # print("MAX")
    # print(np.max(u_crystal_plane[..., 0]))
    # print(np.max(u_crystal_plane[..., 1]))
    
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


def animate_all(l, d1s, d2s, u_atoms, u_crystal_planes, u_points=None, alpha=1.0, points=None, xlimits=None, ylimits=None):
    frames = np.arange(len(d1s))
    if u_points is not None:
        data = list(zip(frames, d1s, d2s, u_atoms, u_crystal_planes, u_points))
    else:
        data = list(zip(frames, d1s, d2s, u_atoms, u_crystal_planes))
    return wzt.visualization.create_animation_2d(
        data,
        lambda *args, **kwargs: plot_function2(l=l, alpha=alpha, points=points, xlim=xlimits, ylim=ylimits, *args, **kwargs),
        figsize=2*np.asarray((10, 10))
    )
