import dataclasses

import numpy as np
import wurtzite as wzt
from examples.icmm8.final.utils_2nd import get_rotation_matrix, \
    update_dislocation


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
        if d1ab is not None and d2ab is not None:
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
        new_d1, _ = update_dislocation(l0, ref_d=d2, d=d1, new_pos=np.squeeze(d1.position))
        new_d1.position[0] = new_d1.position[0] + new_d1s[i][0]
        new_d2, _ = update_dislocation(l0, ref_d=d1, d=d2)
        d1s.append(new_d1)
        d2s.append(new_d2)

    data = list(zip(l0s, d1s, d2s, us, frame_nrs))
    return wzt.visualization.create_animation_2d(
        data,
        lambda *args, **kwargs: plot_function(d1ab=d1ab, d2ab=d2ab, alpha=alpha, display_tees=display_tees, *args, **kwargs),
        figsize=1.0*np.asarray((10, 6.5))
    )