import dataclasses

import numpy as np
import wurtzite as wzt
from utils_2nd import get_rotation_matrix, update_dislocation
import matplotlib.pyplot as plt
import wurtzite
from wurtzite.dislocations import beta_function


def plot_distances(ax, d, a, b):
    a_dist = np.hypot(d.position[0] - a[0], d.position[1] - a[1])
    b_dist = np.hypot(d.position[0] - b[0], d.position[1] - b[1])
    ax.plot([d.position[0], a[0]], [d.position[1], a[1]], zorder=100000, color="black", ls="--")  # , marker="_", lw=2, markersize=10)
    ax.plot([d.position[0], b[0]], [d.position[1], b[1]], zorder=100000, color="black", ls="--")  # , marker="_", lw=2, markersize=10)
    ax.text(a[0] - 2.0, a[1] - 2.0, f"${a_dist:2.2f} \\AA$", zorder=100000, fontsize=8)
    ax.text(b[0] + 0.0, b[1] - 2.0, f"${b_dist:2.2f} \\AA$", zorder=100000, fontsize=8)


def calculate_distance(line, a, b):
    p_x, p_y = line
    # find the start point on the line
    d = np.hypot(p_x - a[0], p_y - a[1])
    start = np.argmin(d)
    d = np.hypot(p_x - b[0], p_y - b[1])
    end = np.argmin(d)
    result = 0
    for i in range(start, end):
        result += np.hypot(p_x[i+1] - p_x[i], p_y[i+1] - p_y[i])
    return result


def beta(x: np.ndarray, be: float, bz: float) -> np.ndarray:
    """
    Calculates Beta function.

    NOTE: this function assumes that the dislocations core is in (0, 0).

    Note: this function will return 1 (diagonal) for atoms which are in the
    center of dislocation core.
    """
    NU = 0.35
    BETA_ONES = np.eye(3)
    RADIUS_FACTOR = 0.5

    x = np.asarray(x)
    if len(x.shape) == 1:
        x = x[np.newaxis, ...]

    x1 = x[..., 0]  # (n_atoms, )
    x2 = x[..., 1]
    x1_2 = x1 ** 2
    x2_2 = x2 ** 2
    r2 = x1_2 + x2_2

    a = be / (4 * np.pi * (1.0 - NU) * r2 * r2)
    # du / dx1
    b11 = (-1) * a * x2 * ((3.0 - 2.0 * NU) * x1_2 + (1.0 - 2.0 * NU) * x2_2)  # (natoms, )
    b21 = (-1) * a * x1 * ((1.0 - 2.0 * NU) * x1_2 + (3.0 - 2.0 * NU) * x2_2)
    b31 = (-1) * bz / (2.0 * np.pi) * x2 / r2
    # du / dx2
    b12 = a * x1 * ((3.0 - 2.0 * NU) * x1_2 + (1.0 - 2.0 * NU) * x2_2)
    b22 = a * x2 * ((1.0 + 2.0 * NU) * x1_2 - (1.0 - 2.0 * NU) * x2_2)
    b32 = bz / (2.0 * np.pi) * x1 / r2
    result = np.repeat(BETA_ONES.copy()[np.newaxis, ...], len(x1), axis=0)
    result[:, 0, 0] = b11
    result[:, 1, 0] = b21
    result[:, 2, 0] = b31
    result[:, 0, 1] = b12
    result[:, 1, 1] = b22
    result[:, 2, 1] = b32

    # Atoms in the center of dislocation core: just equal 1
    # (1-beta will be zero)
    core_center_atoms = r2 < 1e-15  # (natoms, )
    result[core_center_atoms, :, :] = BETA_ONES
    return result  # (natoms, 3, 3)

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
        bet = beta(p_beta.reshape(1, -1), be=be, bz=bz)
        # b = np.eye(3)
        v = np.asarray([p_x[i+1] - p_x[i], p_y[i+1] - p_y[i], 0])
        result += bet.dot(v).squeeze()
    return result, np.linalg.norm(result)


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


def plot_function2(data, l, crystal_planes, fig, ax, alpha, xlim, ylim, highlighted_atom, big_points,
                   all_u_atoms):
    if len(data) == 6:
        frame, d1, d2, u_atoms, u_crystal_planes, u_points = data
    else:
        frame, d1, d2, u_atoms, u_crystal_planes = data
    print(f"Frame: {frame}")
    li = l.translate(u_atoms)
    li = wzt.generate.update_bonds(li)
    if highlighted_atom is not None:
        highlighted_atom = {highlighted_atom}
    wzt.visualization.plot_atoms_2d(
        li, fig=fig, ax=ax, alpha=alpha,
        highlighted_atoms=highlighted_atom
    )
    wzt.visualization.display_tee_2d(ax, d=d1, scale=0.6, fontsize=10)
    wzt.visualization.display_tee_2d(ax, d=d2, scale=0.6, fontsize=10)


    u_atoms[np.isclose(u_atoms, 0.0)] = 1e-5
    for i, crystal_plane in enumerate(crystal_planes):
        u_crystal_plane = u_crystal_planes[i]
        crystal_plane_x, crystal_plane_y = crystal_plane[:, 0], crystal_plane[:, 1]
        crystal_plane_x = crystal_plane_x + u_crystal_plane[:, 0]
        crystal_plane_y = crystal_plane_y + u_crystal_plane[:, 1]

        if i == 0:
            crystal_plane_y = crystal_plane_y[crystal_plane_x < d2.position[0]]
            crystal_plane_x = crystal_plane_x[crystal_plane_x < d2.position[0]]
            color = "indianred"
        else:
            color = "plum"

        ax.plot(crystal_plane_x, crystal_plane_y, color=color)

        # Find atoms that lie on the selected plane
        # plane_atoms = []
        #
        # for i, c in enumerate(li.coordinates):
        #     d = np.hypot(crystal_plane_x - c[0], crystal_plane_y - c[1])
        #     d = np.min(d)
        #     if d < 0.5 and c[2] > 0.0:
        #         plane_atoms.append((i, c))
        # # Sort atoms by idx (???)
        # print(f"plane atoms: {plane_atoms}")
        # plane_atoms = sorted(plane_atoms, key=lambda v: v[1][0])
        #
        # for i in range(1, len(plane_atoms)):
        #     ia, a = plane_atoms[i-1]
        #     ib, b = plane_atoms[i]
        #     # distance = calculate_distance((crystal_plane_x, crystal_plane_y), a, b)
        #     vec, distance = calculate_b_distance((crystal_plane_x, crystal_plane_y), a, b, d1=d1, cell=l.cell)
        #     print(a)
        #     print(vec)
        #     ax.quiver(a[0], a[1], vec[0], vec[1])
        #     px, py = (a[0]+b[0])/2, (a[1]+b[1])/2
        #     ax.text(px-0.6, py+0.0, f"{distance:0.2f}")
            # ax.text(px-0.2, py+0.0, f"{np.hypot(b[0]-a[0], b[1]-a[1]):0.2f}", color="blue")
            # if frame + 1 < all_u_atoms.shape[0]:
            #     ax.text(px-0.6, py-0.5, f"{np.linalg.norm(all_u_atoms[frame+1, ia]):0.2f}", color="red")

        # for each consecutive pair of atoms -- calculate distance along the plane
        # put text next to the right atoms
    if big_points is not None:
        for (x, y, z) in big_points:
            ax.add_patch(plt.Circle((x, y), 0.5, color="gray", zorder=-z))

    ax.set_title(f"Iteration: {frame}")
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


def animate_all(output_dir, l, crystal_planes, d1s, d2s, u_atoms, u_crystal_planes,
                u_points=None, alpha=1.0,  xlimits=None, ylimits=None, highlighted_atom=None,
                output_format="png", big_points=None):
    frames = np.arange(len(d1s))
    if u_points is not None:
        data = list(zip(frames, d1s, d2s, u_atoms, u_crystal_planes, u_points))
    else:
        data = list(zip(frames, d1s, d2s, u_atoms, u_crystal_planes))
    return wzt.visualization.create_animation_frames(
        data,
        lambda data, fig, ax: plot_function2(
            data=data, fig=fig, ax=ax, l=l, crystal_planes=crystal_planes,
            alpha=alpha, xlim=xlimits, ylim=ylimits, highlighted_atom=highlighted_atom,
            big_points=big_points,
            all_u_atoms=u_atoms),
        figsize=(20, 20),
        output_dir=output_dir,
        output_format=output_format
    )