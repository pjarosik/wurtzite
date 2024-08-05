import wurtzite as wzt
import sys
import matplotlib

matplotlib.rc('font', size=14)

output_dir = "4"

nx, ny, nz = 6, 6, 2
l0 = wzt.generate.create_lattice(
    dimensions=(nx, ny, nz),
    cell="B4_ZnS",
)

import numpy as np
from wurtzite.model import Crystal
from typing import Tuple, Union, Sequence

BETA_ONES = np.eye(3)
BETA_ONES[-1, -1] = 0


def newton_raphson(x0, n_iter, f, jacobian, **kwargs):
    x = x0
    xs = []
    xs.append(x0)
    for i in range(n_iter):
        j = jacobian(x, **kwargs)
        o = f(x, **kwargs)
        x = x - 1.0 * np.linalg.inv(j).dot(o)
        xs.append(x)
    return x, xs


def displace_love2(
        crystal: Crystal,
        position: Union[Sequence[float], np.ndarray],
        burgers_vector: Union[Sequence[float], np.ndarray],
        plane: Union[Sequence[float], np.ndarray],
        l_function=wzt.dislocations.love_function,
        bv_fraction: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position)
    burgers_vector = np.asarray(burgers_vector)
    plane = np.asarray(plane)
    cell = crystal.cell

    rt = wzt.dislocations._get_rotation_tensor(
        burgers_vector=burgers_vector,
        plane=plane,
        cell=cell
    )
    rt_inv = np.transpose(rt)

    burgers_vector = bv_fraction * cell.to_cartesian_indices(burgers_vector)
    burgers_vector = burgers_vector.reshape(-1, 1)
    burgers_vector = rt.dot(burgers_vector).squeeze()
    be = np.sqrt(burgers_vector[0] ** 2 + burgers_vector[1] ** 2)
    bz = burgers_vector[2]

    position = position.reshape(-1, 1)
    cd = rt.dot(position).squeeze()  # (3, )

    # Initial setting
    x_all = rt.dot(crystal.coordinates.T).T  # (natoms, 3)
    x_all = x_all - cd.reshape(1, -1)

    # x_distance = x_all-cd

    def f(u, x):
        nonlocal be, bz
        current_x = x + u
        current_x = current_x.reshape(1, -1)
        return u - l_function(current_x, be, bz).squeeze()

    def jacobian(u, x):
        nonlocal be, bz
        current_x = x + u
        current_x = current_x.reshape(1, -1)
        return BETA_ONES - wzt.dislocations.beta_function(current_x, be,bz).squeeze()

    result_u = np.zeros((crystal.n_atoms, 3))
    all_us = []
    for i, coords in enumerate(x_all):
        u0 = np.zeros(3)
        u, us = newton_raphson(x0=u0, n_iter=6, f=f, jacobian=jacobian,
                               x=coords)
        result_u[i] = u
        all_us.append(np.stack(us))

    all_us = np.stack(all_us)  # (n atoms, timestep, 3)
    all_us = np.transpose(all_us, (1, 0, 2))  # (timestep, n atoms, 3)
    # Move to the previous system
    result_u = rt_inv.dot(result_u.T).T
    return result_u, all_us


b0 = [1, 0, 0]
position0 = [4.765+1.8667, 5.53+0.656, 7.5]
# position0 = [5, 6, 7.5]
plane0 = (0, 0, 1)

d = wzt.model.DislocationDef(position=position0, b=b0, plane=plane0)

u0, us = displace_love2(
    crystal=l0,
    position=position0,
    burgers_vector=b0,
    plane=plane0,
    bv_fraction=1.0,
)

i = 0


def plot_distance(axis, a, b, label_pos, reference_point="a"):
    ax, ay = a
    bx, by = b
    ab_distance = np.hypot(ax-bx, ay-by)
    axis.plot((ax, bx), (ay, by), zorder=100000, color="black", ls="--")  # , marker="_", lw=2, markersize=10)
    ref_point = b if reference_point == "b" else a
    axis.text(ref_point[0]+label_pos[0], ref_point[1]+label_pos[1], f"${ab_distance:2.2f} \\AA$", zorder=100000, fontsize=14)


def plot_function(data, fig, ax):
    global i
    u = data
    li = l0.translate(u)
    li = wzt.generate.update_bonds(li)
    wzt.visualization.plot_atoms_2d(li, fig=fig, ax=ax, alpha=0.8)
    wzt.visualization.display_tee_2d(ax, d=d, scale=0.6, fontsize=14)

    a = 202
    b = a - 2*nz

    c = a
    e = c + 2*nz

    plot_distance(ax, li.coordinates[a, :2], li.coordinates[b, :2], (0.5, 0.5))
    plot_distance(ax, li.coordinates[c, :2], li.coordinates[e, :2], (-2.0, 0.5))

    f = b + nz
    g = f - 2*ny*nz
    plot_distance(ax, li.coordinates[f, :2], li.coordinates[g, :2], (-2.2, -1.5))

    h = f + 2*nz
    j = h - 2 * ny * nz
    plot_distance(ax, li.coordinates[h, :2], li.coordinates[j, :2], (0.0, 1.5), reference_point="b")
    ax.set_xlim(-0.4, 13.6)
    ax.set_ylim(-1, 12)
    ax.set_axis_off()
    print(i)
    # ax.set_title(f"Iteration: {i}")
    i += 1

wzt.visualization.create_animation_frames(us, plot_function, figsize=(10, 5), output_dir=output_dir, file_prefix="4_")
