import matplotlib.pyplot as plt

import wurtzite as wzt
import sys

l0 = wzt.generate.create_lattice(
    dimensions=(6, 6, 2),  #  The number of cells
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
        x = x - 0.7 * np.linalg.inv(j).dot(o)
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
        u, us = newton_raphson(x0=u0, n_iter=10, f=f, jacobian=jacobian,
                               x=coords)
        result_u[i] = u
        all_us.append(np.stack(us))

    all_us = np.stack(all_us)  # (n atoms, timestep, 3)
    all_us = np.transpose(all_us, (1, 0, 2))  # (timestep, n atoms, 3)
    # Move to the previous system
    result_u = rt_inv.dot(result_u.T).T
    return result_u, all_us


def plot_function(data, fig, ax):
    u = data
    li = l0.translate(u)
    li = wzt.generate.update_bonds(li)
    wzt.visualization.plot_atoms_2d(li, fig=fig, ax=ax, alpha=0.8)
    wzt.visualization.display_tee_2d(ax, d=d, scale=0.6, fontsize=14)
    #
    a, b = li.coordinates[199], li.coordinates[202]
    a_dist = np.hypot(d.position[0]-a[0], d.position[1]-a[1])
    b_dist = np.hypot(d.position[0]-b[0], d.position[1]-b[1])
    ax.plot([d.position[0], a[0]], [d.position[1], a[1]], zorder=100000, color="black", ls="--")# , marker="_", lw=2, markersize=10)
    ax.plot([d.position[0], b[0]], [d.position[1], b[1]], zorder=100000, color="black", ls="--")# , marker="_", lw=2, markersize=10)
    ax.text(a[0]-0.5, a[1]-1.5, f"${a_dist:2.2f} \\AA$", zorder=100000, fontsize=14)
    ax.text(b[0]-0.5, b[1]-1.5, f"${b_dist:2.2f} \\AA$", zorder=100000, fontsize=14)
    ax.set_xlim(-5, 20)
    ax.set_ylim(-1, 9.5)


b0 = [1, 0, 0]
plane0 = (0, 0, 1)
dxs = np.linspace(0, 5, 50)
dys = np.linspace(0, 5, 50)

i = 0
for dy in dys:
    for dx in dxs:
        print(f"\n\nIteration: {i}/{len(dxs)*len(dys)}\n\n")
        position0 = [4.765+dx, 5.53+dy, 7.5]

        d = wzt.model.DislocationDef(position=position0, b=b0, plane=plane0)
        u0, us = displace_love2(
            crystal=l0,
            position=position0,
            burgers_vector=b0,
            plane=plane0,
            bv_fraction=1.0,
        )
        fig, ax = plt.subplots()
        plot_function(us[-2:-1], fig, ax)
        ax.set_aspect("equal")
        fig.savefig(f"{dx:.3f}-{dy:.3f}.png")
        plt.close(fig)
        i += 1


