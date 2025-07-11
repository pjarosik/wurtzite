import numpy as np
import matplotlib.pyplot as plt
import wurtzite as wzt
from examples.icmm8.article.debug2.visualization import calculate_b_distance
from wurtzite.model import DislocationDef
from wurtzite.model import Crystal
from typing import Tuple, Union, Sequence
import wurtzite
import dataclasses
import scipy.integrate

NU = 0.35
BETA_ONES = np.eye(3)
RADIUS_FACTOR = 0.5


def love_function2(x: np.ndarray, be: float, bz: float) -> np.ndarray:
    x1 = x[..., 0]  # (n_atoms, )
    x2 = x[..., 1]  # (n_atoms, )
    r2 = x1 ** 2 + x2 ** 2
    r = np.sqrt(r2)
    x1_norm = x1 / r
    x2_norm = x2 / r
    r02 = RADIUS_FACTOR * be ** 2

    ux = be / (2 * np.pi) * (np.arctan2(x2_norm, x1_norm) + x1_norm * x2_norm / (2.0 * (1 - NU)))
    uy = -be / (8 * np.pi * (1 - NU)) * ((1.0 - NU - NU) * np.log(r2 / r02) + (x1_norm + x2_norm) * (x1_norm - x2_norm))
    uz = bz / (2 * np.pi) * np.arctan2(x2_norm, x1_norm)

    ux = ux.reshape(-1, 1)  # (natoms, 1)
    uy = uy.reshape(-1, 1)  # (natoms, 1)
    uz = uz.reshape(-1, 1)  # (natoms, 1)
    return np.column_stack((ux, uy, uz))  # (natoms, 3)


def newton_raphson(x0, n_iter, f, jacobian, lr=1.0, **kwargs):
    x = x0
    xs = []
    xs.append(x0)
    for i in range(n_iter):
        j = jacobian(x, **kwargs)
        o = f(x, **kwargs)
        x = x - lr * np.linalg.inv(j).dot(o)
        xs.append(x)
    return x, xs


def calculate_b_integral(start, end, be, bz):
    start = np.asarray(start)
    end = np.asarray(end)
    n = 500
    t = np.linspace(0, 1, n)  # (n, )
    r = np.outer(1 - t, start) + np.outer(t, end) # (n, 3)
    dl = (end - start).reshape(1, -1) * np.gradient(t).reshape(-1, 1)  # (n, 3)
    beta = np.array([(np.eye(3) - wzt.dislocations.beta_function(v.reshape(1, -1), be=be, bz=bz).squeeze())
                     for v in r])  # (n, 3, 3)
    # integrand = np.einsum('ijk,ij->ik', beta, dl)  # (n, 3)
    integrand = []
    for i in range(dl.shape[0]):
        integrand.append(beta[i].dot(dl[i]))
    integrand = np.stack(integrand)  # (n, 3)

    integral = np.sum(integrand, axis=0)
    # integral = scipy.integrate.simps(integrand, axis=0)  # (3, )
    return integral


def calculate_b_integralv2(start, end, be, bz):
    start = np.asarray(start)
    end = np.asarray(end)
    n = 1000
    t = np.linspace(0, 1, n)  # (n, )
    r = np.outer(1 - t, start) + np.outer(t, end) # (n, 3)
    dl = (end - start).reshape(1, -1) * np.gradient(t).reshape(-1, 1)  # (n, 3)
    beta = np.array([(np.eye(3) - wzt.dislocations.beta_function(v.reshape(1, -1), be=be, bz=bz).squeeze())
                     for v in r])  # (n, 3, 3)

    integrand = []
    for i in range(dl.shape[0]):
        integrand.append(beta[i].dot(dl[i]))
    integrand = np.stack(integrand) # (n, 3)
    integral = np.sum(integrand, axis=0)
    return integral


def displace_love3(
        crystal: Crystal,
        position: Union[Sequence[float], np.ndarray],
        burgers_vector: Union[Sequence[float], np.ndarray],
        plane: Union[Sequence[float], np.ndarray],
        l_function=love_function2,
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
        # TODO: trzeba zmienic u: (liczyc calke (1-beta) pomiedzy pozycja poczatkowa a nowym u)
        # TODO trzeba zmieniac l_function: po pierwszej iteracji musi byc użyta f z falka
        u_dash = calculate_b_integral(x, x+u, be=be, bz=bz)
        current_x = x + u
        current_x = current_x.reshape(1, -1)

        return u_dash - l_function(current_x, be, bz).squeeze()

    def jacobian(u, x):
        nonlocal be, bz
        current_x = x + u
        current_x = current_x.reshape(1, -1)
        BETA_ONES = np.eye(3)
        BETA_ONES[-1, -1] = 0
        return BETA_ONES - wzt.dislocations.beta_function(current_x, be, bz).squeeze()

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