import numpy as np
import matplotlib.pyplot as plt
import wurtzite as wzt
from wurtzite.model import DislocationDef
from wurtzite.model import Crystal
from typing import Tuple, Union, Sequence
import wurtzite
import dataclasses
from utils_1st import newton_raphson


NU = 0.35
BETA_ONES = np.eye(3)
RADIUS_FACTOR = 0.25


def get_crystal_surface_y0(l0, dislocation, xy, x0=0, n=100000, y0lim=(-30, 30)):
    bv_angstrom = l0.cell.to_cartesian_indices(dislocation.b)
    bx = bv_angstrom[0]

    def F(x, y, x0, y0, nu, bx, r0):
        return (y - y0 +
                bx / (8 * np.pi * (1 - nu))
                * ((1-2*nu) * np.log((x**2 + y**2)/r0**2)
                   - 2*y**2/(x**2 +y**2)
                   - (1-2*nu)*np.log((x0**2 + y0**2)/r0**2)
                   + 2*y0**2/(x0**2+y0**2)))

    y0s = np.linspace(y0lim[0], y0lim[1], n)
    x, y = xy
    dis_x, dis_y, _ = dislocation.position
    x -= dis_x
    y -= dis_y
    v = F(x, y, 0, y0s, nu=NU, bx=bx, r0=bx)
    return y0s[np.argmin(np.abs(v))]


def get_rotation_matrix(l0, dis_a, dis_b):
    """Returns rotation matrix of dis_b, when dis_a is considered"""
    bv_angstrom = l0.cell.to_cartesian_indices(dis_a.b)
    dis_b_local = np.asarray(dis_b.position) - np.asarray(dis_a.position)

    be = np.hypot(bv_angstrom[0], bv_angstrom[1])
    bz = bv_angstrom[2]
    betas = beta(dis_b_local, be=be, bz=bz)
    F_inv = (BETA_ONES - betas)
    F = np.linalg.inv(F_inv[0, :2, :2])

    ba = dis_a.b[:2]
    ba_rotated = F.dot(ba)
    
    ba = ba_rotated / np.linalg.norm(ba_rotated)
    ba_orto = np.array([[0, -1],
                        [1, 0]]).dot(ba)
    ba_z = np.asarray([0, 0, 1])
    
    rotmatrix = np.eye(3)
    rotmatrix[:2, 0] = ba
    rotmatrix[:2, 1] = ba_orto
    return rotmatrix


def beta(x: np.ndarray, be: float, bz: float) -> np.ndarray:
    """
    Calculates Beta function.

    NOTE: this function assumes that the dislocations core is in (0, 0).

    Note: this function will return 1 (diagonal) for atoms which are in the
    center of dislocation core.
    """
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
    b11 = (-1) * a * x2 * (
                (3.0 - 2.0 * NU) * x1_2 + (1.0 - 2.0 * NU) * x2_2)  # (natoms, )
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


# d2, d1, d2 rotation matrix
def get_betas(l0, dis_a, point, rotation_matrix):
    bv_angstrom = l0.cell.to_cartesian_indices(dis_a.b)
    point = np.asarray(point) - np.asarray(dis_a.position)
    point = rotation_matrix.dot(point)
    be = np.hypot(bv_angstrom[0], bv_angstrom[1])
    bz = bv_angstrom[2]
    betas = np.squeeze(beta(point, be=be, bz=bz))[:2, :2]
    betas = rotation_matrix[:2, :2].T.dot(betas).dot(rotation_matrix[:2, :2])
    return betas


def get_F(betas):
    F_inv = (np.eye(2) - betas)
    F = np.linalg.inv(F_inv)
    F_full = np.eye(3)
    assert len(F.shape) == 2
    F_full[:2, :2] = F
    return F_full


def get_crystal_plane(l0, dis_a, dis_b, xlim=(-100, 100), ylim=(-100, 100), nx=20000, ny=20000):
    """
    Returns crystal plane. dis_a "generates" (distorts) the crystal plane. The y0 is selected in a way
    that dis_b is included in the crystal plane. 
    The coordinates of the crystal plane are in the global  
    """
    y0 = get_crystal_surface_y0(l0, dis_a, dis_b.position[:2])
    bv_angstrom = l0.cell.to_cartesian_indices(dis_a.b)
    plane_d_x, plane_d_y = wurtzite.dislocations.get_crystal_surface_oxy(
        position=dis_a.position, x0=0.0, y0=y0, xlim=xlim, ylim=ylim, nx=nx,
        ny=ny, bx=bv_angstrom[0]
    )
    return (plane_d_x, plane_d_y), y0


def get_d0(plane_d_x_y, point):
    # dis_a: dyslokacja d
    # dis_b: dyslokacja N+1
    # UWAGA: plane_d_x oraz plane_d_y musza byc w ukladzie dis_b!

    plane_d_x, plane_d_y = plane_d_x_y

    # patrz tylko w lewo na plaszczyzne krystalograficzna
    plane_d_y = plane_d_y[plane_d_x < 0]
    plane_d_x = plane_d_x[plane_d_x < 0]

    # point: wspolrzedne w ukl. wspolrzdnych zaczepionym w dis_b
    # r_d0 = np.asarray(point)-np.asarray(dis_b.position)  # vector
    r_d0 = np.asarray(point)
    r_d0_val = np.hypot(*r_d0[:2])  # value

    # Find x, y of of the surface for r_d0
    idx = np.argmin(np.abs(np.hypot(plane_d_x, plane_d_y) - r_d0_val))
    rd0x = plane_d_x[idx]
    rd0y = plane_d_y[idx]
    return np.asarray([rd0x, rd0y])


def to_cartesian(lattice, dislocation):
    return lattice.cell.to_cartesian_indices(dislocation.b)


def to_polar(vector):
    r = np.hypot(vector[0], vector[1])
    theta = np.arctan2(vector[1], vector[0])
    return r, theta


# OPAKOWAC W FUNKCJE, dla dis_a i dis_b, wyznacz r_d, theta_d
# uzyc tych wartosci w ux uy uz

def get_love_compensation(plane_d_x_y, dis_b, point):
    # zaczepiony w N+1 (dis_b)
    vd0 = get_d0(plane_d_x_y, point=point)

    bv_r, bv_theta = to_polar(dis_b.b)
    vd0_r, vd0_theta = to_polar(vd0)

    if vd0_theta < 0:
        vd0_theta = 2 * np.pi + vd0_theta
    # print(f"vd0_theta: {vd0_theta}")
    # Obroc kat r_do zgodne z obroceniem ukladu wspolrzednych dis_b

    # TODO czy na pewno zakomentowane?
    # vd0_theta -= bv_theta
    # point: wspolrzedne w ukl. wspolrzdnych zaczepionym w dis_b
    vd = np.asarray(point)  # -np.asarray(dis_b.position)
    rd, theta = to_polar(vd)
    # TODO czy na pewno zakomentowane?
    # theta -= bv_theta
    
    # delta = theta + (vd0_theta-np.pi)
    delta = vd0_theta - theta
    if delta < 0:
        delta += 2 * np.pi
    elif delta > 2 * np.pi:
        delta -= 2 * np.pi
    return rd, delta


def love_polar_integral(x: np.ndarray, be: float, bz: float) -> np.ndarray:
    """
    Calculates love function.

    :param i: the position of the atom in the list of the atoms
    :return: love function for the given parameters (n_points, 3)
    """
    r, theta = x
    r02 = RADIUS_FACTOR * be**2
    ux = -be / (2 * np.pi) * (theta + np.sin(2 * theta) / (4.0 * (1 - NU))) + be / 2
    uy = - be / (8 * np.pi * (1 - NU)) * ( (1.0 - 2 * NU) * np.log(r**2 / r02) - 2 * (np.sin(theta) ** 2))
    # uy = - be / (8 * np.pi * (1 - NU)) * ( - 2 * (np.sin(theta) ** 2))
    # uy = - be / (8 * np.pi * (1 - NU)) * ( (1.0 - 2 * NU) * np.log(r**2 / r02) + 2 * (np.sin(theta) ** 2))
    uz = -bz / (2 * np.pi) * theta + bz / 2
    return np.array([ux, uy, uz])


def update_dislocation(l0, d, ref_d, new_pos=None):
    # Obroc dis_2 zgodnie z betami wyznaczonymi przez dis_1
    rot_matrix = get_rotation_matrix(l0, ref_d, d)
    # Obroc wektor burgersa dis_2 o macierz obrotu wynikajaca z dyslokacji dis_1
    db = rot_matrix[:2, :2].dot(d.b[:2]).squeeze()
    orig_norm = np.linalg.norm(d.b)
    # Zachowaj oryginalna dlugosc wektora
    db = db / np.linalg.norm(db) * orig_norm
    db = np.asarray(db.tolist() + [0])
    d = dataclasses.replace(d, b=db)
    if new_pos is not None:
        d = dataclasses.replace(d, position=new_pos)
    return d, rot_matrix


def displace_love2_2nd_dis(
        plane_d_x_y,
        crystal: Crystal,
        dis_1, dis_2,
        dis_2_rot_matrix,
        dis1_coordinates=None,
        dis1_rot_matrix=None,
        other_coordinates=None,
        n_iter=10
) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray(dis_2.position)
    burgers_vector = np.asarray(dis_2.b)
    plane = np.asarray(dis_2.plane)
    cell = crystal.cell
    bv_fraction = 1.0

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
    if dis1_coordinates is not None:
        x_all = np.array(dis1_coordinates).reshape(1, 3)
        x_all = rt.dot(x_all.T).T
    elif other_coordinates is not None:
        x_all = rt.dot(other_coordinates.T).T  # (n points, 3)
    else:
        x_all = rt.dot(crystal.coordinates.T).T  # (natoms, 3)
    x_all = x_all - cd.reshape(1, -1)

    def f(u, x):
        nonlocal be, bz, dis_1, dis_2
        current_x = x + u
        current_x = current_x.reshape(1, -1)
        result = get_love_compensation(plane_d_x_y, dis_2, point=current_x.squeeze())
        rd, delta_d = result["rd"], result["delta_d"]
        new_u = love_polar_integral((rd, delta_d), be=be, bz=bz)
        return (u - new_u).squeeze()

    def jacobian(u, x):
        nonlocal be, bz, dis1_coordinates, dis1_rot_matrix
        current_x = x + u
        current_x = current_x.reshape(1, -1)
        betas = get_betas(crystal, dis_2, current_x.squeeze(), dis_2_rot_matrix)
        if dis1_coordinates is None:
            betas += get_betas(crystal, dis_1, current_x.squeeze(), dis1_rot_matrix)
        F_inv = (np.eye(2) - betas)
        ones = np.eye(3)
        ones[:2, :2] = F_inv
        return ones.squeeze()

    if dis1_coordinates is not None:
        result_u = np.zeros((1, 3))
    elif other_coordinates is not None:
        result_u = np.zeros((other_coordinates.shape[0], 3))
    else:
        result_u = np.zeros((crystal.n_atoms, 3))

    all_us = []
    for i, coords in enumerate(x_all):
        u0 = np.zeros(3)
        u, us = newton_raphson(x0=u0, n_iter=n_iter, f=f, jacobian=jacobian, x=coords)
        result_u[i] = u
        all_us.append(np.stack(us))

    all_us = np.stack(all_us)  # (n atoms, timestep, 3)
    all_us = np.transpose(all_us, (1, 0, 2))  # (timestep, n atoms, 3)
    # Move to the previous system
    result_u = rt_inv.dot(result_u.T).T
    # TODO rotate each array from all_us
    return result_u, all_us


def update_dislocation(l0, d, ref_d, new_pos=None):
    # Obroc dis_2 zgodnie z betami wyznaczonymi przez dis_1
    rot_matrix = get_rotation_matrix(l0, ref_d, d)
    # Obroc wektor burgersa dis_2 o macierz obrotu wynikajaca z dyslokacji dis_1
    db = rot_matrix[:2, :2].dot(d.b[:2]).squeeze()
    orig_norm = np.linalg.norm(d.b)
    # Zachowaj oryginalna dlugosc wektora
    db = db/np.linalg.norm(db)*orig_norm
    db = np.asarray(db.tolist() + [0])
    d = dataclasses.replace(d, b=db)
    if new_pos is not None:
        d = dataclasses.replace(d, position=new_pos)
    return d, rot_matrix

