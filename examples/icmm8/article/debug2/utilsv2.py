import json
import os.path
import pickle
import time
from pathlib import Path
import glob

import numpy as np

import visualization

import wurtzite as wzt
from wurtzite.model import DislocationDef
from utils_1st import *
from utils_2nd import *
import pickle



# TODO czy obracac poczatkowe dyslokacje (teowniki), czy obracac obecny? -- na razie obracane sa poczatkowe dyslokacje
# TODO czy dobrze jest uzywany rt oraz rt_inv?
# jacobian: nie sprawdzam czy punkty sa za blisko (zakomentowane) -- rozwazyc odfiltrowanie punktow, dla ktorych tego nie da sie policzyc
# TODO preprocess -- czy nie powinno byc odwrotnie: najpierw przesuniecie do ukladu zaczepionego w rdzeniu dyslokacji, nastepnie obrot?
# TODO czy punkty plaszczyzny powinny być obrocone przed liczeniem kompensacji?
# TODO dis_1_orig itd. wszystkie z orig powinny byc liczone w ukladzie wspolrzednych drugiej dyslokacji -- CHYBA POPRAWIONE
# TODO get_rotation_matrix -- czy nie powinno byc 

import copy


def atan2(y, x):
    if np.isclose(y, 0.0) and np.isclose(x, 0.0):
        # zmiana wzgledem standardu
        return 0.0
    else:
        return np.arctan2(y, x)


def to_polar(vector):
    r = np.hypot(vector[0], vector[1])
    theta = atan2(vector[1], vector[0])
    return r, theta


def _get_rotation_tensor(burgers_vector, plane, cell: wzt.model.UnitCellDef):
    s = cell.to_cartesian_indices(np.asarray(burgers_vector))
    s = _normalize(s)
    m = np.transpose(cell.miller_to_cartesian).dot(plane)
    m = _normalize(m)
    mxs = _normalize(np.cross(m, s))
    mxsxm = _normalize(np.cross(mxs, m))
    return np.array([
        [mxsxm[0], mxsxm[1], mxsxm[2]],
        [mxs[0],   mxs[1],   mxs[2]  ],
        [m[0],     m[1],     m[2]    ],
    ])


def get_rotation_matrix(l0, dis_a, dis_b_pos, debug=False):
    """Returns rotation matrix of dis_b, when dis_a is considered. NOTE dis_b is assumed to be in the local coordinate system of dis_a. """
    be, bz = get_be_bz(l0.cell, dis_a.b)
    # dis_b is assumed to be located in the local coordinate system of dis_a.
    dis_b_local_pos = np.asarray(dis_b_pos).reshape(1, -1) # np.asarray(dis_b.position) - np.asarray(dis_a.position)
    # dis_b_local = np.asarray(dis_b.position) - np.asarray(dis_a.position)
    BETA_ONES = np.eye(3)
    betas = beta(dis_b_local_pos, be=be, bz=bz)
    F_inv = (BETA_ONES - betas)
    F = np.linalg.inv(F_inv[0, :2, :2])

    if debug:
        print(f"F: {F}")

    # TODO upewnic sie, ze ponizsze ma sens
    # ba = dis_a.b[:2]
    ba = np.asarray([1.0, 0.0])

    ba_rotated = F.dot(ba)
    ba = ba_rotated / np.linalg.norm(ba_rotated)
    ba_orto = np.array([[0, -1],
                        [1, 0]]).dot(ba)
    ba_z = np.asarray([0, 0, 1])
    
    rotmatrix = np.eye(3)
    rotmatrix[:2, 0] = ba
    rotmatrix[:2, 1] = ba_orto
    return rotmatrix


def get_d0(crystal_plane, point):
    # dis_a: dyslokacja d
    # dis_b: dyslokacja N+1
    # UWAGA: plane_d_x oraz plane_d_y musza byc w ukladzie dis_b!

    plane_d_x, plane_d_y = crystal_plane

    # patrz tylko w lewo na plaszczyzne krystalograficzna
    plane_d_y = plane_d_y[plane_d_x < 0]
    plane_d_x = plane_d_x[plane_d_x < 0]

    # point: wspolrzedne w ukl. wspolrzdnych zaczepionym w dis_b
    # r_d0 = np.asarray(point)-np.asarray(dis_b.position)  # vector
    r_d0 = np.asarray(point)
    # ponizej promien do punktu point
    r_d0_val = np.hypot(*r_d0[:2])  # value

    # Find x, y of of the surface for r_d0
    idx = np.argmin(np.abs(np.hypot(plane_d_x, plane_d_y) - r_d0_val))
    rd0x = plane_d_x[idx]
    rd0y = plane_d_y[idx]
    return np.asarray([rd0x, rd0y])


def get_love_compensation(crystal_plane, point, debug=False):
    # Crystal plane zaczepiona powinna byc w dis_b!
    vd0 = get_d0((crystal_plane[:, 0], crystal_plane[:, 1]), point=point)
    if debug:
        import matplotlib.pyplot as plt
        plt.plot(crystal_plane[:, 0], crystal_plane[:, 1])
        plt.plot(vd0[0], vd0[1], '-ro')
        plt.plot(point[0], point[1], '-bo')
        plt.plot(0, 0, '-go')
        plt.show()

    vd0_r, vd0_theta = to_polar(vd0)
    vd = np.asarray(point)
    rd, theta = to_polar(vd)
    # Move to [0, 2pi]
    vd0_theta = np.mod(vd0_theta, 2*np.pi)
    theta = np.mod(theta, 2*np.pi)
    
    delta = vd0_theta - theta
    if delta < 0:
        delta += 2*np.pi
    elif delta > 2*np.pi:
        delta -= 2*np.pi
    return rd, delta


def step_newton_raphson(u, f, jacobian_inv, lr=1.0, **kwargs):
    j = jacobian_inv(u, **kwargs)
    o = f(u, **kwargs)
    o = o.reshape(-1, 1, 3)  # (n points, 1, 3)
    o = np.transpose(o, axes=(0, 2, 1))  # (n points, 3, 1)
    delta = lr * j @ o  # (n points, 3, 1)
    delta = np.squeeze(delta)  # (n points, 3)
    u = u - delta  # (n points, 3)
    return u


def get_be_bz(cell, burgers_vector):
    bv_angstrom = cell.to_cartesian_indices(burgers_vector)
    be = np.sqrt(bv_angstrom[0] ** 2 + bv_angstrom[1] ** 2)
    bz = bv_angstrom[2]
    return be, bz


def rotate_dis(crystal, dis_a, dis_b_position, dis_b_bv, debug=False):
    # dis_a => pole beta
    # obracamy dis_b
    # Obroc dis_b po uwzglednieniu beta wynikajacego z dis_a
    # Zachowaj oryginalna dlugosc wektora
    # UWAGA: skladowa srubowa jest zerowana
    
    dis_b_rt = get_rotation_matrix(crystal, dis_a, dis_b_position, debug)
    dis_b_b = dis_b_rt[:2, :2].dot(dis_b_bv[:2]).squeeze()
    orig_norm = np.linalg.norm(dis_b_bv)
    dis_b_b = dis_b_b/np.linalg.norm(dis_b_b)*orig_norm
    dis_b_b = np.asarray(dis_b_b.tolist() + [0])
    return dis_b_b, dis_b_rt


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


DIS_TOLERANCE = 5e-0


def get_betas(crystal, d, point, rotation_matrix, debug=False):
    """
    d - dislocation IN THE LOCAL SYSTEM OF THE CURRENTly introduced dislocation
    """
    if rotation_matrix is None:
        # None means there should be no betas for the given point
        return np.zeros((point.shape[0], 2, 2))
    be, bz = get_be_bz(crystal.cell, d.b)
    point = np.asarray(point)-np.asarray(d.position)
    point = rotation_matrix.dot(point.T).T
    betas = beta(point, be=be, bz=bz)[:, :2, :2]
    rm = rotation_matrix[:2, :2].reshape(1, 2, 2)
    betas = rm.transpose((0, 2, 1)) @ betas @ rm
    # clean-up betas
    # Make sure that any point that is close to the dislocation d is zeroed.
    # Otherwise, we may get very large values (due to division by x**2+y**2 in
    # the beta function).
    # NOTE: point is already in the d local coordinate system
    # (see -d.position above)
    p_d_distance = np.linalg.norm(point[..., :2], axis=1)
    near_points = np.squeeze(np.isclose(p_d_distance, 0.0, atol=DIS_TOLERANCE))
    betas[near_points, :] = 0.0
    # if debug:
    #     print(f"p_d_distance: {p_d_distance}, d_position: {d.position}, point: {point}")
    return betas


def broadcast_eye(n, nrepeats):
    return np.array([np.eye(n)]*nrepeats)


def convert_to_cp_distance(x, u, cp, direction, debug=False):
    """
    Returns the first point of the cp that exceeds the distance determined by the given x.
    """
    x = x.reshape(-1, 3)

    def nnorm(a, b):
        v = np.sum((a-b)**2, axis=1)
        v = np.sqrt(v)
        return v


    target_distance = np.linalg.norm(u[:2])
    # Find the cp point that is closest to the given atom
    # cp: (n points, 3)
    cp_x_dist = nnorm(x[:, :2], cp[:, :2])

    if debug:
        print(f"Minimum distance: {np.min(cp_x_dist)}")

    closest_point = np.argmin(cp_x_dist.squeeze())
    # find the shortest distance
    current_distance = 0
    current_point = closest_point

    if direction == 1:
        # to the right
        r = range(closest_point, cp.shape[0]-1)
    else:
        # to the left
        r = range(closest_point, 1, -1)

    for i in r:
        current_distance += nnorm(cp[i+1, :2].reshape(1, 2), cp[i, :2].reshape(1, 2))
        current_point = cp[i+direction]

        if debug:
            print(f"current distance: {current_distance}, "
                  f"target_distance: {target_distance}, "
                  f"current_point: {current_point}")

        if current_distance > target_distance:
            if debug:
                print(f"INSIDE: x: {x.squeeze()}, current_point: {current_point.squeeze()}")
            result = current_point.squeeze()-x.squeeze()
            result[-1] = 0
            # current_point[..., np.newaxis] = x[..., -1]
            return current_point.squeeze(), result
    raise ValueError("Couldn't find appropriate distance")


initial_atom_position = None

# VALUE FUNCTION AND JACOBIAN FOR ITERATING ATOMS, POINTS AND DISLOCATION POSITIONS
def f_points(crystal, u, x, crystal_plane, d2, cp_atoms=None, debug=False):
    current_x = x + u
    current_x = current_x.reshape(-1, 3)
    us = []
    be, bz = get_be_bz(crystal.cell, d2.b)
    for i, p in enumerate(current_x):
        # if np.isclose(np.linalg.norm(p[..., :2]), 0.0, atol=1):
        #     r, _ = to_polar(p)  # Intentionally ignoring delta -- should be 0
        #     theta = 0.0
        #     new_u = love_polar((r, theta), be=be, bz=bz)
        #     # Intentionally setting ux and uz to 0 for crystal plane (should be 0, not the b/2)
        #     new_u[0] = 0.0  # -be
        #     new_u[2] = 0.0
        # else:
        rd, delta_d = get_love_compensation(crystal_plane, point=p, debug=debug)
        new_u = love_polar((rd, delta_d), be=be, bz=bz)
        # if cp_atoms is not None:
        #     closest_cp_point = np.argmin(np.abs((crystal_plane[:, 0] - p[0])).squeeze())
        #     closest_cp_point = crystal_plane[closest_cp_point]
        #     direction = 1 if closest_cp_point[1] < p[1] else -1
        #     # It should be relative to the beginning!
        #     new_u = convert_to_cp_distance(p, new_u, cp_atoms[i], direction=direction)
        # new_u = p - new_p = x + u_tp1 - x - u_t = u_tp1-u_t
        # and we would like to have u_tp1?
        # ACTUALLY, we need to produce u - f(X + u) in the end.
        # So it means, here we have to have u_tp1-u_t = new_u
        us.append(new_u)
        # if i in {93}:
        #     global initial_atom_position
        #     if initial_atom_position is None:
        #         initial_atom_position = p
        #     print(f"Atom {i}: current_pos: {p}, displacement: {new_u}")
        #     import matplotlib.pyplot as plt
        #     import time
        #     plt.figure(figsize=(10, 4))
        #     r02 = RADIUS_FACTOR * be ** 2
        #     u_y_log = - be / (8 * np.pi * (1 - NU)) * ((1.0 - 2 * NU) * np.log(rd**2 / r02))
        #     print(f"DELTA: rad: {delta_d}, deg: {delta_d*180/np.pi}")
        #     u_y_sin = - be / (8 * np.pi * (1 - NU)) * (-2*(np.sin(delta_d) ** 2))
        #     plt.plot(crystal_plane[:, 0], crystal_plane[:, 1])
        #     plt.plot(cp_atoms[i][:, 0], cp_atoms[i][:, 1])
        #     plt.scatter(initial_atom_position[0], initial_atom_position[1], color="grey", s=75, label="$initial$")
        #     plt.scatter(p[0]+new_u[0], p[1]+new_u[1], color="red", s=75, label="$f_y$")
        #     # plt.scatter(p[0]+new_u[0], p[1]+new_u[1], color="red", s=75, label="$f_y$")
        #     # plt.scatter(p[0]+new_u[0], p[1]+u_y_sin, color="green", s=50, label="$f_{y, sin}$")
        #     # plt.scatter(p[0]+new_u[0], p[1]+u_y_log, color="blue", s=50, label="$f_{y, log}$")
        #     plt.scatter(0, 0)
        #     plt.xlim(-20, 0)
        #     plt.ylim(-0.5, 1.5)
        #     # plt.legend()
        #     plt.savefig(f"oo_{time.time_ns()}.png")
    new_u = np.stack(us)
    # u - f(x + u)
    result = u - new_u
    return result


def jacobian_points_inv(crystal, u, x, d1_rt, d2_rt, d1, d2, debug=False):
    # dis_1_rot_matrix: macierz obrotu 1 pierwszej dyslokacji, po uwzglednieniu bet z drugiej
    current_x = x + u
    current_x = current_x.reshape(1, -1)
    betas_1 = get_betas(crystal, d1, current_x.reshape(-1, 3), d1_rt, debug=debug)
    betas_2 = get_betas(crystal, d2, current_x.reshape(-1, 3), d2_rt, debug=debug)
    betas = betas_1 + betas_2
    F_inv = np.linalg.inv(broadcast_eye(2, betas.shape[0]) - betas)
    ones = broadcast_eye(3, betas.shape[0])
    ones[:, :2, :2] = F_inv
    return ones


# VALUE FUNCTION AND JACOBIAN FOR ITERATING CRYSTAL PLANE POINTS
# NOTE: we sk:Wip here the delta part -- as this is always equal 0
def f_crystal_plane(crystal, u, x, d2):
    current_x = x + u
    current_x = current_x.reshape(-1, 3)
    us = []
    be, bz = get_be_bz(crystal.cell, d2.b)
    for p in current_x:
        p = np.asarray(p)
        r, _ = to_polar(p)  # Intentionally ignoring delta -- should be 0
        theta = 0.0
        new_u = love_polar((r, theta), be=be, bz=bz)
        # Intentially setting ux and uz to 0 for crystal plane (should be 0, not the b/2)
        new_u[0] = 0.0 # -be
        new_u[2] = 0.0
        us.append(new_u)
    new_u = np.stack(us)
    return u-new_u


def jacobian_crystal_plane(*args, **kwargs):
    return jacobian_points_inv(*args, **kwargs)


def get_cp(l, d1, position):
    """
    Returns crystal plane points
    """
    # d1, position: global coordinate system
    start = time.time()
    crystal_plane, y0 = get_crystal_plane(l, d1, position, ylim=(-20, 20), ny=10000, nx=1000)
    cp = np.zeros((len(crystal_plane[0]), 3))  # (n plane points, 3)
    cp[:, 0] = crystal_plane[0]  # x
    cp[:, 1] = crystal_plane[1]  # y
    print(f"Execution time: {time.time() - start}")
    return cp


def displace_all(
        crystal: Crystal,
        d1, d2,
        points=None,
        n_iter=3,
        lr=1.0,
        regenerate=False
) -> Tuple[np.ndarray, np.ndarray]:
    # Wszystkie punkty z _orig -- punkty poczatkowe, ze wspolrzednymi w ukladzie wspolrzednych zaczepionym w dis_2
    # rotate dis2 according to dis1
    # dis_2_rt: macierz obrotu drugiej dyslokacji po uwzglednieniu
    # bet z pierwszej dyslokacja
    crystal_plane = get_cp(crystal, d1=d1, position=d2.position)
    
    if not os.path.exists("cps.pkl") or regenerate:
        print(f"n atoms = {len(crystal.coordinates)}")
        crystal_plane_atoms = []

        for i, c in enumerate(crystal.coordinates):
            print(f"ATOM: {i}/{len(crystal.coordinates)}")
            crystal_plane_atoms.append(get_cp(crystal, d1=d1, position=c))

        pickle.dump(crystal_plane_atoms, open("cps.pkl", "wb"))
    else:
        crystal_plane_atoms = pickle.load(open("cps.pkl", "rb"))

    dis_2_bv, d2_rt = rotate_dis(
        crystal,
        d1,
        dis_b_position=np.asarray(d2.position)-np.asarray(d1.position),
        dis_b_bv=d2.b
    )

    d2_global = dataclasses.replace(d2, b=dis_2_bv)
    d2_local = dataclasses.replace(
        d2_global,
        b=np.asarray([1.0, 0, 0]),
        position=[0.0, 0, 0]
    )
    position = np.asarray(d2_global.position)
    cell = crystal.cell
    bv_fraction = 1.0

    burgers_vector = np.asarray(d2_global.b)
    plane = np.asarray(d2_global.plane)
    rt = wzt.dislocations._get_rotation_tensor(
        burgers_vector=burgers_vector,
        plane=plane,
        cell=cell
    )
    rt_inv = np.transpose(rt)

    position = position.reshape(-1, 1)
    cd = rt.dot(position).squeeze()  # (3, )

    crystal_plane_orig = crystal_plane.copy()
    
    def _preprocess(x):
        x = rt.dot(x.T).T
        x = x - cd.reshape(1, -1)
        return x


    def _postprocess(u):
        return rt_inv.dot(u.T).T

    # W UKLADZIE WSPOLRZEDNYCH DRUGIEJ DYSLOKACJI
    atoms_orig = _preprocess(crystal.coordinates.copy())  # (n atoms, 3)
    d1_pos_orig = _preprocess(np.array(d1.position).reshape(1, 3))  # (1, 3)
    # crystal_plane_orig = crystal_plane_orig-cd.reshape(1, -1)  # (n plane points, 2)
    crystal_plane_orig = _preprocess(crystal_plane_orig.copy())  # (n plane points, 2)
    crystal_plane_atoms_orig = [_preprocess(cp) for cp in crystal_plane_atoms]

    if points is not None:
        points_orig = _preprocess(points.copy())  # (n points, 3)

    # Iterated values
    n_atoms, dims = atoms_orig.shape
    n_crystal_points = crystal_plane.shape[0]
    u_atoms = np.zeros((n_iter+1, n_atoms, dims))
    dis_1s = []
    dis_2s = []
    dis_1s.append(copy.deepcopy(d1))
    dis_2s.append(copy.deepcopy(d2_global))
    u_crystal_planes = np.zeros((n_iter+1, n_crystal_points, dims))
    if points is not None:
        n_points, dims = points.shape
        u_points = np.zeros((n_iter+1, n_points, dims))
    
    u_atoms_current = np.zeros((n_atoms, dims))
    u_d1_current = np.zeros((1, dims))
    u_crystal_plane_current = np.zeros((n_crystal_points, dims))
    if points is not None:
        u_points_current = np.zeros((n_points, dims))

    d1_rt_current = None
    dis_1_orig = dataclasses.replace(d1, position=d1_pos_orig)
    
    # prepend with initial state (i.e. the first element of each list is the initial state).
    d1_current_local = dis_1_orig
    crystal_plane_current = crystal_plane_orig

    crystal_plane_atoms_orig = np.stack(crystal_plane_atoms_orig)
    # (n timesteps, n atoms, n points per plane, 2)
    u_cp_atoms_current = np.zeros((n_iter+1, ) + crystal_plane_atoms_orig.shape)

    previous_displacement = np.asarray([0, 0, 0]).reshape(1, -1)

    u_crystal_plane_atoms = []

    for i in range(1, n_iter+1):
        print(f"Iteration {i}")
        # Do not calculate betas for d1 field
        # Di
        def jac(u, x):
            # dis_2 -- moze byc w ukladzie globalnym, bo kierunek wektora burgersa
            # nie ma znaczenia -- ma znacznie tylko modul
            return jacobian_points_inv(crystal, u, x, d1_rt=None, d2_rt=d2_rt, d1=d1_current_local, d2=d2_local)

        def u_func(u, x):
            return f_points(crystal, u, x, crystal_plane=crystal_plane_current, d2=d2_local, cp_atoms=None)
            
        # - Determine the new position
        u_d1_current = step_newton_raphson(u_d1_current, f=u_func, jacobian_inv=jac, lr=lr, x=d1_pos_orig)
        # - Rotate the dislocation d1 according to the new betas,
        # starting from the dis_1 angle
        d1_current_bv, d1_rt_current = rotate_dis(
            crystal, d2_global,
            dis_b_position=np.asarray(d1.position + u_d1_current) - np.asarray(d2_global.position),
            dis_b_bv=d1.b
        )
        # dis_1_current_bv, dis_1_rt_current = rotate_dis(crystal, dis_2, dis_b_position=dis_1_current.position+u_dis_1_current, dis_b_bv=dis_1.b)
        d1_current_local = dataclasses.replace(
            d1_current_local,
            position=np.squeeze(d1_pos_orig+u_d1_current),
            b=d1_current_bv
        )
        d1_current_global = dataclasses.replace(
            d1_current_local,
            position=np.squeeze(d1.position+u_d1_current),
            b=d1_current_bv
        )
        # PLANE
        def jac_cp(u, x):
            return jacobian_crystal_plane(
                crystal,
                u, x,
                d1_rt=d1_rt_current, d2_rt=d2_rt, d1=d1_current_local, d2=d2_local
            )

        def u_func_cp(u, x):
            return f_crystal_plane(crystal, u, x, d2=d2_global)

        #################### MOVE ALL THE crystal planes and POINTS ACCORDING TO THE d1 displacement
        # print(f"uuu: {uuu}, current: {u_d1_current}")
        u_cp = u_d1_current.reshape(1, 1, -1)
        current_crystal_plane_atoms_orig = crystal_plane_atoms_orig + u_cp

        # Move the plane according to the displacement
        # TODO perhaps translating by u_cp is not needed here?
        def jac_cp_atoms_points(u, x):
            return jacobian_points_inv(
                crystal,
                u, x,
                d1_rt=d1_rt_current,
                d2_rt=d2_rt,
                d1=d1_current_local,
                d2=d2_local,
                debug=False
            )

        def func_cp_atoms_points(u, x):
            return f_points(crystal, u, x, crystal_plane=crystal_plane_current,
                            d2=d2_local, debug=False, cp_atoms=None)

        for k in range(u_cp_atoms_current.shape[1]):
            print(f"{k}/{u_cp_atoms_current.shape[1]}", end="\r")
            # Translate the plane
            start_plane = crystal_plane_atoms_orig[k] + u_cp.reshape(1, -1)
            TOLERANCE = 20
            # Determine points, that are close the d2
            points_close_d2 = np.abs(start_plane[:, 0] - d2_local.position[0]) < TOLERANCE

            current_u = np.zeros(start_plane.shape)
            new_u = np.zeros(start_plane.shape)

            for _ in range(i):
                # Find the new displacement
                uu = step_newton_raphson(current_u[points_close_d2, :],
                                         f=func_cp_atoms_points,
                                         jacobian_inv=jac_cp_atoms_points,
                                         lr=lr, x=start_plane[points_close_d2, :])
                # Keep only dy (we only care about OY movement)
                uu[:, 0] = 0
                uu[:, 2] = 0
                new_u[points_close_d2, :] = uu
                current_u = new_u
            u_cp_atoms_current[i, k] = _postprocess(new_u + u_cp.reshape(1, -1))
            current_crystal_plane_atoms_orig[k] = crystal_plane_atoms_orig[k] + new_u + u_cp
        ####################
        # Wydaje sie nie dawac dobrego rozwiazania
        # Wyznacz polozenia atomow
        # u_crystal_plane_current = step_newton_raphson(
        #     u_crystal_plane_current, f=u_func_cp, jacobian=jac_cp, lr=lr,
        #     x=crystal_plane_orig
        # )
        cpp = get_cp(crystal, d1=d1_current_global, position=d2_global.position)
        # # UWAGA: cpp jest w globalnym ukladzie wspolrzednych
        u_crystal_plane_current = cpp-crystal_plane

        # ATOMS
        u_atoms_current = step_newton_raphson(
            u_atoms_current, f=u_func, jacobian_inv=jac, lr=lr, x=atoms_orig
        )
        for j, x in enumerate(atoms_orig):
            p = x + u_atoms_current[j]
            # Determine if the point is above or below the slicing plane (direction)
            closest_cp_point = np.argmin(np.abs((crystal_plane_current[:, 0] - p[0])).squeeze())
            closest_cp_point = crystal_plane_current[closest_cp_point]
            direction = 1 if closest_cp_point[1] < p[1] else -1
            # It should be relative to the beginning!
            current_p, new_uuu = convert_to_cp_distance(
                x, u_atoms_current[j], current_crystal_plane_atoms_orig[j],
                direction=direction
            )
            u_atoms_current[j] = new_uuu + u_cp.flatten()

        crystal_plane_current = crystal_plane_orig + u_crystal_plane_current

        # OTHER POINTS
        if points is not None:
            pass

        # Store
        u_crystal_planes[i, :, :] = _postprocess(u_crystal_plane_current)
        u_atoms[i, :, :] = _postprocess(u_atoms_current)

        if points is not None:
            u_points[i, :, :] = _postprocess(u_points_current)
        
        dis_1s.append(dataclasses.replace(d1_current_local, position=np.squeeze(d1.position + _postprocess(u_d1_current))))
        dis_2s.append(copy.deepcopy(d2_global))

    if points is not None:
        return dis_1s, dis_2s, u_atoms, u_crystal_planes, u_points, crystal_plane, crystal_plane_atoms, u_cp_atoms_current
    else:
        return dis_1s, dis_2s, u_atoms, u_crystal_planes, None, crystal_plane, crystal_plane_atoms, u_cp_atoms_current
