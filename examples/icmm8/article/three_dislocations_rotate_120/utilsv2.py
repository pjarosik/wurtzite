import json
import pickle
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import scipy.integrate

import visualization

import wurtzite as wzt
from wurtzite.model import DislocationDef
from utils_1st import *
from utils_2nd import *


ANGLE = 60/180*np.pi

# DIS_A_GLOBAL = np.asarray([np.cos(ANGLE), np.sin(ANGLE), 0])
DIS_A_GLOBAL = np.asarray([1, 0, 0])


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
    ba = np.asarray(DIS_A_GLOBAL)[:2]
    print(f"DIS A BA: {ba}")

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
    vd0 = None
    if crystal_plane is not None:
        vd0 = get_d0((crystal_plane[:, 0], crystal_plane[:, 1]), point=point)
    # if debug:
    #     import matplotlib.pyplot as plt
    #     plt.plot(crystal_plane[:, 0], crystal_plane[:, 1])
    #     plt.plot(vd0[0], vd0[1], '-ro')
    #     plt.plot(point[0], point[1], '-bo')
    #     plt.plot(0, 0, '-go')
    #     plt.show()
    if vd0 is not None:
        vd0_r, vd0_theta = to_polar(vd0)
    else:
        vd0_theta = np.pi

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


def step_newton_raphson(u, f, jacobian, lr=1.0, debug=False, **kwargs):
    j = jacobian(u, **kwargs)
    o = f(u, **kwargs)
    o = o.reshape(-1, 1, 3)  # (n points, 1, 3)
    o = np.transpose(o, axes=(0, 2, 1))  # (n points, 3, 1)

    # delta u = J^{-1}*f
    delta = lr*np.linalg.inv(j) @ o  # (n points, 3, 1)
    delta = np.squeeze(delta)  # (n points, 3)
    if debug:
        print(f"Atom 89")
        print(o[89])
        print(j[89])
        print("Delta")
        print(delta[89])
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


DIS_TOLERANCE = 1e-0


def get_betas(crystal, d, point, rotation_matrix, debug=False, dis_tolerance=DIS_TOLERANCE):
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
    near_points = np.squeeze(np.isclose(p_d_distance, 0.0, atol=dis_tolerance))
    betas[near_points, :] = 0.0

    # if debug:
    #     print(f"p_d_distance: {p_d_distance}, d_position: {d.position}, point: {point}")
    return betas


def broadcast_eye(n, nrepeats):
    return np.array([np.eye(n)]*nrepeats)


# VALUE FUNCTION AND JACOBIAN FOR ITERATING ATOMS, POINTS AND DISLOCATION POSITIONS
def f_points(crystal, u, x, crystal_plane, d2, debug=False, rotate=False, d1_local=None, d1_rt=None, d2_rt=None):
    current_x = x + u
    current_x = current_x.reshape(-1, 3)
    us = []
    be, bz = get_be_bz(crystal.cell, d2.b)
    for i, (p, first_x) in enumerate(zip(current_x, x)):
        # if np.isclose(np.linalg.norm(p[..., :2]), 0.0, atol=1):
        #     r, _ = to_polar(p)  # Intentionally ignoring delta -- should be 0
        #     theta = 0.0
        #     new_u = love_polar((r, theta), be=be, bz=bz)
        #     # Intentially setting ux and uz to 0 for crystal plane (should be 0, not the b/2)
        #     new_u[0] = 0.0  # -be
        #     new_u[2] = 0.0
        # else:
        rd, delta_d = get_love_compensation(crystal_plane, point=p, debug=debug)
        new_u = love_polar((rd, delta_d), be=be, bz=bz)

        # Rotate to make the displacement parallel to the crystal plane
        # 1. find the position on the crystal plane (just find the same x)
        if rotate:
            # styczna

            # p_x = p[0]
            # cp_x = crystal_plane[:, 0]
            # cp_y = crystal_plane[:, 1]
            # # 2. get the angle of the tangent
            # cp_pos = np.argmin(np.abs(p_x-cp_x))
            # cp_pos_x, cp_pos_xp1 = cp_x[cp_pos], cp_x[cp_pos+5]
            # cp_pos_y, cp_pos_yp1 = cp_y[cp_pos], cp_y[cp_pos+5]
            # angle = np.arctan2(cp_pos_yp1-cp_pos_y, cp_pos_xp1-cp_pos_x)
            # # 3. rotate by the rotation matrix
            # angle_rt = np.array([[np.cos(angle), -np.sin(angle)],
            #                     [np.sin(angle), np.cos(angle)]])

            # z beta
            # Calcultate the rotation based on the current local position of the point
            p = np.asarray(first_x).reshape(1, -1)
            betas_1 = get_betas(crystal, d1_local, p, d2_rt, debug=debug)
            betas_2 = get_betas(crystal, d2, p, d1_rt, debug=debug)
            betas = betas_1 + betas_2
            betas = np.squeeze(betas)
            F_inv = np.eye(2)-betas
            ba = np.asarray([1.0, 0.0])
            F = np.linalg.inv(F_inv)
            ba_rotated = F.dot(ba)
            ba = ba_rotated/np.linalg.norm(ba_rotated)
            ba_orto = np.array([[0, -1],
                                [1, 0]]).dot(ba)
            angle_rt = np.eye(2)
            angle_rt[:, 0] = ba
            angle_rt[:, 1] = ba_orto
            new_u_rotated = angle_rt.dot(new_u[:2])
            new_u[:2] = new_u_rotated

        us.append(new_u)
        if debug:
            if i in {89}:
                print(f"Atom {i}: displacement: {new_u}")

    new_u = np.stack(us)
    return u - new_u


def jacobian_points(crystal, u, x, d1_rt, d2_rt, d1, d2, debug=False):
    # dis_1_rot_matrix: macierz obrotu 1 pierwszej dyslokacji, po uwzglednieniu bet z drugiej
    current_x = x + u
    current_x = current_x.reshape(1, -1)
    p = current_x
    betas_1 = get_betas(crystal, d1, p.reshape(-1, 3), d1_rt, debug=debug)
    betas_2 = get_betas(crystal, d2, p.reshape(-1, 3), d2_rt, debug=debug)
    betas = betas_1 + betas_2
    F_inv = (broadcast_eye(2, betas.shape[0]) - betas)
    ones = broadcast_eye(3, betas.shape[0])
    ones[:, :2, :2] = F_inv
    if debug:
        print("beta1")
        print(betas_1[89])
        print("beta2")
        print(betas_2[89])
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
    return jacobian_points(*args, **kwargs)


def get_cp(l, d1, d2):
    # d1, d2: global coordinate system
    crystal_plane, y0 = get_crystal_plane(l, d1, d2, ylim=(-50, 50), ny=10000)
    cp = np.zeros((len(crystal_plane[0]), 3))  # (n plane points, 3)
    cp[:, 0] = crystal_plane[0]  # x
    cp[:, 1] = crystal_plane[1]  # y
    return cp, y0


# For CP integration
def get_total_beta(x, y, l, d1_local, d2_local, d1_rt, d2_rt, enable_beta2=True):
    # UWAGA: d1_local i d2_local sa w ukladzie zaczepionym w d2!
    p = np.asarray([x, np.squeeze(y), 0.0])
    betas_1 = get_betas(l, d1_local, p.reshape(-1, 3), d2_rt, dis_tolerance=1e-6)
    betas_2 = get_betas(l, d2_local, p.reshape(-1, 3), d1_rt, dis_tolerance=1e-6)
    betas = betas_1
    if enable_beta2:
        betas = betas + betas_2

    # DEBUG
    betas = betas[0]
    F_inv = np.eye(2) - betas
    # if x == 0.0:
    #     print(f"p: {p}, BETA 2: {F_inv[1, 0]}, {F_inv[1, 1]}")

    return np.asarray([F_inv[1, 0]/F_inv[1, 1]])


def get_cp_integral(l, y_start, d1_local, d2_local, d1_rt, d2_rt, enable_beta2=True):
    # NOTE: y0 must be determined for system located in the d2
    # print(f"y0: {y0}, d1_local: {d1_local}, d2_local: {d2_local}")
    def func(t, y):
        return get_total_beta(
            x=t, y=y,
            d1_local=d1_local, d2_local=d2_local,
            d1_rt=d1_rt, d2_rt=d2_rt, l=l, enable_beta2=enable_beta2
        )
    be, bz = get_be_bz(l.cell, d2_local.b)
    y0 = y_start
    ode_res = scipy.integrate.solve_ivp(func, t_span=(np.squeeze(d1_local.position)[0], -15), y0=[y_start], method="BDF")
    ode_res2 = scipy.integrate.solve_ivp(func, t_span=(np.squeeze(d1_local.position)[0], 2), y0=[y_start], method="BDF")

    d1_local_y = d1_local.position[..., 1]

    np1, np2 = len(ode_res.t), len(ode_res2.t)
    n_points = np1 + np2
    result = np.zeros((n_points, 3))

    # left side
    result[:np1, 0] = np.flip(np.squeeze(ode_res.t))
    result[:np1, 1] = np.flip(np.squeeze(-ode_res.y))

    # right side
    result[np1:, 0] = ode_res2.t
    result[np1:, 1] = -ode_res2.y

    result[:, 1] = result[:, 1] - np.max(result[:, 1]) + d1_local_y + y0
    return result


def find_cp_integral(l, d1_local, d2_local, d1_rt, d2_rt, y0_range=(-5, 5), npoints=200):
    y0s = np.linspace(y0_range[0], y0_range[1], npoints)
    planes = []
    for y0 in y0s:
        plane = get_cp_integral(l=l, y_start=y0, d1_local=d1_local, d2_local=d2_local,
                        d1_rt=d1_rt, d2_rt=d2_rt, enable_beta2=False)
        planes.append(plane)

    ds = []
    # Find the point closest to d2 (0,0,0)
    for plane in planes:
        d = np.hypot(plane[:, 0], plane[:, 1])
        ds.append(np.min(d))
    # Find the plane
    a = np.argmin(ds)
    y0 = y0s[a]
    return get_cp_integral(l=l, y_start=y0, d1_local=d1_local, d2_local=d2_local,
                           d1_rt=d1_rt, d2_rt=d2_rt, enable_beta2=True)


def displace_all(
        crystal: Crystal,
        d1, d2,
        points=None,
        n_iter=3,
        lr=1.0
) -> Tuple[np.ndarray, np.ndarray]:
    # Wszystkie punkty z _orig -- punkty poczatkowe, ze wspolrzednymi w ukladzie wspolrzednych zaczepionym w dis_2
    # rotate dis2 according to dis1
    # dis_2_rt: macierz obrotu drugiej dyslokacji po uwzglednieniu
    # bet z pierwszej dyslokacji
    # crystal_plane, y0 = get_cp(crystal, d1=d1, d2=d2)
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
    # rt = wzt.dislocations._get_rotation_tensor(
    #     burgers_vector=burgers_vector,
    #     plane=plane,
    #     cell=cell
    # )
    # rt = np.eye(3)
    #
    # rt_x = np.asarray(d2_global.b).squeeze()
    # rt_x = rt_x / np.linalg.norm(rt_x)
    # rt_y = rt_x[:2] @ np.asarray([
    #     [0, -1],
    #     [1,  0],
    # ])
    # rt_y = rt_y / np.linalg.norm(rt_y)
    # rt[:, 0] = d2_global.b
    # rt[:2, 1] = rt_y

    rt = np.array([
        [np.cos(-ANGLE), -np.sin(-ANGLE), 0],
        [np.sin(-ANGLE), np.cos(-ANGLE), 0],
        [0, 0, 1]])

    print(f"Rotation tensor")
    print(rt)
    print(f"Rotation tensor inverse")
    rt_inv = np.linalg.inv(rt)
    print(rt_inv)

    position = position.reshape(-1, 1)
    cd = position.squeeze()  # rt.dot(position).squeeze()  # (3, )

    # crystal_plane_orig = crystal_plane.copy()
    
    def _preprocess(x):
        x = x - cd.reshape(1, -1)
        x = rt.dot(x.T).T
        return x

    # def _preprocess_points(x):
    #     x = x - cd.reshape(1, -1)
    #     x = rt.dot(x.T).T
    #     return x

    def _postprocess(u):
        return rt_inv.dot(u.T).T

    def _postprocess_points(x):
        x = rt_inv.dot(x.T).T
        x = x + cd.reshape(1, -1)
        return x

    # W UKLADZIE WSPOLRZEDNYCH DRUGIEJ DYSLOKACJI
    atoms_orig = _preprocess(crystal.coordinates.copy())  # (n atoms, 3)
    d1_pos_orig = _preprocess(np.array(d1.position).reshape(1, 3))  # (1, 3)
    print(f"ROTATION TENSOR: {rt}")
    print(f"BEFORE PREPROCESS: {d1.position}")
    print(f"AFTER PREPROCESS: {d1_pos_orig}")

    # crystal_plane_orig = crystal_plane_orig-cd.reshape(1, -1)  # (n plane points, 3)
    # crystal_plane_orig = _preprocess(crystal_plane_orig.copy())  # (n plane points, 3)

    if points is not None:
        points_orig = _preprocess(points.copy())  # (n points, 3)

    # Iterated values
    n_atoms, dims = atoms_orig.shape
    # n_crystal_points = crystal_plane.shape[0]
    u_atoms = np.zeros((n_iter+1, n_atoms, dims))
    dis_1s = []
    dis_2s = []
    dis_1s.append(copy.deepcopy(d1))
    dis_2s.append(copy.deepcopy(d2_global))
    crystal_planes = []  # np.zeros((n_iter+1, n_crystal_points, dims))

    if points is not None:
        n_points, dims = points.shape
        u_points = np.zeros((n_iter+1, n_points, dims))
    
    u_atoms_current = np.zeros((n_atoms, dims))
    u_d1_current = np.zeros((1, dims))
    if points is not None:
        u_points_current = np.zeros((n_points, dims))

    d1_rt_current = None
    dis_1_orig = dataclasses.replace(d1, position=d1_pos_orig)
    
    # prepend with initial state (i.e. the first element of each list is the initial state).
    d1_current_local = dis_1_orig

    cpp = find_cp_integral(
        l=crystal, d1_local=d1_current_local, d2_local=d2_local,
        d1_rt=d1_rt_current, d2_rt=d2_rt,
    )

    # cpp = np.zeros((20, 3))
    # cpp[:, 0] = np.linspace(-15, 5, 20)

    # plt.plot(d1_current_local.position[..., 0], d1_current_local.position[..., 1], "-ro")
    # plt.plot(d2_local.position[0], d2_local.position[1], "-go")
    # plt.plot(cpp[..., 0], cpp[..., 1])
    # plt.gca().set_aspect("equal")
    # plt.show()
    crystal_planes.append(_postprocess_points(cpp.copy()))

    # Rotate d2 to align with the detected crystal plane
    for i in range(1, n_iter+1):
        print(f"Iteration {i}")
        # Do not calculate betas for d1 field
        # Di
        def jac(u, x):
            # dis_2 -- moze byc w ukladzie globalnym, bo kierunek wektora burgersa
            # nie ma znaczenia -- ma znacznie tylko modul
            return jacobian_points(crystal, u, x, d1_rt=None, d2_rt=d2_rt, d1=d1_current_local, d2=d2_local)

        def u_func(u, x):
            return f_points(crystal, u, x, crystal_plane=None, d2=d2_local)
            
        # - Determine the new position
        u_d1_current = step_newton_raphson(u_d1_current, f=u_func, jacobian=jac, lr=lr, x=d1_pos_orig)
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

        # Wydaje sie nie dawac dobrego rozwiazania
        # Wyznacz polozenia atomow
        # u_crystal_plane_current = step_newton_raphson(
        #     u_crystal_plane_current, f=u_func_cp, jacobian=jac_cp, lr=lr,
        #     x=crystal_plane_orig
        # )
        # cpp = get_cp(crystal, d1=d1_current_global, d2=d2_global) UWAGA: cpp jest w globalnym ukladzie wspolrzednych
        # Get initial y0 for the crystal plane
        # Calculated in the coordinate system of the d2!
        # Find the initial y0
        # d1 initially distorts plane, d2 is included in the plane
        # _, cp_y0 = get_cp(l=crystal, d1=d1, d2=d2_global)
        # ponizsza plaszczyzna jest zaczepiona w d2
        # Move to the d2 local coordinate system
        # uwaga: y0 jest dla ukladu zaczepionego w d1
        # cp_y0 += (d1.position[1]-d2_global.position[1]) # TODO what about the rotation???

        # PONIZEJ punkty zaczepione w d2!
        cpp = find_cp_integral(
            l=crystal, d1_local=d1_current_local, d2_local=d2_local,
            d1_rt=d1_rt_current, d2_rt=d2_rt,
        )

        # plt.plot(d1_current_local.position[..., 0],
        #          d1_current_local.position[..., 1], "-ro")
        # plt.plot(d2_local.position[0], d2_local.position[1], "-go")
        # plt.plot(cpp[..., 0], cpp[..., 1])
        # plt.gca().set_aspect("equal")
        # plt.show()

        # print("CPP x")
        # print(cpp[:, 0])
        # print("CPP y")
        # print(cpp[:, 1])
        crystal_plane_current = cpp
        crystal_planes.append(_postprocess_points(cpp.copy()))

        # POINTS
        # include di field in the jacobian
        def jac(u, x):
            return jacobian_points(
                crystal,
                u, x,
                d1_rt=d1_rt_current,
                d2_rt=d2_rt,
                d1=d1_current_local,
                d2=d2_local,
                debug=False
            )

        def u_func(u, x):
            return f_points(crystal, u, x, crystal_plane=crystal_plane_current, d2=d2_local, rotate=True, debug=False, d1_local=d1_current_local,
                            d1_rt=d1_rt_current, d2_rt=d2_rt)
        # ATOMS
        u_atoms_current = step_newton_raphson(u_atoms_current, f=u_func, jacobian=jac, lr=lr, x=atoms_orig, debug=False)

        # OTHER POINTS
        if points is not None:
            def jac_points(u, x):
                return jacobian_points(
                    crystal,
                    u, x,
                    d1_rt=d1_rt_current,
                    d2_rt=d2_rt,
                    d1=d1_current_local,
                    d2=d2_local,
                    debug=False
                )
            def u_func_points(u, x):
                return f_points(crystal, u, x, crystal_plane=crystal_plane_current, d2=d2_local, debug=False, rotate=False, d1_local=d1_current_local,
                                d1_rt=d1_rt_current, d2_rt=d2_rt)
            u_points_current = step_newton_raphson(u_points_current, f=u_func_points, jacobian=jac_points, lr=lr, x=points_orig)

        # Store
        u_atoms[i, :, :] = _postprocess(u_atoms_current)
        if points is not None:
            u_points[i, :, :] = _postprocess(u_points_current)
        
        dis_1s.append(dataclasses.replace(d1_current_local, position=np.squeeze(d1.position + _postprocess(u_d1_current))))
        dis_2s.append(copy.deepcopy(d2_global))

    if points is not None:
        return dis_1s, dis_2s, u_atoms, crystal_planes, u_points
    else:
        return dis_1s, dis_2s, u_atoms, crystal_planes
