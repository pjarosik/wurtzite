from typing import List, Optional, Set

import wurtzite as wzt
import numpy as np
import matplotlib.pyplot as plt
from utils_1st import displace_love2
import scipy.integrate
import dataclasses
from astar import astar, interpolate_path
import copy
from scipy.integrate import quad_vec
import sys
from utils import get_be_bz, broadcast_eye, get_line, line_integral, \
    MillerIndices

import cupy as cp
from constants import *


# 1. Przetestowac z jedna dyslokacja
# 2. Zaimplementowac wyznaczanie plaszczyzn rozciecia uzywajac calki (to jedyne rozwiazanie, zeby zadzialao dla n dyslokacji), dodac wyznaczanie gora/dol wg tego
# 3. Przetestowac z dwiema dyslokacjami, zgodnie z prosba
# 4. Zaimplementowac lepsze wyznaczanie sciezki


class DisplacementLog:
    """
    Contains the history of all displacements performed by the `displace_all`
    function.

    :param integration_paths: integration paths used to calcuate the displacement
    """

    def __init__(self):
        self.d_states = []
        self.u_atoms = []
        self.glide_planes = []
        self.points = []
        self.integration_paths = []

    def log(self, d_state, u_atoms, glide_planes, points=None, integration_paths=None):
        self.d_states.append(d_state)
        self.u_atoms.append(u_atoms)
        self.glide_planes.append(glide_planes)
        self.points.append(points)
        self.integration_paths.append(integration_paths)

    def get_u_atoms(self):
        return cp.stack(self.u_atoms)

    @property
    def last_d_state(self):
        return self.d_states[-1]

    @property
    def last_u_atoms(self):
        return self.u_atoms[-1]

    @property
    def last_glide_planes(self):
        return self.glide_planes[-1]

    @property
    def last_integration_paths(self):
        return self.integration_paths[-1]


def postprocess_dislocations(d_state, miller):
    new_ds = []
    for i, d in enumerate(d_state.ds):
        current_pos = cp.asarray(d.position).reshape(1, -1)
        new_pos = miller.postprocess_points(current_pos).squeeze()

        b = d.b
        if isinstance(b, cp.ndarray):
            b = b.get()

        d = _set_d(d, b=b)
        d = _set_d(d, position=new_pos.get())
        new_ds.append(d)
    return DislocationsState(ds=new_ds, ds_rt=d_state.ds_rt)


def displace(crystal, dislocations, d_n, n_iters=3, alpha=1.0, skip_np1=False, n_points=30000,
             glide_plane_margin=45, points=(), debug=False):
    """
    Displaces the given crystal lattice according to the displacements
    caused by the dislocation d.

    :param dislocations: a list of dislocations already inserted in the crystal
    :param d_n: dislocation that we are inserting
    """

    # Move all entities into the coordinate system centered in the dislocation d.
    miller = MillerIndices(crystal=crystal, dislocation=d_n)
    # (n atoms, 3)
    # - The introduced dislocation (move to the (0, 0, 0))
    initial_dn_local = dataclasses.replace(d_n, position=[0.0, 0, 0], b=[1.0, 0.0, 0.0])
    # - Other dislocations
    initial_ds_local = [miller.preprocess_dislocation(d) for d in dislocations]
    # - Atoms
    initial_atoms_local = miller.preprocess(cp.asarray(crystal.coordinates))
    all_dislocations_local = initial_ds_local + [initial_dn_local]

    points_local = None
    if points:
        points_local = [miller.preprocess(cp.asarray(p)) for p in points]

    # ALl entities in the d_state is assumed to be located in the coordinate
    # system centered in d_n.
    d_state = DislocationsState(
        ds=all_dislocations_local,
        ds_rt=[cp.eye(2) for _ in range(len(all_dislocations_local))]
    )

    initial_d_state = d_state
    log = DisplacementLog()

    glide_planes = []
    for i in range(len(d_state.ds)):
        glide_plane = find_glide_plane(
            crystal, d_state, margin=glide_plane_margin, dislocation_nr=i)
        glide_planes.append(glide_plane)

    # Initialize with u = 0.
    log.log(d_state=d_state, u_atoms=cp.zeros(shape=initial_atoms_local.shape),
            glide_planes=glide_planes, points=points_local)

    initial_d_state = d_state

    for i in range(n_iters):
        if len(d_state.ds) > 1:
            # More than one dislocation -- we need to update the rotation
            # and location of each previous dislocation.

            # Rotate dislocations (including the currently added one).
            new_ds = []
            new_d_rts = []
            for i, d in enumerate(d_state.ds):
                new_d, new_d_rt = rotate_dislocation(
                    crystal=crystal, d_state=d_state,
                    rotated_dislocation=d,
                    exclude_beta={i}
                )
                new_ds.append(new_d)
                new_d_rts.append(new_d_rt)

            d_state = DislocationsState(ds=new_ds, ds_rt=new_d_rts)

            # Displace dislocations (excluding the currently added one).
            new_ds = []
            for i, d in enumerate(initial_d_state.ds[:-1]):
                initial_p = cp.asarray(d.position)
                current_u, _ = get_u_new(  # (3, )
                    crystal=crystal,
                    x=initial_p.reshape(1, -1),
                    initial_d_state=initial_d_state,
                    current_d_state=d_state,

                    # UWAGA: bierzemy tutaj initial_d_state ze wzgledu na to,
                    #  ze jest uzywany tylko do wyznaczenia x_o, a to jest w konfiguracji pierwotnej

                    d_n=initial_d_state.ds[-1],
                    # Exclude beta for the displaced dislocation
                    exclude_beta={i},
                    skip_np1=False,
                    n_points=n_points
                )
                current_u = current_u.squeeze()
                current_p = initial_p + current_u
                new_d = _set_d(d, position=current_p)
                new_ds.append(new_d)

            # Leave the d_n unmodified.
            new_ds.append(d_state.ds[-1])
            d_state = DislocationsState(ds=new_ds, ds_rt=new_d_rts)

    current_d_state = d_state

    # Find glide planes
    glide_planes = []
    for i in range(len(d_state.ds)):
        glide_plane = find_glide_plane(
            crystal, initial_d_state, margin=glide_plane_margin, dislocation_nr=i)
        glide_planes.append(glide_plane)

    # Update atom locations.
    initial_p = cp.asarray(initial_atoms_local)  # (n atoms, 3)

    u_atoms, aux = get_u_new(  # (n atoms, 3)
        crystal=crystal,
        x=initial_p,
        initial_d_state=initial_d_state,
        current_d_state=current_d_state,
        d_n=initial_d_state.ds[-1],
        skip_np1=skip_np1,
        n_points=n_points,
        debug=debug,
    )
    u_atoms = cp.stack(u_atoms)

    output_points = []
    if points:
        for pp in points_local:
            u_points, _ = get_u_new(  # (n atoms, 3)
                crystal=crystal,
                x=pp,
                initial_d_state=initial_d_state,
                current_d_state=initial_d_state,
                d_n=initial_d_state.ds[-1],
                skip_np1=skip_np1,
                n_points=n_points
            )
            pp = pp + u_points
            output_points.append(pp)
        
    log.log(d_state=current_d_state, u_atoms=u_atoms,
        glide_planes=glide_planes,
        points=output_points,
        integration_paths=aux["integration_paths"]
    )

    # Move back all the dislocations and atoms to the global coordinate system.
    # postprocess
    postprocessed_log = DisplacementLog()
    for u, d, gs, p, ip in zip(log.u_atoms, log.d_states, log.glide_planes, log.points, log.integration_paths):
        u = miller.postprocess(u).get()
        new_gs = []
        for g in gs:
            g = miller.postprocess_points(g).get()
            new_gs.append(g)

        new_points = []
        if points:
            for pp in p:
                if pp is not None:
                    pp = miller.postprocess_points(pp).get()
                    new_points.append(pp)

        new_ip = []
        
        if ip is not None:
            # ip: (n atoms, n points, 3)
            actual_shape = ip.shape
            ip = ip.reshape(-1, 3)  # (n total points, 3)
            ip = miller.postprocess_points(ip).get()
            ip = ip.reshape(actual_shape)  # (n atoms, n points, 3)
            new_ip.append(ip)

        new_ds = []
        for dislocation in d.ds:
            dislocation = miller.postprocess_dislocation(dislocation)
            new_ds.append(dislocation)
        d = dataclasses.replace(d, ds=new_ds)
        postprocessed_log.log(
            d_state=d,
            u_atoms=u,
            glide_planes=new_gs,
            points=new_points,
            integration_paths=new_ip
        )

    return postprocessed_log


def get_u_new(x, initial_d_state, crystal, d_n, skip_np1, current_d_state,
              exclude_beta: set = None, n_points=20000, debug=False):
    """
    :param x_0: the initial position of the atom (before iterating)
    """

    be, bz = get_be_bz(crystal.cell, d_n.b)
    x_o = cp.asarray([0.5 * be.item(), 0.0, 0.0])
    x_dash = x

    if exclude_beta is None:
        exclude_beta = set()

    n_x = x.shape[0]

    paths = []
    for i, x_d in enumerate(x_dash):

        # UWAGA: tutaj jest licze wg konfiguracji pierwotnej, gdyz sciezke
        #  calkowania tez mamy wg konfiguracji pierwotnej (int_{x_o}^{\dash{x}}

        print(f"Integration path: {i}", end="\r")
        p = get_integration_path(
            x_o=x_o, x_dash=x_d, d_state=initial_d_state, n_points=n_points,
            excluded_dislocations=exclude_beta,
        )
        paths.append(p)

    paths = cp.asarray(paths)  # (n atoms, n steps, 2)

    paths_orig = paths.copy()
    paths_orig = cp.pad(paths_orig, ((0, 0), (0, 0), (0, 1)))
    paths_orig[..., 2] = 0

    # Ignore 3rd dimension
    paths = paths[..., :2]  # (n atoms, n steps, 2)
    x_o = x_o[:2]  # (2, )

    # F_2_{\Sigma_N} (!)
    F_2_excluded_beta = exclude_beta.copy()
    F_2_excluded_beta.add(len(initial_d_state.ds) - 1)
    # result = integrate_paths_euler_parallel( # (n traj (atoms), n_steps, 2)

    F1 = lambda x: get_F(
        points=x,
        crystal=crystal,
        d_state=current_d_state,
        exclude_beta=exclude_beta
    )

    F2 = lambda x: get_F_inv(
        points=x,
        crystal=crystal,
        d_state=initial_d_state,
        exclude_beta=F_2_excluded_beta
    )


    # DEBUG
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        for d in initial_d_state.ds:
            d_pos = d.position
            d_pos = d_pos.get() if isinstance(d_pos, cp.ndarray) else d_pos            
            plt.scatter(d_pos[0], d_pos[1], color="red")
        for d in current_d_state.ds:
            d_pos = d.position
            d_pos = d_pos.get() if isinstance(d_pos, cp.ndarray) else d_pos
            plt.scatter(d_pos[0], d_pos[1], color="blue")
        plt.plot(paths.get()[129, :, 0], paths.get()[129, :, 1])
        plt.show()
            
    # if x_dash.shape[056] > 1:
    #     p = paths[19]
    #     v = F1(p)
    #     u = F2(p)
    #     for vv, uu in zip(v, u):
    #         yy = cp.dot(vv, uu)
    #         print(cp.linalg.norm(yy, ord="fro"))

    result = integrate_paths_euler_parallel( # (n traj (atoms), n_steps, 2)
        x0=x_o,
        path_points=paths,
        # F_{\Sigma_{N+1}}
        F1=F1,
        # F^{-1}_{\Sigma_N}
        F2=F2
    )
    result = result[:, -1, :]  # Use the final integration value (n atoms, 2)
    # Just for the backward compatibility -- return displacement instead of the
    # final position.
    result = cp.concatenate((result, cp.zeros((n_x, 1))), axis=1)
    u = result - x
    return u, {"integration_paths": paths_orig}


def get_F_inv(points, crystal, d_state, exclude_beta):
    beta_s = beta_sigma(
        points=points,
        crystal=crystal,
        d_state=d_state,
        exclude_beta=exclude_beta,
    )
    one = broadcast_eye(2, beta_s.shape[0])
    F_inv = (one - beta_s)
    return F_inv


def get_F(points, crystal, d_state, exclude_beta):
    F_inv = get_F_inv(points=points, crystal=crystal, d_state=d_state,
                      exclude_beta=exclude_beta).reshape((-1, 2, 2))
    det_mask = cp.isclose(cp.linalg.det(F_inv), 0)
    F_inv[det_mask, ...] = cp.eye(2)
    F = cp.linalg.inv(F_inv)
    # TODO: should be zero or one?
    F[det_mask, ...] = cp.eye(2)
    return F


def integrate_paths_euler_parallel(x0, path_points, F1, F2):
    """
    Euler integration for many independent paths in parallel on GPU (CuPy).

    Parameters
    ----------
    x0 : cp.ndarray, shape (2,)
        Common initial position for all trajectories.
    path_points : cp.ndarray, shape (n_traj, n_steps+1, 2)
        Path points for each trajectory.
    F1 : callable
        F1(y) -> (n_traj, 2, 2). Vectorized over batch of y.
    F2 : callable
        F2(p) -> (n_traj, 2, 2). Vectorized over batch of p.

    Returns
    -------
    traj : cp.ndarray, shape (n_traj, n_steps+1, 2)
        Trajectories of all initial points along their own path.
    """
    dl_list = cp.diff(path_points, axis=1)  # (n_traj (atoms), n_steps, 2)
    n_traj, n_steps = dl_list.shape[0], dl_list.shape[1]

    traj = cp.zeros((n_traj, n_steps+1, 2), dtype=cp.float64)
    traj[:, 0, :] = x0  # same initial point for all trajectories

    y = cp.broadcast_to(x0, (n_traj, 2)).copy()  # current states for each trajectory
    p = path_points[:, 0, :].copy()

    for i in range(n_steps):
        print(f"Step: {i}", end="\r")
        # F1(y_i) and F2(p_i) for all trajectories in batch
        F1y = F1(y)          # (n_traj, 2, 2)
        F2p = F2(p)          # (n_traj, 2, 2)
        mat = cp.matmul(F1y, F2p)  # (n_traj, 2, 2)

        dy = cp.einsum('nij,nj->ni', mat, dl_list[:, i, :])
        y = y + dy
        traj[:, i+1, :] = y
        p = p + dl_list[:, i, :]   # update path position per trajectory

    return traj


def integrate_paths_trapezoid_parallel(x0, path_points, F1, F2):
    """
    Trapezoidal integration for many independent paths in parallel on GPU (CuPy).

    Parameters
    ----------
    x0 : cp.ndarray, shape (2,)
        Common initial position for all trajectories.
    path_points : cp.ndarray, shape (n_traj, n_steps+1, 2)
        Path points for each trajectory.
    F1 : callable
        F1(y) -> (n_traj, 2, 2). Vectorized over batch of y.
    F2 : callable
        F2(p) -> (n_traj, 2, 2). Vectorized over batch of p.

    Returns
    -------
    traj : cp.ndarray, shape (n_traj, n_steps+1, 2)
        Trajectories of all initial points along their own path.
    """
    dl_list = cp.diff(path_points, axis=1)  # (n_traj, n_steps, 2)
    n_traj, n_steps = dl_list.shape[0], dl_list.shape[1]

    traj = cp.zeros((n_traj, n_steps+1, 2), dtype=cp.float64)
    traj[:, 0, :] = x0

    y = cp.broadcast_to(x0, (n_traj, 2)).copy()
    p = path_points[:, 0, :].copy()

    for i in range(n_steps):
        print(f"Trapezoid step: {i}", end="\r")
        dl = dl_list[:, i, :]

        # --- f(y_i, p_i) ---
        F1y = F1(y)
        F2p = F2(p)
        mat = cp.matmul(F1y, F2p)
        f_i = cp.einsum('nij,nj->ni', mat, dl)

        # provisional step y* and next path point p_{i+1}
        y_star = y + f_i
        p_next = p + dl

        # --- f(y*, p_{i+1}) ---
        F1y_star = F1(y_star)
        F2p_next = F2(p_next)
        mat_star = cp.matmul(F1y_star, F2p_next)
        f_star = cp.einsum('nij,nj->ni', mat_star, dl)

        # trapezoidal update
        dy = 0.5 * (f_i + f_star)
        y = y + dy
        traj[:, i+1, :] = y
        p = p_next

    return traj


def integrate_path_euler(x0, path_points, F1, F2):
    """
    NOTE: assuming, that x0, dl_list, F1, F2 are all 2D.
    """
    dl_list = cp.diff(path_points, axis=0)
    dl_list = cp.asarray(dl_list)
    y = cp.zeros((len(dl_list) + 1, 2))

    y[0] = x0

    # Current position of the traversed path (we start at the position p, then we are moving along the path).
    p = x0

    for i, dl in enumerate(dl_list):
        # F1(x_i) + F2(x^_i)
        mat = F1(y[i]) @ F2(p)
        y[i+1] = y[i] + mat @ dl
        p += dl

    return y


def delta_u(crystal, point, current_u, d_state, d_n, exclude_beta=None):
    """
    Calculate delta u = - jac(psi) * psi

    :param points: the CURRENT positions of the atoms/dislocations
    :param current_us: the CURRENT displacement field of the atoms/dislocations
    :param d_n: the introduced dislocation
    """
    if exclude_beta is None:
        exclude_beta = {}
    u = get_u(
        crystal=crystal,
        point=point,
        d_n=d_n,
        d_state=d_state, exclude_beta=exclude_beta
    )
    psi = current_u - u
    psi = psi.squeeze()
    psi_jac = cp.eye(3) + integrand(
        crystal=crystal,
        points=point.reshape(1, -1),
        d_state=d_state, exclude_beta=exclude_beta)
    psi_jac = psi_jac.squeeze()
    # TODO singularity? anyway, we should see errors if the given array
    #  is not reversible
    psi_jac_inv = cp.linalg.inv(psi_jac)
    return (-1)*cp.matmul(psi_jac_inv, psi)


@dataclasses.dataclass
class DislocationsState:
    """
    Describes the current configuration of dislocations.
    """
    ds: List[wzt.model.DislocationDef]
    ds_rt: List[cp.ndarray]

    def copy(self, **kwargs):
        return DislocationsState(**{**self.__dict__, **kwargs})

    def get_current_d_n(self):
        return self.ds[-1]


def get_integration_path(
        x_o, x_dash, d_state, n_points=30000, debug=False,
        excluded_dislocations=(),
    ):
    excluded_dislocations = set(excluded_dislocations)

    dislocation = d_state.ds[-1]
    d_y = dislocation.position[1]
    direction = cp.sign(x_dash.squeeze()[1] - cp.asarray(dislocation.position)[1])

    d_position = [cp.asarray(d.position).get()
                  for i, d in enumerate(d_state.ds)
                  if i not in excluded_dislocations]
    if direction == 0:
        raise ValueError("there should be no point located exactly at y=0")

    x_o = x_o.get()
    x_dash = x_dash.get()
    direction = direction.get()

    offset = direction * np.asarray([0.0, 2.0, 0.0])
    start = np.asarray(x_o).squeeze() + offset
    goal = x_dash

    DEFAULT_TOLERANCE = 1.0 - 1e-3
    DEFAULT_GLIDE_PLANE_TOLERANCE = 1e-3

    def is_restricted_by_dislocation_square(x, y, tolerance=DEFAULT_TOLERANCE, return_dislocation_nr=False):
        for d_i, d in enumerate(d_position):
            if abs(x-d[0]) <= tolerance and abs(y-d[1]) <= tolerance:
                if return_dislocation_nr:
                    return True, d_i
                else:
                    return True
        if return_dislocation_nr:
            return False, None
        else:
            return False

    def is_restricted_by_glide_plane(x, y, tolerance=DEFAULT_GLIDE_PLANE_TOLERANCE):
        if direction < 0.0:
            return y > d_y - tolerance
        else:
            return y < d_y + tolerance


    def is_restricted(x, y, tolerance_dislocation=DEFAULT_TOLERANCE, tolerance_glide_plane=DEFAULT_GLIDE_PLANE_TOLERANCE):
        return (
            is_restricted_by_dislocation_square(x, y, tolerance=tolerance_dislocation)
         or is_restricted_by_glide_plane(x, y, tolerance=tolerance_glide_plane)
        )

    inside_dis, nr = is_restricted_by_dislocation_square(x=x_dash[0], y=x_dash[1], return_dislocation_nr=True)
    if inside_dis:
        d_conflict_x, d_conflict_y, _ = d_position[nr].squeeze()
        offset = DEFAULT_TOLERANCE + 0.5
        left_point = (d_conflict_x - offset, x_dash[1])
        right_point = (d_conflict_x + offset, x_dash[1])
        up_point = (x_dash[0], d_conflict_y + offset)
        down_point = (x_dash[0], d_conflict_y - offset)
        ppp = (left_point, right_point, up_point, down_point)
        ppp = [np.asarray(p) for p in ppp if not is_restricted(p[0], p[1])]

        if not ppp:
            raise ValueError("Could not find closest point!")

        ppp_dist = [(p, np.sum(np.abs(np.asarray(p - x_dash[:2]))).item()) for p in ppp]
        goal = min(ppp_dist, key=lambda p: p[1])[0]



    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x_o[0], x_o[1], color="red")
        plt.scatter(x_dash[0], x_dash[1], color="green")
        plt.scatter(goal[0], goal[1], color="blue")

        for d in d_position:
            plt.scatter(d[0], d[1], color="orange")

        print("IS RESTRICTED: ", is_restricted(goal[0], goal[1]))

        plt.show()

    points = astar(start=start, goal=goal, is_restricted=is_restricted,
                   step=DEFAULT_TOLERANCE, goal_tol=DEFAULT_TOLERANCE+1e-1)

    points.insert(0, x_o.squeeze()[:2])
    if inside_dis:
        points.append(x_dash.squeeze()[:2])


    points = interpolate_path(points, n_points=n_points)
    points = np.stack(points)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(x_o[0], x_o[1], color="red")
        plt.scatter(x_dash[0], x_dash[1], color="green")

        for d in d_position:
            plt.scatter(d[0], d[1], color="orange")

        plt.plot(np.stack(points)[:, 0], np.stack(points)[:, 1])
        plt.show()

    return points


def get_u(point, d_n, crystal, d_state: DislocationsState,
          exclude_beta: set = None):
    be, bz = get_be_bz(crystal.cell, d_n.b)
    x_o = cp.asarray([0.5 * be.item(), 0.0, 0.0])
    x_dash = point

    if exclude_beta is None:
        exclude_beta = {}

    points = get_integration_path(x_o=x_o, x_dash=x_dash, d_state=d_state)
    vals = integrand(
        points=points,
        crystal=crystal,
        d_state=d_state,
        exclude_beta=exclude_beta
    )
    return line_integral(path=points, vals=vals)


def integrand(points, crystal, d_state, exclude_beta):
    n_points, n_dims = points.shape
    beta_s, beta_i = beta_sigma(
        points=points,
        crystal=crystal,
        d_state=d_state,
        exclude_beta=exclude_beta,
        # the currently inserted dislocation -- the last one in the state
        return_beta=len(d_state.ds)-1
    )

    one = broadcast_eye(2, beta_s.shape[0])
    F_inv = (one - beta_s)

    det_mask = cp.isclose(cp.linalg.det(F_inv), 0)
    F_inv[det_mask, ...] = cp.eye(2)
    F = cp.linalg.inv(F_inv)
    # TODO: should be zero or one?
    F[det_mask, ...] = cp.eye(2)

    result = cp.zeros((n_points, n_dims, n_dims))
    result[:, :2, :2] = cp.matmul(F, beta_i)
    return result


def beta_sigma(points: cp.ndarray, crystal, d_state: DislocationsState,
               exclude_beta: Optional[Set] = None,
               return_beta: Optional[int] = None):
    if exclude_beta is None:
        exclude_beta = {}
    n_points, dims = points.shape
    result = cp.zeros((n_points, 2, 2))

    returned_beta = None
    for i, (d, d_rt) in enumerate(zip(d_state.ds, d_state.ds_rt)):
        if i not in exclude_beta:
            beta = beta_rotated(
                points=points,
                crystal=crystal,
                d=d,
                rotation_matrix=d_rt,
            )
            result += beta
            if return_beta == i:
                returned_beta = beta
    if returned_beta is not None:
        return result, returned_beta
    else:
        return result


def beta_rotated(crystal, d, points, rotation_matrix, dis_tolerance=DIS_TOLERANCE):
    if rotation_matrix is None:
        return cp.zeros((points.shape[0], 2, 2))

    be, bz = get_be_bz(crystal.cell, d.b)
    # Przenieś do układu zaczepionego w dyslokacji (istotne np. dla d1).
    points = cp.asarray(points) - cp.asarray(d.position[:2]).reshape(1, -1)
    points = rotation_matrix.T.dot(points.T).T
    betas = beta(points, be=be, bz=bz)[:, :2, :2]
    rm = rotation_matrix[:2, :2].reshape(1, 2, 2)
    # KLUCZOWA INSTRUKCJA
    betas = rm @ betas @ rm.transpose((0, 2, 1))
    return betas


def beta(x, be, bz):
    x = cp.asarray(x)
    if len(x.shape) == 1:
        x = x[cp.newaxis, ...]

    x1 = x[..., 0]  # (n_atoms, )
    x2 = x[..., 1]  # (n_atoms, )
    x1_2 = x1 ** 2  # (n_atoms, )
    x2_2 = x2 ** 2  # (n_atoms, )
    r2 = x1_2 + x2_2 # (n_atoms, )

    a = be / (4 * cp.pi * (1.0 - NU) * r2 * r2)
    # du / dx1
    b11 = (-1) * a * x2 * (
                (3.0 - 2.0 * NU) * x1_2 + (1.0 - 2.0 * NU) * x2_2)  # (natoms, )
    b21 = (-1) * a * x1 * ((1.0 - 2.0 * NU) * x1_2 + (3.0 - 2.0 * NU) * x2_2)
    b31 = (-1) * bz / (2.0 * cp.pi) * x2 / r2
    # du / dx2
    b12 = a * x1 * ((3.0 - 2.0 * NU) * x1_2 + (1.0 - 2.0 * NU) * x2_2)
    b22 = a * x2 * ((1.0 + 2.0 * NU) * x1_2 - (1.0 - 2.0 * NU) * x2_2)
    b32 = bz / (2.0 * cp.pi) * x1 / r2
    result = cp.repeat(BETA_ONES.copy()[cp.newaxis, ...], len(x1), axis=0)
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


def rotate_dislocation(crystal, d_state, rotated_dislocation, exclude_beta):
    """
    We are rotating according to the reference -- the currently introduced
    dislocation.

    :param reference_dislocation: introduced dislocation
    """
    bv, p = rotated_dislocation.b, rotated_dislocation.position[:2]
    bv = cp.asarray(bv)

    rm = get_rotation_matrix(
        p=p, bv=bv,
        crystal=crystal, d_state=d_state,
        exclude_beta=exclude_beta
    )
    global_rm = get_rotation_matrix(
        p=p, bv=[1.0, 0.0, 0.0],
        crystal=crystal, d_state=d_state,
        exclude_beta=exclude_beta
    )
    # Rotate
    b = global_rm[:2, :2].dot(bv[:2]).squeeze()
    # normalize/restore the initial norm
    orig_norm = cp.linalg.norm(bv)
    b = b/cp.linalg.norm(b)*orig_norm
    b = cp.asarray(b.tolist() + [0])
    new_d = _set_d(rotated_dislocation, b=b)

    return new_d, rm


def estimate_rotation_matrix_from_vector(vector):
    # normalizacja wektora (x, y)

    x, y, z = vector.squeeze()
    norm = np.sqrt(x**2 + y**2)
    if norm == 0:
        raise ValueError("The vector cannot be zero")
    x, y = x / norm, y / norm
    theta = cp.arctan2(y, x)

    return cp.array([
        [cp.cos(theta), -cp.sin(theta)],
        [cp.sin(theta),  cp.cos(theta)]
    ])


def get_rotation_matrix(crystal, d_state, p, bv, exclude_beta):
    """
    Returns the given

    :param dislocation: the dislocation that causes the rotation
    :param p: point where the rotation we want to calculate
    """
    p = cp.asarray(p).reshape(1, -1)
    n_points = p.shape[0]
    BETA_ONES = broadcast_eye(2, n_points)
    betas = beta_sigma(
        points=p,
        crystal=crystal,
        d_state=d_state,
        exclude_beta=exclude_beta
    )
    F_inv = (BETA_ONES - betas)
    F = cp.linalg.inv(F_inv[0, :2, :2])

    ba = cp.asarray(bv).squeeze()[:2]

    ba_rotated = F.dot(ba)
    ba = ba_rotated / cp.linalg.norm(ba_rotated)
    ba_orto = cp.array([[0, -1],
                        [1, 0]]).dot(ba)
    ba_z = cp.asarray([0, 0, 1])

    rotmatrix = cp.eye(2)
    rotmatrix[:2, 0] = ba
    rotmatrix[:2, 1] = ba_orto
    return rotmatrix


def _set_d(dislocation, **kwargs):
    """
    Creates the copy of an object the given dislocation with the given
    kwargs replaced.
    """
    return dataclasses.replace(dislocation, **kwargs)


## Glide plane
def get_glide_plane(crystal, d_state, dislocation_nr, margin=45):
    d_n = d_state.ds[dislocation_nr]
    # NOTE: y0 must be determined for system located in the d2
    def func(t, y):
        point = np.asarray([t, y.item()]).reshape(1, -1)
        betas = beta_sigma(
            points=point,
            crystal=crystal,
            d_state=d_state,
            exclude_beta={dislocation_nr}
        ).get().squeeze()  # TODO avoid GPU -> CPU
        F_inv = np.eye(2) - betas
        return np.asarray([F_inv[1, 0] / F_inv[1, 1]])
    position = d_n.position 
    if isinstance(position, cp.ndarray):
        position = position.get()
    position = np.asarray(position).copy()
    # Just don't start too close to the current dislocation
    x0, y0, _ = np.squeeze(position)
    # TODO: rotate according to the burgers vector (this will not work for e.g. b = (1, 1, 0))
    t_span = (x0, -margin)

    # TODO zmienic ponizsze tak, zeby bylo bardziej ogolne
    ode_res = scipy.integrate.solve_ivp(
        func,
        t_span=t_span, y0=[y0],
        method="BDF",
        rtol=1e-8, atol=1e-10, max_step=0.1
    )
    n_points = len(ode_res.t)
    result = np.zeros((n_points, 3))
    # left side
    result[:, 0] = np.flip(np.squeeze(ode_res.t))
    result[:, 1] = np.flip(np.squeeze(-ode_res.y))
    return result


def find_glide_plane(crystal, d_state, dislocation_nr, margin=45):
    return cp.asarray(get_glide_plane(crystal=crystal, d_state=d_state,
                                      dislocation_nr=dislocation_nr,
                                      margin=margin))