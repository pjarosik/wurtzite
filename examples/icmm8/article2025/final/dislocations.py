from typing import List, Optional, Set

import wurtzite as wzt
import numpy as np
import matplotlib.pyplot as plt
from utils_1st import displace_love2
import scipy.integrate
import dataclasses
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
    """

    def __init__(self):
        self.d_states = []
        self.u_atoms = []

    def log(self, d_state, u_atoms):
        self.d_states.append(d_state)
        self.u_atoms.append(u_atoms)

    def get_u_atoms(self):
        return cp.stack(self.u_atoms)

    @property
    def last_d_state(self):
        return self.d_states[-1]

    @property
    def last_u_atoms(self):
        return self.u_atoms[-1]


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


def displace(crystal, dislocations, d_n, n_iters=3, alpha=1.0):
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
    initial_dn_local = dataclasses.replace(d_n, position=[0.0, 0, 0])
    # - Other dislocations
    d_positions = cp.asarray([d.position for d in dislocations])  # (n disl. 3)
    d_positions = miller.preprocess(d_positions)  # (n disl. 3)
    initial_ds_local = [_set_d(d_i, position=p)
                for d_i, p in zip(dislocations, d_positions)]
    # - Atoms
    initial_atoms_local = miller.preprocess(cp.asarray(crystal.coordinates))
    all_dislocations_local = initial_ds_local + [initial_dn_local]

    # ALl entities in the d_state is assumed to be located in the coordinate
    # system centered in d_n.
    d_state = DislocationsState(
        ds=all_dislocations_local,
        ds_rt=[cp.eye(2) for _ in range(len(all_dislocations_local))]
    )

    initial_d_state = d_state
    log = DisplacementLog()

    # Initialize with u = 0.
    log.log(d_state=d_state, u_atoms=cp.zeros(shape=initial_atoms_local.shape))

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

            # Displace dislocations (a single Newton procedure step).
            # (excluding the currently added one).
            new_ds = []
            for i, d in enumerate(d_state.ds[:-1]):
                current_p = cp.asarray(d.position)
                # initial_p = cp.asarray(initial_ds_local[i].position)
                # current_u = current_p - initial_p
                # du = delta_u(
                #     crystal=crystal,
                #     point=d.position,
                #     current_u=current_u,
                #     d_state=d_state,
                #     d_n=d_state.ds[-1],
                #     # Exclude beta for the displaced dislocation
                #     exclude_beta={i}
                # )
                # current_u = current_u + du
                current_u = get_u_new(  # (3, )
                    crystal=crystal,
                    points=current_p.reshape(1, -1),
                    d_state=d_state,
                    d_n=d_state.ds[-1],
                    # Exclude beta for the displaced dislocation
                    exclude_beta={i}
                ).squeeze()
                current_p = current_p + current_u
                new_d = _set_d(d, position=current_p)
                new_ds.append(new_d)

            new_ds.append(d_state.ds[-1])

            d_state = DislocationsState(ds=new_ds, ds_rt=new_d_rts)

        # Update atom locations.
        atoms_d_state = d_state
        # atoms_d_state = initial_d_state
        initial_p = cp.asarray(cp.coordinates)  # (n atoms, 3)

        u_atoms = get_u_new(  # (n atoms, 3)
            crystal=crystal,
            point=initial_p,
            d_state=atoms_d_state,
            d_n=atoms_d_state.ds[-1],
        )
        u_atoms = cp.stack(u_atoms)
        log.log(d_state=atoms_d_state, u_atoms=u_atoms)

    # Move back all the dislocations and atoms to the global coordinate system.
    # postprocess
    postprocessed_log = DisplacementLog()
    for u, d in zip(log.u_atoms, log.d_states):
        u = miller.postprocess(u).get()
        d = postprocess_dislocations(d_state=d,  miller=miller)

        postprocessed_log.log(d_state=d, u_atoms=u)

    return postprocessed_log


def get_u_new(points, d_state, crystal, d_n, exclude_beta: set = None, n_points=100, debug=False):
    be, bz = get_be_bz(crystal.cell, d_n.b)
    x_o = cp.asarray([0.5 * be.item(), 0.0, 0.0])
    x_dash = points

    if exclude_beta is None:
        exclude_beta = set()

    n_atoms = points.shape[0]

    points = []
    for point in points:
        p = get_integration_path(
            x_o=x_o, x_dash=x_dash, dislocation=d_n, n_points=n_points
        )
        points.append(p)

    # Ignore 3rd dimension
    points = points[:, :2]
    x_o = x_o[:2]

    # F_2_{\Sigma_N} (!)
    F_2_excluded_beta = exclude_beta.copy()
    F_2_excluded_beta.add(len(d_state.ds)-1)

    result = integrate_paths_euler_parallel( # (n traj (atoms), n_steps, 2)
        x0=x_o, path_points=points,
        # F_{\Sigma_{N+1}}
        F1=lambda x: get_F(
            points=x,
            crystal=crystal,
            d_state=d_state,
            exclude_beta=exclude_beta
        ),
        # F_{\Sigma_N}
        F2=lambda x: get_F_inv(
            points=x,
            crystal=crystal,
            d_state=d_state,
            exclude_beta=F_2_excluded_beta
        )
    )
    result = result[:, -1, :]  # Use the final integration value (n atoms, 2)
    # Just for the backward compatibility -- return displacement instead of the
    # final position.
    result = cp.concatenate((result, cp.zeros((n_points, 1))), axis=1)
    u = result - point
    return u


def get_F_inv(points, crystal, d_state, exclude_beta):
    points = points.reshape(1, -1)
    beta_s = beta_sigma(
        points=points,
        crystal=crystal,
        d_state=d_state,
        exclude_beta=exclude_beta,
    )
    one = broadcast_eye(2, beta_s.shape[0])
    F_inv = (one - beta_s)
    return F_inv.squeeze()


def get_F(points, crystal, d_state, exclude_beta):
    F_inv = get_F_inv(points=points, crystal=crystal, d_state=d_state,
                      exclude_beta=exclude_beta).reshape((1, 2, 2))
    det_mask = cp.isclose(cp.linalg.det(F_inv), 0)
    F_inv[det_mask, ...] = cp.eye(2)
    F = cp.linalg.inv(F_inv)
    # TODO: should be zero or one?
    F[det_mask, ...] = cp.eye(2)
    return F.squeeze()


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
    dl_list = cp.diff(path_points, axis=1)  # (n_traj, n_steps, 2)
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


def get_integration_path(x_o, x_dash, dislocation, n_points=1000):
    # TODO replace with RTT
    direction = cp.sign(x_dash.squeeze()[1] - cp.asarray(dislocation.position)[1])
    if direction == 0:
        raise ValueError("there should be no point located exactly at y=0")
    offset = direction * cp.asarray([0.0, 10.0, 0.0])
    x_o1 = cp.asarray(x_o).squeeze() + offset
    x_o2 = cp.asarray([x_dash.squeeze()[0].item(), x_o1[1].item(), 0])

    l1 = get_line(x_o, x_o1, n=n_points)
    l2 = get_line(x_o1, x_o2, n=n_points)
    l3 = get_line(x_o2, x_dash, n=n_points)
    points = cp.concatenate((l1, l2, l3), axis=0)
    return points


def get_u(point, d_n, crystal, d_state: DislocationsState,
          exclude_beta: set = None):
    be, bz = get_be_bz(crystal.cell, d_n.b)
    x_o = cp.asarray([0.5 * be.item(), 0.0, 0.0])
    x_dash = point

    if exclude_beta is None:
        exclude_beta = {}

    points = get_integration_path(x_o=x_o, x_dash=x_dash, dislocation=d_n)
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
    points = rotation_matrix.dot(points.T).T
    betas = beta(points, be=be, bz=bz)[:, :2, :2]
    rm = rotation_matrix[:2, :2].reshape(1, 2, 2)
    # KLUCZOWA INSTRUKCJA
    betas = rm.transpose((0, 2, 1)) @ betas @ rm
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
        p=p,
        crystal=crystal, d_state=d_state,
        exclude_beta=exclude_beta
    )
    b = rm[:2, :2].dot(bv[:2]).squeeze()
    orig_norm = cp.linalg.norm(bv)
    b = b/cp.linalg.norm(b)*orig_norm
    b = cp.asarray(b.tolist() + [0])
    new_d = _set_d(rotated_dislocation, b=b)
    return new_d, rm


def get_rotation_matrix(crystal, d_state, p, exclude_beta):
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

    ba = cp.asarray([1.0, 0.0])

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