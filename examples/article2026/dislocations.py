import os
from gpu_compatibility import xp, d2h, h2d
import numpy as np
from typing import List, Optional, Set

import wurtzite as wzt
import matplotlib.pyplot as plt
import scipy.integrate
import dataclasses
from astar import astar, interpolate_path
import copy
from scipy.integrate import quad_vec
import sys
from utils import get_be_bz, broadcast_eye, get_line, line_integral, \
    MillerIndices

from constants import *
import heapq
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


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
        self.energies = []
        # Convergence history of the iterative scheme solving Eq. (29):
        # a list of (max ||Psi_d||, max ||Delta x_d||) pairs [A], one per
        # iteration.
        self.convergence = []

    def log(self, d_state, u_atoms, glide_planes, points=None, integration_paths=None, energy=None):
        self.d_states.append(d_state)
        self.u_atoms.append(u_atoms)
        self.glide_planes.append(glide_planes)
        self.points.append(points)
        self.integration_paths.append(integration_paths)
        self.energies.append(energy)

    def get_u_atoms(self):
        return xp.stack(self.u_atoms)

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

    @property
    def last_energy(self):
        return self.energies[-1]


def postprocess_dislocations(d_state, miller):
    new_ds = []
    for i, d in enumerate(d_state.ds):
        current_pos = h2d(d.position).reshape(1, -1)
        new_pos = miller.postprocess_points(current_pos).squeeze()

        b = d.b
        b = d2h(b)
        d = _set_d(d, b=b)
        d = _set_d(d, position=d2h(new_pos))
        new_ds.append(d)
    return DislocationsState(ds=new_ds, ds_rt=d_state.ds_rt)


def displace(crystal, dislocations, d_n, n_iters=3, alpha=1.0, skip_np1=False, n_points=30000,
             glide_plane_margin=35, points=(), debug=False, only_inv=False, skip_ds=False, skip_atoms=False,
             same_excluded=False, only_noninv=False, plot_local_planes=False, plot_local=False, custom_paths=dict(),
             method="picard", tol=None):
    """
    Displaces the given crystal lattice according to the displacements
    caused by the dislocation d.

    :param dislocations: a list of dislocations already inserted in the crystal
    :param d_n: dislocation that we are inserting
    :param method: the method used to solve the nonlinear equation set (29)
      for the positions of the already inserted dislocations:
      "picard" -- successive substitution, x^{i+1} = int(...) (the historical
      behaviour of this function),
      "newton" -- the (modified) Newton-Raphson scheme, Eqs. (30)-(33):
      x^{i+1} = x^i - alpha * J^{-1} Psi^i.
    :param tol: convergence criterion [A]: the iterations are stopped as soon as
      max_d ||Delta x_d|| < tol. `None` means: always perform `n_iters`
      iterations.
    """

    # Move all entities into the coordinate system centered in the dislocation d.
    miller = MillerIndices(crystal=crystal, dislocation=d_n)
    # (n atoms, 3)
    # - The introduced dislocation (move to the (0, 0, 0))
    initial_dn_local = dataclasses.replace(d_n, position=[0.0, 0, 0], b=[1.0, 0.0, 0.0])
    # - Other dislocations
    initial_ds_local = [miller.preprocess_dislocation(d) for d in dislocations]
    # - Atoms
    initial_atoms_local = miller.preprocess(h2d(crystal.coordinates))
    all_dislocations_local = initial_ds_local + [initial_dn_local]

    points_local = None
    if points:
        points_local = [miller.preprocess(h2d(p)) for p in points]

    # All entities in the d_state is assumed to be located in the coordinate
    # system centered in d_n.
    d_state = DislocationsState(
        ds=all_dislocations_local,
        ds_rt=[xp.eye(2) for _ in range(len(all_dislocations_local))]
    )

    log = DisplacementLog()

    glide_planes = []
    for i in range(len(d_state.ds)):
        glide_plane = find_glide_plane(
            crystal, d_state, margin=glide_plane_margin, dislocation_nr=i)
        glide_planes.append(glide_plane)

    # Initialize with u = 0.
    log.log(d_state=d_state, u_atoms=xp.zeros(shape=initial_atoms_local.shape),
            glide_planes=glide_planes, points=points_local)

    initial_d_state = d_state

    # Convergence history of the iterative scheme solving Eq. (29).
    # One entry per iteration: (max ||Psi_d||, max ||Delta x_d||) [A].
    convergence = []

    for i in range(n_iters):
        if len(d_state.ds) > 1 and not skip_ds:
            # More than one dislocation -- we need to update the rotation
            # and location of each previous dislocation.

            # Rotate dislocations (including the currently added one).
            new_ds = []
            new_d_rts = []
            for j, d in enumerate(d_state.ds):
                if method == "picard":
                    # obrot tylko raz
                    rotated = d
                else:
                    # The Burgers vector is always rotated starting from its
                    # REFERENCE orientation, so that the rotations do not
                    # accumulate over the subsequent iterations.
                    rotated = _set_d(initial_d_state.ds[j], position=d.position)
                new_d = rotate_dislocation(
                    crystal=crystal, d_state=d_state,
                    rotated_dislocation=rotated,
                    exclude_beta={j}
                )
                new_ds.append(new_d)
                # IGNORE
                new_d_rts.append(xp.eye(2))

            d_state = DislocationsState(ds=new_ds, ds_rt=new_d_rts)

            # Number of dislocations already present in the lattice -- their
            # positions are the unknowns of the equation set (29). The position
            # of the inserted dislocation (the last one) is fixed.
            n_prev = len(initial_d_state.ds) - 1

            # Residuals Psi_d, Eq. (32), and the derivatives d Psi_d/d x_n,
            # Eq. (33), evaluated at the current iterate.
            psis = []
            jacs = []
            integrals = []
            for j in range(n_prev):
                # The integration path always ends in the reference position
                # \hat{x}_d of the dislocation.
                reference_p = h2d(initial_d_state.ds[j].position)
                jacobian_dislocations = None
                if method == "newton":
                    jacobian_dislocations = set(range(n_prev)) - {j}
                current_u, aux_d = get_u_new(  # (3, )
                    crystal=crystal,
                    x=reference_p.reshape(1, -1),
                    initial_d_state=initial_d_state,
                    current_d_state=d_state,
                    d_n=initial_d_state.ds[-1],
                    # Exclude beta for the displaced dislocation
                    exclude_beta={j},
                    skip_np1=False,
                    n_points=n_points,
                    jacobian_dislocations=jacobian_dislocations,
                )
                # The value of the line integral of Eq. (29).
                integral = reference_p + current_u.squeeze()
                integrals.append(integral)
                psis.append(integral - h2d(d_state.ds[j].position))
                jacs.append(aux_d["jacobian"])

            if method == "picard":
                # x^{i+1} = int(...), i.e. the integral is taken as the new
                # position, and the correction is measured with respect to the
                # reference position of the dislocation.
                deltas = [integrals[j] - h2d(initial_d_state.ds[j].position)
                          for j in range(n_prev)]
            elif method == "newton":
                deltas = newton_raphson_step(psis=psis, jacs=jacs, alpha=alpha)
            else:
                raise ValueError(f"Unknown method: {method}")

            residual = max(float(xp.linalg.norm(h2d(p)[:2])) for p in psis)
            step = max(float(xp.linalg.norm(h2d(dx)[:2])) for dx in deltas)
            convergence.append((residual, step))
            print(f"Iteration {i}: max||Psi|| = {residual:.3e} A, "
                  f"max||dx|| = {step:.3e} A")

            new_ds = []
            for j in range(n_prev):
                if method == "picard":
                    # NOTE: historically the Burgers vector determined in the
                    # rotation step above was dropped for the previous
                    # dislocations, and the correction was always referred to
                    # the reference position.
                    d = initial_d_state.ds[j]
                else:
                    d = d_state.ds[j]
                new_p = h2d(d.position) + deltas[j]
                new_ds.append(_set_d(d, position=new_p))

            # Leave the d_n unmodified.
            new_ds.append(d_state.ds[-1])
            d_state = DislocationsState(ds=new_ds, ds_rt=new_d_rts)

            if tol is not None and step < tol:
                print(f"Converged after {i+1} iteration(s) "
                      f"(max||dx|| = {step:.3e} A < {tol:.3e} A).")
                break

    current_d_state = d_state

    # Find glide planes
    glide_planes = []
    for i in range(len(d_state.ds)):
        glide_plane = find_glide_plane(
            crystal, initial_d_state, margin=glide_plane_margin, dislocation_nr=i, debug=True)
        glide_planes.append(glide_plane)

    if plot_local:
        # DEBUG PLOTTING

        import matplotlib.pyplot as plt
        import wurtzite as wzt
        local_crystal = dataclasses.replace(
            crystal,
            coordinates=d2h(initial_atoms_local)
        )
        # Plotting beta.
        npoints = 1000
        x = np.linspace(-40, 30, npoints)
        y = np.linspace(-40, 40, npoints)
        X, Y = np.meshgrid(x, y)

        pp = np.column_stack([X.ravel(), Y.ravel()])
        pp = h2d(pp)
        _, bb = beta_sigma(points=pp, crystal=crystal, d_state=initial_d_state, return_beta=0)

        checkpoints = [(0, 0), (0, 1), (1, 0), (1, 1)]
        titles = [r"$\beta_{1, " + f"{c0+1}{c1+1}" + "}$" for c0, c1 in checkpoints]

        fig, axes = plt.subplots(2, 2, constrained_layout=True)

        for checkpoint, title in zip(checkpoints, titles):
            ax = axes[checkpoint[0], checkpoint[1]]
            wzt.visualization.plot_atoms_2d(
                local_crystal, xlim=(-45, 45), ylim=(-15, 25), fig=fig, ax=ax
            )
            ax.set_title(title)
            bbb = bb[:, checkpoint[0], checkpoint[1]]
            bbb = bbb.reshape(X.shape)
            cs = ax.contourf(
                X, Y, d2h(xp.log10(np.abs(bbb)+1e-9)), levels=50,
                cmap="jet",
                zorder=-1000
            )
            fig.colorbar(cs, ax=ax, orientation="vertical")
            for d in initial_d_state.ds:
                if isinstance(d.position, xp.ndarray):
                    d = dataclasses.replace(d, position=d2h(d.position))
                if isinstance(d.b, xp.ndarray):
                    d = dataclasses.replace(d, b=d2h(d.b))
                wzt.visualization.display_tee_2d(ax, d, scale=0.5)

            def make_format():
                def format_coord(x_mouse, y_mouse):
                    col = np.searchsorted(x, x_mouse) - 1
                    row = np.searchsorted(y, y_mouse) - 1
                    if 0 <= col < len(x) and 0 <= row < len(y):
                        z = bbb[row, col]
                        return f"x={x_mouse:.2f}, y={y_mouse:.2f}, z={z:.3f}"
                    else:
                        return f"x={x_mouse:.2f}, y={y_mouse:.2f}"
                return format_coord

            ax.format_coord = make_format()
        plt.show()

    if plot_local_planes:
        # DEBUG PLOTTING
        import matplotlib.pyplot as plt
        import wurtzite as wzt
        local_crystal = dataclasses.replace(
            crystal,
            coordinates=d2h(initial_atoms_local)
        )
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

        wzt.visualization.plot_atoms_2d(
                local_crystal, xlim=(-45, 45), ylim=(-15, 25), fig=fig, ax=ax
            )
        for d in initial_d_state.ds:
            if isinstance(d.position, xp.ndarray):
                d = dataclasses.replace(d, position=d2h(d.position))
            if isinstance(d.b, xp.ndarray):
                d = dataclasses.replace(d, b=d2h(d.b))
            wzt.visualization.display_tee_2d(ax, d, scale=0.5)

        ax.plot(glide_planes[-1][:, 0], glide_planes[-1][:, 1])
        plt.show()



    # Update atom locations.
    initial_p = h2d(initial_atoms_local)  # (n atoms, 3)

    u_atoms = xp.zeros(crystal.coordinates.shape)
    aux = {"integration_paths": None, "energies": None}

    if not skip_atoms:
        u_atoms, aux = get_u_new(  # (n atoms, 3)
            crystal=crystal,
            x=initial_p,
            initial_d_state=initial_d_state,
            current_d_state=current_d_state,
            d_n=initial_d_state.ds[-1],
            skip_np1=skip_np1,
            n_points=n_points,
            debug=debug,
            only_inv=only_inv,
            glide_planes=glide_planes,
            same_excluded=same_excluded,
            only_noninv=only_noninv,
            path_via_energy=True,
            atom_local_coordinates=initial_p,
            custom_paths=custom_paths
        )
    u_atoms = xp.stack(u_atoms)

    output_points = []
    if points:
        for pp in points_local:
            u_points, _ = get_u_new(  # (n atoms, 3)
                crystal=crystal,
                x=pp,
                initial_d_state=initial_d_state,
                current_d_state=current_d_state,
                d_n=initial_d_state.ds[-1],
                skip_np1=skip_np1,
                n_points=n_points,
                same_excluded=same_excluded,
                only_noninv=only_noninv,
            )
            pp = pp + u_points
            output_points.append(pp)

    log.log(d_state=current_d_state, u_atoms=u_atoms,
        glide_planes=glide_planes,
        points=output_points,
        integration_paths=aux["integration_paths"],
        energy=aux["energies"]
    )

    # Move back all the dislocations and atoms to the global coordinate system.
    # postprocess
    postprocessed_log = DisplacementLog()
    for u, d, gs, p, ip in zip(log.u_atoms, log.d_states, log.glide_planes, log.points, log.integration_paths):
        u = d2h(miller.postprocess(u))
        new_gs = []
        for g in gs:
            g = d2h(miller.postprocess_points(g))
            new_gs.append(g)

        new_points = []
        if points:
            for pp in p:
                if pp is not None:
                    pp = d2h(miller.postprocess_points(pp))
                    new_points.append(pp)

        new_ip = []

        if ip is not None:
            # ip: (n atoms, n points, 3)
            actual_shape = ip.shape
            ip = ip.reshape(-1, 3)  # (n total points, 3)
            ip = d2h(miller.postprocess_points(ip))
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
    postprocessed_log.convergence = convergence

    return postprocessed_log


def get_u_new(x, initial_d_state, crystal, d_n, skip_np1, current_d_state,
              exclude_beta: set = None, n_points=20000, debug=False, only_inv=False,
              glide_planes=None, same_excluded=False, only_noninv=False,
              path_via_energy=False, atom_local_coordinates=None, custom_paths=dict(),
              jacobian_dislocations=None):
    """
    :param x_0: the initial position of the atom (before iterating)
    :param jacobian_dislocations: a collection of dislocation numbers for which
      the derivative of the computed line integral with respect to the
      dislocation position should be determined (Eq. (33)); the derivatives are
      returned in aux["jacobian"] as a dictionary
      dislocation number -> (n x, 2, 2) array

    NOTE: the crystal.coordinates are in the GLOBAL coordinate system; please
    atom_local_coordinates instead.
    """

    be, bz = get_be_bz(crystal.cell, d_n.b)
    x_o = h2d([0.5 * be.item(), 0.0, 0.0])
    x_dash = x

    if exclude_beta is None:
        exclude_beta = set()

    n_x = x.shape[0]

    paths = []
    energies = h2d([])
    if path_via_energy:
        energies = calculate_energies(l=crystal, d_state=initial_d_state,
                                      local_coordinates=atom_local_coordinates)
    energies = h2d(energies)

    for i, x_d in enumerate(x_dash):
        # UWAGA: tutaj jest licze wg konfiguracji pierwotnej, gdyz sciezke
        #  calkowania tez mamy wg konfiguracji pierwotnej (int_{x_o}^{\dash{x}}

        print(f"Integration path: {i}", end="\r")
        custom_path_fn = custom_paths.get(i)
        if glide_planes is not None:
            glide_plane = glide_planes[-1]
        else:
            glide_plane = None
        if custom_path_fn:
            p = custom_path_fn(
                x_o=x_o, x_dash=x_d, d_state=initial_d_state, n_points=n_points,
                energies=energies, excluded_dislocations=exclude_beta,
                glide_plane=glide_plane, lattice=crystal,
                atom_local_coordinates=atom_local_coordinates
            )
        elif not path_via_energy:
            p = get_integration_path(
                x_o=x_o, x_dash=x_d, d_state=initial_d_state, n_points=n_points,
                excluded_dislocations=exclude_beta, glide_plane=glide_plane
            )
        else:
            p = get_integration_path_via_energy(
                x_o=x_o, x_dash=x_d, d_state=initial_d_state, n_points=n_points,
                energies=energies, excluded_dislocations=exclude_beta,
                glide_plane=glide_plane, lattice=crystal,
                atom_local_coordinates=atom_local_coordinates
            )
        paths.append(p)

            # TODO only for debug purposes!
        # if i == 223 and only_inv:
        #     print("APPENDING ONLY!")
        #     p = paths[221].copy()
        #     # print(p.shape)
        #     # print(x_d)
        #     p[-1, :] = x_d[:2]
        #     # p = np.concatenate((p, x_d[:2].reshape(1, 2)))
        #     # print(p.shape)
    # if only_inv:
    #     p = paths[223].copy()
    #     p[-1, :] = x_dash[221, :2]
    #     paths[221] = p

    paths = h2d(paths)  # (n atoms, n steps, 2)

    paths_orig = paths.copy()
    paths_orig = xp.pad(paths_orig, ((0, 0), (0, 0), (0, 1)))
    paths_orig[..., 2] = 0

    # Ignore 3rd dimension
    paths = paths[..., :2]  # (n atoms, n steps, 2)
    x_o = x_o[:2]  # (2, )

    # F_2_{\Sigma_N} (!)
    F_2_excluded_beta = exclude_beta.copy()
    F_2_excluded_beta.add(len(initial_d_state.ds) - 1)

    F1_exclude_beta = exclude_beta
    if same_excluded:
        F1_exclude_beta = F_2_excluded_beta

    F1 = lambda x: get_F(
        points=x,
        crystal=crystal,
        d_state=current_d_state,
        exclude_beta=F1_exclude_beta
    )

    F2 = lambda x: get_F_inv(
        points=x,
        crystal=crystal,
        d_state=initial_d_state,
        exclude_beta=F_2_excluded_beta
    )

    jacobian = None
    if jacobian_dislocations:
        dbetas = {
            n: (lambda y, n=n: dbeta_sigma_single(
                points=y, crystal=crystal, d_state=current_d_state,
                dislocation_nr=n))
            for n in jacobian_dislocations
        }
        result, jacobian = integrate_paths_euler_parallel_with_jacobian(
            x0=x_o, path_points=paths, F1=F1, F2=F2, dbetas=dbetas
        )
    else:
        if only_inv:
            func = integrate_paths_euler_parallel_only_inv
        elif only_noninv:
            func = integrate_paths_euler_parallel_only_noninv
        else:
            func = integrate_paths_euler_parallel

        result = func( # (n traj (atoms), n_steps, 2)
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
    # NOTE: the line integration is performed in the plane perpendicular to the
    # dislocation lines, so the out-of-plane component of the displacement
    # vanishes (the screw component of the Burgers vector is carried by the
    # \beta_{z\cdot} components of the distortion field, not by this integral).
    u = xp.zeros(x.shape)
    u[:, :2] = result - x[:, :2]
    return u, {"integration_paths": paths_orig, "energies": energies,
               "jacobian": jacobian}


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
    det_mask = xp.isclose(xp.linalg.det(F_inv), 0)
    F_inv[det_mask, ...] = xp.eye(2)
    F = xp.linalg.inv(F_inv)
    # TODO: should be zero or one?
    F[det_mask, ...] = xp.eye(2)
    return F


def integrate_paths_euler_parallel_only_inv(x0, path_points, F1, F2):
    """
    Euler integration for many independent paths in parallel on GPU (CuPy).

    Parameters
    ----------
    x0 : xp.ndarray, shape (2,)
        Common initial position for all trajectories.
    path_points : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Path points for each trajectory.
    F1 : callable
        F1(y) -> (n_traj, 2, 2). Vectorized over batch of y.
    F2 : callable
        F2(p) -> (n_traj, 2, 2). Vectorized over batch of p.

    Returns
    -------
    traj : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Trajectories of all initial points along their own path.
    """
    dl_list = xp.diff(path_points, axis=1)  # (n_traj (atoms), n_steps, 2)
    n_traj, n_steps = dl_list.shape[0], dl_list.shape[1]

    traj = xp.zeros((n_traj, n_steps + 1, 2), dtype=xp.float64)
    traj[:, 0, :] = x0  # same initial point for all trajectories

    y = xp.broadcast_to(x0, (n_traj, 2)).copy()  # current states for each trajectory
    p = path_points[:, 0, :].copy()

    for i in range(n_steps):
        print(f"Step: {i}", end="\r")
        # F1(y_i) and F2(p_i) for all trajectories in batch
        # TODO F1y = F1(y)          # (n_traj, 2, 2)
        F2p = F2(p)          # (n_traj, 2, 2)
        # TODO mat = xp.matmul(F1y, F2p)  # (n_traj, 2, 2)
        mat = F2p
        dy = xp.einsum('nij,nj->ni', mat, dl_list[:, i, :])
        y = y + dy
        traj[:, i+1, :] = y
        p = p + dl_list[:, i, :]   # update path position per trajectory

    return traj


def integrate_paths_euler_parallel_only_noninv(x0, path_points, F1, F2):
    """
    ONLY NON INV.
    Euler integration for many independent paths in parallel on GPU (CuPy).

    Parameters
    ----------
    x0 : xp.ndarray, shape (2,)
        Common initial position for all trajectories.
    path_points : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Path points for each trajectory.
    F1 : callable
        F1(y) -> (n_traj, 2, 2). Vectorized over batch of y.
    F2 : callable
        F2(p) -> (n_traj, 2, 2). Vectorized over batch of p.

    Returns
    -------
    traj : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Trajectories of all initial points along their own path.
    """
    dl_list = xp.diff(path_points, axis=1)  # (n_traj (atoms), n_steps, 2)
    n_traj, n_steps = dl_list.shape[0], dl_list.shape[1]

    traj = xp.zeros((n_traj, n_steps + 1, 2), dtype=xp.float64)
    traj[:, 0, :] = x0  # same initial point for all trajectories

    y = xp.broadcast_to(x0, (n_traj, 2)).copy()  # current states for each trajectory
    p = path_points[:, 0, :].copy()

    for i in range(n_steps):
        print(f"Step: {i}", end="\r")
        # F1(y_i) and F2(p_i) for all trajectories in batch
        F1y = F1(y)          # (n_traj, 2, 2)
        # F2p = F2(p)          # (n_traj, 2, 2)
        # mat = xp.matmul(F1y, F2p)  # (n_traj, 2, 2)
        mat = F1y

        dy = xp.einsum('nij,nj->ni', mat, dl_list[:, i, :])
        y = y + dy
        traj[:, i+1, :] = y
        p = p + dl_list[:, i, :]   # update path position per trajectory

    return traj


def integrate_paths_euler_parallel(x0, path_points, F1, F2):
    """
    Euler integration for many independent paths in parallel on GPU (CuPy).

    Parameters
    ----------
    x0 : xp.ndarray, shape (2,)
        Common initial position for all trajectories.
    path_points : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Path points for each trajectory.
    F1 : callable
        F1(y) -> (n_traj, 2, 2). Vectorized over batch of y.
    F2 : callable
        F2(p) -> (n_traj, 2, 2). Vectorized over batch of p.

    Returns
    -------
    traj : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Trajectories of all initial points along their own path.
    """
    dl_list = xp.diff(path_points, axis=1)  # (n_traj (atoms), n_steps, 2)
    n_traj, n_steps = dl_list.shape[0], dl_list.shape[1]

    traj = xp.zeros((n_traj, n_steps + 1, 2), dtype=xp.float64)
    traj[:, 0, :] = x0  # same initial point for all trajectories

    y = xp.broadcast_to(x0, (n_traj, 2)).copy()  # current states for each trajectory
    p = path_points[:, 0, :].copy()

    for i in range(n_steps):
        print(f"Step: {i}", end="\r")
        # F1(y_i) and F2(p_i) for all trajectories in batch
        F1y = F1(y)          # (n_traj, 2, 2)
        F2p = F2(p)          # (n_traj, 2, 2)
        mat = xp.matmul(F1y, F2p)  # (n_traj, 2, 2)

        dy = xp.einsum('nij,nj->ni', mat, dl_list[:, i, :])
        y = y + dy
        traj[:, i+1, :] = y
        p = p + dl_list[:, i, :]   # update path position per trajectory

    return traj


def integrate_paths_trapezoid_parallel(x0, path_points, F1, F2):
    """
    Trapezoidal integration for many independent paths in parallel on GPU (CuPy).

    Parameters
    ----------
    x0 : xp.ndarray, shape (2,)
        Common initial position for all trajectories.
    path_points : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Path points for each trajectory.
    F1 : callable
        F1(y) -> (n_traj, 2, 2). Vectorized over batch of y.
    F2 : callable
        F2(p) -> (n_traj, 2, 2). Vectorized over batch of p.

    Returns
    -------
    traj : xp.ndarray, shape (n_traj, n_steps+1, 2)
        Trajectories of all initial points along their own path.
    """
    dl_list = xp.diff(path_points, axis=1)  # (n_traj, n_steps, 2)
    n_traj, n_steps = dl_list.shape[0], dl_list.shape[1]

    traj = xp.zeros((n_traj, n_steps + 1, 2), dtype=xp.float64)
    traj[:, 0, :] = x0

    y = xp.broadcast_to(x0, (n_traj, 2)).copy()
    p = path_points[:, 0, :].copy()

    for i in range(n_steps):
        print(f"Trapezoid step: {i}", end="\r")
        dl = dl_list[:, i, :]

        # --- f(y_i, p_i) ---
        F1y = F1(y)
        F2p = F2(p)
        mat = xp.matmul(F1y, F2p)
        f_i = xp.einsum('nij,nj->ni', mat, dl)

        # provisional step y* and next path point p_{i+1}
        y_star = y + f_i
        p_next = p + dl

        # --- f(y*, p_{i+1}) ---
        F1y_star = F1(y_star)
        F2p_next = F2(p_next)
        mat_star = xp.matmul(F1y_star, F2p_next)
        f_star = xp.einsum('nij,nj->ni', mat_star, dl)

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
    dl_list = xp.diff(path_points, axis=0)
    dl_list = h2d(dl_list)
    y = xp.zeros((len(dl_list) + 1, 2))

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
    psi_jac = xp.eye(3) + integrand(
        crystal=crystal,
        points=point.reshape(1, -1),
        d_state=d_state, exclude_beta=exclude_beta)
    psi_jac = psi_jac.squeeze()
    # TODO singularity? anyway, we should see errors if the given array
    #  is not reversible
    psi_jac_inv = xp.linalg.inv(psi_jac)
    return (-1)*xp.matmul(psi_jac_inv, psi)


@dataclasses.dataclass
class DislocationsState:
    """
    Describes the current configuration of dislocations.
    """
    ds: List[wzt.model.DislocationDef]
    ds_rt: List[xp.ndarray]

    def copy(self, **kwargs):
        return DislocationsState(**{**self.__dict__, **kwargs})

    def get_current_d_n(self):
        return self.ds[-1]


def get_integration_path(
        x_o, x_dash, d_state, n_points=30000, debug=False,
        excluded_dislocations=(),
        glide_plane=None,
        lattice=None
    ):
    excluded_dislocations = set(excluded_dislocations)

    dislocation = d_state.ds[-1]
    d_y = dislocation.position[1]
    # TODO tutaj powinno byc znalezienie najblizszego punktu na linii rozciecia

    # Initially assume straight line.
    direction = xp.sign(x_dash.squeeze()[1] - h2d(dislocation.position)[1])
    # if glide_plane is not None:
    #     gp_l, gp_r = np.min(glide_plane[:, 0]), np.max(glide_plane[:, 1])
    #     xx = x_dash.squeeze()[0]
    #     if xx > gp_l and xx < gp_r:
    #         x_dist = np.abs(glide_plane[:, 0] - x_dash.squeeze()[0])
    #         assert np.min(x_dist) < 1
    #         idx = np.argmin(x_dist)
    #         y_closest = glide_plane[idx, 1]
    #         direction = xp.sign(x_dash.squeeze()[1] - y_closest)

    d_position = [d2h(h2d(d.position))
                  for i, d in enumerate(d_state.ds)
                  if i not in excluded_dislocations]
    if direction == 0:
        raise ValueError("there should be no point located exactly at y=0")

    x_o = d2h(x_o)
    x_dash = d2h(x_dash)
    direction = d2h(direction)

    offset = direction * np.asarray([0.0, 2.0, 0.0])
    start = np.asarray(x_o).squeeze() + offset
    goal = x_dash

    # TODO zmienione na 2!
    DEFAULT_TOLERANCE = 2.0 - 1e-3
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

    points = astar(start=start, goal=goal, is_restricted=is_restricted,
                   step=DEFAULT_TOLERANCE, goal_tol=DEFAULT_TOLERANCE+1e-1)

    points.insert(0, x_o.squeeze()[:2])
    if inside_dis:
        points.append(x_dash.squeeze()[:2])


    points = interpolate_path(points, n_points=n_points)
    points = np.stack(points)
    return points


def get_u(point, d_n, crystal, d_state: DislocationsState,
          exclude_beta: set = None):
    be, bz = get_be_bz(crystal.cell, d_n.b)
    x_o = h2d([0.5 * be.item(), 0.0, 0.0])
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

    det_mask = xp.isclose(xp.linalg.det(F_inv), 0)
    F_inv[det_mask, ...] = xp.eye(2)
    F = xp.linalg.inv(F_inv)
    # TODO: should be zero or one?
    F[det_mask, ...] = xp.eye(2)

    result = xp.zeros((n_points, n_dims, n_dims))
    result[:, :2, :2] = xp.matmul(F, beta_i)
    return result


def calculate_rotation_matrix_for_vector(v):
    """
    Calculates matrix to the local coordinate system according to the
    given burgers vector v.

    Wynikowa macierz to:

    # [v, orto_v, 0]
    # [v, orto_v, 0]
    # [v, orto_v, 1]

    gdzie:
     - v to przekazany w parametrach wektor (wektor Burgersa),
     - orto_v to wektor v obrócony o 90 stopni:

     orto_v = [[0, -1]  * v
               [1,  0]]

    """
    # TODO ignoring OZ
    v = h2d(v)[:2]
    ba = v / xp.linalg.norm(v)

    ba_orto = xp.array([[0, -1],
                        [1, 0]]).dot(ba)
    ba_z = h2d([0, 0, 1])
    # R_i =
    rotmatrix = xp.eye(2)
    rotmatrix[:2, 0] = ba
    rotmatrix[:2, 1] = ba_orto
    return rotmatrix


def beta_sigma(points: xp.ndarray, crystal, d_state: DislocationsState,
               exclude_beta: Optional[Set] = None,
               return_beta: Optional[int] = None, return_all_beta: bool = False, debug=False):

    # Points oraz d_state (opis stanu dyslokacji): punkty są w układzie współrzędnych:
    # - zaczepionym w wstawianej dyslokacji (ostatnia dysloakacja z d_state.ds)
    # - oś OX jest styczna do wektora burgersa ostatniej dyslokacji, PRZED OBROTEM DYSLOKACJI.


    if exclude_beta is None:
        exclude_beta = {}
    n_points, dims = points.shape
    result = xp.zeros((n_points, 2, 2))

    returned_beta = None
    betas = []

    for i, (d, _) in enumerate(zip(d_state.ds, d_state.ds_rt)):

        # Dla każdej dyslokacji `d`:

        if i not in exclude_beta:
            # Oblicz macierz obrotu z obecnego układu punktów / dyslokacji
            # (patrz pierwszy komentarz tej funkcji) do układu, gdzie
            # OX jest równoległa z wektorem Burgersa
            rt = calculate_rotation_matrix_for_vector(v=d.b)
            # Policz beta.
            beta = beta_rotated(
                points=points,
                crystal=crystal,
                d=d,
                rotation_matrix=rt,
            )
            # Dodaj betę do wynikowej sumy.
            result += beta
            if return_beta == i:
                returned_beta = beta
            betas.append(beta)
    if returned_beta is not None:
        return result, returned_beta
    elif return_all_beta:
        return result, betas
    else:
        return result


def beta_rotated(crystal, d, points, rotation_matrix, dis_tolerance=DIS_TOLERANCE):
    """
    Oblicza macierz b dla punktów `points`, dla dyslokacji d (używamy jej położenia)
    oraz dla macierzy obrotu do danej dyslokacji `rotation_matrix`.

    Polożenie dyslokacji: `d.position`, skalar
    Wektor b dyslokacji: `d.b`, krawędziowa: be, śrubowa: bz, skalary
    Obecny obrót dyslokacji: `rotation_matrix`: macierz o wymiarach (3, 3)
    Punkty, dla których liczymy b: `points`, macierz o wymiarach (liczba punktów, 3)

    wykonywane instrukcje:

    points := points - d.position
    points = (rotation_matrix.T * points.T).T
    b = beta(points, be, bz)
    b = rotation_matrix * b * rotation_matrix.T
    """

    if rotation_matrix is None:
        return xp.zeros((points.shape[0], 2, 2))

    be, bz = get_be_bz(crystal.cell, d.b)
    # Przenieś do układu zaczepionego w dyslokacji (istotne np. dla d1).
    points = h2d(points) - h2d(d.position[:2]).reshape(1, -1)
    points = rotation_matrix.T.dot(points.T).T
    betas = beta(points, be=be, bz=bz)[:, :2, :2]
    rm = rotation_matrix[:2, :2].reshape(1, 2, 2)
    # KLUCZOWA INSTRUKCJA
    betas = rm @ betas @ rm.transpose((0, 2, 1))
    return betas


def beta(x, be, bz):
    x = h2d(x)
    if len(x.shape) == 1:
        x = x[xp.newaxis, ...]

    x1 = x[..., 0]  # (n_atoms, )
    x2 = x[..., 1]  # (n_atoms, )
    x1_2 = x1 ** 2  # (n_atoms, )
    x2_2 = x2 ** 2  # (n_atoms, )
    r2 = x1_2 + x2_2 # (n_atoms, )

    a = be / (4 * xp.pi * (1.0 - NU) * r2 * r2)
    # du / dx1
    b11 = (-1) * a * x2 * (
                (3.0 - 2.0 * NU) * x1_2 + (1.0 - 2.0 * NU) * x2_2)  # (natoms, )
    b21 = (-1) * a * x1 * ((1.0 - 2.0 * NU) * x1_2 + (3.0 - 2.0 * NU) * x2_2)
    b31 = (-1) * bz / (2.0 * xp.pi) * x2 / r2
    # du / dx2
    b12 = a * x1 * ((3.0 - 2.0 * NU) * x1_2 + (1.0 - 2.0 * NU) * x2_2)
    b22 = a * x2 * ((1.0 + 2.0 * NU) * x1_2 - (1.0 - 2.0 * NU) * x2_2)
    b32 = bz / (2.0 * xp.pi) * x1 / r2
    result = xp.repeat(BETA_ONES.copy()[xp.newaxis, ...], len(x1), axis=0)
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


def newton_raphson_step(psis, jacs, alpha=1.0):
    """
    A single step of the Newton-Raphson scheme of Eqs. (30)-(33).

    The (2N x 2N) Jacobian of the equation set (29) is assembled from
    d Psi_d / d x_n, where (Eq. (33))

        d Psi_d/d x_d = -1,
        d Psi_d/d x_n = d/dx_n int F^{d'}_{Sigma_{N+1}} F^{-1}_{Sigma_N^d} dl
                        (d != n),

    and the correction factors read Delta x = -J^{-1} Psi.

    :param psis: a list of N residuals Psi_d (each of them a vector of length
      >= 2; only the in-plane components are taken into account)
    :param jacs: a list of N dictionaries: n -> (1, 2, 2) array with
      d (int)_i/d x_n^k; `None` means that all the off-diagonal derivatives are
      neglected (the scheme degenerates then to successive substitution)
    :param alpha: relaxation (multiplicity) factor
    :return: a list of N corrections Delta x_d (vectors of length 3)
    """
    n_prev = len(psis)
    jacobian = xp.zeros((2 * n_prev, 2 * n_prev), dtype=xp.float64)
    residual = xp.zeros(2 * n_prev, dtype=xp.float64)

    for d in range(n_prev):
        residual[2 * d:2 * d + 2] = h2d(psis[d])[:2]
        # d Psi_d / d x_d = -1
        jacobian[2 * d:2 * d + 2, 2 * d:2 * d + 2] = -xp.eye(2)
        if not jacs[d]:
            continue
        for n, value in jacs[d].items():
            if n == d:
                continue
            jacobian[2 * d:2 * d + 2, 2 * n:2 * n + 2] = \
                xp.asarray(value).reshape(2, 2)

    deltas_flat = -alpha * xp.linalg.solve(jacobian, residual)

    deltas = []
    for d in range(n_prev):
        delta = xp.zeros(3, dtype=xp.float64)
        delta[:2] = deltas_flat[2 * d:2 * d + 2]
        deltas.append(delta)
    return deltas


def dbeta(x, be, bz):
    """
    Analytic gradient of the elemental lattice distortion field
    (Eq. (18) of the paper), i.e. d beta_{ij} / d x_k.

    NOTE: this is the derivative with respect to the FIELD coordinate x.
    The derivative with respect to the dislocation position x_d
    (the one appearing in Eq. (33)) is the opposite one, since
    beta_d(x) = beta(x - x_d).

    :param x: points (n points, >= 2), local coordinate system of the
      dislocation (i.e. OX parallel to the edge component of the Burgers vector)
    :return: array (n points, 3, 3, 2), where [:, i, j, k] = d beta_{ij}/d x_k
    """
    x = h2d(x)
    if len(x.shape) == 1:
        x = x[xp.newaxis, ...]

    x1 = x[..., 0]
    x2 = x[..., 1]
    x1_2 = x1 ** 2
    x2_2 = x2 ** 2
    r2 = x1_2 + x2_2

    # be/(4 pi (1-nu) r^6)
    c = be / (4.0 * xp.pi * (1.0 - NU) * r2 * r2 * r2)
    # bz/(2 pi r^4)
    e = bz / (2.0 * xp.pi * r2 * r2)

    two_nu_r2 = 2.0 * NU * r2
    quad = 2.0 * NU * (x1_2 * x1_2 - x2_2 * x2_2)

    db11_dx1 = -2.0 * c * x1 * x2 * (two_nu_r2 - 3.0 * x1_2 + x2_2)
    db11_dx2 = c * (quad - 3.0 * x1_2 * x1_2 + 6.0 * x1_2 * x2_2 + x2_2 * x2_2)
    db21_dx1 = -c * (quad - x1_2 * x1_2 - 6.0 * x1_2 * x2_2 + 3.0 * x2_2 * x2_2)
    db21_dx2 = -2.0 * c * x1 * x2 * (two_nu_r2 + x1_2 - 3.0 * x2_2)
    db12_dx2 = 2.0 * c * x1 * x2 * (two_nu_r2 - 5.0 * x1_2 - x2_2)
    db22_dx2 = c * (quad + x1_2 * x1_2 - 6.0 * x1_2 * x2_2 + x2_2 * x2_2)
    db31_dx1 = 2.0 * e * x1 * x2
    db31_dx2 = -e * (x1_2 - x2_2)
    db32_dx2 = -2.0 * e * x1 * x2

    result = xp.zeros((len(x1), 3, 3, 2))
    result[:, 0, 0, 0] = db11_dx1
    result[:, 0, 0, 1] = db11_dx2
    result[:, 1, 0, 0] = db21_dx1
    result[:, 1, 0, 1] = db21_dx2
    result[:, 2, 0, 0] = db31_dx1
    result[:, 2, 0, 1] = db31_dx2
    # beta is a gradient of a displacement field, hence
    # d beta_{i1}/d x2 == d beta_{i2}/d x1.
    result[:, 0, 1, 0] = db11_dx2
    result[:, 0, 1, 1] = db12_dx2
    result[:, 1, 1, 0] = db21_dx2
    result[:, 1, 1, 1] = db22_dx2
    result[:, 2, 1, 0] = db31_dx2
    result[:, 2, 1, 1] = db32_dx2

    # The core singularity is regularized in the same way as in `beta`.
    core_center_atoms = r2 < 1e-15
    result[core_center_atoms, ...] = 0.0
    return result  # (n points, 3, 3, 2)


def dbeta_rotated(crystal, d, points, rotation_matrix):
    """
    d beta_{ij}/d x_k of a single dislocation `d`, expressed in the coordinate
    system in which `points` are given (cf. `beta_rotated`).

    beta_d(x) = R beta(R^T (x - x_d)) R^T, hence
    d beta_d,ij / d x_k = R_ia R_jb R_kc  d beta_ab / d x'_c.

    :return: (n points, 2, 2, 2), where [:, i, j, k] = d beta_{ij}/d x_k
    """
    if rotation_matrix is None:
        return xp.zeros((points.shape[0], 2, 2, 2))

    be, bz = get_be_bz(crystal.cell, d.b)
    points = h2d(points) - h2d(d.position[:2]).reshape(1, -1)
    points = rotation_matrix.T.dot(points.T).T
    dbetas = dbeta(points, be=be, bz=bz)[:, :2, :2, :]  # (n, 2, 2, 2)
    rm = rotation_matrix[:2, :2]
    return xp.einsum("ia,jb,kc,nabc->nijk", rm, rm, rm, dbetas)


def dbeta_sigma_single(points, crystal, d_state: DislocationsState,
                       dislocation_nr: int):
    """
    d beta_n / d x of a single (n = `dislocation_nr`) dislocation of `d_state`,
    evaluated at `points`.

    :return: (n points, 2, 2, 2), where [:, i, j, k] = d beta_{ij}/d x_k
    """
    d = d_state.ds[dislocation_nr]
    rt = calculate_rotation_matrix_for_vector(v=d.b)
    return dbeta_rotated(
        points=points, crystal=crystal, d=d, rotation_matrix=rt
    )


def integrate_paths_euler_parallel_with_jacobian(x0, path_points, F1, F2, dbetas):
    """
    The same Euler integration as `integrate_paths_euler_parallel`, but it
    additionally accumulates the derivatives of the resulting line integral
    with respect to the positions of the dislocations listed in `dbetas`
    (Eq. (33) of the paper).

    d/dx_n int F_{Sigma_{N+1}} F^{-1}_{Sigma_N} dl
        = - int F_{Sigma_{N+1}} (d beta_n/d x) F_{Sigma_{N+1}} F^{-1}_{Sigma_N} dl

    (the minus sign comes from beta_n(x) = beta(x - x_n)).

    :param dbetas: a dictionary: dislocation number -> callable f(y), which
      returns (n_traj, 2, 2, 2) array of d beta_n/d x evaluated at y
    :return: (traj, jac), where jac is a dictionary:
      dislocation number -> (n_traj, 2, 2) array of d (int)_i / d x_n^k
    """
    dl_list = xp.diff(path_points, axis=1)  # (n_traj, n_steps, 2)
    n_traj, n_steps = dl_list.shape[0], dl_list.shape[1]

    traj = xp.zeros((n_traj, n_steps + 1, 2), dtype=xp.float64)
    traj[:, 0, :] = x0

    jac = {n: xp.zeros((n_traj, 2, 2), dtype=xp.float64) for n in dbetas}

    y = xp.broadcast_to(x0, (n_traj, 2)).copy()
    p = path_points[:, 0, :].copy()

    for i in range(n_steps):
        print(f"Step (with jacobian): {i}", end="\r")
        F1y = F1(y)  # (n_traj, 2, 2)
        F2p = F2(p)  # (n_traj, 2, 2)
        mat = xp.matmul(F1y, F2p)  # (n_traj, 2, 2)
        dl = dl_list[:, i, :]

        for n, dbeta_fn in dbetas.items():
            # (n_traj, 2, 2, 2): [:, i, j, k] = d beta_n,ij / d x_k
            g = dbeta_fn(y)
            # F1 (d beta_n/d x_k) F1 F2 dl, for each k
            m = xp.einsum("nia,nabk,nbc,ncj,nj->nik", F1y, g, F1y, F2p, dl)
            jac[n] = jac[n] - m

        dy = xp.einsum('nij,nj->ni', mat, dl)
        y = y + dy
        traj[:, i + 1, :] = y
        p = p + dl

    return traj, jac


def rotate_dislocation(crystal, d_state, rotated_dislocation, exclude_beta):
    """
    We are rotating according to the reference -- the currently introduced
    dislocation.

    :param reference_dislocation: introduced dislocation
    """
    bv, p = rotated_dislocation.b, rotated_dislocation.position[:2]
    bv = h2d(bv)

    p = h2d(p).reshape(1, -1)
    n_points = p.shape[0]
    BETA_ONES = broadcast_eye(2, n_points)
    betas = beta_sigma(
        points=p,
        crystal=crystal,
        d_state=d_state,
        exclude_beta=exclude_beta
    )
    F_inv = (BETA_ONES - betas)
    F = xp.linalg.inv(F_inv[0, :2, :2])
    bv = h2d(bv).squeeze()[:2]
    bv_rotated = F.dot(bv)
    orig_norm = xp.linalg.norm(bv)
    # Rescale the vector to the original norm.
    new_b = bv_rotated / xp.linalg.norm(bv_rotated) * orig_norm
    new_b = h2d(new_b.tolist() + [0])
    new_d = _set_d(rotated_dislocation, b=new_b)
    # TODO po co ten obrot tutaj?
    # global_rm = get_rotation_matrix(
    #     p=p, bv=[1.0, 0.0, 0.0],
    #     crystal=crystal, d_state=d_state,
    #     exclude_beta=exclude_beta
    # )
    # Rotate
    # b = global_rm[:2, :2].dot(bv[:2]).squeeze()
    # normalize/restore the initial norm
    return new_d


def estimate_rotation_matrix_from_vector(vector):
    # normalizacja wektora (x, y)

    x, y, z = vector.squeeze()
    norm = np.sqrt(x**2 + y**2)
    if norm == 0:
        raise ValueError("The vector cannot be zero")
    x, y = x / norm, y / norm
    theta = xp.arctan2(y, x)

    return xp.array([
        [xp.cos(theta), -xp.sin(theta)],
        [xp.sin(theta), xp.cos(theta)]
    ])


def get_rotation_matrix(crystal, d_state, p, bv, exclude_beta):
    """
    Returns the given

    :param dislocation: the dislocation that causes the rotation
    :param p: point where the rotation we want to calculate
    """
    p = h2d(p).reshape(1, -1)
    n_points = p.shape[0]
    BETA_ONES = broadcast_eye(2, n_points)
    betas = beta_sigma(
        points=p,
        crystal=crystal,
        d_state=d_state,
        exclude_beta=exclude_beta
    )
    F_inv = (BETA_ONES - betas)
    F = xp.linalg.inv(F_inv[0, :2, :2])

    ba = h2d(bv).squeeze()[:2]

    ba_rotated = F.dot(ba)
    ba = ba_rotated / xp.linalg.norm(ba_rotated)
    ba_orto = xp.array([[0, -1],
                        [1, 0]]).dot(ba)
    ba_z = h2d([0, 0, 1])

    rotmatrix = xp.eye(2)
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
def get_glide_plane(crystal, d_state, dislocation_nr, margin=45, debug=False):
    d_n = d_state.ds[dislocation_nr]
    # NOTE: y0 must be determined for the system located in d2
    f21 = []
    f22 = []
    ts = []
    bs21 = []
    bs22 = []
    def func(t, y):
        point = np.asarray([t, y.item()]).reshape(1, -1)
        betas, all_betas = beta_sigma(
            points=point,
            crystal=crystal,
            d_state=d_state,
            exclude_beta={dislocation_nr},
            return_all_beta=True,
        )
        betas = d2h(betas).squeeze()
        F_inv = np.eye(2) - betas
        ts.append(t)
        f21.append(betas[1, 0])
        f22.append(betas[1, 1])
        bs21.append(tuple([b.squeeze()[1, 0] for b in all_betas]))
        bs22.append(tuple([b.squeeze()[1, 1] for b in all_betas]))

        return -np.asarray([F_inv[1, 0] / F_inv[1, 1]])
    position = d_n.position
    if isinstance(position, xp.ndarray):
        position = d2h(position)
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
        rtol=1e-8, atol=1e-10, max_step=1e-1
    )
    n_points = len(ode_res.t)
    result = np.zeros((n_points, 3))
    # left side
    result[:, 0] = np.flip(np.squeeze(ode_res.t))
    result[:, 1] = np.flip(np.squeeze(ode_res.y))
    return result


def find_glide_plane(crystal, d_state, dislocation_nr, margin=45, debug=False):
    return h2d(get_glide_plane(crystal=crystal, d_state=d_state,
                               dislocation_nr=dislocation_nr,
                               margin=margin, debug=debug))


def calculate_energies(l, local_coordinates, d_state):
    """
    local_coordinates: atom coordinates in the current local coordinate system (i.e. for the currently inserted dislocation)
    """
    c12 = 160
    c44 = 81
    c11 = c12 + 2 * c44

    C = np.asarray([
        [c11, c12, 0],
        [c12, c11, 0],
        [0, 0, c44]
    ])
    energies = []
    for i, c in enumerate(local_coordinates):

        c = c[:2].reshape(1, -1)
        F = get_F(points=c, crystal=l, d_state=d_state, exclude_beta={})
        # F = (1-beta_\Sigma)^{-1}

        # beta_{\Sigma} = \Sigma \beta
        # R beta(R^T x ) R^T

        R, U = scipy.linalg.polar(d2h(F).squeeze())
        # F = RU
        eps = U - np.eye(2)
        # epsilon = U - 1
        eps_xx = eps[0, 0]
        eps_yy = eps[1, 1]
        gamma_xy = eps[0, 1] + eps[1, 0]

        # [eps_xx, eps_yy, gamma_xy]
        v = np.asarray([eps_xx, eps_yy, gamma_xy]).reshape(1, 3)
        E = 0.5 * (v @ C) @ v.T
        energies.append(E.squeeze())
    return np.asarray(energies)


def calculate_stress(l, local_coordinates, d_state):
    """
    local_coordinates: atom coordinates in the current local coordinate system (i.e. for the currently inserted dislocation)
    """
    c12 = 160
    c44 = 81
    c11 = c12 + 2 * c44

    C = np.asarray([
        [c11, c12, 0],
        [c12, c11, 0],
        [0, 0, c44]
    ])
    sigmas = []
    for i, c in enumerate(local_coordinates):

        c = c[:2].reshape(1, -1)
        F = get_F(points=c, crystal=l, d_state=d_state, exclude_beta={})
        # F = (1-beta_\Sigma)^{-1}

        # beta_{\Sigma} = \Sigma \beta
        # R beta(R^T x ) R^T

        R, U = scipy.linalg.polar(d2h(F).squeeze())
        # F = RU
        eps = U - np.eye(2)
        # epsilon = U - 1
        eps_xx = eps[0, 0]
        eps_yy = eps[1, 1]
        gamma_xy = eps[0, 1] + eps[1, 0]

        # v = [eps_xx, eps_yy, gamma_xy]
        v = np.asarray([eps_xx, eps_yy, gamma_xy]).reshape(1, 3)
        # Biot stress
        # 1/2 * (C * v) + (v^T * C)
        sigma = 0.5 * (C @ v.T) + (v @ C)
        sigmas.append(sigma.squeeze())
    return np.asarray(sigmas)


def get_integration_path_via_energy(
        x_o, x_dash, d_state, energies, atom_local_coordinates, n_points=30000,
        excluded_dislocations=None,
        glide_plane=None,
        lattice=None,
    ):
    """
    Finds the integral path based on the energies in each node.
    NOTE! the lattice.coordinates are in the GLOBAL COORDINATE System;
    use local_coordinates to use the correct coordinate system
    """
    assert not excluded_dislocations, "this function can be used only for atoms"
    assert lattice, "lattice is required to calculate integration path via energy"

    d_n = d_state.ds[-1]

    # Initially assume straight line.
    direction = xp.sign(x_dash.squeeze()[1] - h2d(d_n.position)[1])
    if glide_plane is not None:
        gp_l, gp_r = np.min(glide_plane[:, 0]), np.max(glide_plane[:, 0])
        xx = x_dash.squeeze()[0]

        if xx > gp_l and xx < gp_r:
            x_dist = np.abs(glide_plane[:, 0] - x_dash.squeeze()[0])
            if np.min(x_dist) >= 1:
                print(gp_l)
                print(gp_r)
                print(xx)
            assert np.min(x_dist) < 1
            idx = np.argmin(x_dist)
            y_closest = glide_plane[idx, 1]
            direction = xp.sign(x_dash.squeeze()[1] - y_closest)

    if direction == 0:
        raise ValueError("there should be no point located exactly at y=0")

    x_o = h2d(x_o)[:2]
    x_dash = h2d(x_dash)[:2]

    # remove from the lattice all atoms that are above/below the glide plane
    c = h2d(atom_local_coordinates)[:, :2]
    c = xp.ascontiguousarray(c)

    c, energies, edges = filter_lattice_by_curve(
        curve=glide_plane, coordinates=c, edges=lattice.bonds, weights=energies,
        direction=direction)

    # Find the index of the atom closest to the x_dash
    dists = xp.linalg.norm(c - x_dash.reshape(1, -1), axis=1)
    closest_x_dash = xp.argmin(dists)  # index

    # Find the index of the atom on the right side of x_o that is closest to x_dash
    c_right = c.copy()
    c_right[xp.argwhere(c[:, 0] <= 0)] = -xp.inf
    dists = xp.linalg.norm(c_right - x_o.reshape(1, -1), axis=1)
    closest_x_o = xp.argmin(dists)  # index

    # Find the shortest path between closest_x_dash and closest_x_o
    path = dijkstra_vertex_weights_scipy(
        weights=d2h(energies),
        # vertices=np.arange(c.shape[0]),
        edges=d2h(edges),
        start=d2h(closest_x_o),
        end=d2h(closest_x_dash),
    )

    points = []
    # include x_o and x_dash
    if not xp.isclose(xp.linalg.norm(x_o.squeeze() - closest_x_o.squeeze()), 0):
        points.append(x_o)

    for i in path:
        points.append(c[i])

    if not xp.isclose(xp.linalg.norm(x_dash.squeeze() - closest_x_dash.squeeze()), 0):
        points.append(x_dash)

    points = xp.stack(points)
    points = h2d(interpolate_path(d2h(points), n_points=n_points))
    return points


def get_integration_path_via_energy_always_direction(
        x_o, x_dash, d_state, energies, atom_local_coordinates, n_points=30000,
        excluded_dislocations=None,
        glide_plane=None,
        lattice=None,
        direction=None
    ):
    """
    Finds the integral path based on the energies in each node.
    NOTE! the lattice.coordinates are in the GLOBAL COORDINATE System;
    use local_coordinates to use the correct coordinate system
    """
    assert not excluded_dislocations, "this function can be used only for atoms"
    assert lattice, "lattice is required to calculate integration path via energy"

    d_n = d_state.ds[-1]

    # Initially assume straight line.
    direction = xp.sign(x_dash.squeeze()[1] - h2d(d_n.position)[1])
    if glide_plane is not None:
        gp_l, gp_r = np.min(glide_plane[:, 0]), np.max(glide_plane[:, 1])
        xx = x_dash.squeeze()[0]

        if xx > gp_l and xx < gp_r:
            x_dist = np.abs(glide_plane[:, 0] - x_dash.squeeze()[0])
            assert np.min(x_dist) < 1
            idx = np.argmin(x_dist)
            y_closest = glide_plane[idx, 1]
            direction = xp.sign(x_dash.squeeze()[1] - y_closest)

    if direction == 0:
        raise ValueError("there should be no point located exactly at y=0")

    x_o = h2d(x_o)[:2]
    x_dash = h2d(x_dash)[:2]

    # remove from the lattice all atoms that are above/below the glide plane
    c = h2d(atom_local_coordinates)[:, :2]
    c = xp.ascontiguousarray(c)

    c, energies, edges = filter_lattice_by_curve(
        curve=glide_plane, coordinates=c, edges=lattice.bonds, weights=energies,
        direction=direction)


    # Find the index of the atom closest to the x_dash
    dists = xp.linalg.norm(c - x_dash.reshape(1, -1), axis=1)
    closest_x_dash = xp.argmin(dists)  # index

    # Find the index of the atom on the right side of x_o that is closest to x_dash
    c_right = c.copy()
    c_right[xp.argwhere(c[:, 0] <= 0)] = -xp.inf
    dists = xp.linalg.norm(c_right - x_o.reshape(1, -1), axis=1)
    closest_x_o = xp.argmin(dists)  # index

    # Find the shortest path between closest_x_dash and closest_x_o
    path = dijkstra_vertex_weights_scipy(
        weights=d2h(energies),
        # vertices=np.arange(c.shape[0]),
        edges=d2h(edges),
        start=d2h(closest_x_o),
        end=d2h(closest_x_dash),
    )

    points = []
    # include x_o and x_dash
    if not xp.isclose(xp.linalg.norm(x_o.squeeze() - closest_x_o.squeeze()), 0):
        points.append(x_o)
    for i in path:
        points.append(c[i])

    if not xp.isclose(xp.linalg.norm(x_dash.squeeze() - closest_x_dash.squeeze()), 0):
        points.append(x_dash)

    points = xp.stack(points)
    points = h2d(interpolate_path(d2h(points), n_points=n_points))
    return points


def dijkstra_vertex_weights(weights, vertices, edges, start, end):
    """
    weights:  np.ndarray (n,) - wagi wierzchołków (float)
    vertices: np.ndarray (n,) - indeksy wierzchołków (0..n-1)
    edges:    np.ndarray (m, 2) - krawędzie nieskierowane
    start:    int - wierzchołek startowy
    end:      int - wierzchołek końcowy
    """

    n = len(vertices)
    start = start.item()
    end = end.item()

    # lista sąsiedztwa
    adj = [[] for _ in range(n)]
    for u, v in edges:
        u = u.item()
        v = v.item()
        adj[u].append(v)
        adj[v].append(u)

    # inicjalizacja
    dist = xp.full(n, xp.inf)
    prev = xp.full(n, -1, dtype=int)

    dist[start] = weights[start]

    pq = [(dist[start], start)]  # (koszt, wierzchołek)

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        if u == end:
            break

        for v in adj[u]:
            new_dist = dist[u] + weights[v]
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    # odtwarzanie ścieżki
    if dist[end] == xp.inf:
        return []  # brak ścieżki

    path = []
    v = end
    while v != -1:
        path.append(v)
        v = prev[v]

    return path[::-1]


def dijkstra_vertex_weights_scipy(weights, edges
                                  , start, end):
    n = len(weights)

    row = edges[:, 0]
    col = edges[:, 1]
    data = weights[col]

    # graf nieskierowany → dodajemy obie strony
    row = np.concatenate([row, col])
    col = np.concatenate([col, row[:len(edges)]])
    data = np.concatenate([data, weights[row[:len(edges)]]])

    graph = csr_matrix((data, (row, col)), shape=(n, n))

    dist, predecessors = dijkstra(
        graph,
        directed=False,
        indices=start,
        return_predecessors=True
    )

    if np.isinf(dist[end]):
        return []

    # rekonstrukcja ścieżki
    path = []
    v = end
    while v != -9999:
        path.append(v)
        v = predecessors[v]

    path.reverse()
    return path


def filter_lattice_by_curve(curve, coordinates, edges, weights, direction):
    """
    Filtruje punkty p względem krzywej c, usuwając punkty powyżej lub poniżej krzywej.

    Parametry:
        c : ndarray (m, 2)
            Współrzędne krzywej (x, y)
        p : ndarray (n, 2)
            Współrzędne punktów
        edges : ndarray (n, 2)
            krawędzie w grafie

        keep : str
            'below'  -> zachowuje punkty poniżej krzywej
            'above'  -> zachowuje punkty powyżej krzywej

    Zwraca:
        filtered_points : ndarray
            Przefiltrowane punkty
    """

    Px, Py = coordinates[:, 0], coordinates[:, 1]
    Cx, Cy = curve[:, 0], curve[:, 1]

    idx = xp.argsort(Cx)
    Cx_sorted = Cx[idx]
    Cy_sorted = Cy[idx]

    Px = xp.ascontiguousarray(Px)
    Cx_sorted = xp.ascontiguousarray(Cx_sorted)
    Cy_sorted = xp.ascontiguousarray(Cy_sorted)
    curve_y = xp.interp(Px, Cx_sorted, Cy_sorted)

    if direction == -1:
        mask = Py < curve_y
    elif direction == 1:
        mask = Py > curve_y
    else:
        raise ValueError(f"Invalid direction: {direction}")

    weights = weights.copy()
    edges = h2d(edges.copy())
    # Make the unmasked vertices unachievable

    weights[xp.logical_not(mask)] = np.inf
    coordinates[np.logical_not(mask)] = h2d([xp.inf, xp.inf])

    # Indices of atoms that should be kept connected.
    mask_idx = h2d(list(set(np.argwhere(mask).squeeze().tolist())))
    # We keep only bonds where all atoms are available.
    mask = xp.isin(edges[:, 0], mask_idx) & xp.isin(edges[:, 1], mask_idx)
    edges = edges[mask]
    return coordinates, weights, edges
