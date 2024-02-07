"""
Functions and utils related to atomic dislocations.
"""
from typing import Tuple, Union, Sequence

import numpy as np
import scipy.optimize

from wurtzite.definitions import UnitCellDef
from wurtzite.model import Crystal
from wurtzite.utils import is_vector


# radius of inmobile ring relative to which the atoms in the core move up
RADIUS_FACTOR = 1.0
NU = 0.35

BETA_ONES = np.diag(np.ones(3))  # (3, 3) diagonal matrix for beta function


def _assert_is_vector(a: np.ndarray):
    if not is_vector(a):
        raise ValueError(f"{a} is not a vector")


def _normalize(v):
    return v/np.linalg.norm(v)


def _get_rotation_tensor(burgers_vector, plane, cell: UnitCellDef):
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


def love_function(x: np.ndarray, be: float, bz: float) -> np.ndarray:
    """
    Calculates love function.

    :return: love function for the given parameters (n_points, 3)
    """
    x1 = x[..., 0]  # (n_atoms, )
    x2 = x[..., 1]  # (n_atoms, )
    r2 = x1**2 + x2**2
    r = np.sqrt(r2)
    x1_norm = x1/r
    x2_norm = x2/r
    r02 = RADIUS_FACTOR*be**2

    ux = be/(2*np.pi)*(
        np.arctan2(x2_norm, x1_norm)
        + x1_norm*x2_norm/(2.0*(1-NU))
    )
    uy = -be/(8*np.pi*(1-NU)) * (
        (1.0-NU-NU)*np.log(r2/r02)
        + (x1_norm+x2_norm)*(x1_norm-x2_norm)
    )
    uz = bz/(2*np.pi)*np.arctan2(x2_norm, x1_norm)

    ux = ux.reshape(-1, 1)  # (natoms, 1)
    uy = uy.reshape(-1, 1)  # (natoms, 1)
    uz = uz.reshape(-1, 1)  # (natoms, 1)
    return np.column_stack((ux, uy, uz))  # (natoms, 3)


def beta_function(x: np.ndarray, be: float, bz: float) -> np.ndarray:
    """
    Calculates Beta function.
    Note: this function will return 1 (diagonal) for atoms which are in the
    center of dislocation core.
    """
    x1 = x[..., 0]  # (n_atoms, )
    x2 = x[..., 1]
    x1_2 = x1**2
    x2_2 = x2**2
    r2 = x1_2+x2_2

    a = be/(4*np.pi*(1.0-NU)*r2*r2)
    # du / dx1
    b11 = (-1)*a*x2*((3.0-2.0*NU)*x1_2 + (1.0-2.0*NU)*x2_2)  # (natoms, )
    b21 = (-1)*a*x1*((1.0-2.0*NU)*x1_2 + (3.0-2.0*NU)*x2_2)
    b31 = (-1)*bz/(2.0*np.pi) * x2/r2
    # du / dx2
    b12 = a*x1*((3.0-2.0*NU)*x1_2 + (1.0-2.0*NU)*x2_2)
    b22 = a*x2*((1.0+2.0*NU)*x1_2-(1.0-2.0*NU)*x2_2)
    b32 = bz/(2.0*np.pi) * x1/r2
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


def displace_love(
        crystal: Crystal,
        position: Union[Sequence[float], np.ndarray],
        burgers_vector: Union[Sequence[float], np.ndarray],
        plane: Union[Sequence[float], np.ndarray],
        bv_fraction: float = 1.0,
        tolerance: float = 1e-7,
        method="hybr",
        options=()
) -> np.ndarray:
    """
    Calculates displacement using the approach described in the paper:

    Dislocation core reconstruction based on finite deformation approach and its
    application to 4h-sic crystal, J. Cholewinski et al. 2014
    http://dx.doi.org/10.1615/IntJMultCompEng.2014010679

    :param crystal: crystal lattice for which the displacement
        should be calculated
    :param position: position of the displacement (x, y, z) [A]
    :param burgers_vector: Burgers vector (miller indices)
    :param plane: dislocation plane (miller indices)
    :param bv_fraction: Burgers vector fraction
    :param tolerance: solver tolrance
    :param method: solver method, see scipy.optimize.root for more information
    :param options: additional parameters that should be passed to the solver
    """

    position = np.asarray(position)
    burgers_vector = np.asarray(burgers_vector)
    plane = np.asarray(plane)
    cell = crystal.cell

    _assert_is_vector(position)
    _assert_is_vector(burgers_vector)
    _assert_is_vector(plane)

    rt = _get_rotation_tensor(
        burgers_vector=burgers_vector,
        plane=plane,
        cell=cell
    )
    rt_inv = np.transpose(rt)

    burgers_vector = bv_fraction * cell.to_cartesian_indices(burgers_vector)
    burgers_vector = burgers_vector.reshape(-1, 1)
    burgers_vector = rt.dot(burgers_vector).squeeze()
    be = np.sqrt(burgers_vector[0]**2 + burgers_vector[1]**2)
    bz = burgers_vector[2]

    position = position.reshape(-1, 1)
    cd = rt.dot(position).squeeze()  # (3, )

    # Initial setting
    x_all = rt.dot(crystal.coordinates.T).T  # (natoms, 3)
    x_all = x_all-cd.reshape(1, -1)
    # x_distance = x_all-cd

    def f(u, x):
        nonlocal be, bz
        current_x = x+u
        current_x = current_x.reshape(1, -1)
        return u-love_function(current_x, be, bz).squeeze()

    def jacobian(u, x):
        nonlocal be, bz
        # TODO czy wlasciwy jacobian we wlasciwym kierunku?
        current_x = x + u
        current_x = current_x.reshape(1, -1)
        return BETA_ONES-beta_function(current_x, be, bz).squeeze()

    result_u = np.zeros((crystal.n_atoms, 3))

    for i, coords in enumerate(x_all):
        u0 = np.zeros(3)
        result = scipy.optimize.root(
            f,
            x0=u0,
            jac=jacobian,
            args=(coords, ),
            tol=tolerance,
            method=method,
            options=options
        )
        if not result.success:
            raise ValueError("Could not calculate displacement, "
                            f"root finder message: {result.message}")
        else:
            result_u[i] = result.x
    # Move to the previous system 
    result_u = rt_inv.dot(result_u.T).T
    return result_u


