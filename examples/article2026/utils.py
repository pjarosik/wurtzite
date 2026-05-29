import numpy as np
from gpu_compatibility import xp, d2h, h2d
import wurtzite as wzt
import dataclasses


def line_integral(path, vals):
    """
    Liczy przybliżoną wartość całki krzywoliniowej, gdy F(r) zwraca macierz (d,d).

    ∫ F(r) dr ≈ Σ ( (F_i + F_{i+1})/2 ) @ (p_{i+1} - p_i)

    Parameters:
    - path: (n, d) xp.ndarray – kolejne punkty ścieżki
    - F_vals: (n, d, d) xp.ndarray – wartości funkcji macierzowej w punktach ścieżki

    Returns:
    - Całkowita wartość całki krzywoliniowej: wektor (d,)
    """
    path = xp.asarray(path, dtype=xp.float64)
    vals = xp.asarray(vals, dtype=xp.float64)

    # (n-1, d)
    deltas = path[1:] - path[:-1]   # dr

    # (n-1, d, d)
    avg = 0.5 * (vals[1:] + vals[:-1])

    # (n-1, d, d) @ (n-1, d, 1) -> (n-1, d, 1)
    contribs = xp.matmul(avg, deltas[..., None])  # (n-1, d, 1)
    integral = contribs.sum(axis=0).squeeze()  # (d,)
    return integral


class MillerIndices:

    def __init__(self, crystal, dislocation):
        self.cell = crystal.cell
        position = xp.asarray(dislocation.position)
        burgers_vector = xp.asarray(dislocation.b)
        plane = xp.asarray(dislocation.plane)
        position = position.reshape(-1, 1)
        self.rt = get_rigid_rotation_tensor_miller(
            burgers_vector=burgers_vector,
            plane=plane,
            cell=self.cell
        )
        self.rt_inv = xp.transpose(self.rt)
        self.cd = self.rt.dot(position).squeeze()  # (3, )

    def preprocess(self, x):
        if x.size == 0:
            return x

        x = self.rt.dot(x.T).T
        x = x - self.cd.reshape(1, -1)
        return x

    def preprocess_dislocation(self, d):
        new_position = self.preprocess(xp.asarray(d.position))
        # b is stored in Miller indices in the user-facing representation;
        # convert to Cartesian first, then rotate into the local frame.
        # Within the local frame we keep b in Cartesian so that get_be_bz
        # and rotate_dislocation operate on physically meaningful magnitudes.
        cart_b = self.cell.to_cartesian_indices(d2h(xp.asarray(d.b)))
        new_b = self.rt.dot(h2d(cart_b).T).T
        return dataclasses.replace(
            d,
            position=new_position.squeeze(),
            b=new_b.squeeze()
        )

    def preprocess_vector(self, v):
        return self.rt.dot(v.T).T

    def postprocess(self, u):
        return self.rt_inv.dot(u.T).T

    def postprocess_points(self, x):
        x = x + self.cd.reshape(1, -1)
        return self.rt_inv.dot(x.T).T

    def postprocess_dislocation(self, dislocation):
        # b in the local frame is Cartesian; rotate back to global Cartesian,
        # then convert to Miller indices for the user-facing representation.
        b_local_cart = xp.asarray(dislocation.b).reshape(1, -1)
        b_global_cart = self.postprocess(b_local_cart)
        cartesian_to_miller = self.cell.cartesian_to_miller
        new_b = cartesian_to_miller.dot(d2h(b_global_cart).T).T
        new_position = self.postprocess_points(
            x=xp.asarray(dislocation.position).reshape(1, -1)
        )
        return dataclasses.replace(
            dislocation,
            position=d2h(new_position).squeeze(),
            b=new_b.squeeze()
        )


def broadcast_eye(n, nrepeats):
    return xp.array([xp.eye(n)]*nrepeats)


def get_be_bz(cell, burgers_vector):
    # burgers_vector is expected to be in Cartesian (the convention used
    # for the local-frame dislocations produced by MillerIndices).
    if isinstance(burgers_vector, xp.ndarray):
        burgers_vector = d2h(burgers_vector)
    burgers_vector = np.asarray(burgers_vector)
    be = np.sqrt(burgers_vector[0] ** 2 + burgers_vector[1] ** 2)
    bz = burgers_vector[2]
    return be, bz


def get_rigid_rotation_tensor_miller(burgers_vector, plane, cell: wzt.model.UnitCellDef):
    normalize = wzt.dislocations._normalize
    s = cell.to_cartesian_indices(d2h(burgers_vector))
    s = normalize(s)
    m = np.transpose(cell.miller_to_cartesian).dot(d2h(plane))
    m = normalize(m)
    mxs = normalize(np.cross(m, s))
    mxsxm = normalize(np.cross(mxs, m))
    return xp.array([
        [mxsxm[0], mxsxm[1], mxsxm[2]],
        [mxs[0],   mxs[1],   mxs[2]  ],
        [m[0],     m[1],     m[2]    ],
    ])


def get_line(a, b, n=10000):
    t = xp.linspace(0, 1, n)[:, None]  # (n,1)
    points = a + t * (b - a)  # (n,3)
    return points
