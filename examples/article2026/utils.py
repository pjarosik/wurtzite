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
        cell = crystal.cell
        position = xp.asarray(dislocation.position)
        burgers_vector = xp.asarray(dislocation.b)
        plane = xp.asarray(dislocation.plane)
        position = position.reshape(-1, 1)
        self.rt = get_rigid_rotation_tensor_miller(
            burgers_vector=burgers_vector,
            plane=plane,
            cell=cell
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
        new_b = self.preprocess_vector(xp.asarray(d.b))
        extra = {}
        if getattr(d, "half_plane", None) is not None:
            extra["half_plane"] = self.preprocess_vector(
                xp.asarray(d.half_plane)).squeeze()
        if getattr(d, "b_ref", None) is not None:
            extra["b_ref"] = self.preprocess_vector(
                xp.asarray(d.b_ref)).squeeze()
        # F_sigma is a rank-2 tensor, so it follows a change of frame as
        # R F R^T rather than as a vector.
        if getattr(d, "F_sigma", None) is not None:
            r = self.rt[:2, :2]
            extra["F_sigma"] = r.dot(xp.asarray(d.F_sigma)).dot(r.T)
        return dataclasses.replace(
            d,
            position=new_position.squeeze(),
            b=new_b.squeeze(),
            **extra
        )

    def preprocess_vector(self, v):
        return self.rt.dot(v.T).T

    def postprocess(self, u):
        return self.rt_inv.dot(u.T).T

    def postprocess_points(self, x):
        x = x + self.cd.reshape(1, -1)
        return self.rt_inv.dot(x.T).T

    def postprocess_dislocation(self, dislocation):
        new_b = self.postprocess(u=xp.asarray(dislocation.b).reshape(1, -1))
        new_position = self.postprocess_points(
            x=xp.asarray(dislocation.position).reshape(1, -1)
        )
        extra = {}
        if getattr(dislocation, "half_plane", None) is not None:
            # a direction -- rotated back like b, not translated like a point
            extra["half_plane"] = d2h(self.postprocess(
                u=xp.asarray(dislocation.half_plane).reshape(1, -1))).squeeze()
        if getattr(dislocation, "b_ref", None) is not None:
            extra["b_ref"] = d2h(self.postprocess(
                u=xp.asarray(dislocation.b_ref).reshape(1, -1))).squeeze()
        # cf. preprocess_dislocation: a rank-2 tensor, carried back as
        # R^T F R.
        if getattr(dislocation, "F_sigma", None) is not None:
            r = self.rt_inv[:2, :2]
            extra["F_sigma"] = d2h(
                r.dot(xp.asarray(dislocation.F_sigma)).dot(r.T))
        return dataclasses.replace(
            dislocation,
            position=d2h(new_position).squeeze(),
            b=d2h(new_b).squeeze(),
            **extra
        )


def broadcast_eye(n, nrepeats):
    return xp.array([xp.eye(n)]*nrepeats)


def get_be_bz(cell, burgers_vector):
    if isinstance(burgers_vector, xp.ndarray):
        burgers_vector = d2h(burgers_vector)
    burgers_vector = np.asarray(burgers_vector)
    bv_angstrom = cell.to_cartesian_indices(burgers_vector)
    be = np.sqrt(bv_angstrom[0] ** 2 + bv_angstrom[1] ** 2)
    bz = bv_angstrom[2]
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
