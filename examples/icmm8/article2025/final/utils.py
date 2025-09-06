import cupy as cp
import numpy as np

class MillerIndices:

    def __init__(self, crystal, d2_global):
        cell = crystal.cell
        position = cp.asarray(d2_global.position)
        burgers_vector = cp.asarray(d2_global.b)
        plane = cp.asarray(d2_global.plane)
        position = position.reshape(-1, 1)
        self.rt = get_rigid_rotation_tensor_miller(
            burgers_vector=burgers_vector,
            plane=plane,
            cell=cell
        )
        self.rt_inv = cp.transpose(self.rt)
        self.cd = self.rt.dot(position).squeeze()  # (3, )

    def preprocess(self, x):
        x = self.rt.dot(x.T).T
        x = x - self.cd.reshape(1, -1)
        return x

    def postprocess(self, u):
        return self.rt_inv.dot(u.T).T

    def postprocess_points(self, x):
        x = x + self.cd.reshape(1, -1)
        return self.rt_inv.dot(x.T).T


def broadcast_eye(n, nrepeats):
    return cp.array([cp.eye(n)]*nrepeats)


def get_be_bz(cell, burgers_vector):
    if isinstance(burgers_vector, cp.ndarray):
        burgers_vector = burgers_vector.get()
    burgers_vector = np.asarray(burgers_vector)
    bv_angstrom = cell.to_cartesian_indices(burgers_vector)
    be = np.sqrt(bv_angstrom[0] ** 2 + bv_angstrom[1] ** 2)
    bz = bv_angstrom[2]
    return be, bz


def get_rigid_rotation_tensor_miller(burgers_vector, plane, cell: wzt.model.UnitCellDef):
    normalize = wzt.dislocations._normalize
    s = cell.to_cartesian_indices(burgers_vector.get())
    s = normalize(s)
    m = np.transpose(cell.miller_to_cartesian).dot(plane.get())
    m = normalize(m)
    mxs = normalize(np.cross(m, s))
    mxsxm = normalize(np.cross(mxs, m))
    return cp.array([
        [mxsxm[0], mxsxm[1], mxsxm[2]],
        [mxs[0],   mxs[1],   mxs[2]  ],
        [m[0],     m[1],     m[2]    ],
    ])


def get_rotation_matrix(l0, dis_a, p, debug=False):
    be, bz = get_be_bz(l0.cell, dis_a.b)
    dis_b_local_pos = cp.asarray(p).reshape(1, -1)
    BETA_ONES = cp.eye(3)
    betas = beta(dis_b_local_pos, be=be, bz=bz)
    F_inv = (BETA_ONES - betas)
    F = cp.linalg.inv(F_inv[0, :2, :2])

    ba = cp.asarray([1.0, 0.0])

    ba_rotated = F.dot(ba)
    ba = ba_rotated / cp.linalg.norm(ba_rotated)
    ba_orto = cp.array([[0, -1],
                        [1, 0]]).dot(ba)
    ba_z = cp.asarray([0, 0, 1])

    rotmatrix = cp.eye(3)
    rotmatrix[:2, 0] = ba
    rotmatrix[:2, 1] = ba_orto
    return rotmatrix