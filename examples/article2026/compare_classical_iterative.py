"""
Quantitative comparison of the classical and the iterative (finite deformation)
reconstruction of the atomistic models of dislocations.

The classical solution is obtained by a direct superposition of the analytic
displacement fields, Eqs. (14)-(16), evaluated at the reference (perfect
lattice) positions of the atoms.  The iterative solution is obtained with the
scheme discussed in Sec. "Dislocation insertion along deformed slip plane",
i.e. by the line integration of the differential form
d x = F_{Sigma_{N+1}} F^{-1}_{Sigma_N} d l, combined with the (modified)
Newton-Raphson update of the dislocation positions, Eqs. (30)-(33).

The following measures are reported:

* deviation of the atom positions between both solutions (mean/max/RMS) [A],
* nearest-neighbour bond lengths with respect to the ideal Ga-N bond of the
  perfect lattice (mean absolute deviation, max deviation, standard deviation),
* the elastic energy stored in the reconstructed lattice, estimated from the
  deformation gradient fitted (in the least-squares sense) to the neighbourhood
  of every atom,
* the convergence history of the iterative scheme.

Usage:
    PYTHONPATH=<wurtzite repo> python3 compare_classical_iterative.py
"""
import json
import math
from datetime import datetime

import numpy as np
import scipy.linalg
import scipy.spatial

import wurtzite as wzt
from dislocations import displace
from gpu_compatibility import d2h

# Elastic constants of GaN [GPa] (the same values as in `calculate_energies`).
C12 = 160.0
C44 = 81.0
C11 = C12 + 2 * C44
C_MATRIX = np.asarray([
    [C11, C12, 0.0],
    [C12, C11, 0.0],
    [0.0, 0.0, C44],
])
# 1 GPa * A^3 expressed in eV.
GPA_A3_TO_EV = 6.241509074e-3

# Radius [A] within which the neighbours used to fit the local deformation
# gradient are searched for.
NEIGHBOUR_CUTOFF = 3.5
# Atoms closer than this distance [A] to any dislocation core are excluded from
# the statistics -- the analytic solution is singular there.
CORE_EXCLUSION_RADIUS = 4.0
# Atoms closer than this distance [A] to the boundary of the lattice are
# excluded from the statistics (incomplete neighbourhoods).
BOUNDARY_MARGIN = 4.0
# Maximum relative change of an inter-atomic distance which is still considered
# to be an elastic distortion; a larger change means that the atom lies on the
# slip surface (its neighbour has been separated by a whole Burgers vector).
SLIP_TOLERANCE = 0.25

# Number of integration points of the line integrals.
N_POINTS = 1000
# Maximum number of iterations and the convergence criterion of the scheme
# solving Eq. (29).
N_ITERS = 20
TOL = 1e-3
# Relaxation (multiplicity) factor of the Newton-Raphson corrections.
ALPHA = 0.6


def create_lattice():
    return wzt.generate.create_lattice(dimensions=(19, 13, 1), cell="B4_GaN")


def get_dislocations(lattice):
    """
    The same configuration as the one used in `6_dislocations_8.py`.
    """
    ox = lattice.cell.dimensions[1] * 0.5
    oy = lattice.cell.dimensions[1] * math.sqrt(3) / 2
    offset_0 = np.asarray([2 * lattice.cell.dimensions[0] + ox, oy, 0])
    return [
        wzt.model.DislocationDef(
            label="$d_1$", b=[1, 0, 0],
            position=[3.190 + 0.8 + 6 * lattice.cell.dimensions[0], 10, 7.5] + offset_0,
            plane=(0, 0, 1), color="brown"),
        wzt.model.DislocationDef(
            label="$d_2$", b=[1, 0, 0], position=[2.35, 9.66, 0] + offset_0,
            plane=(0, 0, 1), color="brown"),
        wzt.model.DislocationDef(
            label="$d_3$", b=[0, 1, 0], position=[8.53, 17.61, 0] + offset_0,
            plane=(0, 0, 1), color="brown"),
    ]


def classical_displacement(lattice, dislocations):
    """
    The classical solution: a superposition of the analytic displacement fields
    of single dislocations, evaluated at the reference positions of the atoms.
    """
    u = np.zeros(lattice.coordinates.shape)
    for d in dislocations:
        u += wzt.dislocations.displace_love_single(
            crystal=lattice, dislocation=d)
    return u


def iterative_displacement(lattice, dislocations, method="newton"):
    """
    The iterative solution: the dislocations are inserted one by one into the
    already deformed lattice.

    :return: (total displacement of the atoms, convergence history)
    """
    l = lattice
    inserted = []
    convergence = []
    for d in dislocations:
        log = displace(
            crystal=l, dislocations=inserted, d_n=d,
            n_iters=N_ITERS, n_points=N_POINTS,
            method=method, alpha=ALPHA, tol=TOL,
        )
        inserted = log.last_d_state.ds
        l = l.translate(log.last_u_atoms)
        convergence.append([tuple(float(v) for v in c) for c in log.convergence])
    return l.coordinates - lattice.coordinates, convergence


def get_selection_mask(lattice, dislocations):
    """
    Atoms taken into account in the statistics: neither too close to a
    dislocation core nor to the boundary of the modelled lattice.
    """
    xy = lattice.coordinates[:, :2]
    mask = np.ones(len(xy), dtype=bool)
    for d in dislocations:
        p = np.asarray(d.position, dtype=float)[:2]
        mask &= np.linalg.norm(xy - p.reshape(1, -1), axis=1) > CORE_EXCLUSION_RADIUS
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    mask &= np.all(xy > lo.reshape(1, -1) + BOUNDARY_MARGIN, axis=1)
    mask &= np.all(xy < hi.reshape(1, -1) - BOUNDARY_MARGIN, axis=1)
    return mask


def get_neighbours(reference_coordinates, cutoff=NEIGHBOUR_CUTOFF):
    """
    :return: a list of neighbour index arrays, one per atom
    """
    tree = scipy.spatial.cKDTree(reference_coordinates)
    pairs = tree.query_ball_point(reference_coordinates, r=cutoff)
    return [np.asarray([j for j in p if j != i], dtype=int)
            for i, p in enumerate(pairs)]


def fit_deformation_gradients(reference_coordinates, coordinates, neighbours,
                              slip_tolerance=SLIP_TOLERANCE):
    """
    Least-squares estimate of the in-plane deformation gradient F in the
    neighbourhood of every atom: F = argmin sum_k |F dX_k - dx_k|^2.

    Atoms adjacent to the slip surface are skipped: for them, at least one pair
    which was a neighbour pair in the perfect lattice has been separated by a
    whole Burgers vector, so the deformation of their neighbourhood is no longer
    described by a single (elastic) deformation gradient. Such an atom is
    detected by a neighbour whose distance changed by more than
    `slip_tolerance` (relative).

    :return: (n atoms, 2, 2) array; np.nan for atoms adjacent to the slip
      surface and for atoms with a degenerate neighbourhood
    """
    n_atoms = len(reference_coordinates)
    result = np.full((n_atoms, 2, 2), np.nan)
    for i in range(n_atoms):
        idx = neighbours[i]
        if len(idx) < 3:
            continue
        dX_3d = reference_coordinates[idx] - reference_coordinates[i]
        dx_3d = coordinates[idx] - coordinates[i]
        length_X = np.linalg.norm(dX_3d, axis=1)
        length_x = np.linalg.norm(dx_3d, axis=1)
        if np.any(np.abs(length_x / np.maximum(length_X, 1e-9) - 1.0)
                  > slip_tolerance):
            continue
        dX, dx = dX_3d[:, :2], dx_3d[:, :2]
        if np.linalg.matrix_rank(dX, tol=1e-6) < 2:
            continue
        f, *_ = np.linalg.lstsq(dX, dx, rcond=None)
        result[i] = f.T
    return result


def elastic_energy(deformation_gradients, atomic_volume):
    """
    Elastic energy stored in the lattice, estimated from the local deformation
    gradients.  The strain is measured as eps = U - 1, where F = R U is the
    polar decomposition of F (the same measure as in `calculate_energies`).

    :return: (mean energy density [GPa], total energy [eV], per-atom energy
      density [GPa])
    """
    densities = []
    for f in deformation_gradients:
        if not np.all(np.isfinite(f)):
            continue
        _, u = scipy.linalg.polar(f)
        eps = u - np.eye(2)
        v = np.asarray([eps[0, 0], eps[1, 1], eps[0, 1] + eps[1, 0]])
        densities.append(0.5 * v @ C_MATRIX @ v)
    densities = np.asarray(densities)
    total_ev = float(densities.sum() * atomic_volume * GPA_A3_TO_EV)
    return float(densities.mean()), total_ev, densities


def bond_statistics(coordinates, reference_coordinates):
    """
    Statistics of the nearest-neighbour (Ga-N) distances measured in the
    RECONSTRUCTED configuration and compared with the nearest-neighbour distance
    of the perfect lattice.

    The neighbours are searched for in the reconstructed configuration, so that
    the measure is not affected by the slip discontinuity introduced on the
    glide plane (two atoms which were neighbours in the perfect lattice may be
    separated by a whole Burgers vector after the insertion of a dislocation --
    this is the plastic slip, not a distortion of the lattice).

    :return: dict with the mean absolute deviation, max deviation and standard
      deviation of the nearest-neighbour distance [A]
    """
    d0 = float(np.median(
        scipy.spatial.cKDTree(reference_coordinates).query(
            reference_coordinates, k=2)[0][:, 1]))
    d = scipy.spatial.cKDTree(coordinates).query(coordinates, k=2)[0][:, 1]
    diff = d - d0
    return {
        "n_atoms": int(len(d)),
        "reference_distance": d0,
        "mean_abs_deviation": float(np.abs(diff).mean()),
        "max_abs_deviation": float(np.abs(diff).max()),
        "std_deviation": float(diff.std()),
    }


def compare(n_dislocations):
    l0 = create_lattice()
    dislocations = get_dislocations(l0)[:n_dislocations]

    print(f"=== {n_dislocations} dislocation(s) ===", flush=True)
    u_classical = classical_displacement(l0, dislocations)
    u_iterative, convergence = iterative_displacement(l0, dislocations)
    u_iterative = d2h(np.asarray(u_iterative))

    x_ref = np.asarray(l0.coordinates)
    x_classical = x_ref + u_classical
    x_iterative = x_ref + u_iterative

    mask = get_selection_mask(l0, dislocations)
    difference = (x_iterative - x_classical)[:, :2]
    deviation = np.linalg.norm(difference, axis=1)
    # Both solutions fix a different material point: the classical one assumes
    # u = 0 at infinity, the iterative one keeps the particle x_o immobile.  The
    # resulting rigid translation of the whole lattice does not change the shape
    # of the reconstructed lattice, so it is reported separately.
    offset = difference[mask].mean(axis=0)
    deviation_no_offset = np.linalg.norm(difference - offset.reshape(1, -1), axis=1)

    neighbours = get_neighbours(x_ref)
    f_classical = fit_deformation_gradients(x_ref, x_classical, neighbours)
    f_iterative = fit_deformation_gradients(x_ref, x_iterative, neighbours)

    # Volume per atom of the wurtzite lattice (4 atoms per hexagonal unit cell
    # of the volume a^2 sin(60 deg) c) and the resulting thickness of the
    # modelled slab along the dislocation lines.
    a, _, c = l0.cell.dimensions
    atomic_volume = float(a * a * math.sin(math.pi / 3) * c / 4)
    area = float(scipy.spatial.ConvexHull(x_ref[:, :2]).volume)
    thickness = len(x_ref) * atomic_volume / area

    # The same set of atoms is used for both solutions, so that the energies
    # are directly comparable.
    energy_mask = (mask
                   & np.all(np.isfinite(f_classical), axis=(1, 2))
                   & np.all(np.isfinite(f_iterative), axis=(1, 2)))
    density_c, total_c, _ = elastic_energy(f_classical[energy_mask], atomic_volume)
    density_i, total_i, _ = elastic_energy(f_iterative[energy_mask], atomic_volume)

    result = {
        "n_dislocations": n_dislocations,
        "n_atoms": int(len(x_ref)),
        "n_atoms_in_statistics": int(mask.sum()),
        "atom_deviation_all": {
            "mean": float(deviation.mean()),
            "max": float(deviation.max()),
            "rms": float(np.sqrt((deviation ** 2).mean())),
        },
        "atom_deviation_selected": {
            "mean": float(deviation[mask].mean()),
            "max": float(deviation[mask].max()),
            "rms": float(np.sqrt((deviation[mask] ** 2).mean())),
        },
        "rigid_offset": [float(v) for v in offset],
        "atom_deviation_selected_without_rigid_offset": {
            "mean": float(deviation_no_offset[mask].mean()),
            "max": float(deviation_no_offset[mask].max()),
            "rms": float(np.sqrt((deviation_no_offset[mask] ** 2).mean())),
        },
        "bonds_perfect": bond_statistics(x_ref, x_ref),
        "bonds_classical": bond_statistics(x_classical, x_ref),
        "bonds_iterative": bond_statistics(x_iterative, x_ref),
        "atomic_volume_A3": float(atomic_volume),
        "slab_thickness_A": float(thickness),
        "n_atoms_in_energy_statistics": int(energy_mask.sum()),
        "energy_classical": {"mean_density_GPa": density_c,
                             "total_eV": total_c,
                             "eV_per_A_of_line": total_c / thickness},
        "energy_iterative": {"mean_density_GPa": density_i,
                             "total_eV": total_i,
                             "eV_per_A_of_line": total_i / thickness},
        "convergence": convergence,
    }
    print(json.dumps(result, indent=2), flush=True)
    coordinates = {
        "reference": x_ref,
        "classical": x_classical,
        "iterative": x_iterative,
        "mask": mask,
    }
    return result, coordinates


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = []
    for n in (1, 2, 3):
        result, coordinates = compare(n)
        results.append(result)
        np.savez(f"comparison_{timestamp}_{n}_dislocations.npz", **coordinates)
    filename = f"comparison_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
