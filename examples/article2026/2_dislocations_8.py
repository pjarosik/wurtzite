import dataclasses
import pickle
from datetime import datetime

from dislocations import *
from glide_planes import clip_glide_plane, draw_glide_plane
import wurtzite as wzt
import math
from pathlib import Path
import numpy as np
import params

np.seterr(invalid="raise")

# INITIAL CONFIGURATION CONFIGURATION.
l0 = wzt.generate.create_lattice(
    dimensions=(19, 13, 1),  # (nx, ny, nz)
    cell="B4_GaN",
)

# PARAMETERS.
# Number of integration points.
n_points = 1000
# Maximum number of iterations of the scheme solving Eq. (29).
n_iters = 20
# Method used to solve Eq. (29) for the positions of the dislocations already
# inserted into the lattice:
# - "newton": the modified Newton-Raphson scheme, Eqs. (30)-(33),
# - "picard": successive substitution.
method = "newton"
# Relaxation (multiplicity) factor of the Newton-Raphson corrections.
# alpha = 1.0 is the classical Newton-Raphson scheme; alpha < 1 turns it into
# the modified (relaxed) one.
alpha = 1.0
# Convergence criterion [A]: stop as soon as max_d ||Delta x_d|| < tol.
tol = 1e-3

# Display parameters
# If True, the current configuration will be displayed after each dislocation
# is inserted.
# Otherwise, the current dislocation is only saved to the {name}_{timestamp}.svg
# file.
show_img = True
# Field of view: the OX/OY limits are derived from the bounding box of the
# dislocation cores and of the integration path drawn below (see
# `field_of_view`),
# so that the figures show the neighbourhood of the dislocations instead of the
# whole -- mostly undistorted -- crystal. `fov_margin` is the margin left around
# that bounding box [A].
fov_margin = 8.0
# Width of the figure [inches]; the height is derived from the field of view,
# so that the aspect ratio of the figure matches the aspect ratio of the data
# (plot_atoms_2d draws with aspect="equal").
fig_width = 12.0
# Size of the dislocation tees (the inverted-T symbols).
tee_scale = 0.6

# --- The glide planes ----------------------------------------------------
# Numbers (1-based) of the dislocations whose glide plane is drawn. A glide
# plane is the trace of the cut along which the dislocation was inserted, drawn
# as a pair of opposed harpoons -- one per lip -- pointing the ways the two lips
# slip; see glide_planes.draw_glide_plane. Each is drawn in the panel in which
# its dislocation appears, in the colour of that dislocation, i.e. of its tee.
#
# Only the cut of the dislocation inserted in the panel is available here.
# Carrying the earlier cuts through the later insertions -- so that the steps
# the later cuts leave on them become visible -- is what 6_dislocations_8.py
# does; this figure is about the integration path, and needs no such thing.
glide_plane_dislocation_nrs = (2,)
# The harpoon style. Same values as in 6_dislocations_8.py, so that a glide
# plane looks the same in both figures; they are passed explicitly, because
# glide_planes.py carries its own defaults which must not silently override
# what is set here.
gp_linewidth = 1.225
gp_lip_offset = 0.157
gp_head_length = 0.665
gp_head_angle = 40.0
gp_head_min_length = 4.0
gp_head_spacing = 2.2
gp_style = dict(linewidth=gp_linewidth, lip_offset=gp_lip_offset,
                head_length=gp_head_length, head_angle=gp_head_angle,
                head_min_length=gp_head_min_length,
                head_spacing=gp_head_spacing)
# The glide plane is not drawn closer than this [A] to any dislocation core:
# there the two lips of the cut merge and a single line stops representing the
# cut at all (see clip_glide_plane).
gp_core_radius = 1.0

# --- The integration path ------------------------------------------------
# The displacement of every atom is obtained by integrating Eq. (29) along a
# path running from x_o -- a point next to the core of the dislocation being
# inserted -- to the reference position \hat{x} of that atom. `displace`
# computes such a path for every atom; the one drawn belongs to the atom
# nearest to `d_{path_anchor_nr}.position + path_atom_offset`, i.e. an atom to
# the LEFT of d_1 and one row BELOW it.
#
# The path is anchored to a dislocation of its own (`path_anchor_nr`) rather
# than to the one being inserted, so that the same atom keeps being shown no
# matter in which order the dislocations are inserted.
#
# NOTE: the path is drawn in the panel showing the configuration the integral
# is taken OVER, i.e. the lattice as it stands BEFORE the insertion the path
# belongs to -- one panel earlier than that insertion. That is the
# configuration the path and the reference position \hat{x} live in. Drawing
# it on the panel of its own insertion would put it in the wrong configuration:
# the atoms move by up to |b|/2 = 1.59 A as the dislocation goes in, so the end
# of the path would miss the very atom it belongs to (measured here: 1.26 A
# away, i.e. closer to a different atom than to its own).
#
# For the same reason x_o -- the start of the path, next to the core of the
# dislocation being inserted -- falls where that dislocation is about to
# appear, and so in a panel that does not show it yet.
# Number (1-based) of the dislocation whose insertion the path belongs to.
path_dislocation_nr = 2
# Number (1-based) of the dislocation the drawn atom is picked relative to.
path_anchor_nr = 1
# Where that atom sits, as a lattice translation (Miller indices) from the core
# of dislocation `path_anchor_nr`: three cells along -a_1 and one along -a_2,
# i.e. to the left of the core and one row below it. Decreasing the second
# index by one moves the target one unit cell down, increasing it moves it one
# up.
#
# The target is given as a lattice translation rather than as a plain
# displacement in OX/OY so that it lands on a site of the crystal. The basal
# plane of the wurtzite cell has gamma = 120 deg, so a_2 is not along OY: a
# step of one row in +OY alone falls 0.85 A from the nearest atom, whereas
# every lattice translation falls within 0.3 A of one.
path_atom_miller = (-3, -1, 0)
path_atom_offset = np.asarray(
    l0.cell.to_cartesian_indices(np.asarray(path_atom_miller)), dtype=float)
# Red, so that the path is not confused with the (black) bonds it runs over.
path_color = "red"
path_linewidth = 2.2
path_linestyle = (0, (5, 2.5))
# If True, the panel carrying the path also shows the tee of the dislocation
# whose insertion the path belongs to -- d_2 here -- even though that panel
# shows the lattice BEFORE the insertion, so the dislocation is not in it yet.
# Without it the figure is hard to read: x_o, the start of the path, sits right
# next to a core that would otherwise not be drawn at all. The tee is placed at
# the reference position of the core, i.e. where the cut starts, and its
# Burgers direction is taken in the undeformed lattice (`cell=` of
# display_tee_2d), because the dislocation has not deformed anything yet.
path_show_next_dislocation = True
# If True, the same panel also shows the integration paths that end in the
# reference position of a dislocation ALREADY present in the lattice, rather
# than in an atom. Those are the paths whose line integral, Eq. (29), gives
# each such dislocation its new position, i.e. what the Newton-Raphson scheme
# iterates on; with two dislocations there is exactly one, from x_o to d_1.
# They start at the same x_o as the atom paths.
path_show_dislocation_paths = True
# Violet, so that a path ending in a dislocation is not taken for the (red) one
# ending in an atom.
path_dislocation_color = "#7d3ac1"
# Label put at the end of such a path; {label} is the label of the dislocation
# the path ends in, e.g. "$d_1$".
path_dislocation_end_label = r"$\hat{{x}}_{{{label}}}$"

# offset_0: The offset that will be added to the position of each
# dislocation.
# The purpose of the offset was to move all the dislocations to satisfy
# requirements for the figures generated in the target article,
# i.e. two unit cells margin at the top, etc.
ny = 2
ox = l0.cell.dimensions[1] * 0.5
oy = l0.cell.dimensions[1] * math.sqrt(3) / 2
offset_0 = np.asarray([2 * l0.cell.dimensions[0] + ox, oy, 0])

# One unit cell "above" in the coordinate system of the lattice, i.e. the
# lattice translation a_2 = [0, 1, 0] (Miller), which points 120 deg away from
# a_1 and therefore upwards. Using the lattice vector rather than a plain
# shift along OY keeps the core of d_1 on an equivalent site of the crystal.
cell_up = np.asarray(l0.cell.to_cartesian_indices(np.asarray([0, 1, 0])),
                     dtype=float)

# --- The core type of d_1 -----------------------------------------------
# The core type -- the number of atomic columns in the ring that closes around
# the core -- is decided by the position the dislocation line takes inside the
# unit cell, cf. the two singularity points marked in Fig. 1 of the paper,
# which give the 5:7 and the 8 core out of the very same displacement field.
# `d1_core_offset` moves the line off the site it would otherwise take, in the
# frame of the dislocation itself: the first component runs along the Burgers
# vector b (in units of the lattice parameter a, its period in that direction),
# the second along the trace of the extra half-plane (in units of a*sqrt(3)/2,
# the spacing of the atomic rows).
#
# The offset is a choice of SITE within the cell, and nothing else: the fact
# that the lattice a dislocation is inserted into has already been displaced by
# the ones inserted before it is taken care of separately and automatically, by
# `follow_lattice` below. Here the bare position already gives the 8-ring core.
d1_core_offset = (0.0, 0.0)

_b1 = np.asarray(l0.cell.to_cartesian_indices(np.asarray([0, 1, 0])),
                 dtype=float)[:2]
_b1_hat = _b1 / np.linalg.norm(_b1)
# The trace of the extra half-plane, i.e. b turned by +90 deg.
_h1_hat = np.asarray([-_b1_hat[1], _b1_hat[0]])
core_offset_1 = np.append(
    d1_core_offset[0] * l0.cell.dimensions[0] * _b1_hat
    + d1_core_offset[1] * l0.cell.dimensions[0] * math.sqrt(3) / 2 * _h1_hat,
    0.0)

# Filename prefix (will be prepended to all filenames generated by this script).
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename_prefix = f"2_dislocations_{timestamp}"

# One colour per dislocation, used for its tee. The colours follow the LABEL,
# as they do in 6_dislocations_8.py -- d_1 brown, d_2 blue -- so a reader who
# knows the six-dislocation figure reads the tees here the same way. They do
# not follow the individual dislocations across the two scripts: the one drawn
# in brown here is the left-hand one, which in 6_dislocations_8.py is d_2 and
# therefore blue.
DISLOCATION_COLORS = [
    "brown",    # d_1
    "#2a78d6",  # d_2 -- blue
]

# NOTE: the dislocations are inserted in the order they are listed here, and
# they are labelled in that same order. The one on the LEFT -- the one carrying
# the 8-ring core and the drawn integration path -- is therefore d_1 and goes
# into the perfect crystal, and the one on the RIGHT is d_2 and goes into the
# lattice already deformed by d_1.
dislocations = [
    wzt.model.DislocationDef(
        # dislocation label (only for the presentation purpose)
        label="$d_1$",
        # Burger's vector: b = a_2.
        b=[0, 1, 0],
        # dislocation core position: the core of 6_dislocations_8.py's d_2,
        # moved one unit cell up, on the site that gives an 8-ring core.
        position=[2.35, 9.66, 0] + offset_0 + cell_up + core_offset_1,
        # dislocation plane
        plane=(0, 0, 1),
        # dislocation tee color (only for the presentation purpose)
        color=DISLOCATION_COLORS[0]
    ),
    wzt.model.DislocationDef(
        label="$d_2$",
        # b = a_1, i.e. the Burgers vector of d_1 turned by -120 deg (the
        # Burgers vectors are given in Miller indices, and the basal plane of
        # the wurtzite cell has gamma = 120 deg, so [1, 0, 0] is exactly
        # [0, 1, 0] turned by -120 deg).
        b=[1, 0, 0],
        position=[3.190 + 0.8 + 6 * l0.cell.dimensions[0], 10, 7.5] + offset_0,
        plane=(0, 0, 1),
        color=DISLOCATION_COLORS[1]
    ),
]
# Numbers of the dislocations for which the (interactive, and quite expensive)
# diagnostic plots of the beta field and the glide planes should be displayed,
# e.g. {1}. Empty set = batch run.
debug_plots = set()

path_atom_target = (np.asarray(dislocations[path_anchor_nr - 1].position,
                              dtype=float) + path_atom_offset)


def follow_lattice(position, crystal, n_neighbours=12):
    """
    Carries a dislocation core position along with the lattice.

    The core positions above are given in the coordinates of the PERFECT
    crystal, but what they are meant to pick out is a site of the MATERIAL --
    the place inside the unit cell that decides the core type, cf.
    `d1_core_offset`. By the time a dislocation is inserted, the ones inserted
    before it have already displaced that piece of the lattice, so a position
    left in the coordinates of the perfect crystal no longer points at the site
    it was chosen for, and the core comes out shifted inside its own ring.

    Here the effect is 0.95 A -- about half a Ga-N bond (1.95 A) -- at the site
    of d_2, and it is nearly a rigid translation: the displacement d_1 imposes
    varies by only 0.05 A over the atoms surrounding that site. So the position
    is simply moved by the mean displacement of the `n_neighbours` atoms nearest
    to it, which puts the core back where it sits in the perfect lattice.

    :param position: the core position in the coordinates of the perfect crystal
    :param crystal: the lattice the dislocation is about to be inserted into
    """
    p = np.asarray(position, dtype=float)
    a0 = np.asarray(l0.coordinates, dtype=float)
    u = np.asarray(crystal.coordinates, dtype=float) - a0
    near = np.argsort(np.linalg.norm(a0[:, :2] - p[:2], axis=1))[:n_neighbours]
    # Only in the plane of the figure: the out-of-plane coordinate of a core is
    # arbitrary (the lines are straight and perpendicular to it).
    return p + np.append(u[near].mean(axis=0)[:2], 0.0)


def field_of_view(extra=()):
    """
    The OX/OY limits and the figure size shared by all the panels.

    The field of view is the bounding box of the dislocation cores, of the atom
    whose integration path is drawn and of everything passed in `extra`,
    extended by `fov_margin` [A] and clipped to the extent of the crystal (so
    that no vacuum is shown around the lattice).

    The integration paths belong in `extra`: they are not confined to the
    neighbourhood of the cores at all. A path starts next to the core of the
    dislocation being inserted and has to reach the atom while avoiding the
    other cores and the cut, so it can sweep far around them -- here it dips to
    within 1 A of the bottom of the crystal, which the box around the cores
    alone would have cut off. All the panels share these limits, so the path
    has to fit even in the panels that do not show it.

    :param extra: point sets, each (n, >= 2), that have to fit in the figure
    """
    points = [np.asarray(d.position, dtype=float)[:2].reshape(1, 2)
              for d in dislocations]
    points.append(path_atom_target[:2].reshape(1, 2))
    points += [np.asarray(e, dtype=float).reshape(-1, np.shape(e)[-1])[:, :2]
               for e in extra]
    points = np.concatenate(points)
    atoms = np.asarray(l0.coordinates, dtype=float)
    xlim = (max(points[:, 0].min() - fov_margin, atoms[:, 0].min()),
            min(points[:, 0].max() + fov_margin, atoms[:, 0].max()))
    ylim = (max(points[:, 1].min() - fov_margin, atoms[:, 1].min()),
            min(points[:, 1].max() + fov_margin, atoms[:, 1].max()))
    # Keep the figure aspect ratio equal to the aspect ratio of the field of
    # view (plot_atoms_2d draws with aspect="equal").
    figsize = (fig_width,
               fig_width * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))
    return xlim, ylim, figsize


def draw_integration_path(ax, path, color=path_color,
                          linewidth=path_linewidth, linestyle=path_linestyle,
                          end_label=r"$\hat{x}$", zorder=9000):
    """
    Draws one integration path of Eq. (29), from x_o (next to the core of the
    dislocation being inserted) to the reference position \\hat{x} of an atom
    or -- see `path_show_dislocation_paths` -- of a dislocation.

    :param path: the polyline of the path, (n, >= 2), global coordinates
    :param end_label: what to write at the end of the path
    """
    p = np.asarray(path, dtype=float)
    ax.plot(p[:, 0], p[:, 1], color=color, linewidth=linewidth,
            linestyle=linestyle, zorder=zorder)
    # x_o -- the beginning of the path.
    ax.plot(p[0, 0], p[0, 1], marker="o", markersize=5, color=color,
            zorder=zorder)
    label_box = dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                     alpha=0.8)
    ax.annotate(r"$x_o$", xy=(p[0, 0], p[0, 1]), xytext=(4, 4),
                textcoords="offset points", zorder=zorder, bbox=label_box)
    # \hat{x} -- the end of the path, i.e. the atom.
    ax.plot(p[-1, 0], p[-1, 1], marker="o", markersize=5, color=color,
            zorder=zorder)
    ax.annotate(end_label, xy=(p[-1, 0], p[-1, 1]), xytext=(4, 4),
                textcoords="offset points", zorder=zorder, bbox=label_box)


def main(params):
    current_state = params.get("init_state")
    start_dislocation = params.get("start_dislocation")
    is_save_state = not params.get("skip_save_state")

    if not current_state:
        # We start from the perfect configuration
        l = l0  # Current lattice
        current_dislocations = []  # Dislocations already applied
        n_ready_dislocations = 0
    else:
        # Start from the state read from the .pkl file
        l = current_state["l"][start_dislocation]  # Current lattice
        # Dislocations already applied
        current_dislocations = current_state["dislocations"][start_dislocation]
        n_ready_dislocations = len(current_dislocations)

    ls = []
    all_dislocation_states = []
    all_convergence = []
    all_integration_paths = []
    all_glide_planes = []
    # The integration path picked for drawing, and the panel it belongs in.
    drawn_path = None
    # The dislocation whose insertion that path belongs to, as it stands in the
    # configuration the path lives in, i.e. before it is inserted.
    drawn_path_dislocation = None
    # The integration paths that end in the reference position of a dislocation
    # already present, paired with that dislocation: [(d, path), ...].
    drawn_dislocation_paths = []

    # ---- Insert the dislocations one by one. Nothing is drawn yet: the field
    # of view has to contain the integration path, which is only known once the
    # displacement has been computed, and all the panels must share it.
    for i in range(n_ready_dislocations, len(dislocations)):
        d = dislocations[i]
        print(f"------------------------------------------ DISLOCATION: {i+1}")
        if current_dislocations:
            # The lattice this one goes into has already been deformed by the
            # dislocations inserted before it -- move the core along with it.
            moved = follow_lattice(d.position, l)
            print(f"  core position carried with the lattice: "
                  f"{np.round(np.asarray(d.position, float), 3)} -> "
                  f"{np.round(moved, 3)}")
            d = dataclasses.replace(d, position=moved)
        # Calculate displacement + auxiliary information
        log = displace(
            crystal=l,
            dislocations=current_dislocations,
            d_n=d,
            n_iters=n_iters,
            n_points=n_points,
            method=method,
            alpha=alpha,
            tol=tol,
            plot_local=(i in debug_plots),
            plot_local_planes=(i in debug_plots)
        )
        all_convergence.append(log.convergence)
        # Displacement.
        u = log.last_u_atoms
        # Log of the last state
        # (where dislocations are located, glide planes, etc.)
        last_d_state = log.last_d_state
        current_dislocations = last_d_state.ds
        # The configuration the integration paths are expressed in, i.e. the one
        # the line integral of Eq. (29) is taken over.
        l_before = l
        # Translate current lattice according to the displace.
        l = l.translate(u)
        # Update bonds between atoms.
        l = wzt.generate.update_bonds(l, tolerance=0.55)

        # The integration path of a single atom: the one nearest to
        # `path_atom_target`, i.e. to the point `path_atom_offset` away from the
        # core of the dislocation number `path_anchor_nr`. The atom is looked up
        # in `l_before`, because that -- and not the configuration produced by
        # this insertion -- is the one the paths were computed in.
        # `log.last_integration_paths` is [(n atoms, n points, 3)], in the same
        # order as the coordinates of that lattice.
        paths = log.last_integration_paths
        if paths and i + 1 == path_dislocation_nr:
            paths = np.asarray(paths[0])
            before = np.asarray(l_before.coordinates, dtype=float)
            atom_nr = int(np.argmin(
                np.linalg.norm(before[:, :2] - path_atom_target[:2], axis=1)))
            print(f"Integration path selected for atom {atom_nr} at "
                  f"{before[atom_nr]} (target: {path_atom_target})")
            drawn_path = paths[atom_nr]
            drawn_path_dislocation = d
            # The paths that end in the cores already present, in the order of
            # `current_dislocations` (the inserted one is last, and has none).
            drawn_dislocation_paths = list(
                zip(current_dislocations, log.last_dislocation_paths or []))

        # The cut this dislocation was inserted along, already carried by
        # `displace` into the configuration this panel shows.
        plane = None
        if i + 1 in glide_plane_dislocation_nrs:
            plane = log.last_glide_planes_displaced[-1]

        ls.append(l)
        all_dislocation_states.append(current_dislocations)
        all_integration_paths.append(None)
        all_glide_planes.append(plane)

    # The path belongs in the panel showing the configuration it was computed
    # in, i.e. the one drawn BEFORE the insertion it belongs to -- see the
    # NOTE next to `path_dislocation_nr`.
    if drawn_path is not None:
        panel = path_dislocation_nr - 2 - n_ready_dislocations
        if panel < 0:
            print("WARNING: the configuration the integration path was "
                  "computed in is not among the panels of this run, so the "
                  "path is not drawn.")
        else:
            all_integration_paths[panel] = drawn_path
            print(f"Integration path drawn in panel {panel + 1} "
                  f"(the configuration it was computed in).")

    # ---- Draw the panels, all of them in the same field of view.
    xlim, ylim, figsize = field_of_view(
        [p for p in all_integration_paths if p is not None]
        + [q for _, q in drawn_dislocation_paths])
    print(f"Field of view: xlim={xlim}, ylim={ylim}, figsize={figsize}")

    for i, (l, ds, path, plane) in enumerate(
            zip(ls, all_dislocation_states, all_integration_paths,
                all_glide_planes),
            start=n_ready_dislocations):
        # Draw atoms.
        fig, ax = wzt.visualization.plot_atoms_2d(l, xlim=xlim, ylim=ylim,
                                                  figsize=figsize)
        # Draw the glide plane of the dislocation inserted in this panel (it is
        # the last of `ds`), in the colour of its tee.
        if plane is not None:
            d_gp = ds[-1]
            cores = [dd.position for dd in ds]
            for segment in clip_glide_plane(plane, l.coordinates, cores,
                                            core_radius=gp_core_radius,
                                            bbox=(xlim, ylim)):
                draw_glide_plane(ax, segment, b=d_gp.b,
                                 half_plane=getattr(d_gp, "half_plane", None),
                                 color=d_gp.color, **gp_style)
        # Draw tees.
        for d_drawn in ds:
            wzt.visualization.display_tee_2d(ax, d_drawn, scale=tee_scale)
        if path is not None:
            if path_show_next_dislocation and drawn_path_dislocation is not None:
                # The dislocation the path belongs to, marked where it is about
                # to be inserted -- see `path_show_next_dislocation`.
                wzt.visualization.display_tee_2d(
                    ax, drawn_path_dislocation, scale=tee_scale, cell=l0.cell)
            if path_show_dislocation_paths:
                for d_end, d_path in drawn_dislocation_paths:
                    draw_integration_path(
                        ax, d_path, color=path_dislocation_color,
                        end_label=path_dislocation_end_label.format(
                            label=d_end.label.strip("$")))
            draw_integration_path(ax, path)

        # Save to the image files (.svg for viewing, .pdf ready to be included
        # in the LaTeX sources of the paper).
        for extension in ("svg", "pdf"):
            filename = f"{filename_prefix}_dislocation_{i}.{extension}"
            fig.savefig(filename, bbox_inches="tight")
            print(f"Image saved to {filename}")
        # Show image, if enabled.
        if show_img:
            plt.show()

    print("Convergence history (max||Psi|| [A], max||dx|| [A]) per dislocation:")
    for i, c in enumerate(all_convergence, start=n_ready_dislocations + 1):
        print(f"  dislocation {i}: {c}")

    if is_save_state:
        with open(f"state_{timestamp}.pkl", "wb") as f:
            state = {
                "l": ls,
                "dislocations": all_dislocation_states,
                "convergence": all_convergence,
                # The integration path drawn in each panel (None if there was
                # none), in global coordinates.
                "integration_paths": all_integration_paths,
                # The glide plane drawn in each panel (None if there was none),
                # unclipped, in global coordinates.
                "glide_planes": all_glide_planes,
                # The integration paths ending in a dislocation core, as drawn:
                # [(dislocation, path), ...] in global coordinates.
                "dislocation_paths": drawn_dislocation_paths,
                "method": method,
                "n_points": n_points,
                "alpha": alpha,
                "tol": tol,
                "colors": DISLOCATION_COLORS,
            }
            pickle.dump(state, f)


if __name__ == "__main__":
    default_params = params.read_default_params()
    main(default_params)
