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
# the modified (relaxed) one. The residual contracts with the ratio (1 - alpha),
# so the classical scheme needs 2-4 iterations here instead of the 7-9 needed
# at alpha = 0.6, and all the values tried (0.6, 0.8, 1.0) converge to the same
# configuration (cores to 1.3e-3 A, atoms to 5.3e-4 A).
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
# dislocation cores (see `xlim`/`ylim` below), so that the figures show the
# neighbourhood of the dislocations instead of the whole -- mostly undistorted
# -- crystal. `fov_margin` is the margin left around the outermost cores [A].
fov_margin = 5.0
# Width of the figure [inches]; the height is derived from the field of view,
# so that the aspect ratio of the figure matches the aspect ratio of the data
# (plot_atoms_2d draws with aspect="equal").
fig_width = 12.0
# Glide plane line width [points]
linewidth = 1.225
# Colour of a glide plane whose dislocation does not define one.
gp_color = "tab:blue"
# Size of the dislocation tees (the inverted-T symbols).
tee_scale = 0.6
# The glide plane is drawn as a pair of opposed harpoons (like the LaTeX
# \rightleftharpoons), one per lip of the cut, showing the direction in which
# each lip slips. `gp_lip_offset` [A] is the distance of each harpoon from the
# traced cut -- it stays well inside the interplanar gap (~1.84 A), so the two
# harpoons remain between the same pair of atomic planes as the cut itself.
# Each glide plane is drawn in the colour of its own dislocation, i.e. in the
# colour of its tee (see the `color` field of the definitions below).
gp_lip_offset = 0.157
# Length [A] and half-angle [deg] of the barb of a harpoon head. On a SOLID
# glide plane the barb is kept short on purpose: it must not reach the
# neighbouring head of the OTHER lip, which points the opposite way and grows
# towards it, or the two fuse into a single chevron reading as one arrowhead
# across both lines. The condition is
# 2 * gp_head_length * cos(gp_head_angle) < gp_head_spacing / 2, and
# draw_glide_plane() enforces it. On a dashed glide plane the heads sit at
# opposite ends of a dash and cannot meet.
gp_head_length = 0.665
gp_head_angle = 40.0
# A harpoon head is only drawn on pieces of the glide plane at least this long
# [A]; shorter pieces are drawn as plain lines.
gp_head_min_length = 4.0
# Distance [A] between consecutive harpoon heads on one lip, so that the
# direction of the slip can be read anywhere along a glide plane and not only
# where it ends. The two lips take alternate stations, i.e. a head of one falls
# half of this distance from the heads of the other.
gp_head_spacing = 2.2
# The glide planes of the dislocations inserted EARLIER are drawn dashed and
# half transparent, so that the one belonging to the dislocation inserted in
# the panel at hand -- solid and opaque -- can be told from those that are only
# being carried along.
#
# The dashes are laid out in the plane of the figure, in Angstroem, and each
# dash carries exactly one harpoon head, at the end its lip slips towards. A
# dash and the one facing it on the other lip therefore read as a single
# \rightleftharpoons. `gp_dash_length` is thus the length of one harpoon, and
# the barb is kept to `gp_head_dash_fraction` of it, so that the two stay in
# proportion however often the harpoons are made to repeat.
gp_previous_alpha = 0.5
gp_dash_length = 0.7
gp_dash_gap = 0.4
gp_head_dash_fraction = 0.36
# If True, every panel shows the glide planes of ALL the dislocations inserted
# so far, not only the one inserted last. The glide planes of the earlier
# dislocations are carried through the displacement field of every subsequent
# insertion, so the steps left on them by the later cuts are visible.
gp_show_previous = True
# The parameters above, in the form draw_glide_plane() takes them. They are
# passed explicitly at every call: glide_planes.py carries its own defaults,
# and they must not silently override what is set here.
gp_style = dict(linewidth=linewidth, lip_offset=gp_lip_offset,
                head_length=gp_head_length, head_angle=gp_head_angle,
                head_min_length=gp_head_min_length,
                head_spacing=gp_head_spacing,
                dash_length=gp_dash_length, dash_gap=gp_dash_gap,
                head_dash_fraction=gp_head_dash_fraction)
# The glide plane is not drawn closer than this [A] to any dislocation core:
# there the two lips of the cut merge and a single line stops representing the
# cut at all (see clip_glide_plane). About half a Ga-N bond, so the line stops
# just clear of the core atoms. This used to be 4.0 A, to hide the staircase
# that displace_glide_plane() produced before it was made to weight the atom
# displacements smoothly; measured against the current curves the polyline is
# as smooth at 0.5 A from a core (turn <= 0.39 deg per 0.1 A step) as it is in
# the bulk, so the exclusion is now a statement about the physics only.
gp_core_radius = 1.0

# offset_0: The offset that will be added to the position of each
# dislocation.
# The purpose of the offset was to move all the dislocations to satisfy
# requirements for the figures generated in the target article,
# i.e. two unit cells margin at the top, etc.
ny = 2
ox = l0.cell.dimensions[1] * 0.5
oy = l0.cell.dimensions[1] * math.sqrt(3) / 2
offset_0 = np.asarray([2 * l0.cell.dimensions[0] + ox, oy, 0])

# Filename prefix (will be prepended to all filenames generated by this script).
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename_prefix = f"6_dislocations_{timestamp}"

# One colour per dislocation, used for its tee AND for its glide plane, so that
# a line can be traced back to the dislocation it belongs to. Slots 2-6 are the
# categorical palette of the convergence plot, in the same order, so that a
# dislocation wears the same colour in both figures -- plot_convergence.py reads
# this list back from the state file. d_1 is brown: it carries no convergence
# history (there is nothing to solve for when it is inserted into the perfect
# lattice), so it never appears in the convergence plot and does not take one of
# its slots.
DISLOCATION_COLORS = [
    "brown",    # d_1
    "#2a78d6",  # d_2 -- blue
    "#eb6834",  # d_3 -- orange
    "#1baf7a",  # d_4 -- aqua
    "#eda100",  # d_5 -- yellow
    "#e87ba4",  # d_6 -- magenta
]

dislocations = [
    wzt.model.DislocationDef(
        # dislocation label (only for the presentation purpose)
        label="$d_1$",
        # Burger's vector
        b=[1, 0, 0],
        # dislocation core position
        position=[3.190 + 0.8 + 6 * l0.cell.dimensions[0], 10, 7.5] + offset_0,
        # dislocation plane
        plane=(0, 0, 1),
        # dislocation tee color (only for the presentation purpose)
        color=DISLOCATION_COLORS[0]
    ),
    wzt.model.DislocationDef(
        label="$d_2$",
        b=[1, 0, 0],
        position=[2.35, 9.66, 0] + offset_0,
        plane=(0, 0, 1),
        color=DISLOCATION_COLORS[1]
    ),
    wzt.model.DislocationDef(
        label="$d_3$",
        b=[0, 1, 0],
        position=[8.53, 17.61, 0] + offset_0,
        plane=(0, 0, 1),
        color=DISLOCATION_COLORS[2]
    ),
    wzt.model.DislocationDef(
        label="$d_4$",
        b=[1, 0, 0],
        position=[2.82, 21.54, 0],
        plane=(0, 0, 1),
        color=DISLOCATION_COLORS[3]
    ),
    wzt.model.DislocationDef(
        label="$d_5$",
        b=[1, 1, 0],
        position=[18.57, 18.92, 0] + offset_0,
        plane=(0, 0, 1),
        color=DISLOCATION_COLORS[4]
    ),
    wzt.model.DislocationDef(
        label="$d_6$",
        b=[0, -1, 0],
        position=[14.01, 4.74, 0] + offset_0,
        plane=(0, 0, 1),
        color=DISLOCATION_COLORS[5]
    )
]
# How far the cut is traced from the core [A]. `get_glide_plane` integrates the
# ODE over t_span = (0, -glide_plane_margin) in the local frame of the
# dislocation, so this is a hard end to the curve: with the default 35 the cut
# ended a few A inside the crystal, and both the drawn line and the tear in the
# lattice stopped there in mid-air.
#
# It is not only a drawing matter. The curve is what
# get_integration_path_via_energy splits the lattice graph along; an atom whose
# x falls outside the traced range gets its side from a straight line through
# the core instead. So the cut has to reach past the crystal, and the value is
# derived rather than guessed: the distance from the farthest core to the
# farthest corner of the crystal, plus a margin. clip_glide_plane trims
# whatever overshoots the atoms when drawing.
_atoms_xy = np.asarray(l0.coordinates, dtype=float)[:, :2]
_corners = np.asarray([[x, y]
                       for x in (_atoms_xy[:, 0].min(), _atoms_xy[:, 0].max())
                       for y in (_atoms_xy[:, 1].min(), _atoms_xy[:, 1].max())])
glide_plane_margin = 5.0 + max(
    np.linalg.norm(_corners - np.asarray(d.position, dtype=float)[:2], axis=1).max()
    for d in dislocations)
print(f"Glide plane traced over {glide_plane_margin:.1f} A from each core")

# Numbers of the dislocations for which the (interactive, and quite expensive)
# diagnostic plots of the beta field and the glide planes should be displayed,
# e.g. {5}. Empty set = batch run.
debug_plots = set()

# Field of view: the bounding box of the dislocation cores, extended by
# `fov_margin` and clipped to the extent of the crystal (so that no vacuum is
# shown around the lattice).
_cores = np.stack([np.asarray(d.position, dtype=float) for d in dislocations])
_atoms = np.asarray(l0.coordinates, dtype=float)
xlim = (max(_cores[:, 0].min() - fov_margin, _atoms[:, 0].min()),
        min(_cores[:, 0].max() + fov_margin, _atoms[:, 0].max()))
ylim = (max(_cores[:, 1].min() - fov_margin, _atoms[:, 1].min()),
        min(_cores[:, 1].max() + fov_margin, _atoms[:, 1].max()))
# Keep the figure aspect ratio equal to the aspect ratio of the field of view.
figsize = (fig_width,
           fig_width * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))
print(f"Field of view: xlim={xlim}, ylim={ylim}, figsize={figsize}")



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
        # Number of dislocations already applied. For example, if already
        n_ready_dislocations = len(current_dislocations)

    ls = []
    all_dislocation_states = []
    all_convergence = []
    all_glide_planes = []
    # The glide planes of the dislocations inserted so far, in the coordinate
    # system of the figures, one polyline per dislocation. They are carried
    # through the displacement field of every subsequent insertion (see below),
    # so they keep the steps left on them by the cuts made in the meantime.
    carried_planes = []

    for i in range(n_ready_dislocations, len(dislocations)):
        d = dislocations[i]
        print(f"------------------------------------------ DISLOCATION: {i+1}")
        # Calculate displacement + auxiliary information
        log = displace(
            crystal=l,
            dislocations=current_dislocations,
            d_n=d,
            n_iters=n_iters,
            n_points=n_points,
            glide_plane_margin=glide_plane_margin,
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
        # The configuration the glide planes of the earlier dislocations are
        # still expressed in.
        l_before = l
        # Translate current lattice according to the displace.
        l = l.translate(u)
        # Update bonds between atoms.
        l = wzt.generate.update_bonds(l, tolerance=0.55)
        # Draw atoms.
        fig, ax = wzt.visualization.plot_atoms_2d(l, xlim=xlim, ylim=ylim, figsize=figsize)
        # Draw tees.
        for d in current_dislocations:
            wzt.visualization.display_tee_2d(ax, d, scale=tee_scale)

        # The glide planes carried into the configuration that is actually
        # drawn (i.e. after the current dislocation has been inserted). The
        # plane of the dislocation inserted now comes from `displace`, which
        # traced it in the configuration deformed by the earlier dislocations
        # and then carried it through its own displacement field. The planes of
        # the earlier dislocations are carried through that same field here --
        # they are material surfaces, so the cut made now tears them, and the
        # step this leaves is exactly what shows that the glide plane of, say,
        # d_1 has itself been dislocated by d_3.
        # The cut that the dislocation inserted now makes in the lattice, traced
        # in the configuration the earlier glide planes are still expressed in.
        # It is what tears them, so it has to be kept out of the averaging --
        # otherwise the step it leaves is smeared over ~2 A and reads as a bend.
        new_cut = log.last_glide_planes[-1]
        carried_planes = [
            displace_glide_plane(g, atoms_local=l_before.coordinates, u_atoms=u,
                                 other_cuts=[new_cut])
            for g in carried_planes
        ]
        carried_planes.append(log.last_glide_planes_displaced[-1])

        cores = [d.position for d in current_dislocations]
        drawn = list(zip(current_dislocations, carried_planes))
        if not gp_show_previous:
            drawn = drawn[-1:]
        segments_per_dislocation = []
        for k, (d, plane) in enumerate(drawn):
            segments = clip_glide_plane(plane, l.coordinates, cores,
                                        core_radius=gp_core_radius,
                                        bbox=(xlim, ylim))
            segments_per_dislocation.append(segments)
            # d_1 ... d_{i-1} dashed, d_i solid.
            is_last = (k == len(drawn) - 1)
            for p in segments:
                draw_glide_plane(ax, p, b=d.b,
                                 half_plane=getattr(d, "half_plane", None),
                                 color=d.color or gp_color,
                                 dashed=not is_last,
                                 alpha=1.0 if is_last else gp_previous_alpha,
                                 **gp_style)
        all_glide_planes.append(segments_per_dislocation)

        # Save to the image files (.svg for viewing, .pdf ready to be included
        # in the LaTeX sources of the paper).
        for extension in ("svg", "pdf"):
            filename = f"{filename_prefix}_dislocation_{i}.{extension}"
            fig.savefig(filename, bbox_inches="tight")
            print(f"Image saved to {filename}")
        # Show image, if enabled.
        if show_img:
            plt.show()

        ls.append(l)
        all_dislocation_states.append(current_dislocations)

    print("Convergence history (max||Psi|| [A], max||dx|| [A]) per dislocation:")
    for i, c in enumerate(all_convergence, start=n_ready_dislocations + 1):
        print(f"  dislocation {i}: {c}")

    if is_save_state:
        with open(f"state_{timestamp}.pkl", "wb") as f:
            state = {
                "l": ls,
                "dislocations": all_dislocation_states,
                "convergence": all_convergence,
                # The glide planes as drawn in the figures (clipped to the
                # crystal): one entry per panel, and inside it one list of
                # polyline pieces per dislocation drawn in that panel.
                "glide_planes": all_glide_planes,
                "method": method,
                "n_points": n_points,
                "alpha": alpha,
                "tol": tol,
                # So that the convergence plot can paint every dislocation in
                # the colour it wears in the lattice figures.
                "colors": DISLOCATION_COLORS,
            }
            pickle.dump(state, f)


if __name__ == "__main__":
    default_params = params.read_default_params()
    main(default_params)
