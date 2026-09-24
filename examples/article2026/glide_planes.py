"""
Drawing of the glide planes -- the traces of the cuts along which the
dislocations were inserted -- shared by the example scripts of this directory
(6_dislocations_8.py, 2_dislocations_8.py), so that a glide plane looks the
same wherever it appears.

The defaults below are the values the figures of the paper are drawn with; a
script that wants to depart from them passes its own.
"""
import numpy as np

# Colour of a glide plane whose dislocation does not define one.
DEFAULT_COLOR = "tab:blue"


def clip_glide_plane(glide_plane, coordinates, cores, atom_cutoff=2.0,
                     core_radius=4.0, min_vertices=5, bbox=None,
                     bbox_inset=0.3):
    """
    Selects the parts of the glide plane polyline that are worth drawing, and
    returns them as a list of separate polylines.

    Two parts are dropped:

    1. Everything outside the crystal. `find_glide_plane` integrates the glide
       plane over a fixed margin (`glide_plane_margin`) expressed in the local
       coordinate system of the dislocation, without any reference to the extent
       of the lattice. Because `create_lattice` produces a skewed (parallelogram)
       slab, the boundary of the crystal at the height of the glide plane depends
       on the dislocation, and the fixed margin overshoots it -- the line would
       be drawn over the vacuum next to the lattice. Vertices further than
       `atom_cutoff` [A] from any atom are dropped.

    2. Everything closer than `core_radius` [A] to a dislocation core.
       `displace_glide_plane` carries the polyline into the current configuration
       by averaging the displacements of the atoms just above and just below the
       cut. Close to a core the two lips of the cut merge, that average stops
       being meaningful, and the polyline visibly kinks (up to ~40 deg, whereas
       the polyline coming out of the ODE is smooth to within 0.4 deg). The core
       is also exactly where the continuum description does not apply, so the
       glide plane is simply not drawn there.

    3. Everything outside `bbox` = ((xmin, xmax), (ymin, ymax)), the field of
       view of the figure, less a `bbox_inset` [A] margin. Matplotlib would clip
       those parts anyway, but then the harpoon head marking the end of the line
       (see draw_glide_plane) would be clipped away together with them.

    The polyline may be cut into several pieces -- it is dropped near EVERY core,
    not only near the core of its own dislocation -- hence a list is returned.
    Pieces shorter than `min_vertices` vertices are discarded.
    """
    p = np.asarray(glide_plane)
    p = p.reshape(-1, p.shape[-1])
    xy = np.asarray(coordinates)[:, :2]
    cores = np.asarray(cores, dtype=float)[:, :2]

    to_atom = np.min(
        np.linalg.norm(p[:, None, :2] - xy[None, :, :], axis=2), axis=1)
    to_core = np.min(
        np.linalg.norm(p[:, None, :2] - cores[None], axis=2), axis=1)
    keep = (to_atom <= atom_cutoff) & (to_core >= core_radius)
    if bbox is not None:
        (bx0, bx1), (by0, by1) = bbox
        keep &= ((p[:, 0] >= bx0 + bbox_inset) & (p[:, 0] <= bx1 - bbox_inset)
                 & (p[:, 1] >= by0 + bbox_inset) & (p[:, 1] <= by1 - bbox_inset))

    segments = []
    start = None
    for i, flag in enumerate(np.append(keep, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_vertices:
                segments.append(p[start:i])
            start = None
    return segments


def _polyline_frame(p):
    """
    Unit tangents and unit normals at the vertices of a 2D polyline, obtained
    from the central difference of the neighbouring vertices.
    """
    xy = np.asarray(p, dtype=float)[:, :2]
    t = np.empty_like(xy)
    t[1:-1] = xy[2:] - xy[:-2]
    t[0] = xy[1] - xy[0]
    t[-1] = xy[-1] - xy[-2]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return xy, t, n


def _slice_polyline(lip, walked, s0, s1):
    """
    The piece of a polyline between the arc lengths `s0` and `s1`, with the two
    ends interpolated so that the piece has exactly the requested length.
    """
    i0, i1 = np.searchsorted(walked, [s0, s1])
    inner = lip[i0:i1]

    def at(s):
        j = int(np.clip(np.searchsorted(walked, s), 1, len(walked) - 1))
        span = max(walked[j] - walked[j - 1], 1e-12)
        f = (s - walked[j - 1]) / span
        return lip[j - 1] + f * (lip[j] - lip[j - 1])

    return np.vstack([at(s0), inner, at(s1)]) if len(inner) else \
        np.vstack([at(s0), at(s1)])


def draw_glide_plane(ax, segment, b, half_plane, color=DEFAULT_COLOR,
                     linewidth=1.75, lip_offset=0.157,
                     head_length=0.665, head_angle=40.0,
                     head_min_length=4.0,
                     head_spacing=2.2, dashed=False,
                     dash_length=0.7, dash_gap=0.4, head_dash_fraction=0.36,
                     alpha=1.0, zorder=None):
    """
    Draws one piece of a glide plane as a pair of opposed harpoons, i.e. the
    curved counterpart of \\rightleftharpoons.

    A glide plane is the trace of the cut along which the dislocation was
    inserted, and the two lips of that cut slip by the Burgers vector with
    respect to each other. Drawing the cut as a single line says nothing about
    that slip, so each lip is drawn separately, offset by `lip_offset` [A] from
    the traced cut and carrying harpoon heads that point the way that lip moves.

    The direction of the slip follows from the elemental field: with the
    Burgers vector along +x1 the cut runs along -x1 (see get_glide_plane), the
    extra half-plane sits above it (tr beta ~ -b x2, i.e. the material above the
    cut is compressed), and u1 = b/(2 pi) atan2(x2, x1) jumps by +b as the cut
    is crossed from below. The lip on the side of the extra half-plane
    therefore slips by +b/2 and the opposite one by -b/2. `half_plane` tells
    which side that is in the current (deformed) configuration.

    A DASHED piece (`dashed=True`) is drawn dash by dash, in the plane of the
    figure rather than through a matplotlib dash pattern, so that each dash can
    carry exactly one head, at the end the lip slips towards. A dash of the one
    lip and the dash facing it on the other therefore read as a single
    \\rightleftharpoons: two strokes of equal length, barbed at opposite ends.
    Matplotlib's own dashes could not do this -- they are measured in points,
    the heads in Angstroem, so the two would drift apart as soon as the scale
    of the figure changed.

    :param segment: the polyline of one piece of the glide plane, (n, >=2)
    :param b: the Burgers vector of the dislocation, global coordinates
    :param half_plane: the trace of the extra half-plane, global coordinates
    :param dashed: draw the lips as dashes, one harpoon head per dash
    :param dash_length: length [A] of one dash, i.e. of one harpoon
    :param dash_gap: gap [A] between consecutive dashes
    :param head_dash_fraction: the largest share of a dash the barb may take
      along the line; it keeps the harpoon in proportion whatever the dash
      length, so that shortening the dashes does not turn them into chevrons
    :param alpha: opacity of the whole glyph
    """
    xy, t, n = _polyline_frame(segment)
    if len(xy) < 2:
        return
    b_hat = np.asarray(b, dtype=float)[:2]
    b_hat = b_hat / max(np.linalg.norm(b_hat), 1e-12)
    if half_plane is None:
        # Undeformed-lattice approximation, cf. display_tee_2d().
        up = np.asarray([-b_hat[1], b_hat[0]])
    else:
        up = np.asarray(half_plane, dtype=float)[:2]
    # Orient the normals towards the extra half-plane.
    if np.dot(n.mean(axis=0), up) < 0:
        n = -n
    length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    a = np.deg2rad(head_angle)
    if dashed:
        # The barb must fit inside its own dash and still leave enough of the
        # stroke to read as the shaft of the harpoon.
        head_length = min(head_length,
                          head_dash_fraction * dash_length / np.cos(a))
    else:
        # On a solid lip the barbs of the two lips point the opposite ways and
        # are laid half a spacing apart, so they grow towards each other: keep
        # each of them short enough that they cannot meet.
        head_length = min(head_length, 0.25 * head_spacing / np.cos(a))
    style = dict(color=color, linewidth=linewidth, solid_capstyle="butt",
                 alpha=alpha, zorder=zorder)

    for side in (+1.0, -1.0):
        # side = +1: the lip on the side of the extra half-plane, which slips
        # by +b/2; side = -1: the opposite lip, which slips by -b/2.
        lip = xy + side * lip_offset * n
        walked = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(lip, axis=0), axis=1))))
        slip = side * b_hat
        at_last = np.dot(lip[-1] - lip[0], slip) > 0

        def head(station):
            """The barb of the head whose tip sits at arc length `station`."""
            i = int(np.argmin(np.abs(walked - station)))
            tip = lip[i]
            # Direction in which the lip runs towards the slip at that vertex.
            forward = t[i] if at_last else -t[i]
            # The barb sits on the outer side of the pair, as in
            # \\rightleftharpoons.
            outward = side * n[i]
            barb = tip + head_length * (-np.cos(a) * forward
                                        + np.sin(a) * outward)
            ax.plot([tip[0], barb[0]], [tip[1], barb[1]], **style)

        if dashed:
            # Whole dashes only: a dash cut short by the end of the piece would
            # carry a head at a place where the line merely stops being drawn.
            period = dash_length + dash_gap
            for k in range(int((walked[-1] + dash_gap) // period)):
                s0 = k * period
                s1 = s0 + dash_length
                piece = _slice_polyline(lip, walked, s0, s1)
                ax.plot(piece[:, 0], piece[:, 1], **style)
                if length >= head_min_length:
                    head(s1 if at_last else s0)
            continue

        ax.plot(lip[:, 0], lip[:, 1], **style)
        if length < head_min_length:
            continue
        # The heads sit on a grid laid along the piece, with the two lips taking
        # alternate stations. The barbs of a pair point in opposite directions,
        # so two heads facing each other at the same place would line up into a
        # single slash across both lines; alternating keeps them half a spacing
        # apart. The grid stops half a spacing short of either end, so that no
        # head sits where the line has been cut off: the ends of a piece are an
        # artefact of the clipping (see clip_glide_plane), not a feature of the
        # glide plane, and a head there reads as an arrow pointing at something.
        grid = np.arange(0.5 * head_spacing, length - 0.25 * head_spacing,
                         0.5 * head_spacing)
        stations = grid[::2] if side > 0 else grid[1::2]
        for station in stations:
            head(station)
