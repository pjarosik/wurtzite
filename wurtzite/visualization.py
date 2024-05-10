"""
Crystal visualization toolkit. This module is intended to be used with Jupyter
notebooks.
"""
import math
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines
import panel as pn
import vtk
import numpy as np
from openbabel import openbabel

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

import wurtzite.model
from wurtzite.model import Crystal

pn.extension("vtk")

# Copied from openbabel implementation
# i-th row: r, g, b for atom with i+1 number
_ATOM_COLORS = [
    0.07, 0.50, 0.70,
    0.75, 0.75, 0.75,
    0.85, 1.00, 1.00,
    0.80, 0.50, 1.00,
    0.76, 1.00, 0.00,
    1.00, 0.71, 0.71,
    0.40, 0.40, 0.40,
    0.05, 0.05, 1.00,
    1.00, 0.05, 0.05,
    0.50, 0.70, 1.00,
    0.70, 0.89, 0.96,
    0.67, 0.36, 0.95,
    0.54, 1.00, 0.00,
    0.75, 0.65, 0.65,
    0.50, 0.60, 0.60,
    1.00, 1.00, 0.00,
    1.00, 1.00, 0.00,
    0.12, 0.94, 0.12,
    0.50, 0.82, 0.89,
    0.56, 0.25, 0.83,
    0.24, 1.00, 0.00,
    0.90, 0.90, 0.90,
    0.75, 0.76, 0.78,
    0.65, 0.65, 0.67,
    0.54, 0.60, 0.78,
    0.61, 0.48, 0.78,
    0.88, 0.40, 0.20,
    0.94, 0.56, 0.63,
    0.31, 0.82, 0.31,
    0.78, 0.50, 0.20,
    0.49, 0.50, 0.69,
    0.76, 0.56, 0.56,
    0.40, 0.56, 0.56,
    0.74, 0.50, 0.89,
    1.00, 0.63, 0.00,
    0.65, 0.16, 0.16,
    0.36, 0.72, 0.82,
    0.44, 0.18, 0.69,
    0.00, 1.00, 0.00,
    0.58, 1.00, 1.00,
    0.58, 0.88, 0.88,
    0.45, 0.76, 0.79,
    0.33, 0.71, 0.71,
    0.23, 0.62, 0.62,
    0.14, 0.56, 0.56,
    0.04, 0.49, 0.55,
    0.00, 0.41, 0.52,
    0.88, 0.88, 1.00,
    1.00, 0.85, 0.56,
    0.65, 0.46, 0.45,
    0.40, 0.50, 0.50,
    0.62, 0.39, 0.71,
    0.83, 0.48, 0.00,
    0.58, 0.00, 0.58,
    0.26, 0.62, 0.69,
    0.34, 0.09, 0.56,
    0.00, 0.79, 0.00,
    0.44, 0.83, 1.00,
    1.00, 1.00, 0.78,
    0.85, 1.00, 0.78,
    0.78, 1.00, 0.78,
    0.64, 1.00, 0.78,
    0.56, 1.00, 0.78,
    0.38, 1.00, 0.78,
    0.27, 1.00, 0.78,
    0.19, 1.00, 0.78,
    0.12, 1.00, 0.78,
    0.00, 1.00, 0.61,
    0.00, 0.90, 0.46,
    0.00, 0.83, 0.32,
    0.00, 0.75, 0.22,
    0.00, 0.67, 0.14,
    0.30, 0.76, 1.00,
    0.30, 0.65, 1.00,
    0.13, 0.58, 0.84,
    0.15, 0.49, 0.67,
    0.15, 0.40, 0.59,
    0.09, 0.33, 0.53,
    0.90, 0.85, 0.68,
    0.80, 0.82, 0.12,
    0.71, 0.71, 0.76,
    0.65, 0.33, 0.30,
    0.34, 0.35, 0.38,
    0.62, 0.31, 0.71,
    0.67, 0.36, 0.00,
    0.46, 0.31, 0.27,
    0.26, 0.51, 0.59,
    0.26, 0.00, 0.40,
    0.00, 0.49, 0.00,
    0.44, 0.67, 0.98,
    0.00, 0.73, 1.00,
    0.00, 0.63, 1.00,
    0.00, 0.56, 1.00,
    0.00, 0.50, 1.00,
    0.00, 0.42, 1.00,
    0.33, 0.36, 0.95,
    0.47, 0.36, 0.89,
    0.54, 0.31, 0.89,
    0.63, 0.21, 0.83,
    0.70, 0.12, 0.83,
    0.70, 0.12, 0.73,
    0.70, 0.05, 0.65,
    0.74, 0.05, 0.53,
    0.78, 0.00, 0.40,
    0.80, 0.00, 0.35,
    0.82, 0.00, 0.31,
    0.85, 0.00, 0.27,
    0.88, 0.00, 0.22,
    0.90, 0.00, 0.18,
    0.92, 0.00, 0.15,
    0.93, 0.00, 0.14,
    0.94, 0.00, 0.13,
    0.95, 0.00, 0.12,
    0.96, 0.00, 0.11,
    0.97, 0.00, 0.10,
    0.98, 0.00, 0.09,
    0.99, 0.00, 0.08,
    0.99, 0.00, 0.07,
    0.99, 0.00, 0.06,
]
_ATOM_COLORS = np.asarray(_ATOM_COLORS)
_ATOM_COLORS = _ATOM_COLORS.reshape(-1, 3)


def get_atom_color_rgb(nr) -> Tuple[float, float, float]:
    return _ATOM_COLORS[nr-1]


class VtkVisualizer:

    def __init__(
            self,
            window_size: Tuple[int, int] = (400, 400),
            resolution_per_atom: float = 600000.0,
            background_color: str = "SlateGray",
            n_ticks: int = 10,
            show_axes=True
    ):
        self.geom_pane = None
        self.background_color = background_color
        self.resolution_per_atom = resolution_per_atom
        self.window_size = window_size
        self.n_ticks = n_ticks
        self.show_axes = show_axes

    def render_molecule(self, molecule: wurtzite.model.Molecule):
        colors = vtk.vtkNamedColors()
        n_atoms = molecule.n_atoms
        n_bonds = molecule.n_bonds

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(colors.GetColor3d(self.background_color))
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.SetSize(*self.window_size)
        self.renderWindow.SetOffScreenRendering(1)

        # Convert crystal to vtkPolyData
        moleculeVtk = vtkPolyData()

        # Atoms:
        points = vtkPoints()

        vdw = vtk.vtkFloatArray()
        vdw.SetNumberOfComponents(3)
        vdw.Allocate(3 * n_atoms)
        vdw.SetName("radius")

        atom_color = vtk.vtkUnsignedCharArray()
        atom_color.SetNumberOfComponents(3)
        atom_color.Allocate(3 * n_atoms)
        atom_color.SetName("rgb_colors")

        for atom in molecule.get_atoms():
            atom_pos = atom.coordinates
            atomic_number = atom.atomic_number
            radius = openbabel.GetVdwRad(int(atomic_number))
            # Color
            rgb = (_ATOM_COLORS[atomic_number-1] * 255)
            rgb = np.round(rgb).astype(int)
            r, g, b = rgb
            points.InsertNextPoint(atom_pos[0], atom_pos[1], atom_pos[2])
            vdw.InsertNextTuple3(radius, radius, radius)
            atom_color.InsertNextTuple3(r, g, b)

        moleculeVtk.SetPoints(points)
        moleculeVtk.GetPointData().SetVectors(vdw)
        moleculeVtk.GetPointData().SetScalars(atom_color)
        resolution = math.sqrt(self.resolution_per_atom / n_atoms)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(0, 0, 0)
        sphere.SetRadius(1)
        sphere.SetThetaResolution(int(resolution))
        sphere.SetPhiResolution(int(resolution))

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(moleculeVtk)
        glyph.SetOrient(1)
        glyph.SetColorMode(1)
        glyph.SetScaleMode(2)
        glyph.SetScaleFactor(0.25)
        glyph.SetSourceConnection(sphere.GetOutputPort(0))

        atomMapper = vtk.vtkPolyDataMapper()
        atomMapper.SetInputConnection(glyph.GetOutputPort(0))
        atomMapper.UseLookupTableScalarRangeOff()
        atomMapper.ScalarVisibilityOn()
        atomMapper.SetScalarModeToDefault()

        atom = vtk.vtkActor()
        atom.SetMapper(atomMapper)
        atom.GetProperty().SetRepresentationToSurface()
        atom.GetProperty().SetInterpolationToGouraud()
        atom.GetProperty().SetAmbient(0.1)
        atom.GetProperty().SetDiffuse(0.7)
        atom.GetProperty().SetSpecular(0.5)
        atom.GetProperty().SetSpecularPower(80)
        atom.GetProperty().SetSpecularColor(colors.GetColor3d("White"))
        self.renderer.AddActor(atom)

        # Bonds:
        if n_bonds > 0:
            bonds = vtk.vtkCellArray()

            for bond in molecule.get_bonds():
                bonds.InsertNextCell(2)
                # NOTE: assumming that OBMolAtomIter returns the sequence of
                # atoms according to its Id.
                bonds.InsertCellPoint(int(bond.a_id))
                bonds.InsertCellPoint(int(bond.b_id))

            moleculeVtk.SetLines(bonds)

            tube = vtk.vtkTubeFilter()
            tube.SetInputData(moleculeVtk)
            tube.SetNumberOfSides(int(resolution))
            tube.CappingOff()
            tube.SetRadius(0.05)
            tube.SetVaryRadius(0)
            tube.SetRadiusFactor(1)

            bondMapper = vtk.vtkPolyDataMapper()
            bondMapper.SetInputConnection(tube.GetOutputPort(0) )
            bondMapper.UseLookupTableScalarRangeOff()
            bondMapper.ScalarVisibilityOff()
            bondMapper.SetScalarModeToDefault()

            bond = vtk.vtkActor()
            bond.SetMapper(bondMapper)
            bond.GetProperty().SetRepresentationToSurface()
            bond.GetProperty().SetInterpolationToGouraud()
            bond.GetProperty().SetAmbient(0.1)
            bond.GetProperty().SetDiffuse(0.7)
            bond.GetProperty().SetSpecular(0.5)
            bond.GetProperty().SetSpecularPower(80)
            bond.GetProperty().SetSpecularColor(colors.GetColor3d("White"))
            self.renderer.AddActor(bond)

        xs = molecule.coordinates[:, 0]
        ys = molecule.coordinates[:, 1]
        zs = molecule.coordinates[:, 2]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        z_min, z_max = np.min(zs), np.max(zs)

        def get_ticker(minimum, maximum, label):
            ticks = np.linspace(minimum, maximum, self.n_ticks)
            labels = [f"{value:.1f}" for value in ticks]
            labels[-1] = f"{label} {labels[-1]}"
            # NOTE: it is important here to return list
            # (and not the numpy array object)
            return dict(ticks=ticks.tolist(), labels=labels)

        xticker = get_ticker(x_min, x_max, label="OX")
        yticker = get_ticker(y_min, y_max, label="OY")
        zticker = get_ticker(z_min, z_max, label="OZ")
        origin = (
            np.min(xticker["ticks"]),
            np.min(yticker["ticks"]),
            np.min(zticker["ticks"])
        )
        if self.show_axes:
            axes = dict(
                xticker=xticker,
                yticker=yticker,
                zticker=zticker,
                show_grid=True,
                origin=origin,
                grid_opacity=0.3,
                axes_opacity=0.3
            )
        else:
            axes = None

        self.geom_pane = pn.pane.VTK(
            self.renderWindow,
            width=self.window_size[0],
            height=self.window_size[1],
            orientation_widget=True,
            axes=axes,
            enable_keybindings=True
        )
        return self.geom_pane


def render(molecule: wurtzite.model.Molecule, **kwargs):
    visualizer = VtkVisualizer(**kwargs)
    return visualizer.render_molecule(molecule)


def vectors_to_rgb(vectors):
    """Credits: https://stackoverflow.com/questions/19576495/color-matplotlib-quiver-field-according-to-magnitude-and-direction"""
    angles = np.arctan2(vectors[..., 1], vectors[..., 0])
    lengths = np.sqrt(np.square(vectors[..., 1]) + np.square(vectors[..., 0]))
    max_abs = np.max(lengths)

    # normalize angle
    result = []
    for angle, length in zip(angles, lengths):
        angle = angle%(2*np.pi)
        if angle < 0:
            angle += 2 * np.pi
        color = matplotlib.colors.hsv_to_rgb((angle/2/np.pi, length/max_abs, length/max_abs))
        result.append(color)
    return np.asarray(result)



def plot_atoms_2d(lattice, offset=5, figsize=None, xlim=None, ylim=None, xlabel=None, ylabel=None): 
    """
    Display the lattice on the 2D plane.

    The lattice atoms are drawn on a 2D plane, in the order determined by the z coordinate, from the 
    larges z value to the smalest. In other words, atoms with the smalest z coordinate are will be in the
    forerground, all the other will be drawn in the background.

    Currently, this function does not scale atoms depending on the z coordinate -- the only factor that
    impacts the atom size is its atomic number (which determines Van Der Vals radius). 

    When the figsize is None, the size of the figure is automatically adapted to display the lattice 
    with the given `offset` value. 

    :param lattice: the lattice to display
    :param offset: the offset to apply to the figure size, relative to the ($\AA$ units). 
    :return: matplotlib figure and axis with drawn atoms 
    """
    coords = lattice.coordinates
    radiuses = [openbabel.GetVdwRad(int(nr)) for nr in lattice.atomic_number]
    colors = [get_atom_color_rgb(nr) for nr in lattice.atomic_number]
    circles = [plt.Circle((x, y), r*0.2, color=c, zorder=z) 
               for (x, y, z), r, c  in zip(coords, radiuses, colors)]
    bonds = [(coords[b[0]], coords[b[1]]) for b in lattice.bonds]
    # The offset was selected so that the bond is shown in the background of atoms.
    bond_offset = -0.5
    lines = [matplotlib.lines.Line2D((bs[0], be[0]), (bs[1], be[1]), zorder=(bs[2]+be[2])/2+bond_offset) 
             for bs, be in bonds if not np.allclose(bs, be)] 
    fig, ax = plt.subplots()
    
    for l in lines:
        ax.add_line(l)
    for c in circles:
        ax.add_patch(c)
    if xlim is None or ylim is None:
        xlim = [np.min(coords[:, 0])-offset, np.max(coords[:, 0])+offset]
        ylim = [np.min(coords[:, 1])-offset, np.max(coords[:, 1])+offset]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if figsize is None:
        # keep the correct aspect ratio
        x_min, x_max = xlim
        xw = xlim[1]-xlim[0]
        yw = ylim[1]-ylim[0]
        # 1A == 0.2 inch
        figsize = (xw*0.2, yw*0.2)
    fig.set_size_inches(*figsize)
    ax.set_xlabel("OX ($\AA$)")
    ax.set_ylabel("OY ($\AA$)")
    return fig, ax


def plot_displacement(lattice, u, xlabel="OX ($\AA$)", ylabel="OY ($\AA$)", title="Dislocation field (u), OXY", dislocation_core_position=None):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    y_dim = 1
    x_dim = 0
    plt.quiver(
        lattice.coordinates[..., x_dim], lattice.coordinates[..., y_dim], 
        u[..., x_dim], u[..., y_dim], 
        color=vectors_to_rgb(u[..., (x_dim, y_dim)])
    )
    if dislocation_core_position is not None:
        pos_x, pos_y = dislocation_core_position
        plt.scatter(pos_x, pos_y, s=200.5, c="brown", marker="$\\bot$")
    fig.tight_layout()
    return fig, ax



def plot_crystal_surface_y(ax, dislocation_position=(0, 0, 0), x0=0, y0=0, nx=2000, ny=2000, xlim=(-10, 10), ylim=(-1, 1), bx=1,
                           color=None, linestyle=None, label=None):
    def F(x, y, x0, y0, nu, bx, r0):
        return y-y0+bx/(8*np.pi*(1-nu))*((1-2*nu)*np.log((x**2 + y**2)/r0**2)-2*y**2/(x**2 +y**2)
                                    -(1-2*nu)*np.log((x0**2 + y0**2)/r0**2) +2*y0**2/(x0**2+y0**2))
    
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(-1, 1, ny)
    xy = np.meshgrid(x, y, indexing="ij")

    xy = np.stack(xy).reshape(2, -1)

    v = F(xy[0, :], xy[1, :], x0, y0, nu=0.35, bx=bx, r0=bx)
    v = v.reshape(nx, ny)  # F(x, y)
    mask = np.argmin(np.abs(v), axis=1)
    ys = []
    for i in mask:
        ys.append(y[i])
    ys = np.stack(ys)
    x += dislocation_position[0]
    ys += dislocation_position[1]
    ax.plot(x, ys, color=color, linestyle=linestyle, label=label)

    