"""
Crystal visualization toolkit. This module is intended to be used with Jupyter
notebooks.
"""
import math
from typing import Tuple

import panel as pn
import vtk
import numpy as np
from openbabel import openbabel

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

import crystalpy.model
from crystalpy.model import Crystal

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
    1.00, 0.50, 0.00,
    0.70, 0.70, 0.00,
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


class VtkVisualizer:

    def __init__(
            self,
            window_size: Tuple[int, int] = (400, 400),
            resolution_per_atom: float = 600000.0,
            background_color: str = "SlateGray",
            n_ticks: int =10,
            show_axes=True
    ):
        self.geom_pane = None
        self.background_color = background_color
        self.resolution_per_atom = resolution_per_atom
        self.window_size = window_size
        self.n_ticks = n_ticks
        self.show_axes = show_axes

    def render_crystal(self, crystal: crystalpy.model.Crystal):
        colors = vtk.vtkNamedColors()
        n_atoms = crystal.n_atoms
        n_bonds = crystal.n_bonds

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(colors.GetColor3d(self.background_color))
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.SetSize(*self.window_size)

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

        for atom in crystal.get_atoms():
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

            for bond in crystal.get_bonds():
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

        xs = crystal.coordinates[:, 0]
        ys = crystal.coordinates[:, 1]
        zs = crystal.coordinates[:, 2]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        z_min, z_max = np.min(zs), np.min(zs)

        def get_ticker(minimum, maximum, label):
            ticks = np.linspace(x_min, x_max, self.n_ticks)
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


def render(crystal: Crystal, **kwargs):
    visualizer = VtkVisualizer(**kwargs)
    return visualizer.render_crystal(crystal)
    pass
