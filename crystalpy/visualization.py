"""
Crystal visualization toolkit. This module is intended to be used with Jupyter
notebooks.
"""
from crystalpy.model import Crystal
import panel as pn
import vtk
print("Using Panel VTK")
pn.extension("vtk")


def visualize(crystal: Crystal):
    pass