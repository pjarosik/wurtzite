import json
import pickle
from pathlib import Path
import glob
import visualization
import sys

import wurtzite as wzt
from wurtzite.model import DislocationDef
from utils_1st import *
from utils_2nd import *
from utilsv2 import *
from visualization import animate_all



if not Path("lattice.pkl").exists():
    l0 = wzt.generate.create_lattice(
        dimensions=(10, 5, 2),
        cell="B4_AlN",
    )
    pickle.dump(l0, open("lattice.pkl", "wb"))
else:
    l0 = pickle.load(open("lattice.pkl", "rb"))


current_points = None

dis_1 = DislocationDef(
    b=[1, 0, 0],
    position=[3.890+1.0*l0.cell.dimensions[0], 4.03+0.35, 7.5],
    plane=(0, 0, 1),
    label="$d_1$",
    color="brown"
)

dis_2 = DislocationDef(
    b=[1, 0, 0],
    position= np.asarray(dis_1.position)+np.array([2*l0.cell.dimensions[0], 0.0, 0.0]),  # [23.88-2*3.811, 5.13, 7.5],
    plane=(0, 0, 1),
    label="$d_2$",
    color="brown"
)

u0, all_us = displace_love2(
    crystal=l0,
    position=dis_1.position,
    burgers_vector=dis_1.b,
    plane=dis_1.plane,
    bv_fraction=1.0,
)

print(get_be_bz(l0.cell, np.asarray(dis_1.b)))

