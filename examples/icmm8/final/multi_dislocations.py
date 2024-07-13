import json
import pickle
from pathlib import Path
import glob
import visualization

import wurtzite as wzt
from wurtzite.model import DislocationDef
from utils_1st import *
from utils_2nd import *


if not Path("lattice.pkl").exists():
    l0 = wzt.generate.create_lattice(
        dimensions=(10, 10, 2),
        cell="B4_ZnS",
    )
    pickle.dump(l0, open("lattice.pkl", "wb"))
else:
    l0 = pickle.load(open("lattice.pkl", "rb"))


directories = glob.glob("resultdir_*")
dir_nr = 0
if len(directories) > 0:
    nrs = [int(name.split("_")[1]) for name in directories]
    nrs = sorted(nrs)
    print(f"found dirs with numbers: {nrs}")
    dir_nr = np.max(nrs) + 1
output_dir = Path(f"resultdir_{dir_nr}")
output_dir.mkdir(parents=True, exist_ok=True)

# DRUGA DYSLOKACJA
dis_1 = DislocationDef(
    b=[1, 0, 0],
    position=[5-2.135, 5.43, 7.5],
    plane=(0, 0, 1),
    label="$d_1$",
    color="brown"
)
dis_2 = DislocationDef(
    b=[-1, 0, 0],
    position=[23.88-3*3.811, 5.43, 7.5],
    plane=(0, 0, 1),
    label="$d_2$",
    color="brown"
)
json.dump(dis_1.__dict__, open(str(output_dir / "d1.json"), "w"))
json.dump(dis_2.__dict__, open(str(output_dir / "d2.json"), "w"))


# PIERWSZA DYSLOKACJA
print("FIRST DISLOCATION")
u0, all_us = displace_love2(
    crystal=l0,
    position=dis_1.position,
    burgers_vector=dis_1.b,
    plane=dis_1.plane,
    bv_fraction=1.0,
)

l1 = l0.translate(u0)
l1 = wzt.generate.update_bonds(l1)
fig, ax = wzt.visualization.plot_atoms_2d(l1, alpha=0.5)
wzt.visualization.display_tee_2d(ax, dis_1, scale=0.5, line_width=10)

visualization.plot_distances(ax, dis_1, l1.coordinates[487], l1.coordinates[491])
plt.savefig(str(output_dir / "d1.png"))

print(dis_2.b)

print("SECOND DISLOCATION")
print("DETERMINING NEW PLANE")
# WYZNACZ PLASZCZYZNE ORAZ OBROT d2 wzgledem d1
# w ukladzie zaczepionym w dis_2
plane_d_x_y, y0 = get_crystal_plane(l0, dis_1, dis_2)
print("DETERMINED")
# Obroc dis_2 zgodnie z betami wyznaczonymi przez dis_1
dis_2_rot_matrix = get_rotation_matrix(l0, dis_1, dis_2)
print(dis_2_rot_matrix)
# Obroc wektor burgersa dis_2 o macierz obrotu wynikajaca
# z dyslokacji dis_1
dis_2_b = dis_2_rot_matrix[:2, :2].dot(dis_2.b[:2]).squeeze()
orig_norm = np.linalg.norm(dis_2.b)
# Zachowaj oryginalna dlugosc wektora
dis_2_b = dis_2_b/np.linalg.norm(dis_2_b)*orig_norm
dis_2_b = np.asarray(dis_2_b.tolist() + [0])
dis_2 = dataclasses.replace(dis_2, b=dis_2_b)

print(f"after: {dis_2.b}")


# wyznacz nowa pozycje dis1
new_d1_pos, new_d1s = displace_love2_2nd_dis(
    plane_d_x_y,
    crystal=l1,
    dis_1=dis_1, dis_2=dis_2,
    dis1_coordinates=dis_1.position,
    dis_2_rot_matrix=dis_2_rot_matrix
)

new_d1, new_d1_rot = update_dislocation(
    l0, d=dis_1, ref_d=dis_2,
    new_pos=dis_1.position + new_d1_pos.squeeze()
)
u1, u1s = displace_love2_2nd_dis(
    plane_d_x_y,
    crystal=l1,
    dis_1=new_d1, dis_2=dis_2,
    dis1_rot_matrix=new_d1_rot,
    dis_2_rot_matrix=dis_2_rot_matrix
)

l2 = l1.translate(u1)
wzt.visualization.plot_displacement(l2, u1)

fig, ax = plt.subplots()
l2 = wzt.generate.update_bonds(l2)
wzt.visualization.plot_atoms_2d(l2, alpha=0.4, offset=3, fig=fig, ax=ax) #, plot_atom_nr=True)

# d2 laziness ...
# dis_2 = dataclasses.replace(dis_2, position=dis_2.position + [])
wzt.visualization.display_tee_2d(ax, dis_2, scale=0.5, line_width=10)
wzt.visualization.display_tee_2d(ax, new_d1, scale=0.5, line_width=10)
# visualization.plot_distances(ax, new_d1, l2.coordinates[487], l2.coordinates[491])
# visualization.plot_distances(ax, dis_2, l2.coordinates[499], l2.coordinates[503])
ax.set_aspect("auto")
fig.savefig(str(output_dir / "d1d2.png"))

pickle.dump((l1, dis_1, dis_2, u1s, new_d1s), open(str(output_dir / "test_vis.pkl"), "wb"))


