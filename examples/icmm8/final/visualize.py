import dataclasses
import pickle
import visualization
import matplotlib.animation
import matplotlib.pyplot as plt
import sys
import numpy as np

dir_nr = sys.argv[1]
l0, d1, d2, us, new_d1s = pickle.load(open(f"resultdir_{dir_nr}/test_vis.pkl", "rb"))

# Rotate by 180 degrees to get the consistent
d2 = dataclasses.replace(d2, b=np.asarray([-d2.b[0], -d2.b[1], d2.b[2]]))
print(d2.b)
anim = visualization.animate(l0, d1, d2, us[1:], new_d1s[1:],
                             # d1ab=(487, 491), d2ab=(499, 503),
                             d1ab=None, d2ab=None,
                             # d2_xoffset=-0.06,
                             alpha=1.0, display_tees=True)
anim.save(f"resultdir_{dir_nr}/two_dislocations.gif")

# anim = visualization.animate(l0, d1, d2, us, new_d1s,
#                              # d1ab=(487, 491), d2ab=(409, 503),
#                              d1ab=None, d2ab=None,
#                              # d2_xoffset=-0.06,
#                              alpha=1.0, display_tees=False
#                              )
# anim.save(f"resultdir_{dir_nr}/two_dislocations_atoms_only.gif")
