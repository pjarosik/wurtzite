import pickle
import visualization
import matplotlib.animation
import matplotlib.pyplot as plt
import sys

dir_nr = sys.argv[1]
l0, d1, d2, us, new_d1s = pickle.load(open(f"resultdir_{dir_nr}/test_vis.pkl", "rb"))

anim = visualization.animate(l0, d1, d2, us[1:], new_d1s[1:],
                             d1ab=(487, 491), d2ab=(499, 503),
                             # d2_xoffset=-0.06,
                             alpha=0.3, display_tees=True
                             )
anim.save(f"resultdir_{dir_nr}/two_dislocations_atoms_background.gif")

anim = visualization.animate(l0, d1, d2, us, new_d1s,
                             d1ab=(487, 491), d2ab=(409, 503),
                             d2_xoffset=-0.06,
                             alpha=1.0, display_tees=False
                             )
anim.save(f"resultdir_{dir_nr}/two_dislocations_atoms_only.gif")
