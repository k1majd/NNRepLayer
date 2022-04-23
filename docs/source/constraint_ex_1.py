import polytope as pc
import numpy as np

A = np.array([
            [-0.70710678, -0.70710678],
            [ 0.70710678, -0.70710678],
            [-0.70710678,  0.70710678],
            [ 0.70710678,  0.70710678]
            ])

b = np.array([
            [-2.31053391],
            [ 1.225     ],
            [ 1.225     ],
            [ 4.76053391]
            ])

p = pc.Polytope(A,b)


from polytope.polytope import bounding_box
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0,5)
ax.set_ylim(0,5)
p.plot(ax)
plt.tight_layout()
plt.savefig("constraint_1_visual.png", dpi = 500, format = 'png')