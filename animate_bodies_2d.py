from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation


t = np.linspace(0, 10, 100)
y = np.sin(t)

ys = np.load('positions_N_bodies.npy')
st = np.load('steps.npy')
print(st)
rs = ys[:, 6:9]

fig, axis = plt.subplots()
axis.set_xlim([int(min(rs[:, 0]) - max(rs[:, 0] // 2)), int(max(rs[:, 0]) + max(rs[:, 0]) // 2)])
axis.set_ylim([int(min(rs[:, 1]) - max(rs[:, 1] // 2)), int(max(rs[:, 1]) + max(rs[:, 1]) // 2)])
# axis.set_zlim([min(rs[:, 2]), max(rs[:, 2])])
animated_orbit, = axis.plot([], [], color = 'blue')
animated_body, = axis.plot([], [], 'o', markersize = 4, color = 'red')

def update_data(frame):
    animated_orbit.set_data(rs[:frame, 0], rs[:frame, 1])
    animated_body.set_data(rs[frame, 0], rs[frame, 1])
    
    return animated_body, animated_orbit

animation = FuncAnimation(
                            fig = fig, 
                            func = update_data,
                            frames = range(0, len(rs), 10000),
                            interval = .01,    
)

plt.show()