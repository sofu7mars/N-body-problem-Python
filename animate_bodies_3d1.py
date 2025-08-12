from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation


t = np.linspace(0, 10, 100)
y = np.sin(t)
m1 = 78.972e22   
m2 = 7.348e22 

ys = np.load('positions2.npy')
st = np.load('steps2.npy')
print(st)
rs = ys[:, :3]
rs1 = ys[:, 6:9]

COM = (m1 * ys[:,0:3] + m2 * ys[:,6:9]) / (m1 + m2)
print(COM)


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
padding_factor = 0.1

all_positions = np.vstack((rs, rs1))

x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])

ax.set_xlim(x_min - abs(x_max - x_min) * padding_factor, x_max + abs(x_max - x_min) * padding_factor)
ax.set_ylim(y_min - abs(y_max - y_min) * padding_factor, y_max + abs(y_max - y_min) * padding_factor)
ax.set_zlim(z_min - abs(z_max - z_min) * padding_factor, z_max + abs(z_max - z_min) * padding_factor)

animated_orbit, = ax.plot([], [], [], color = 'blue')
animated_body, = ax.plot([], [], [], 'o', markersize = 4, color = 'red')
animated_orbit1, = ax.plot([], [], [], color = 'blue')
animated_body1, = ax.plot([], [], [], 'o', markersize = 4, color = 'yellow')
animated_COM, = ax.plot([], [], [],  color = 'green')

def update_data(frame):
    print(f"Frame {frame}")
    print("Body 1 pos:", rs[frame])
    print("Body 2 pos:", rs1[frame])
    animated_orbit.set_data(rs[:frame, 0], rs[:frame, 1])
    animated_orbit.set_3d_properties(rs[:frame, 2])
    animated_body.set_data(rs[frame, 0], rs[frame, 1])
    animated_body.set_3d_properties(rs[frame, 2])

    animated_orbit1.set_data(rs1[:frame, 0], rs1[:frame, 1])
    animated_orbit1.set_3d_properties(rs1[:frame, 2])
    animated_body1.set_data(rs1[frame, 0], rs1[frame, 1])
    animated_body1.set_3d_properties(rs1[frame, 2])

    animated_COM.set_data(COM[:frame, 0], COM[:frame, 1])
    animated_COM.set_3d_properties(COM[:frame, 2])

    
    return animated_body, animated_orbit, animated_body1, animated_orbit1, animated_COM

animation = FuncAnimation(
                            fig = fig, 
                            func = update_data,
                            frames = range(0, len(rs), 2000),
                            interval = 20,
                            blit = False,    
)

plt.show()