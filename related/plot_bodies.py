from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation

positions_so_far = []

def init():
    orbit_line.set_data([], [])
    orbit_line.set_3d_properties([])
    moon_point.set_data([], [])
    moon_point.set_3d_properties([])
    positions_so_far.clear()
    return orbit_line, moon_point

def update(frame):
    pos = positions[frame]
    positions_so_far.append(pos)
    arr = np.array(positions_so_far)

    orbit_line.set_data(arr[:, 0], arr[:, 1])
    orbit_line.set_3d_properties(arr[:, 2])

    moon_point.set_data(pos[0], pos[1])
    moon_point.set_3d_properties(pos[2])

    return orbit_line, moon_point

def animateBodies(positions):
    
    global orbit_line, moon_point, ax, fig

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    margin = 1.1
    ax.set_xlim(np.min(positions[:, 0]) * margin, np.max(positions[:, 0]) * margin)
    ax.set_ylim(np.min(positions[:, 1]) * margin, np.max(positions[:, 1]) * margin)
    ax.set_zlim(np.min(positions[:, 2]) * margin, np.max(positions[:, 2]) * margin)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("Moon Orbit Simulation")

    ax.scatter(0, 0, 0, color='orange', s=100, label='Earth')

    orbit_line, = ax.plot([], [], [], lw=1, label='Moon Orbit')
    moon_point, = ax.plot([], [], [], 'o', color='blue', label='Moon')

    ani = FuncAnimation(fig, update, frames=range(len(positions)),
                        init_func=init, blit=False, interval=50)

    plt.legend()
    plt.show()

def plot(array):
    ax = plt.axes(projection="3d")
    x_data = array[:, 0]
    y_data = array[:, 1]
    z_data = array[:, 2]
    ax.plot(x_data, y_data, z_data, label='Moon orbit')
    ax.scatter([0], [0], [0], color='orange', label='Earth', s=100)
    plt.show()

def plot2d(array):
    x = array[:, 0]
    y = array[:, 1]

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, label='Moon orbit')
    plt.scatter(0, 0, color='orange', label='Earth', s=100)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('2D orbit plot')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    ys = np.load('positions.npy')
    st = np.load('steps.npy')
    rs = ys[:, :3]

    

    # Animate the orbit simulation
    animateBodies(rs)
