from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation


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

def animate3d(earth, moon):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    padding_factor = 0.5



    all_positions = np.vstack((earth, moon))

    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])

    ax.set_xlim(x_min - abs(x_max - x_min) * padding_factor, x_max + abs(x_max - x_min) * padding_factor)
    ax.set_ylim(y_min - abs(y_max - y_min) * padding_factor, y_max + abs(y_max - y_min) * padding_factor)
    ax.set_zlim(z_min - abs(z_max - z_min) * padding_factor, z_max + abs(z_max - z_min) * padding_factor)

    animated_orbit, = ax.plot([], [], [], color = 'blue')
    animated_body, = ax.plot([], [], [], 'o', markersize = 15, color = 'red')
    animated_orbit1, = ax.plot([], [], [], color = 'blue')
    animated_body1, = ax.plot([], [], [], 'o', markersize = 8, color = 'yellow')
    animated_orbit2, = ax.plot([], [], [], color = 'blue')
    animated_body2, = ax.plot([], [], [], 'o', markersize = 4, color = 'green')

    # ax.scatter(0, 0, 0, color = 'orange', s = 100, label = 'Earth')
    def update_data(frame):
        nonlocal animation
        if frame > 2000000:
            animation.event_source.stop()
        # print(f"Frame {frame}")
        # print("Body 1 pos:", rs[frame])
        # print("Body 2 pos:", rs1[frame])
        animated_orbit.set_data(sun[:frame, 0], sun[:frame, 1])
        animated_orbit.set_3d_properties(sun[:frame, 2])
        animated_body.set_data(sun[frame, 0], sun[frame, 1])
        animated_body.set_3d_properties(sun[frame, 2])

        animated_orbit1.set_data(earth[:frame, 0], earth[:frame, 1])
        animated_orbit1.set_3d_properties(earth[:frame, 2])
        animated_body1.set_data(earth[frame, 0], earth[frame, 1])
        animated_body1.set_3d_properties(earth[frame, 2])

        animated_orbit2.set_data(moon[:frame, 0], moon[:frame, 1])
        animated_orbit2.set_3d_properties(moon[:frame, 2])
        animated_body2.set_data(moon[frame, 0], moon[frame, 1])
        animated_body2.set_3d_properties(moon[frame, 2])


        
        return animated_body, animated_orbit, animated_body1, animated_orbit1, animated_body2, animated_orbit2

    animation = FuncAnimation(
                                fig = fig, 
                                func = update_data,
                                frames = range(0, len(sun), 5000),
                                interval = 20,
                                blit = False,    
    )

    plt.show()

if __name__ == '__main__':
    ys = np.load('positions_N_bodies.npy')
    
    sun = ys[:, :3]
    earth = ys[:, 3:6]
    moon = ys[:, 6:9]
    plot2d(earth)
    animate3d(earth, moon)
    