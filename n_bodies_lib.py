from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import random

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype = float)
        self.velocity = np.array(velocity, dtype = float)

class Universe:
    def __init__(self, G, t_max, dt):
        self.t = t_max
        self.dt = dt
        self.G = G
        self.bodies = []
        self.ys = []
        self.ts = []
        

    def add_body(self, body):
        self.bodies.append(body)

    def create_N_bodies_with_random_pos_vel(self, n_bodies, mass = 1, masses = []):
        if n_bodies != len(masses):
            raise Exception("WARNING: Masses more than bodies")
        for _, mass in zip(range(n_bodies), masses):
            body = Body(
            mass = mass,
            position = [2 * (random.random() - 1), 2 * random.random() - 1, 2 * random.random() -1],
            velocity = [0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1)]) 
            self.add_body(body)

    def total_mass(self):
        return sum(b.mass for b in self.bodies)
    
    def center_of_mass(self):
        M = self.total_mass()
        if M == 0:
            return np.zeros(3)
        
        r_com = sum(b.mass * b.position for b in self.bodies) / M
        v_com = sum(b.mass * b.velocity for b in self.bodies) / M
        return r_com, v_com

    def calculate_velocity(self):
        for b in self.bodies:
            other_masses = self.total_mass() - b.mass
            r_vecs = [other.position - b.position for other in self.bodies if other is not b]
            print(r_vecs)
            distance = [np.linalg.norm(rv) for rv in r_vecs]
            if len(distance) == 0:
                continue
            print(f"Distances: {distance}")
            min_index = np.argmin(distance)
            print(f"Min index: {min_index}")
            
            r_vec = r_vecs[min_index]
            r_norm = distance[min_index]
            if r_norm < 1e-15:
                continue

            r_unit = r_vec / r_norm
            print(f"r_unit: {r_unit}")
            v_mag = np.sqrt(self.G * other_masses / r_norm)
            v_direction = np.array([-r_unit[1], r_unit[0], 0])
            b.velocity = v_direction * v_mag

    def calculate_velocity2(self):
        G = self.G
        r = self.bodies[1].position - self.bodies[0].position
        r_earth = np.linalg.norm(r)
        v_earth = np.sqrt(G * self.bodies[0].mass / r_earth)
        self.bodies[1].velocity = np.array([0, v_earth, 0])

        r_moon = np.linalg.norm(self.bodies[2].position - self.bodies[1].position)
        v_moon = np.sqrt(G * self.bodies[1].mass / r_moon)
        self.bodies[2].velocity = self.bodies[1].velocity + np.array([0, v_moon, 0])

    def n_body_diffy1(self, t, y):
        
        n = len(self.bodies)
        positions = y[:3*n].reshape((n, 3))
        
        # print(f'Posiions shape: {positions.shape}')
        velocities = y[3*n:].reshape((n, 3))
        accelerations = np.zeros_like(positions)
        for i in range(len(self.bodies)):
            for j in range(len(self.bodies)):
                if i != j:
                    rji = positions[i] - positions[j]
                    distance = np.linalg.norm(rji)
                    accelerations[i] += -self.G * self.bodies[j].mass *rji / distance**3 

        dydt = np.concatenate((velocities.flatten(), accelerations.flatten()))
        return dydt
    
    def n_body_diffy_vectorized(self, t, y):
 
        n = len(self.bodies)
        positions = y[:3*n].reshape((n, 3))
        velocities = y[3*n:].reshape((n, 3))

        accelerations = np.zeros_like(positions)
        
        r = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distance = np.linalg.norm(r, axis = 2) + 1e-10

        np.fill_diagonal(distance, np.inf)

        inv_dist3 = 1.0 / distance ** 3

        masses = np.array([body.mass for body in self.bodies])  

        accelerations = -self.G * np.einsum('j,ijk->ik', masses, r * inv_dist3[:, :, np.newaxis])

        dydt = np.concatenate((velocities.flatten(), accelerations.flatten()))
        return dydt
     
    def init_solver(self, y0):
        solver = ode(self.n_body_diffy_vectorized)
        solver.set_integrator('lsoda')
        solver.set_initial_value(y0, 0)
        return solver
    
    def solve(self):
        y0 = np.array([])
        y0_positions = np.concatenate([b.position for b in self.bodies])
        y0_velocitys = np.concatenate([b.velocity for b in self.bodies])
        y0 = np.concatenate([y0_positions, y0_velocitys])
        n_steps = int(np.ceil(self.t / self.dt))
        n = len(y0) // 6
        n_bodies = len(self.bodies)
        self.ys = np.zeros((n_steps, 6 * n_bodies))

        self.ts = np.zeros(n_steps)

        solver = self.init_solver(y0)
        print(f"Solver Y shape: {solver.y.shape}")
        self.ys[0] = y0
        self.ts[0] = 0
        step = 1
        # print(f"YS: {ys[0]}")
        # print(ys.shape)

        with tqdm(total = n_steps - 1, desc = "Calculation Progress", unit = "step", dynamic_ncols = True, ascii = True, leave = True) as pbar:
            while solver.successful() and step < n_steps:
                solver.integrate(solver.t + self.dt)
                self.ts[step] = solver.t
                self.ys[step] = solver.y
                step += 1
                pbar.update(1)

        rs1 = self.ys[:, :3]
        np.save('positions_N_bodies.npy', self.ys)
        return self.ts, self.ys
    
    def animate3d(self, trace_lines = True, fading_trace_lines = True, padding_factor = 0, track_body_index = None):
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        all_positions = self.ys.reshape(-1, 3)
        x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
        y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
        z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])

        ax.set_xlim(x_min - abs(x_max - x_min) * padding_factor,
                    x_max + abs(x_max - x_min) * padding_factor)

        ax.set_ylim(y_min - abs(y_max - y_min) * padding_factor, 
                    y_max + abs(y_max - y_min) * padding_factor)

        ax.set_zlim(z_min - abs(z_max - z_min) * padding_factor, 
                    z_max + abs(z_max - z_min) * padding_factor)

        animated_bodies = []
        animated_orbits = []
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        used_colors = set()
        available_colors = list(set(colors) - used_colors)
        n_bodies = len(self.bodies)
        for i, _ in enumerate(self.bodies):
            available_colors = list(set(colors) - used_colors)
            if available_colors:
                c = random.choice(available_colors)
                used_colors.add(c)
            else:
                c = random.choice(colors)
            if track_body_index == i:
                body_line, = ax.plot([], [], [], 'o', markersize = 5, color = c)
            else:
                body_line, = ax.plot([], [], [], 'o', markersize = 2, color = c)
            animated_bodies.append(body_line)
            if trace_lines:
                orbit_line, = ax.plot([], [], [], color = c)
                animated_orbits.append(orbit_line)
        
        def update_frame(frame):
            for i, animated_body in enumerate(animated_bodies):
                animated_body.set_data(self.ys[frame, i * 3], self.ys[frame, i * 3 + 1])
                animated_body.set_3d_properties(self.ys[frame, i * 3 + 2])
            if trace_lines:
                if fading_trace_lines:
                    start = max(0, frame - len(self.ys) // 15)
                for i, animated_orbit in enumerate(animated_orbits):
                    animated_orbit.set_data(self.ys[start : frame, i * 3], 
                                            self.ys[start : frame, i * 3 + 1])
                    animated_orbit.set_3d_properties(self.ys[start : frame, i * 3 + 2])
            
            if track_body_index != None:
                x, y, z = self.ys[frame, track_body_index * 3 : track_body_index * 3 + 3]
                padding = 1.0
                ax.set_xlim(x - padding, x + padding)
                ax.set_ylim(y - padding, y + padding)
                ax.set_zlim(z - padding, z + padding)

            return animated_bodies, animated_orbits

        animation = FuncAnimation(
                fig = fig,
                func = update_frame,
                frames = range(0, len(self.ys)),
                interval = 40, 
                blit = False,
        )
        plt.show()







    


    