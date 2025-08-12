from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype = float)
        self.velocity = np.array(velocity, dtype = float)

class Universe:
    def __init__(self, G):
        self.G = G
        self.bodies = []

    def add_body(self, body):
        self.bodies.append(body)

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

        
def init_solver(y0):
    solver = ode(n_body_diffy_q)
    solver.set_integrator('lsoda')
    solver.set_initial_value(y0, 0)
    return solver
    

def solve(dt, t_max, y0):
    n_steps = int(np.ceil(t_max / dt))
    n = len(y0) // 6
    n_bodies = len(universe.bodies)
    ys = np.zeros((n_steps, 6 * n_bodies))

    ts = np.zeros(n_steps)

    solver = init_solver(y0)
    print(f"Solver Y shape: {solver.y.shape}")
    ys[0] = y0
    ts[0] = 0
    step = 1
    print(f"YS: {ys[0]}")
    print(ys.shape)


    while solver.successful() and step < n_steps:
        solver.integrate(solver.t + dt)
        ts[step] = solver.t
        ys[step] = solver.y
        step += 1

    rs1 = ys[:, :3]
    return ts, ys


def n_body_diffy_q(t, y):
    n = len(universe.bodies)
    G = universe.G
    positions = y[:3*n].reshape((n, 3))
    print(f'Posiions shape: {positions.shape}')
    velocities = y[3*n:].reshape((n, 3))
  
    accelerations = np.zeros_like(positions)
    print(f'Accelerations shape: {accelerations.shape}')
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = positions[j] - positions[i]
                dist = np.linalg.norm(r_vec)
                accelerations[i] += G * universe.bodies[j].mass * r_vec / dist**3

    dydt = np.concatenate((velocities.flatten(), accelerations.flatten()))
    return dydt

if __name__ == '__main__':
    universe = Universe(6.67430e-11)

    sun = Body(
        mass = 1.989e30,
        position = [0, 0, 0],
        velocity = [0, 0, 0]
    )

    earth = Body(
        mass = 5.972e24,
        position = [1.496e11, 0, 0],
        velocity = [0, 0, 0]
    )

    moon = Body(
        mass = 7.348e22,
        position = [3.844e8 + 1.496e11, 0, 0],
        velocity = [0, 0, 0]
    )
    universe.add_body(sun)
    universe.add_body(earth)
    universe.add_body(moon)

    universe.calculate_velocity2()
    
    for i, b in enumerate(universe.bodies):
        print(f"Body {i}: pos = {b.position}, vel = {b.velocity}")
    y0 = np.array([])
    y0_positions = np.concatenate([b.position for b in universe.bodies])
    y0_velocitys = np.concatenate([b.velocity for b in universe.bodies])
    y0 = np.concatenate([y0_positions, y0_velocitys])
    print('\n\n')
    print(y0)
    print('\n\n')

    print(f"Y0: {y0.shape}")

    dt = 10  
    t_max = 12 * 30 * 24 * 3600  

    ts, ys = solve(dt, t_max, y0)
    print(f'Ys shape: {ys.shape}')
    np.save('positions_N_bodies.npy', ys)


    