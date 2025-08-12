from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

G = 6.67430e-11 

m1 = 5.972e24   
m2 = 7.348e22   
r2_mag = 314400000.0 

r1 = [0.0, 0.0, 0.0]
r2 = [r2_mag, 0.0, 0.0]

v1 = [0.0, 0.0, 0.0]
v2_mag = float(np.sqrt(G * m1 / r2_mag))
v2 = [0.0, v2_mag, 0.0]       

def diffy_q(t, y):
    
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])
    norm = np.linalg.norm(r)
    F = G * m1 * m2 / norm ** 3

    ax, ay, az = -r * F / m2

    return [vx, vy, vz, ax, ay, az]


if __name__ == '__main__':
    dt = 10  
    t_max = 30 * 24 * 3600      

    n_steps = int(np.ceil(t_max / dt))

    ys = np.zeros((n_steps, 6))
    ts = np.zeros((n_steps, 1))

    y0 = r2 + v2
    print(y0)
    ys[0] = np.array(y0)
    step = 1

    solver = ode(diffy_q)
    solver.set_integrator('lsoda')
    solver.set_initial_value(y0, 0)
    

    while solver.successful() and step < n_steps:
        solver.integrate(solver.t + dt)
        ts[step] = solver.t
        ys[step] = solver.y
        step += 1

    rs = ys[:, :3]
    np.save('positions.npy', ys)
    np.save('steps.npy', ts)
