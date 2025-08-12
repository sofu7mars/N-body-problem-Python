from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def rotate_x(vec, theta):
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]]
    )
    return R @ vec

def diffy_q(t, y):
    
    r1 = y[0:3]
    v1 = y[3:6]
    r2 = y[6:9]
    v2 = y[9:12]

    r = r2 - r1
    norm = np.linalg.norm(r)

    a1 = r * G * m2 / norm ** 3
    a2 = -r * G * m1 / norm ** 3

    return np.concatenate((v1, a1, v2, a2))


if __name__ == '__main__':
    G = 6.67430e-11 

    m1 = 25.972e22   
    m2 = 7.348e22 
    r = 38440000.0 

    r1 = [-m2 / (m1 + m2) * r, 0.0, 0.0]
    r2 = [m1 / (m1 + m2) * r, 0.0, 0.0]

    v = np.sqrt(G * (m1 + m2) / r)

    v1 = [0.0, -m2 / (m1 + m2) * v, 0.0]
    v2 = [0.0, m1 / (m1 + m2) * v, 0.0]
    theta = np.radians(30)
    
    r1 = rotate_x(r1, theta)
    v1 = rotate_x(v1, theta)
    print(r1)
    print(v1)
    dt = 10  

    t_max = 30 * 24 * 10600      

    n_steps = int(np.ceil(t_max / dt))

    ys = np.zeros((n_steps, 12))
    ts = np.zeros((n_steps, 1))

    y0 = np.concatenate((r1, v1, r2, v2))
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

    np.save('positions2.npy', ys)
    np.save('steps2.npy', ts)
