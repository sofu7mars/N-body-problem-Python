import numpy as np
import matplotlib.pyplot as plt


G = 6.67430e-11 


m1 = 5.972e24   
m2 = 7.348e22   


r1 = np.array([0.0, 0.0])
r2 = np.array([384400000.0, 0.0])  


v1 = np.array([0.0, 0.0])
v2 = np.array([0.0, 1022.0])       


dt = 1000  
t_max = 30 * 24 * 3600  


r1_list = []
r2_list = []

t = 0

while t < t_max:
    r = r2 - r1
    dist = np.linalg.norm(r)
    force_dir = r / dist

  
    F = G * m1 * m2 / dist**2

   
    a1 = force_dir * F / m1
    a2 = -force_dir * F / m2

   
    v1 += a1 * dt
    v2 += a2 * dt

   
    r1 += v1 * dt
    r2 += v2 * dt

    
    r1_list.append(r1.copy())
    r2_list.append(r2.copy())

    t += dt


r1_array = np.array(r1_list)
r2_array = np.array(r2_list)


plt.plot(r1_array[:,0], r1_array[:,1], label='Body 1 (Earth)')
plt.plot(r2_array[:,0], r2_array[:,1], label='Body 2 (Moon)')
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.legend()
plt.axis('equal')
plt.title('2-body Problem Simulation (Earth-Moon)')
plt.show()