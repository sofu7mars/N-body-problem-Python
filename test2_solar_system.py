from n_bodies_lib import Universe, Body
import random
import cProfile

if __name__ == '__main__':

    universe = Universe(6.67430e-11, 12 * 30 * 24 * 3600, 10)

    sun = Body(
        mass = 1.989e30,
        position = [0, 0, 0],
        velocity = [0, 0, 0]
    )

    earth = Body(
        mass = 5.972e24,
        position = [1.496e11, 0, 0],
        velocity = [0, 29784.8, 0]
    )

    moon = Body(
        mass = 7.348e22,
        position = [3.844e8 + 1.496e11, 0, 0],
        velocity = [0, 29784.8 + 1022.0, 0]
    )
    universe.add_body(sun)
    universe.add_body(earth)
    universe.add_body(moon)


    universe.solve()
    universe.animate3d(trace_lines = True, fading_trace_lines = True, padding_factor = 10,
                    track_body_index = 1, animation_step = 100)
