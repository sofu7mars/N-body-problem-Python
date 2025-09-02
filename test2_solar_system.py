from n_bodies_lib import Universe, Body
import random
import cProfile

if __name__ == '__main__':

    universe = Universe(6.67430e-11, 12 * 30 * 24 * 3600, 10)

    universe.sun_earth_moon()

    universe.add_body(Body(
        name = 'Mars',
        mass = 6.93e23,
        position = [2.28e11, 0, 0],
        velocity = [0, 24100, 0]
    ))

    for b in universe.bodies:
        print(b.name)

    universe.solve()
    universe.plot2d(from_file = False, axises = 'xy', body_index = 1)
    universe.animate3d(from_file = True, trace_lines = True, fading_trace_lines = True, trace_length_dev = 25, padding_factor = 1,
                    track_body_index = 2, animation_step = 1000)
