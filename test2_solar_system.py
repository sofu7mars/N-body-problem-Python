from n_bodies_lib import Universe, Body
import random
import cProfile

if __name__ == '__main__':

    universe = Universe(6.67430e-11, 12 * 30 * 24 * 3600, 10)

    # sun = Body(
    #     mass = 1.989e30,
    #     position = [0, 0, 0],
    #     velocity = [0, 0, 0]
    # )

    universe.sun_earth_moon()

    universe.solve()
    universe.plot2d('xy', 2)
    universe.animate3d(trace_lines = True, fading_trace_lines = True, padding_factor = 0,
                    track_body_index = None, animation_step = 1000)
