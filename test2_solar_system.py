from n_bodies_lib import Universe, Body
import random
import cProfile

if __name__ == '__main__':

    universe = Universe(1, 12 * 30 * 24 * 3600, 10)

    universe.create_N_bodies_with_random_pos_vel(5, masses = [1, 1.2, 1.5, 2.4, 2.1])

    universe.solve()
    universe.plot2d(axises = 'xy', body_index = 2)
    universe.animate3d(trace_lines = True, fading_trace_lines = True, trace_length_dev = 50, padding_factor = 0.5,
                    track_body_index = None, animation_step = 1)
