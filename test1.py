from n_bodies_lib import Universe, Body
import random
import cProfile

if __name__ == '__main__':

    universe = Universe(1, 10, 0.001)

    universe.create_N_bodies_with_random_pos_vel(n_bodies = 7, masses = [0.5, 0.7, 1, 0.9, 0.63, 0.6, 0.8])


    universe.solve()
    universe.animate3d(
            from_file = False,
            trace_length_dev = 25,
            padding_factor = 1,
            track_body_index = 2,
            animation_step = 1
    )