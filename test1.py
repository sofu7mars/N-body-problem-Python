from n_bodies_lib import Universe, Body
import random
import cProfile

if __name__ == '__main__':

    universe = Universe(1, 10, 0.001)

    universe.create_N_bodies_with_random_pos_vel(n_bodies = 4, masses = [1.7, 5, 2.5, 2.1])


    # universe.solve()
    universe.animate3d(
            from_file = True,
            trace_length_dev = 25,
            padding_factor = 1,
            track_body_index = 2,
            animation_step = 1
    )