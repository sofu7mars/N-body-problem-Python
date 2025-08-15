from n_bodies_lib import Universe, Body
import random

if __name__ == '__main__':

    universe = Universe(1, 30, 0.01)

    sun = Body(
        mass = 1,
        position = [2 * (random.random() - 1), 2 * random.random() - 1, 2 * random.random() -1],
        velocity = [0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1)]
    )
    print(sun.position)

    earth = Body(
        mass = 0.1,
        position = [2 * (random.random() - 1), 2 * random.random() - 1, 2 * random.random() -1],
        velocity = [0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1)]
    )
    print(earth.position)

    moon = Body(
        mass = 2,
        position = [2 * (random.random() - 1), 2 * random.random() - 1, 2 * random.random() -1],
        velocity = [0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1), 0.1 * (2 * random.random() -1)]
    )

    universe.add_body(sun)
    universe.add_body(earth)
    universe.add_body(moon)

    universe.solve()

    universe.animate3d()