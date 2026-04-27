import numpy as np

MAX_SPEED = 5.0  # meters per second


def compute_control(target):
    velocity = target - current_position

    # Safety speed limiting
    speed = np.linalg.norm(velocity)

    if speed > MAX_SPEED:
        velocity = velocity / speed * MAX_SPEED

    return velocity
