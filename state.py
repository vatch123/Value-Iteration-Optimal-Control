import numpy as np

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()


def error_next_state(cur_state, control, ref_car_state, next_ref_car_state, time_step, noise = True):
    theta = cur_state[2]
    alpha = ref_car_state[2]
    residual = ref_car_state - next_ref_car_state

    rot_3d_z = np.array([[np.cos(theta + alpha), 0], [np.sin(theta + alpha), 0], [0, 1]])
    f = rot_3d_z @ control

    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))

    if noise:
        return cur_state + time_step*f.flatten() + residual + w
    else:
        return cur_state + time_step*f.flatten() + residual
