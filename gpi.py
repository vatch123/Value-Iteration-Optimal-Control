import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from state import *
from main import lissajous
from time import time
from utils import visualize
from itertools import product
from scipy.stats import multivariate_normal

time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time
period = 100

v_max = 1
v_min = 0
w_max = 1
w_min = -1

gamma = 0.9

Q = np.diag([10, 10, 0])  # State cost matrix
R = np.diag([1, 1])  # Control cost matrix
q = 1


coord_step = 10
t_grid = np.arange(0,100)
x_grid = np.linspace(-3, 3, coord_step)
y_grid = np.linspace(-3, 3, coord_step)
theta_grid = np.linspace(-np.pi, np.pi, coord_step)

cont_step = 10
v_grid = np.linspace(0, 1, cont_step)
w_grid = np.linspace(-1, 1, cont_step)


states = np.array(list(product(t_grid, x_grid, y_grid, theta_grid)))
controls = np.array(list(product(v_grid, w_grid)))

num_states = states.shape[0]
num_controls = controls.shape[0]


def wrap_to_pi(angle):
    wrapped_angle = np.arctan2(np.sin(angle), np.cos(angle))
    return wrapped_angle

def create_obstacle(x_obs, y_obs, r):
    obstacle = np.array([x_obs,y_obs,0])
    Lambda = np.diag([1,1,0])

    def boundary_function(cur_car_state):
        return (cur_car_state - obstacle).T @ Lambda @ (cur_car_state - obstacle) - r**2
    
    return boundary_function


circle1 = create_obstacle(-2,-2,0.5)  
circle2 = create_obstacle(1,2,0.5)


def stage_cost(lissajous):
    print("Evaluating the cost function...")
    try:
        L = np.load('stage_cost_matrix.npy')
    except:
        L = np.zeros((num_states, num_controls))
        for i in tqdm(range(num_states)):
            e = states[i, 1:]
            t = states[i, 0]
            ref_curr = lissajous(t)
            for j in range(num_controls):
                u = controls[j,:]
                c_curr = ref_curr + e
                L[i,j] = e.T @ Q @ e + u.T @ R @ u + (1 - np.cos(e[2]))**2
                if (circle1(c_curr) < 0 or circle2(c_curr) < 0):
                    L[i,j] = 1e9
                elif c_curr[0] < x_grid[0] or c_curr[0] > x_grid[-1] or c_curr[1] < y_grid[0] or c_curr[1] > y_grid[-1]:
                    L[i,j] = 1e9
        np.save('stage_cost_matrix.npy', L)
    return L

L = stage_cost(lissajous)


def find_closest_state(state):
    t, x, y, theta = state
    xi = np.argmin(np.abs(x_grid - x))
    yi = np.argmin(np.abs(y_grid - y))
    thetai = np.argmin(np.abs(theta_grid - theta))
    return int(t*coord_step**3 + xi*coord_step**2 + yi*coord_step + thetai)

def get_close_states(state, noise_cov):
    points = np.vstack([state]*6)
    points[0, 1] += noise_cov[0]
    points[1, 1] -= noise_cov[0]
    points[2, 2] += noise_cov[1]
    points[3, 2] -= noise_cov[1]
    points[4, 3] += noise_cov[2]
    points[5, 3] -= noise_cov[2]

    return points


def transitions(lissajous):
    print("Evaluating the transition matrix")
    try:
        transition_idx = np.load('transition_idx.npy')
        transition_probs = np.load('transition_probs.npy')
    except:
        transition_idx = np.zeros((num_states, num_controls, 6))
        transition_probs = np.zeros((num_states, num_controls, 6))
        for idx in tqdm(range(num_states)):
            e = states[idx,:]
            t = e[0]
            ref_curr = lissajous(t)
            ref_next = lissajous(t+1)
            theta = e[2]
            alpha = ref_curr[2]
            residual = ref_curr - ref_next

            rot_3d_z = np.array([[np.cos(theta + alpha), 0], [np.sin(theta + alpha), 0], [0, 1]])
            f = time_step * rot_3d_z @ controls.T
            next_pos = e[1:, None] + f + residual[:,None]
            next_pos[2,:] = wrap_to_pi(next_pos[2,:])

            next_st = np.vstack((np.ones(controls.shape[0])*((t+1) % period), next_pos))
            next_st_idx = np.apply_along_axis(find_closest_state, 0, next_st)

            noise_cov = [0.04, 0.04, 0.004]
            for i in range(next_st_idx.shape[0]):
                points = get_close_states(next_st[:, i], noise_cov)
                probs = multivariate_normal.pdf(points[:,1:], states[next_st_idx[i], 1:], np.diag(noise_cov))
                probs = probs / np.sum(probs)
                p_idx = np.apply_along_axis(find_closest_state, 1, points)
                
                transition_idx[idx, i, :] = p_idx
                transition_probs[idx, i, :] = probs
                
        np.save('transition_idx.npy', transition_idx)
        np.save('transition_probs.npy', transition_probs)
    return transition_idx, transition_probs

transition_idx, transition_probs = transitions(lissajous)

V_fn = np.zeros(num_states)
Q_fn = np.zeros((num_states, num_controls))


print("Running Value Iteration...")
while True:
    V_prev = np.copy(V_fn)
    Q_fn = L + gamma * np.sum(transition_probs * V_fn[transition_idx.astype(np.int32)], axis=2)
    V_fn = np.min(Q_fn, 1)
    print("Value change: ", np.max(np.abs(V_fn-V_prev)))
    if np.max(np.abs(V_fn-V_prev)) < 1:
        break



traj = []

x_init = 1.5
y_init = 0.0
theta_init = np.pi/2

reference_traj = np.vstack([lissajous(k) for k in t_grid]).T

c_curr = np.array([x_init, y_init, theta_init])
e_curr = c_curr - reference_traj[:,0]

traj.append(c_curr)
times = []

pb = tqdm(t_grid)
error = 0
for t in pb:
    if t + 1 >= reference_traj.shape[1]:
        break
    t1 = time()

    # Get the optimal control trajectory
    st = np.zeros(4)
    st[0] = t
    st[1:] = e_curr
    st_idx = find_closest_state(st)
    u = controls[np.argmin(Q_fn[st_idx,:])]

    # Apply the first control input to the system
    e_next = error_next_state(e_curr, u, reference_traj[:,t], reference_traj[:,t+1], time_step)
    e_next[2] = wrap_to_pi(e_next[2])

    c_curr = e_next + reference_traj[:,t+1]
    # c_curr[2] = wrap_to_pi(c_curr[2])
    
    e_curr = e_next
    t2 = time()
    times.append(t2-t1)

    traj.append(c_curr)
    error = error + np.linalg.norm(c_curr - reference_traj[:,t])

traj = np.array(traj)
error = error / t

obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
visualize(traj, reference_traj.T, obstacles, times, time_step, save=True)
print(f"Tracking Error: {error}")