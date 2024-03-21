import matplotlib.pyplot as plt
from tqdm import tqdm
import casadi
import numpy as np
from state import *
from main import lissajous
from time import time
from utils import visualize


Q = np.diag([1, 1, 0])  # State cost matrix
Qf = np.diag([0, 0, 0]) # Terminal state cost matrix
R = np.diag([1, 1])  # Control cost matrix
q = 1

time_step = 1 # time between steps in seconds
sim_time = 100   # simulation time
N = 10

v_max = 1
v_min = 0
w_max = 1
w_min = -1

epsilon = 0.6

def create_obstacle(x_obs, y_obs, r):
    obstacle = casadi.MX(np.array([x_obs,y_obs,0]))
    Lambda = casadi.MX(np.diag([1,1,0]))

    def boundary_function(cur_car_state):
        return (cur_car_state - obstacle).T @ Lambda @ (cur_car_state - obstacle) - r**2
    
    return boundary_function

def wrap_to_pi(angle):
    wrapped_angle = np.arctan2(np.sin(angle), np.cos(angle))
    return wrapped_angle

def next_state(e, u, r, rn, time_step):

    theta = e[2]
    alpha_t = r[2]
    G = casadi.MX(np.zeros((3,2)))
    G[0,0] = casadi.cos(theta + alpha_t)
    G[1,0] = casadi.sin(theta + alpha_t)
    G[2,1] = 1

    e_next = e + time_step * G @ u + r - rn
    e_next[2] = wrap_to_pi(e_next[2])
    return e_next


circle1 = create_obstacle(-2,-2,0.5)  
circle2 = create_obstacle(1,2,0.5) 


# Create an optimization problem for the MPC
opti = casadi.Opti()


# Set the initial state constraint
e0 = opti.parameter(3,1)
t_tau = opti.parameter(1)
reference = opti.parameter(3, 100)

# X = opti.variable(3,N+1)
U = opti.variable(2, N)    # Control trajectory

# Set the system dynamics constraints
cost = 0
et = e0

for k in range(N-1):
    e_next = next_state(et, U[:, k], reference[:, t_tau+k], reference[:,t_tau+k+1], time_step)

    cost += e_next.T @ Q @ e_next + q * (1 - casadi.cos(e_next[2]))**2 + U[:, k].T @ R @ U[:, k]
    # cost += -10000* (casadi.fmin(0, circle1(reference[:, k+1] + e_next)) + casadi.fmin(0, circle2(reference[:, k+1] + e_next)))
    opti.subject_to(circle1(reference[:, k+1] + e_next) >= epsilon)
    opti.subject_to(circle2(reference[:, k+1] + e_next) >= epsilon)

    et = e_next

cost += e_next.T @ Qf @ e_next

# Set the control bounds
opti.subject_to(opti.bounded(v_min - 0.4, U[0, :], v_max))      # Linear velocity control bounds
opti.subject_to(opti.bounded(w_min - 0.4, U[1, :], w_max))      # Angular velocity control bounds

# Set the objective function
opti.minimize(cost)

# Set up the solver
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-3}
opti.solver('ipopt', opts)
# Set the tolerance for constraints
opti.solver_options = {'constr_viol_tol': 1e-6}


states = []
pb = tqdm(np.arange(0, sim_time, time_step))

x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
reference_traj = np.vstack([lissajous(k) for k,_ in enumerate(np.arange(0, sim_time, time_step))]).T

c_curr = np.array([x_init, y_init, theta_init])
e_curr = c_curr - reference_traj[:,0]

states.append(c_curr)
times = []
controls = []

error = 0
for idx,t in enumerate(pb):
    if idx + N > reference_traj.shape[1]:
        break
    t1 = time()
    opti.set_value(e0, e_curr)
    opti.set_value(reference, reference_traj)
    opti.set_value(t_tau, idx)
    opti.set_initial(U, np.zeros((2,N)))
    opti.solve()

    # Get the optimal control trajectory
    u_opt = opti.value(U)[:,0]
    controls.append(u_opt)

    # Apply the first control input to the system
    e_next = error_next_state(e_curr, u_opt, reference_traj[:,idx], reference_traj[:,idx+1], time_step)
    c_curr = e_next + reference_traj[:,idx+1]
    e_curr = e_next
    t2 = time()
    times.append(t2-t1)
    # Print the current state
    pb.set_description(f"Time: {t:.1f}, Loss: {opti.value(cost):.2f}")
    states.append(c_curr)
    error = error + np.linalg.norm(c_curr - reference_traj[:,idx])

states = np.array(states)
controls = np.array(controls)
error = error / idx

states = np.array(states)
obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
visualize(states, reference_traj.T, obstacles, times, time_step, save=True)
print(f"Tracking Error: {error}")
