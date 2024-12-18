import casadi as cs
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Parameters
dim = 3  # Dimensions of the point mass (3D case)
mass = 0.033
F_max = 4*0.12 # 4* individual max thrust
a_max = F_max/mass
v_max = 5 # m/s
n_steps = 100  # Number of time steps
angle_max = 45 # angle in deg in [0,90) with which the drone can pass through the gate
g = np.array([0,0,-9.81])

# Waypoints
waypoints = np.array(
    [
        [1.0, 1.0, 0.05],
        [ 0.45, -1.0, 0.56],
        [ 1.0, -1.55 , 1.11],
        [ 0.0, 0.5, 0.56],
        [ -0.5, -0.5 , 1.11],
    ]
)
waypoints_rot = np.array(
    [
        [1.0, 1.0, 0.0], # first direction doesn't matter
        [0.0, -1.0, 0.0], # other directions need to have to be normalized
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)
# waypoints_rot = None
T_init = 10 # initial guess for the time
# waypoint_times = np.linspace(0, T_init, len(waypoints))
waypoint_times = np.array([0, 0.25, 0.5, 0.75, 1])*T_init

# Interpolate waypoints to create a warm-start trajectory
spline = CubicSpline(waypoint_times, waypoints, axis=0)

# Warm-start trajectory
def warm_start(t):
    return spline(t)

# CasADi optimization
opti = cs.Opti()

# Decision variables
X = opti.variable(dim, n_steps)  # Positions
V = opti.variable(dim, n_steps)  # Velocities
A = opti.variable(dim, n_steps)  # Forces
T = opti.variable()  # Total time (to be minimized)

# Time step size
dt = T / n_steps

# Objective function: minimize total time
opti.minimize(T)

for t in range(n_steps):
    # Dynamics constraints
    if t < n_steps-1:
        opti.subject_to(X[:, t + 1] == X[:, t] + V[:, t] * dt)
        opti.subject_to(V[:, t + 1] == V[:, t] + (A[:, t] + g) * dt)
    # Velocity constraint
    opti.subject_to(V[:, t].T@V[:, t] <= v_max**2) 
    # Acceleration constraint
    opti.subject_to(A[:, t].T@A[:, t] <= a_max**2)

# Boundary conditions
opti.subject_to(X[:, 0] == warm_start(0))
opti.subject_to(V[:, 0] == 0)  # Start at rest
opti.subject_to(X[:, -1] == warm_start(T_init))
# opti.subject_to(V[:, -1] == 0)  # End at rest

# Waypoint constraints
for i, t_wp in enumerate(waypoint_times):
    idx = int(t_wp/T_init * (n_steps - 1))
    # Position
    opti.subject_to(X[:, idx] == waypoints[i, :])
    # Orientation
    if i>0 and waypoints_rot is not None: # initial orientation doesnt matter
        opti.subject_to(V[:, idx]/cs.norm_2(V[:, idx])*waypoints_rot[i, :] >= np.arccos(angle_max * np.pi/180))


# Add constraint to ensure positive total time
opti.subject_to(T > 0)

# Solver options
# opti.solver("ipopt")
p_opts = {"expand":True}
s_opts = {"max_iter": 1000}
opti.solver("ipopt", p_opts, s_opts)

# Warm start values
# warm_X = np.array([spline(i / n_steps * T_init) for i in range(n_steps)]).T
warm_X = np.array(spline(np.linspace(0,T_init, n_steps))).T
warm_V = np.gradient(warm_X, axis=1) * (n_steps / T_init)
warm_A = np.gradient(warm_V, axis=1) * (n_steps / T_init) - g.reshape(3,1)
opti.set_initial(X, warm_X)
opti.set_initial(V, warm_V)
opti.set_initial(A, warm_A)
opti.set_initial(T, 10.0)

# Solve
try:
    opti.solve()

except Exception as e:
    print("Solver failed:", e)

X_sol = opti.debug.value(X)
V_sol = opti.debug.value(V)
A_sol = opti.debug.value(A)
T_sol = opti.debug.value(T)

# Plot results
time_grid = np.linspace(0, T_sol, n_steps)

plt.figure(figsize=(12, 8))

# Plot trajectory
ax = plt.subplot(2, 2, 1, projection='3d')
ax.plot(warm_X[0, :], warm_X[1, :], warm_X[2, :], label="Warm started trajectory")
ax.plot(X_sol[0, :], X_sol[1, :], X_sol[2, :], label="Optimized trajectory")
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='r', label="Waypoints")
ax.set_title("Trajectory")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()

# Plot forces
plt.subplot(2, 2, 2)
plt.plot(time_grid, A_sol[0, :], label="Acceleration x")
plt.plot(time_grid, A_sol[1, :], label="Acceleration y")
plt.plot(time_grid, A_sol[2, :], label="Acceleration z")
plt.plot(time_grid, np.linalg.norm(A_sol+g.reshape(3,1), axis=0), "--", label="Acceleration abs")
plt.plot(time_grid, np.full_like(time_grid, a_max), ".", label="Acceleration max")
plt.title("Acceleration")
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.legend()

# Plot velocities
plt.subplot(2, 2, 3)
plt.plot(time_grid, V_sol[0, :], label="Velocity x")
plt.plot(time_grid, V_sol[1, :], label="Velocity y")
plt.plot(time_grid, V_sol[2, :], label="Velocity z")
plt.title("Velocities")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend()

# Plot positions
plt.subplot(2, 2, 4)
plt.plot(time_grid, X_sol[0, :], label="Position x")
plt.plot(time_grid, X_sol[1, :], label="Position y")
plt.plot(time_grid, X_sol[2, :], label="Position z")
plt.title("Positions")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()

plt.tight_layout()
plt.show()