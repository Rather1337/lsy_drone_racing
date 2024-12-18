import casadi as cs
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Parameters
dim = 3  # Dimensions of the point mass (3D case)
mass = 0.033
F_max = 4*0.10 # 4* individual max thrust in N
a_max = F_max/mass
v_max = 5 # m/s
n_steps = 50  # Number of time steps
g = np.array([0,0,-9.81])
tol_waypoint = 0.02 # m, how close is the trajectory supposed to get to the waypoint

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
T_init = 10.0 # initial guess for the time
waypoint_times = np.linspace(0, T_init, len(waypoints)) 
waypoint_times = np.array([0, 0.2, 0.5, 0.8, 1])*T_init

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
MU = opti.variable(len(waypoints), n_steps)  # Slack variables for gate passing
LAMBDA = opti.variable(len(waypoints), n_steps)  # Slack variables for gate order

# Time step size
dt = T / n_steps

# Objective function: minimize total time
alpha = 1e-6
opti.minimize(T + alpha * cs.sumsqr(A))


for i in range(n_steps):
    # Dynamics constraints
    if i<n_steps-1:
        opti.subject_to(X[:, i + 1] == X[:, i] + V[:, i] * dt ) #+ 0.5*A[:, i]*dt**2
        opti.subject_to(V[:, i + 1] == V[:, i] + (A[:, i] + g) * dt)
        opti.subject_to( LAMBDA[:, i+1] - LAMBDA[:, i] + MU[:, i] == 0 )
    opti.subject_to(X[2, i] >= 0.0) # don't crash into the floor
    opti.subject_to(V[:, i].T@V[:, i] <= v_max**2)
    opti.subject_to(A[:, i].T@A[:, i] <= a_max**2)

    # Waypoint constraints
    # https://rpg.ifi.uzh.ch/docs/ScienceRobotics21_Foehn.pdf constraints 12

    opti.subject_to(MU[:,i] >= 0)
    opti.subject_to(LAMBDA[:,i] >= 0)

    for w in range(len(waypoints)):
        opti.subject_to(MU[w,i] * ((X[:,i]-waypoints[w, :]).T@(X[:,i]-waypoints[w, :]) - tol_waypoint**2) == 0) # Distance to gate <= tol_waypoint
        if w < len(waypoints)-1:
            opti.subject_to( LAMBDA[w,i] - LAMBDA[w+1,i] <= 0 )



# Boundary conditions
opti.subject_to(X[:, 0] == warm_start(0))
opti.subject_to(V[:, 0] == 0)  # Start at rest
opti.subject_to(X[:, -1] == warm_start(T_init))
opti.subject_to(LAMBDA[:,0] - 1 == 0)
opti.subject_to(LAMBDA[:,-1] == 0)
# opti.subject_to(V[:, -1] == 0)  # End at rest


# Add constraint to ensure positive total time
opti.subject_to(T > 0)

# Solver options
p_opts = {}
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
warm_MU = np.zeros((len(waypoints), n_steps))
warm_LAMBDA = np.zeros_like(warm_MU)
for i, t_wp in enumerate(waypoint_times):
    idx = int(t_wp/T_init * (n_steps - 1))
    warm_MU[i,idx] = 1
    warm_LAMBDA[i,:idx+1] = 1
opti.set_initial(MU, warm_MU)
opti.set_initial(LAMBDA, warm_LAMBDA)

opti.set_initial(T, T_init)


# Solve
try:
    opti.solve()

except Exception as e:
    print("Solver failed:", e)

X_sol = opti.debug.value(X)
V_sol = opti.debug.value(V)
A_sol = opti.debug.value(A)
T_sol = opti.debug.value(T)
MU_sol = opti.debug.value(MU)
LAMBDA_sol = opti.debug.value(LAMBDA)
print(MU_sol)
print(LAMBDA_sol)

# X_sol = warm_X
# V_sol = warm_V
# A_sol = warm_A

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
plt.plot(time_grid, A_sol[2, :]-9.81, label="Acceleration z")
plt.plot(time_grid, warm_A[0, :], "--", label="Warm Acceleration x")
plt.plot(time_grid, warm_A[1, :], "--", label="Warm Acceleration y")
plt.plot(time_grid, warm_A[2, :], "--", label="Warm Acceleration z")
plt.plot(time_grid, np.linalg.norm(A_sol, axis=0), "--", label="Acceleration abs") #A_sol+g.reshape(3,1)
plt.plot(time_grid, np.full_like(time_grid, a_max), ".", label="Acceleration max")
plt.title("Acceleration")
plt.xlabel("Time")
plt.ylabel("Force")
plt.legend()

# Plot velocities
plt.subplot(2, 2, 3)
plt.plot(time_grid, V_sol[0, :], label="Velocity x")
plt.plot(time_grid, V_sol[1, :], label="Velocity y")
plt.plot(time_grid, V_sol[2, :], label="Velocity z")
plt.plot(time_grid, warm_V[0, :], "--", label="Warm Velocity x")
plt.plot(time_grid, warm_V[1, :], "--", label="Warm Velocity y")
plt.plot(time_grid, warm_V[2, :], "--", label="Warm Velocity z")
plt.title("Velocities")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.legend()

# Plot positions
plt.subplot(2, 2, 4)
plt.plot(time_grid, X_sol[0, :], label="Position x")
plt.plot(time_grid, X_sol[1, :], label="Position y")
plt.plot(time_grid, X_sol[2, :], label="Position z")
plt.plot(time_grid, warm_X[0, :], "--", label="Warm Position x")
plt.plot(time_grid, warm_X[1, :], "--", label="Warm Position y")
plt.plot(time_grid, warm_X[2, :], "--", label="Warm Position z")
plt.title("Positions")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()

plt.tight_layout()
plt.show()

plt.plot(np.arange(len(MU_sol[0])), np.sum(LAMBDA_sol, axis=0) , label="LAMBDA")
plt.plot(np.arange(len(MU_sol[0])), np.sum(MU_sol, axis=0) , label="MU")
plt.xlabel("Step")
plt.ylabel("Gates to pass")
plt.legend()
plt.show()