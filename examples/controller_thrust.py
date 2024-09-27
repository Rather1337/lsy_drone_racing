"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

from pathlib import Path

import math
import numpy as np
import numpy.typing as npt
from stable_baselines3 import PPO

from scipy.interpolate import CubicSpline, interp1d, splrep, splev, make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.wrapper import ObsWrapper
from lsy_drone_racing.utils.utils import draw_trajectory, draw_segment_of_traj

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation
import json
from munch import Munch
from pathlib import Path

#import sys
#from pathlib import Path
#sys.path.insert(Path(__file__).parent / "../examples")
import sys
import os

# TODO: Very ugly, maybe move the helpers inside the packet?
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from examples.helper import rotation2euler, euler2rotation, clamp, rad2deg

# data logging simple
dd = {}
dd["pos_des"] = []
dd["pos"] = []


# Crazyswarm position controller slightly modified - from https://github.com/utiasDSL/crazyswarm-import/blob/ad2f7ea987f458a504248a1754b124ba39fc2f21/ros_ws/src/crazyswarm/scripts/position_ctl_m.py
class PositionController:
    def __init__(self, quad=None):
        #if not quad is None:
        #    # get parameters
        #    self.params = quad.params
        #    self.controller_params = quad.controller
        self.params = self.get_params()

        self.i_error = 0

        # abc-formula coefficients for thrust to pwm conversion
        # pwm = a * thrust^2 + b * thrust + c
        self.a_coeff = -1.1264
        self.b_coeff = 2.2541
        self.c_coeff = 0.0209
        self.pwm_max = 65535.0

    def get_params(self):
        '''
        Load parameters from parameters.json

        :return:
            params - dictionary with simulation, quadrotor physics, and controller parameters
        '''

        # load parameters and convert to . dict format
        load_params = json.load(open(Path(__file__).parent / "parameters.json"))
        params = Munch.fromDict(load_params)

        return params
    
    def thrust2pwm(self, thrust):
        """Convert thrust to pwm using a quadratic function."""
        pwm = self.a_coeff * thrust * thrust + self.b_coeff * thrust + self.c_coeff
        pwm = np.maximum(pwm, 0.0)
        pwm = np.minimum(pwm, 1.0)
        thrust_pwm = pwm * self.pwm_max
        return thrust_pwm
    
    def pwm2thrust(self, pwm):
        """Convert pwm to thrust using a quadratic function."""
        pwm_scaled = pwm / self.pwm_max
        # solve quadratic equation using abc formula
        thrust = (-self.b_coeff + np.sqrt(self.b_coeff**2 - 4 * self.a_coeff * (self.c_coeff - pwm_scaled))) / (2 * self.a_coeff)
        return thrust

    def compute_action(self, measured_pos, measured_rpy, measured_vel, desired_pos, desired_yaw, desired_vel, dt):
        """Compute the thrust and euler angles for the drone to reach the desired position.
        
        Args:
            measured_pos (np.array): current position of the drone
            measured_rpy (np.array): current roll, pitch, yaw angles of the drone
            desired_pos (np.array): desired position of the drone
            desired_yaw (float): desired yaw angle of the drone in radians
            dt (float): time step
            
        Returns:
            thrust_desired (float): desired thrust
            euler_desired (np.array): desired euler angles
        """
        current_R = euler2rotation(measured_rpy[0], measured_rpy[1], measured_rpy[2])

        # compute position and velocity error
        pos_error = desired_pos - measured_pos
        vel_error = desired_vel - measured_vel

        # update integral error
        self.i_error += pos_error * dt
        self.i_error = clamp(self.i_error, np.array(self.params.pos_ctl.i_range))

        # compute target thrust
        target_thrust = np.zeros(3)

        target_thrust += self.params.pos_ctl.kp * pos_error
        target_thrust += self.params.pos_ctl.ki * self.i_error
        target_thrust += self.params.pos_ctl.kd * vel_error
        # target_thrust += params.quad.m * desired_acc
        target_thrust[2] += self.params.quad.m * self.params.quad.g
        
        # update z_axis
        z_axis = current_R[:,2]

        # update current thrust
        current_thrust = target_thrust.dot(z_axis)
        current_thrust = max(current_thrust, 0.3 * self.params.quad.m * self.params.quad.g)
        current_thrust = min(current_thrust, 1.8 * self.params.quad.m * self.params.quad.g)
        # print('current_thrust:', current_thrust)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(desired_yaw), math.sin(desired_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = rotation2euler(R_desired)

        thrust_desired = self.thrust2pwm(current_thrust)

        return thrust_desired, euler_desired

    def position_controller_reset(self):
        self.i_error = np.zeros(3)


class Controller(BaseController):
    """Template controller class."""

    def __init__(self, initial_obs: npt.NDArray[np.floating], initial_info: dict):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            initial_info: Additional environment information from the reset.
        """
        super().__init__(initial_obs, initial_info)

        self.initial_obs = initial_obs
        self.initial_info = initial_info

        self.thrust_ctrl = PositionController()
        self.counter = 0
        self.cmd_type="thrust"

        start_position = initial_obs[:3]
        gate_positions = initial_info["gates.pos"]
        gates_orientations = initial_info["gates.rpy"]
        gates_in_range = initial_info["gates.in_range"]
        obstacle_positions = initial_info["obstacles.pos"]
        obstacles_in_range = initial_info["obstacles.in_range"]

        waypoints = []
        waypoints.append([1, 1, 0.05])
        gates = gate_positions 
        z_low = 0.525 
        z_high = 1.0 
        waypoints.append([1, 0, z_low])
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.7,
                (gates[0][1] + gates[1][1]) / 2 - 0.3,
                (z_low + z_high) / 3,
            ]
        )
        waypoints.append(
            [
                (gates[0][0] + gates[1][0]) / 2 - 0.5,
                (gates[0][1] + gates[1][1]) / 2 - 0.6,
                (z_low + z_high) / 2,
            ]
        )
        waypoints.append([gates[1][0] - 0.4, gates[1][1] - 0.4, z_high])
        waypoints.append([gates[1][0], gates[1][1], z_high]) # added
        waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.4, z_high])
        waypoints.append([gates[2][0]+0.2, gates[2][1] - 0.5, z_low])
        waypoints.append([gates[2][0] -0.3, gates[2][1]+0.5, z_low]) # added
        waypoints.append([gates[2][0], gates[2][1] + 0.2, z_high + 0.2])
        waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
        waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
        waypoints = np.array(waypoints)
        print(f"waypoints: {waypoints}")

        xs = waypoints[:,0]
        ys = waypoints[:,1]
        zs = waypoints[:,2]

        # scale trajecotry between 0 and 1 for now.
        ts = np.linspace(0,1,np.shape(waypoints)[0])
        cs_x = CubicSpline(ts,xs)
        cs_y = CubicSpline(ts,ys)
        cs_z = CubicSpline(ts,zs)

        # control input is 30Hz, this way we can say how fast he should run through the course.
        des_completion_time = 10 
        ts = np.linspace(0, 1, int(30 * des_completion_time))

        self.traj = []

        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        # create another trajectory sample freqency for the plotting
        ts_plot = np.linspace(0,1,10_000)

        # Draw interpolated Trajectory
        draw_trajectory(initial_info, waypoints, cs_x(ts_plot), cs_y(ts_plot), cs_z(ts_plot), num_plot_points=200)

        self.traj.append(start_position)


    def compute_control(
        self, obs: npt.NDArray[np.floating], info: dict | None = None
    ) -> npt.NDarray[np.floating]:
        """Compute the next desired position and orientation of the drone.

        INSTRUCTIONS:
            Re-implement this method to return the target pose to be sent from Crazyswarm to the
            Crazyflie using the `cmdFullState` call.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone pose [x_des, y_des, z_des, yaw_des] as a numpy array.
        """
        pos = obs[:3]
        rpy = obs[3:6]
        vel = obs[6:9]
        ang_vel = obs[9:12]

        des_pos = np.array([self.x_des[self.counter], self.y_des[self.counter], self.z_des[self.counter]])
        des_vel = np.zeros(3) 
        des_yaw = 0.0
        dt = 1/500 # TODO: Verify Simulation Freq. and control freq(this) with firmware ctrl freq

        dd["pos_des"].append(des_pos)
        dd["pos"].append(pos)

        # thrust controller gives us the thrust as pwm and the rpy as radians.
        thrust_des, rpy_des = self.thrust_ctrl.compute_action(pos, rpy, vel, des_pos, des_yaw, des_vel, dt)

        # TODO: move conversion from rad to degree & inverting pitch maybe out of controller?
        # convert rpy to degrees
        rpy_des = rad2deg(rpy_des)
        action = np.array([rpy_des[0], -1*rpy_des[1], rpy_des[2], thrust_des])

        self.counter +=1

        # draw actual drone trajectory in green - increases simulation time significantly however
        draw_segment_of_traj(self.initial_info, self.traj[-1], pos, [0,1,0,1])

        self.traj.append(pos)

        return action

    @staticmethod
    def action_transform(
        action: npt.NDArray[np.floating], obs: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        drone_pos = obs[:3]
        return drone_pos + action

    def episode_reset(self):
        self._last_action = np.zeros(3)
