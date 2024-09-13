# general
import numpy as np
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def rad2deg(rad):
    return rad * 180 / math.pi


def deg2rad(deg):
    return deg * math.pi / 180


def rotation_x(angle):
    """ find rotation matrix about x """

    # angle in rad
    c = math.cos(angle)
    s = math.sin(angle)

    R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])

    return R

def rotation_y(angle):
    """ find rotation matrix about y"""

    # angle in rad
    c = math.cos(angle)
    s = math.sin(angle)

    R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])

    return R

def rotation_z(angle):
    """ find rotation matrix about z"""

    # angle in rad
    c = math.cos(angle)
    s = math.sin(angle)

    R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])

    return R

def euler2quat(roll, pitch, yaw):
    """ convert euler angles to quaternion """

    r = Rotation.from_euler('XYZ', [roll, pitch, yaw], degrees=False)
    x,y,z,w = r.as_quat()
    quat = np.array([x,y,z, w])

    return quat

def quat2rotation(quat):
    """ convert quaternion to rotation matrix """

    w,x,y,z = quat
    r = Rotation.from_quat([x,y,z,w])

    return r.as_matrix()

def quat2euler(x, y, z, w):
    '''
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    '''
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians
    

def rotation2euler(R):
    """ convert rotation matrix to euler """

    r = Rotation.from_matrix(R)

    euler = r.as_euler('XYZ', degrees=False)

    if np.any(euler > math.pi) or np.any(euler < -math.pi):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("eot2euler ERROR!!!!!!!!!!!!!!!!!!!")

    return euler


def euler2rotation(roll, pitch, yaw):
    """ convert euler angles to rotation matrix """

    R = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)

    return R


def clamp(i_error, i_range):
    clamped = i_error[:]

    clamped = np.maximum(np.minimum(i_error, i_range), -i_range)

    return clamped

def plot_states(log_times, log_state):
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)

    ax1.plot(log_times, log_state[:,0], 'r')
    ax1.plot(log_times, log_state[:,1], 'g')
    ax1.plot(log_times, log_state[:,2], 'b')

    ax2.plot(log_times, log_state[:,3], 'r')
    ax2.plot(log_times, log_state[:,4], 'g')
    ax2.plot(log_times, log_state[:,5], 'b')

    ax3.plot(log_times, log_state[:,6], 'r')
    ax3.plot(log_times, log_state[:,7], 'g')
    ax3.plot(log_times, log_state[:,8], 'b')

    ax4.plot(log_times, log_state[:,9], 'r')
    ax4.plot(log_times, log_state[:,10], 'g')
    ax4.plot(log_times, log_state[:,11], 'b')

    ax5.plot(log_times, log_state[:,12], 'r')
    ax5.plot(log_times, log_state[:,13], 'g')
    ax5.plot(log_times, log_state[:,14], 'b')
    ax5.plot(log_times, log_state[:,15], 'k')

    plt.show()

def plot_tracking(log_times, log_state, log_reference, log_desired):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)

    ax1.plot(log_times, log_state[:,0], 'b')
    ax1.plot(log_times, log_reference[:,0], 'g--')
    ax1.plot(log_times, log_desired[:,0], 'r--')
    ax1.set_ylabel('x [m]')

    ax2.plot(log_times, log_state[:,1], 'b')
    ax2.plot(log_times, log_reference[:,1], 'g--')
    ax2.plot(log_times, log_desired[:,1], 'r--')
    ax2.set_ylabel('y [m]')

    ax3.plot(log_times, log_state[:,2], 'b')
    ax3.plot(log_times, log_reference[:,2], 'g--')
    ax3.plot(log_times, log_desired[:,2], 'r--')
    ax3.set_ylabel('z [m]')

    ax3.set_xlabel('time [s]')

    plt.show()


def plot_tracking_comparison(log_times_baseline, log_state_baseline, log_desired_baseline, log_times_learning, log_state_learning, log_reference_learning, log_desired_learning, save_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, dpi=160)

    ax1.plot(log_times_learning, log_desired_learning[:,0], '--', color='r', label='desired')
    ax1.plot(log_times_baseline, log_state_baseline[:,0], '-', color=(0.8, 0.8, 0.8), label='baseline')
    ax1.plot(log_times_learning, log_reference_learning[:,0], 'g--', label='reference')
    ax1.plot(log_times_learning, log_state_learning[:,0], '-', color='b', label='learning')
    ax1.set_ylabel('x [m]')

    ax2.plot(log_times_learning, log_desired_learning[:,1], 'r--', label='desired')
    ax2.plot(log_times_baseline, log_state_baseline[:, 1], '-', color=(0.8, 0.8, 0.8), label='baseline')
    ax2.plot(log_times_learning, log_reference_learning[:,1], 'g--', label='reference')
    ax2.plot(log_times_learning, log_state_learning[:,1], 'b', label='learning')
    ax2.set_ylabel('y [m]')

    ax3.plot(log_times_learning, log_desired_learning[:,2], 'r--', label='desired')
    ax3.plot(log_times_baseline, log_state_baseline[:, 2], '-', color=(0.8, 0.8, 0.8), label='baseline')
    ax3.plot(log_times_learning, log_reference_learning[:,2], 'g--', label='reference')
    ax3.plot(log_times_learning, log_state_learning[:,2], 'b', label='learning')
    ax3.set_ylabel('z [m]')

    ax3.set_xlabel('time [s]')

    ax1.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode='expand', ncol=4)

    # plt.show()
    plt.savefig(save_dir)

def update_full_state(state):
    '''
    Construct an array containing the full state of the quadrotor

    :param state: state dictionary with quadrotor state as attributes
    :return:
    full_state - an array containing full state of the quadrotor
    '''

    full_state = np.array([state.x, state.y, state.z,
                           state.vx, state.vy, state.vz,
                           state.roll, state.pitch, state.yaw,
                           state.p, state.q, state.r])
    return full_state