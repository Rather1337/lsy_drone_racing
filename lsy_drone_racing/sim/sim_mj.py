"""Quadrotor environment using PyBullet physics.

Based on UTIAS Dynamic Systems Lab's gym-pybullet-drones:
    * https://github.com/utiasDSL/gym-pybullet-drones
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np
import pybullet_data
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.utils import seeding
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.sim.drone import Drone
from lsy_drone_racing.sim.noise import NoiseList
from lsy_drone_racing.sim.physics import GRAVITY, Physics, PhysicsMode, force_torques
from lsy_drone_racing.sim.symbolic import SymbolicModel, symbolic
from lsy_drone_racing.utils import map2pi

logger = logging.getLogger(__name__)


class Sim:
    """Drone simulation based on gym-pybullet-drones."""

    mj_path = Path(__file__).resolve().parent / "bitcraze_crazyflie_2/scene.xml"

    def __init__(
        self,
        track: dict,
        sim_freq: int = 500,
        ctrl_freq: int = 500,
        disturbances: dict = {},
        randomization: dict = {},
        gui: bool = False,
        n_drones: int = 1,
        physics: PhysicsMode = PhysicsMode.PYB,
    ):
        """Initialization method for BenchmarkEnv.

        Args:
            track: The configuration of gates and obstacles. Must contain at least the initial drone
                state and can contain gates and obstacles.
            sim_freq: The frequency at which PyBullet steps (a multiple of ctrl_freq).
            ctrl_freq: The frequency at which the onboard drone controller recalculates the rotor
                rmps.
            disturbances: Dictionary to specify disturbances being used.
            randomization: Dictionary to specify randomization of the environment.
            gui: Option to show PyBullet's GUI.
            camera_view: The camera pose for the GUI.
            n_drones: The number of drones in the simulation. Only supports 1 at the moment.
            physics: The physics backend to use for the simulation. For more information, see the
                PhysicsMode enum.
        """
        self.np_random = np.random.default_rng()
        assert n_drones == 1, "Only one drone is supported at the moment."
        self.drone = Drone(controller="mellinger")
        self.n_drones = n_drones
        if not sim_freq % ctrl_freq == 0:
            raise ValueError(f"Sim frequency {sim_freq} must be multiple of {ctrl_freq}")
        self._n_substeps = sim_freq // ctrl_freq
        self.sim_freq = sim_freq
        self.ctrl_freq = ctrl_freq
        self.gui = gui

        self.physics = Physics(1 / sim_freq, PhysicsMode(physics))

        # Create the state and action spaces of the simulation. Note that the state space is
        # different from the observation space of any derived environment.
        min_thrust, max_thrust = self.drone.params.min_thrust, self.drone.params.max_thrust
        self.action_space = spaces.Box(low=min_thrust, high=max_thrust, shape=(4,))
        # pos in meters, rpy in radians, vel in m/s ang_vel in rad/s
        rpy_max = np.array([85 / 180 * np.pi, 85 / 180 * np.pi, np.pi], np.float32)  # Yaw unbounded
        max_flt = np.full(3, np.finfo(np.float32).max, np.float32)
        pos_low, pos_high = np.array([-3, -3, 0]), np.array([3, 3, 2.5])
        # State space uses 64-bit floats for better compatibility with pycffirmware.
        self.state_space = spaces.Dict(
            {
                "pos": spaces.Box(low=pos_low, high=pos_high, dtype=np.float64),
                "rpy": spaces.Box(low=-rpy_max, high=rpy_max, dtype=np.float64),
                "vel": spaces.Box(low=-max_flt, high=max_flt, dtype=np.float64),
                "ang_vel": spaces.Box(low=-max_flt, high=max_flt, dtype=np.float64),
            }
        )
        self.disturbance_config = disturbances
        self.disturbances = self._setup_disturbances(disturbances)
        self.randomization = randomization

        assert isinstance(track.drone, dict), "Expected drone state as dictionary."
        for key, val in track.drone.items():
            assert hasattr(self.drone, "init_" + key), f"Unknown key '{key}' in drone init state."
            setattr(self.drone, "init_" + key, np.array(val, dtype=float))

        self.gates = {f"gate{i}": g.toDict() for i, g in enumerate(track.get("gates", {}))}
        # Add nominal values to not loose the default when randomizing.
        for k, gate in self.gates.items():
            self.gates[k].update({"nominal." + k: v for k, v in gate.items()})
        self.n_gates = len(self.gates)

        self.obstacles = {
            f"obstacle{i}": o.toDict() for i, o in enumerate(track.get("obstacles", {}))
        }
        for i, obstacle in self.obstacles.items():
            self.obstacles[i].update({"nominal." + k: v for k, v in obstacle.items()})
        self.n_obstacles = len(self.obstacles)

        res = (640, 480)
        self.model, self.data = initialize_simulation(self.mj_path, self.gates, self.obstacles, res)

        # Initialize helpers for rendering the simulation.
        self.renderer = None
        if self.gui:
            default_camera_config = {}
            self.renderer = MujocoRenderer(self.model, self.data, default_camera_config)

        # Helper variables
        self.reset()  # TODO: Avoid double reset.

    def reset(self):
        """Reset the simulation to its original state."""
        for mode in self.disturbances.keys():
            self.disturbances[mode].reset()
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_drone()
        self._sync_mj()
        self.drone.reset()
        if self.gui:
            self.renderer.render("human")

    def step(self, desired_thrust: NDArray[np.floating]):
        """Advance the environment by one control step.

        Args:
            desired_thrust: The desired thrust for the drone.
        """
        self.drone.desired_thrust[:] = desired_thrust
        rpm = self._thrust_to_rpm(desired_thrust)  # Pre-process/clip the action
        disturb_force = np.zeros(3)
        if "dynamics" in self.disturbances:  # Add dynamics disturbance force.
            disturb_force = self.disturbances["dynamics"].apply(disturb_force)
        for _ in range(self._n_substeps):
            self.drone.rpm[:] = rpm  # Save the last applied action (e.g. to compute drag)
            for i, ft in force_torques(self.drone, rpm, self.physics.mode, self.physics.dt):
                body_id = self.model.body(f"drone:{i}").id
                self.data.xfrc_applied[body_id, :3] = ft.f
                self.data.xfrc_applied[body_id, 3:] = ft.t
            self.data.xfrc_applied[self.model.body("drone:4").id, :3] = disturb_force
            if self.physics.mode != PhysicsMode.DYN:
                mujoco.mj_step(self.model, self.data, nstep=1)
            else:
                mujoco.mj_forward(self.model, self.data)
            self._sync_mj()
        if self.gui:
            self.renderer.render("human")

    @property
    def collisions(self) -> NDArray[np.int_]:
        """Return the collisions with the drone."""
        return self.data.contact.geom.copy()

    def in_range(self, bodies: list[str], target_body: str, dist: float) -> NDArray[np.bool_]:
        """Return a mask array of objects within a certain distance of the drone."""
        body_pos = np.array([self.data.body(name).xpos for name in bodies])
        target_pos = self.data.body(target_body).xpos
        return np.linalg.norm(body_pos - target_pos, axis=1) < dist

    def seed(self, seed: int | None = None) -> int | None:
        """Set up a random number generator for a given seed."""
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        for noise in self.disturbances.values():
            noise.seed(seed)
        return seed

    def _setup_disturbances(self, disturbances: dict | None = None) -> dict[str, NoiseList]:
        """Creates attributes and action spaces for the disturbances.

        Args:
            disturbances: A dictionary of disturbance configurations for the environment.

        Returns:
            A dictionary of NoiseList that fuse disturbances for each mode.
        """
        dist = {}
        if disturbances is None:  # Default: no passive disturbances.
            return dist
        modes = {"action": {"dim": spaces.flatdim(self.action_space)}, "dynamics": {"dim": 3}}
        for mode, spec in disturbances.items():
            assert mode in modes, "Disturbance mode not available."
            spec["dim"] = modes[mode]["dim"]
            dist[mode] = NoiseList.from_specs([spec])
        return dist

    def _reset_pybullet(self):
        """Reset PyBullet's simulation environment."""
        p.resetSimulation(physicsClientId=self.pyb_client)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=self.pyb_client)
        p.setRealTimeSimulation(0, physicsClientId=self.pyb_client)
        p.setTimeStep(1 / self.settings.sim_freq, physicsClientId=self.pyb_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.pyb_client)
        # Load ground plane, drone and obstacles models.
        i = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.pyb_client)
        self.sim_objects["plane"] = i

        self.drone.id = p.loadURDF(
            str(self.URDF_DIR / "cf2x.urdf"),
            self.drone.init_pos,
            p.getQuaternionFromEuler(self.drone.init_rpy),
            flags=p.URDF_USE_INERTIA_FROM_FILE,  # Use URDF inertia tensor.
            physicsClientId=self.pyb_client,
        )
        # Load obstacles into the simulation and perturb their poses if configured.
        self._reset_obstacles()
        self._reset_gates()
        self._sync_mj()

    def _reset_obstacles(self):
        """Reset the obstacles in the simulation."""
        for i, obstacle in self.obstacles.items():
            pos_offset = np.zeros(3)
            if obstacle_pos := self.randomization.get("obstacle_pos"):
                distrib = getattr(self.np_random, obstacle_pos.get("type"))
                kwargs = {k: v for k, v in obstacle_pos.items() if k != "type"}
                pos_offset = distrib(**kwargs)
            self.obstacles[i]["pos"] = np.array(obstacle["nominal.pos"]) + pos_offset
            self.obstacles[i]["id"] = self._load_urdf_into_sim(
                self.URDF_DIR / "obstacle.urdf",
                self.obstacles[i]["pos"] + pos_offset,
                marker=str(i),
            )
            self.sim_objects[f"obstacle_{i}"] = self.obstacles[i]["id"]

    def _reset_gates(self):
        """Reset the gates in the simulation."""
        assert False
        for i, gate in self.gates.items():
            pos_offset = np.zeros_like(gate["nominal.pos"])
            if gate_pos := self.randomization.get("gate_pos"):
                distrib = getattr(self.np_random, gate_pos.get("type"))
                pos_offset = distrib(**{k: v for k, v in gate_pos.items() if k != "type"})
            rpy_offset = np.zeros(3)
            if gate_rpy := self.randomization.get("gate_rpy"):
                distrib = getattr(self.np_random, gate_rpy.get("type"))
                rpy_offset = distrib(**{k: v for k, v in gate_rpy.items() if k != "type"})
            gate["pos"] = np.array(gate["nominal.pos"]) + pos_offset
            gate["rpy"] = map2pi(np.array(gate["nominal.rpy"]) + rpy_offset)  # Ensure [-pi, pi]
            gate["id"] = self._load_urdf_into_sim(
                self.URDF_DIR / "gate.urdf", gate["pos"], gate["rpy"], marker=str(i)
            )
            self.sim_objects[f"gate_{i}"] = gate["id"]

    def _randomize_drone(self):
        """Randomize the drone's position, orientation and physical properties."""
        inertia_diag = self.drone.nominal_params.J.diagonal()
        if drone_inertia := self.randomization.get("drone_inertia"):
            distrib = getattr(self.np_random, drone_inertia.type)
            kwargs = {k: v for k, v in drone_inertia.items() if k != "type"}
            inertia_diag = inertia_diag + distrib(**kwargs)
            assert all(inertia_diag > 0), "Negative randomized inertial properties."
        self.drone.params.J = np.diag(inertia_diag)

        mass = self.drone.nominal_params.mass
        if drone_mass := self.randomization.get("drone_mass"):
            distrib = getattr(self.np_random, drone_mass.type)
            mass += distrib(**{k: v for k, v in drone_mass.items() if k != "type"})
            assert mass > 0, "Negative randomized mass."
        self.drone.params.mass = mass
        self.model.body("drone").mass = mass  # TODO: Verify this changes dynamics
        self.model.body("drone").inertia = inertia_diag

        pos = self.drone.init_pos.copy()
        if drone_pos := self.randomization.get("drone_pos"):
            distrib = getattr(self.np_random, drone_pos.type)
            pos += distrib(**{k: v for k, v in drone_pos.items() if k != "type"})

        rpy = self.drone.init_rpy.copy()
        if drone_rpy := self.randomization.get("drone_rpy"):
            distrib = getattr(self.np_random, drone_rpy.type)
            kwargs = {k: v for k, v in drone_rpy.items() if k != "type"}
            rpy = np.clip(rpy + distrib(**kwargs), -np.pi, np.pi)

        self.model.body("drone").pos = pos
        # Change from scipy quat (x y z w) to mujoco quat (w x y z)
        self.model.body("drone").quat = R.from_euler("XYZ", rpy).as_quat()[[3, 0, 1, 2]]
        self.data.cvel[self.model.body("drone").id][:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _sync_mj(self):
        """Read state values from PyBullet and synchronize the drone buffers with it.

        We cache the state values in the drone class to avoid calling PyBullet too frequently.
        """
        pos, quat = self.data.body("drone").xpos, self.data.body("drone").xquat
        self.drone.pos[:] = np.array(pos, float)
        self.drone.rpy[:] = R.from_quat(quat[[1, 2, 3, 0]]).as_euler("XYZ")
        cvel = self.data.cvel[self.model.body("drone").id]
        self.drone.vel[:] = cvel[:3].copy()
        self.drone.ang_vel[:] = cvel[3:].copy()

    def symbolic(self) -> SymbolicModel:
        """Create a symbolic (CasADi) model for dynamics, observation, and cost.

        Returns:
            CasADi symbolic model of the environment.
        """
        return symbolic(self.drone, 1 / self.settings.sim_freq)

    def _thrust_to_rpm(self, thrust: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert the desired_thrust into motor RPMs.

        Args:
            thrust: The desired thrust per motor.

        Returns:
            The motors' RPMs to apply to the quadrotor.
        """
        thrust = np.clip(thrust, self.drone.params.min_thrust, self.drone.params.max_thrust)
        if "action" in self.disturbances:
            thrust = self.disturbances["action"].apply(thrust)
        thrust = np.clip(thrust, 0, None)  # Make sure thrust is not negative after disturbances
        pwm = (
            np.sqrt(thrust / self.drone.params.kf) - self.drone.params.pwm2rpm_const
        ) / self.drone.params.pwm2rpm_scale
        pwm = np.clip(pwm, self.drone.params.min_pwm, self.drone.params.max_pwm)
        return self.drone.params.pwm2rpm_const + self.drone.params.pwm2rpm_scale * pwm


# TODO: Check with actual dimensions
GATE_GEOMS = [
    {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.01, 0.01, 0.01], "pos": [0, 0, 0]},
    {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.25, 0.025, 0.025], "pos": [0, 0, -0.225]},
    {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.25, 0.025, 0.025], "pos": [0, 0, 0.225]},
    {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.025, 0.025, 0.25], "pos": [-0.225, 0, 0]},
    {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.025, 0.025, 0.25], "pos": [0.225, 0, 0]},
    {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.025, 0.025, 0.4], "pos": [0, 0, -0.6]},
]

OBSTACLE_GEOMS = [
    {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": [0.025, 0.025, 0.475], "pos": [0, 0, 0]}
]


def initialize_simulation(
    path: Path, gates: dict, obstacles: dict, res: tuple[int, int]
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Initialize MuJoCo simulation data structures `mjModel` and `mjData`.

    Load the original MuJoCo specs from the XML file, dynamically add gates and obstacles to the
    worldbody, and recompile the model.

    Args:
        path: Path to the MuJoCo XML model file.
        res: Simulation resolution (width x height).

    Returns:
        The model and data structures.
    """
    spec = mujoco.MjSpec()
    spec.from_file(str(path))

    for name, gate in gates.items():
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = gate["pos"]
        body.quat = R.from_euler("XYZ", gate["rpy"]).as_quat()[[3, 0, 1, 2]]  # Scalar first
        for j, geom in enumerate(GATE_GEOMS):
            mj_geom = body.add_geom()
            mj_geom.size[:3] = geom["size"]
            mj_geom.pos = geom["pos"]
            mj_geom.type = geom["type"]
            mj_geom.name = f"{name}_geom{j}"

    for name, obstacle in obstacles.items():
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = obstacle["pos"]
        for j, geom in enumerate(OBSTACLE_GEOMS):
            mj_geom = body.add_geom()
            mj_geom.size[:3] = geom["size"]
            mj_geom.pos = geom["pos"]
            mj_geom.pos[2] -= geom["size"][2]  # Obstacle position is given at the top -> convert
            mj_geom.type = geom["type"]
            mj_geom.name = f"{name}_geom{j}"

    model = spec.compile()
    model.vis.global_.offwidth = res[0]
    model.vis.global_.offheight = res[1]
    data = mujoco.MjData(model)
    return model, data
