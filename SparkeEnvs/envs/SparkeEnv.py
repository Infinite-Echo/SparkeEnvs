import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
from . import Sparke
import time
from ..utils import reward_utils

NUM_SUBSTEPS = 5

'''Possible observations:
Observation	                    40
----------------------------------
Last Sent Motor Positions	    12
Approximate Base Position	    3
Approximate Base Orientation	4
Approximate Base Velocity	    6
Target Base Z	                1
Target Base Orientation	        4
Target Base Velocity	        6
Feet Contact	                4'''

OBS_X_VEL = 12 + 3 + 4
OBS_Y_VEL = 12 + 3 + 4 + 1
OBS_YAW_VEL = 12 + 3 + 4 + 5
OBS_BASE_POS = 12
OBS_BASE_ORIENTATION = 12 + 3
OBS_TARGET_Z = 12 + 3 + 4 + 6
OBS_TARGET_X_VEL = 12 + 3 + 4 + 6 + 1 + 4
OBS_TARGET_Y_VEL = 12 + 3 + 4 + 6 + 1 + 4 + 1

GENERAL_Z_HEIGHT = 0.17818074236278436
MIN_Z_HEIGHT = 0.1
MAX_Z_HEIGHT = 0.25
YAW_DRIFT_TOLERANCE = np.pi/36

class SparkeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render_mode=None, 
                 hard_reset=True,
                 action_repeat=1,
                 use_sub_steps=True,
                 time_step=0.004166667,
                 max_episode_steps=250,
                 x_vel_weight=1.0,
                 yaw_vel_weight=1.0,
                 drift_weight=2.0,
                 base_orientation_weight=0.1,
                 height_weight=0.01,
                 ):
        super().__init__()

        self.render_mode = render_mode
        self._action_repeat = action_repeat
        self._time_step = time_step
        self._max_episode_steps = max_episode_steps
        self._num_bullet_solver_iterations = 300
        self._last_frame_time = 0.0
        self._x_vel_weight = x_vel_weight
        self._yaw_vel_weight = yaw_vel_weight
        self._drift_weight = drift_weight
        self._base_orientation_weight = base_orientation_weight
        self._height_weight = height_weight

        if use_sub_steps:
            self._time_step /= NUM_SUBSTEPS
            self._num_bullet_solver_iterations /= NUM_SUBSTEPS
            self._action_repeat *= NUM_SUBSTEPS
        
        if self.render_mode:
            self._pybullet_client = bc.BulletClient(connection_mode=p.GUI, options="--opengl2")
        else:
            self._pybullet_client = bc.BulletClient()

        self._prev_reward = None
        
        # First Time Setup Must be Hard Reset
        self._hard_reset = True
        self.reset()

        self.action_space = spaces.Box(
            low=self._get_motor_bounds(high_bound=False),
            high=self._get_motor_bounds(high_bound=True),
            shape=(12,),
            dtype=np.float64
            )
        
        self.observation_space = spaces.Box(
            low=self._get_obs_low_bound(),
            high=self._get_obs_high_bound(),
            shape=(40,),
            dtype=np.float64
            )

        self._hard_reset = hard_reset
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self._hard_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self._pybullet_client.loadURDF(f"{pybullet_data.getDataPath()}/plane.urdf") #optionally
            self._pybullet_client.setGravity(0,0,-9.8)
            self.robot = Sparke.Sparke(self._pybullet_client)
        else:
            self.robot.reset(reload_urdf=False)

        self._eps_step_counter = 0
        self._cummulative_reward = 0.0
        self._terminated = False
        self._set_new_goal()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if self.render_mode == 'human':
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._action_repeat * self._time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        for _ in range(self._action_repeat):
            self.robot.apply_action(action)
            self._pybullet_client.stepSimulation()
        
        self._eps_step_counter += 1
        observation = self._get_obs()
        reward = self._get_reward(observation)
        terminated = self._terminated
        if self._eps_step_counter >= self._max_episode_steps:
            truncated = True
        else:
            truncated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self, mode, close=False):
        return np.array([])

    def close(self):
        self._pybullet_client.disconnect()
        pass

    def config(self, args):
        self._args = args
        
    def _get_obs(self) -> np.ndarray:
        observation = self.robot.get_obs(
            target_base_orientation=self._goal_orientation,
            target_base_vel=self._goal_vel,
            target_base_z=GENERAL_Z_HEIGHT
        )
        return observation
    
    def _get_info(self):
        info = {}
        return info

    def _get_reward(self, observation: np.ndarray):

        x_vel_reward = reward_utils.calc_gaussian_reward(
            max_reward=1.0,
            value=observation[OBS_X_VEL],
            target_value=observation[OBS_TARGET_X_VEL],
            std_dev=0.05
        )

        # yaw_vel_reward = reward_utils.calc_gaussian_reward(
        #     max_reward=1.0,
        #     value=observation[OBS_YAW_VEL],
        #     target_value=0.0,
        #     std_dev=0.2
        # )

        yaw_vel_reward = -abs(observation[OBS_YAW_VEL])

        base_orientation = self._pybullet_client.getEulerFromQuaternion(observation[OBS_BASE_ORIENTATION:(OBS_BASE_ORIENTATION+4)])
        orientation_reward = -(abs(observation[OBS_BASE_ORIENTATION]) + abs(observation[OBS_BASE_ORIENTATION + 1]))

        drift_reward = -abs(observation[OBS_BASE_POS + 1])

        height_reward = -(abs(GENERAL_Z_HEIGHT - observation[OBS_BASE_POS + 2]))

        if observation[OBS_BASE_POS + 2] <= MIN_Z_HEIGHT:
            self._terminated = True
        elif observation[OBS_BASE_POS + 2] >= MAX_Z_HEIGHT:
            self._terminated = True

        # print(f'X Velocity Reward: {x_vel_reward * self._x_vel_weight}')
        # print(f'Yaw Velocity Reward: {yaw_vel_reward * self._yaw_vel_weight}')
        # print(f'Orientation Reward: {orientation_reward * self._base_orientation_weight}')
        # print(f'Drift Reward: {drift_reward * self._drift_weight}')
        # print(f'Height Reward: {height_reward * self._height_weight}')

        reward = np.sum([
            x_vel_reward * self._x_vel_weight,
            yaw_vel_reward * self._yaw_vel_weight,
            orientation_reward * self._base_orientation_weight,
            drift_reward * self._drift_weight,
            height_reward * self._height_weight
        ], dtype=np.float64)
        
        return reward

    def _set_new_goal(self):
        self._goal_vel = np.zeros((6,), dtype=np.float64)
        self._goal_vel[0] = self.np_random.uniform(0.05, 0.25)
        self._goal_z = GENERAL_Z_HEIGHT # figure out standing height
        self._goal_orientation = np.array([0, 0, 0, 1], dtype=np.float64)

    def _get_obs_high_bound(self):
        '''
        Possible observations:
        Observation	                    40
        ----------------------------------
        Last Sent Motor Positions	    12
        Approximate Base Position	    3
        Approximate Base Orientation	4
        Approximate Base Velocity	    6
        Target Base Z	                1
        Target Base Orientation	        4
        Target Base Velocity	        6
        Feet Contact	                4

        '''
        high_bound = self._get_motor_bounds(high_bound=True)
        high_bound = np.append(high_bound, np.full((3,), np.inf, dtype=np.float64))
        high_bound = np.append(high_bound, np.full((4,), 1.0, dtype=np.float64))
        high_bound = np.append(high_bound, np.full((6,), np.inf, dtype=np.float64))
        high_bound = np.append(high_bound, np.full((1,), np.inf, dtype=np.float64))
        high_bound = np.append(high_bound, np.full((4,), 1.0, dtype=np.float64))
        high_bound = np.append(high_bound, np.full((6,), 0.25, dtype=np.float64))
        high_bound = np.append(high_bound, np.full((4,), 1.0, dtype=np.float64))
        return high_bound
    
    def _get_obs_low_bound(self):
        '''
        Possible observations:
        Observation	                    40
        ----------------------------------
        Last Sent Motor Positions	    12
        Approximate Base Position	    3
        Approximate Base Orientation	4
        Approximate Base Velocity	    6
        Target Base Z	                1
        Target Base Orientation	        4
        Target Base Velocity	        6
        Feet Contact	                4

        '''
        low_bound = self._get_motor_bounds(high_bound=False)
        low_bound = np.append(low_bound, np.full((3,), -np.inf, dtype=np.float64))
        low_bound = np.append(low_bound, np.full((4,), -1.0, dtype=np.float64))
        low_bound = np.append(low_bound, np.full((6,), -np.inf, dtype=np.float64))
        low_bound = np.append(low_bound, np.full((1,), -np.inf, dtype=np.float64))
        low_bound = np.append(low_bound, np.full((4,), -1.0, dtype=np.float64))
        low_bound = np.append(low_bound, np.full((6,), -0.25, dtype=np.float64))
        low_bound = np.append(low_bound, np.full((4,), 0.0, dtype=np.float64))
        return low_bound

    def _get_motor_bounds(self, high_bound: bool = False) -> np.ndarray:
        robot_id = self.robot.robot

        if high_bound:
            bound_index = 9
        else:
            bound_index = 8
        
        motor_bounds = np.ndarray((12,), dtype=np.float64)
        for i in range(self._pybullet_client.getNumJoints(robot_id)):
            joint_info = self._pybullet_client.getJointInfo(robot_id, i)
            motor_bounds[i] = joint_info[bound_index]
        
        return motor_bounds
