import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data
from . import Sparke
import time

NUM_SUBSTEPS = 5

GENERAL_Z_HEIGHT = 0.17818074236278436
MIN_Z_HEIGHT = 0.1
YAW_DRIFT_TOLERANCE = np.pi/36

class SparkeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}

    def __init__(self, render_mode=None, 
                 hard_reset=True,
                 action_repeat=1,
                 use_sub_steps=True,
                 time_step=0.01,
                 max_episode_steps=1000,):
        super().__init__()

        self.render_mode = render_mode
        self._action_repeat = action_repeat
        self._time_step = time_step
        self._max_episode_steps = max_episode_steps
        self._num_bullet_solver_iterations = 300
        self._last_frame_time = 0.0
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
            # set time step
            # set gravity, etc
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
        self._prev_reward = None
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
            reward = -100.0
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
        
    def _get_obs(self):
        observation = self.robot.get_obs(
            target_base_orientation=self._goal_orientation,
            target_base_vel=self._goal_vel,
            target_base_z=GENERAL_Z_HEIGHT
        )
        return observation
    
    def _get_info(self):
        info = {}
        return info

    def _get_reward(self, observation):
        reward = 0.0
        # increase reward as current vel gets closer to target vel
        
        current_vel = observation[19:25]
        for i in range(6):
            diff = abs(self._goal_vel[i] - current_vel[i])

            if round(diff, 2) <= round(self._best_vel_diffs[i],2):
                reward += 50.0 * (self._best_vel_diffs[i] - diff)
                self._best_vel_diffs[i] = diff
            else:
                reward += 50.0 * (self._best_vel_diffs[i] - diff)
            
            if round(diff, 2) <= 0.01:
                reward += 100.0

        if not ((GENERAL_Z_HEIGHT - 0.02) < observation[14] < (GENERAL_Z_HEIGHT + 0.02)):
            reward -= 1.0

        base_orientation = self._pybullet_client.getEulerFromQuaternion(observation[15:19])
        yaw = base_orientation[2]
        # if not (-YAW_DRIFT_TOLERANCE < yaw < YAW_DRIFT_TOLERANCE):
        #     self._terminated = True
        #     return -100.0

        # punish and terminate if base Z goes below minimum
        if observation[14] <= MIN_Z_HEIGHT:
            self._terminated = True
            return -100.0
        
        return reward

    def _set_new_goal(self):
        self._goal_vel = np.zeros((6,), dtype=np.float64)
        self._goal_vel[0] = self.np_random.uniform(0.05, 0.25)
        self._goal_z = GENERAL_Z_HEIGHT # figure out standing height
        self._goal_orientation = np.array([0, 0, 0, 1], dtype=np.float64)
        self._best_vel_diffs = np.zeros((6,), dtype=np.float64)

    def _calc_goal_distance(self, current_pos, goal_pos):
        return np.linalg.norm(current_pos[0:2] - goal_pos[0:2])

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
