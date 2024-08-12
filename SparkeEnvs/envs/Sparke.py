import gymnasium as gym
import pybullet as p
from pybullet_utils import bullet_client as bc
import numpy as np
import pybullet_data
from SparkeEnvs.utils.urdf_utils import create_motor_dict, create_foot_dict

INIT_POSITION = [0, 0, 0.2]
INIT_ORIENTATION = [0, 0, 0, 1]
BASE_LINK_ID = -1
SERVO_OPERATING_SPEED = 9.519898 #rads/s based on 0.11sec/60deg from servo datasheet
POS_GAIN = 0.4
VEL_GAIN = 0.3

class Sparke():
    def __init__(self, pybullet_client, control_mode = p.POSITION_CONTROL) -> None:
        self._pybullet_client = pybullet_client
        self._control_mode = control_mode
        self.reset()
    
    def reset(self, reload_urdf=True):
        if reload_urdf:
            self.robot = self._pybullet_client.loadURDF(
                f'/home/echo/workspaces/rl_ws/SparkeEnvs/bullet_sparke.urdf', 
                INIT_POSITION,
                flags=(
                    p.URDF_MERGE_FIXED_LINKS |
                    p.URDF_USE_SELF_COLLISION |
                    p.URDF_USE_INERTIA_FROM_FILE |
                    p.URDF_MAINTAIN_LINK_ORDER
                )
            )
            self.set_link_dynamics()
            self._motor_dict = create_motor_dict(self._pybullet_client, self.robot)
            self._foot_dict = create_foot_dict(self._pybullet_client, self.robot)
        else:
            self._pybullet_client.resetBasePositionAndOrientation(self.robot, 
                                                                  INIT_POSITION, INIT_ORIENTATION)
            self._pybullet_client.resetBaseVelocity(self.robot, [0,0,0], [0,0,0])
            self._reset_motors()
        self._prev_motor_cmds = np.zeros((12,), dtype=np.float64)

    def get_obs(self, target_base_vel, 
                target_base_z, 
                target_base_orientation) -> np.ndarray:
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
        observation = np.append(self._prev_motor_cmds, self._get_base_pos())
        observation = np.append(observation, self._get_base_orientation())
        observation = np.append(observation, self._get_base_vel())
        observation = np.append(observation, target_base_z)
        observation = np.append(observation, target_base_orientation)
        observation = np.append(observation, target_base_vel)
        observation = np.append(observation, self._get_feet_contacts().astype(np.float64))
        return observation

    def apply_action(self, motor_cmds):
        motors = list(self._motor_dict.values())
        self._pybullet_client.setJointMotorControlArray(self.robot, motors, 
                                controlMode=self._control_mode,
                                targetPositions=motor_cmds, 
                                targetVelocities=np.full((12,), SERVO_OPERATING_SPEED, dtype=np.float64), 
                                forces=np.full((12,), 500.0, dtype=np.float64),
                                positionGains=np.full((12,), POS_GAIN, dtype=np.float64),
                                velocityGains=np.full((12,), VEL_GAIN, dtype=np.float64))
        self._prev_motor_cmds = motor_cmds

    def _reset_motors(self):
        for key in self._motor_dict:
            motor = self._motor_dict[key]
            self._pybullet_client.setJointMotorControl2(self.robot, motor, 
                                    controlMode=self._control_mode,
                                    targetPosition=0, targetVelocity=0, force=500.0)

    def _get_base_vel(self):
        lin_vel, ang_vel = self._pybullet_client.getBaseVelocity(self.robot)
        vel = list(lin_vel)
        vel.extend(list(ang_vel))
        vel = np.array(vel, dtype=np.float64)
        return vel
    
    def _get_base_pos(self):
        position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.robot))
        position = np.array(list(position), dtype=np.float64)
        return position
    
    def _get_base_orientation(self):
        _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.robot))
        orientation = np.array(list(orientation), dtype=np.float64)
        return orientation
    
    def _get_motor_vels(self) -> np.ndarray:
        motor_velocities = np.zeros((12,), dtype=np.float64)
        for i, motor in zip(range(12), self._motor_dict.keys()):
            motor_state = self._pybullet_client.getJointState(self.robot, 
                                                       self._motor_dict[motor])
            motor_velocities[i] = motor_state[1]
        return motor_velocities
    
    def _get_motor_positions(self) -> np.ndarray:
        motor_positions = np.zeros((12,), dtype=np.float64)
        for i, motor in zip(range(12), self._motor_dict.keys()):
            motor_state = self._pybullet_client.getJointState(self.robot, 
                                                       self._motor_dict[motor])
            motor_positions[i] = motor_state[0]
        return motor_positions
    
    def _get_feet_contacts(self) -> np.ndarray:
        self._pybullet_client.performCollisionDetection()
        contacts = np.zeros((4,), dtype=bool)
        for i, foot, in zip(range(4), self._foot_dict.keys()):
            contact_points = self._pybullet_client.getContactPoints(
                bodyA=self.robot,
                linkIndexA=self._foot_dict[foot]
            )
            for contact in contact_points:
                if contact[1] != contact[2]:
                    contacts[i] = True
                    break
                else:
                    contacts[i] = False
        return contacts

    def set_link_dynamics(self):
        for i in range(self._pybullet_client.getNumJoints(self.robot)):
            self._pybullet_client.changeDynamics(
                bodyUniqueId=self.robot,
                linkIndex=i,      # This specifies the link, not the joint
                lateralFriction=0.8,       # Set lateral friction (Coulomb friction)
                rollingFriction=0.001,     # Set rolling friction
                spinningFriction=0.01,     # Spinning friction (if relevant)
                frictionAnchor=True        # Enable friction anchors for stability
            )

if __name__ == '__main__':
    client = bc.BulletClient(connection_mode=p.GUI, options="--opengl2")
    ground = client.loadURDF(f"{pybullet_data.getDataPath()}/plane.urdf") #optionally
    client.changeDynamics(
        bodyUniqueId=ground,
        linkIndex=-1,            # The plane is a single link, so use -1
        lateralFriction=1.0,     # Set a higher or lower value depending on your needs
        rollingFriction=0.001,   # Rolling friction (if relevant)
        spinningFriction=0.01    # Spinning friction (if relevant)
    )
    client.setGravity(0,0,-9.8)
    test = Sparke(pybullet_client=client)
    test._pybullet_client.setRealTimeSimulation(1)
    
    action_space = gym.spaces.Box(-np.pi, np.pi, (12,), np.float64)
    action = action_space.sample()
    print(f'Action: {action}')
    robot = test.robot

    client.addUserDebugLine([1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], 1.0, 0)

    button_id = client.addUserDebugParameter("button", 1, 0, 1)
    # test.apply_action(action)
    # print(test._get_motor_positions())
    print(test._get_feet_contacts().astype(np.float64))

    params = client.getPhysicsEngineParameters()
    for param in params:
        print(f'{param}: {params[param]}')

    low_motor_bounds = np.ndarray((12,), dtype=np.float64)
    high_motor_bounds = np.ndarray((12,), dtype=np.float64)
    for i in range(client.getNumJoints(robot)):
        joint_info = client.getJointInfo(robot, i)
        low_motor_bounds[i] = joint_info[8]
        high_motor_bounds[i] = joint_info[9]

    motor_slider_ids = []
    for i in range(12):
        motor_slider_ids.append(client.addUserDebugParameter(f"motor_{i}_slider",low_motor_bounds[i],high_motor_bounds[i],
                                     0.0))

    test.reset(reload_urdf=False)

    motor_cmds = np.ndarray((12,), dtype=np.float64)
    new_value = client.readUserDebugParameter(button_id)
    value = new_value
    try:
        while True:
            for i in range(12):
                motor_cmds[i] = client.readUserDebugParameter(motor_slider_ids[i])
            test.apply_action(motor_cmds=motor_cmds)
            new_value = client.readUserDebugParameter(button_id)
            # test.apply_action(np.full((12,), -0.743689, dtype=np.float64))
            if value != new_value:
                value = new_value
                # print(test._get_motor_positions())
                print(test._get_feet_contacts())
                print(test._foot_dict)
                print(client.getJointInfo(robot, 0))
                print(client.getBasePositionAndOrientation(robot))
                print(test._get_motor_positions())
                test.reset(reload_urdf=False)
                
            # test.apply_action(action)
            pass
    except KeyboardInterrupt:
        pass
    finally:
        test._pybullet_client.disconnect()