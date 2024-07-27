import numpy as np
import pybullet as p
from pybullet_utils import bullet_client as bc

REVOLUTE_JOINT = 0

def create_motor_dict(client, robot_id) -> dict:
    motor_dict = {}
    for i in range(client.getNumJoints(robot_id)):
        joint_info = client.getJointInfo(robot_id, i)
        if joint_info[2] == REVOLUTE_JOINT:
            motor_name = joint_info[12].decode(encoding='utf-8')
            motor_name = motor_name.rsplit('_')
            motor_name = f'{motor_name[0][0]}{motor_name[1][0]}_{motor_name[2]}'
            motor_dict[motor_name] = i
    return motor_dict

def create_foot_dict(client, robot_id) -> dict:
    foot_dict = {}
    for i in range(client.getNumJoints(robot_id)):
        joint_info = client.getJointInfo(robot_id, i)
        joint_name = joint_info[12].decode(encoding='utf-8')
        if "foot" in joint_name:
            joint_name = joint_name.rsplit('_')
            joint_name = f'{joint_name[0][0]}{joint_name[1][0]}_{joint_name[2]}'
            foot_dict[joint_name] = i
    return foot_dict