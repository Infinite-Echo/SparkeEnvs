import numpy as np
import pybullet as p
from pybullet_utils import bullet_client as bc

def calc_gaussian_reward(max_reward, value, target_value, std_dev) -> float:
    numerator = (value - target_value)**2
    denominator = 2*(std_dev**2)
    reward = max_reward * np.exp(-(numerator/denominator))
    return reward