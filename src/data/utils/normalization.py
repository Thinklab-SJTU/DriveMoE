import numpy as np
import torch
import json

class Normalize():
    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    def __init__(self, percentile_path):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.percentile_path = percentile_path
        # load static
        with open(self.percentile_path, 'r') as f:
            loaded_data = json.load(f)
            self.speed_01 = loaded_data['speed'][0]
            self.speed_99 = loaded_data['speed'][1]
            self.acceleration_01 = loaded_data['acceleration'][0]
            self.acceleration_99 = loaded_data['acceleration'][1]
            self.angular_velocity_01 = loaded_data['angular_velocity'][0]
            self.angular_velocity_99 = loaded_data['angular_velocity'][1]
            self.theta_01 = loaded_data['theta'][0]
            self.theta_99 = loaded_data['theta'][1]
            self.fur_x_01 = loaded_data['fur_x'][0]
            self.fur_x_99 = loaded_data['fur_x'][1]
            self.fur_y_01 = loaded_data['fur_y'][0]
            self.fur_y_99 = loaded_data['fur_y'][1]
            self.target_far_x_01 = loaded_data['command_far_x'][0]
            self.target_far_x_99 = loaded_data['command_far_x'][1]
            self.target_far_y_01 = loaded_data['command_far_y'][0]
            self.target_far_y_99 = loaded_data['command_far_y'][1]
            self.target_near_x_01 = loaded_data['command_near_x'][0]
            self.target_near_x_99 = loaded_data['command_near_x'][1]
            self.target_near_y_01 = loaded_data['command_near_y'][0]
            self.target_near_y_99 = loaded_data['command_near_y'][1]

    def standard_normalize(self, true_np_array, percentile_01, percentile_99):
        return 2 * (true_np_array - percentile_01) / (percentile_99 - percentile_01) - 1
    
    def recover_trajectory(self, normalized_np_array, percentile_01, percentile_99):
        return (normalized_np_array + 1) * (percentile_99 - percentile_01) / 2 + percentile_01

    def prepare_state(self, all_data):
        his_speed = self.standard_normalize(all_data['his_speed'], self.speed_01, self.speed_99)
        his_acceleration = self.standard_normalize(all_data['his_acceleration'], self.acceleration_01, self.acceleration_99)
        his_angular_velocity = self.standard_normalize(all_data['his_angular_velocity'], self.angular_velocity_01, self.angular_velocity_99)
        his_theta = self.standard_normalize(all_data['his_theta'], self.theta_01, self.theta_99)
        target_far_x = self.standard_normalize(all_data['x_command_far'], self.target_far_x_01, self.target_far_x_99)
        target_far_y = self.standard_normalize(all_data['y_command_far'], self.target_far_y_01, self.target_far_y_99)
        state = torch.tensor(np.concatenate((
                    his_speed[:, np.newaxis],
                    his_acceleration,
                    his_angular_velocity,
                    his_theta[:, np.newaxis],
                    target_far_x[:, np.newaxis],
                    target_far_y[:, np.newaxis],
                ), axis=1))
        return state

    def prepare_traj(self, all_data):
        fur_x = self.standard_normalize(all_data['fur_x'], self.fur_x_01, self.fur_x_99)
        fur_y = self.standard_normalize(all_data['fur_y'], self.fur_y_01, self.fur_y_99)
        trajectory = torch.tensor(np.concatenate((fur_x[:, :, np.newaxis], fur_y[:, :, np.newaxis]), axis=2))
        return trajectory
    
    def infer_traj(self, result):
        pred_x = self.recover_trajectory(result[0, :, 0], self.fur_x_01, self.fur_x_99)
        pred_y = self.recover_trajectory(result[0, :, 1], self.fur_y_01, self.fur_y_99)
        return pred_x, pred_y