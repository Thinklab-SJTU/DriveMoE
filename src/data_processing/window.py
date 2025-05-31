import os
import json
import time
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf
from tqdm import *
import argparse

class Bench2DriveGenerater:
    def __init__(self, work_dir, is_train, window_size, horizon):
        if is_train:
            self.is_train = True
            self.fixed_camera_save_path = os.path.join(work_dir, 'b2d_fixed_camera')
            os.makedirs(self.fixed_camera_save_path, exist_ok=True)
            self.dynamic_camera_save_path = os.path.join(work_dir, 'b2d_dynamic_camera', 'train')
            self.pkl_file_path = os.path.join(work_dir, 'b2d_dynamic_camera', 'train', 'train_data.pkl')
        else:
            self.is_train = False
            self.dynamic_camera_save_path = os.path.join(work_dir, 'b2d_dynamic_camera', 'val')
            self.pkl_file_path = os.path.join(work_dir, 'b2d_dynamic_camera', 'val', 'val.pkl')
        with open(self.pkl_file_path, 'rb') as f:
            all_episode_data = pickle.load(f)
        self.data = self._get_data(all_episode_data, window_size=window_size, horizon=horizon)
        self.dataset_size = 0
        os.remove(self.pkl_file_path)

    def _get_data(self, all_episode_data, window_size, horizon):
        all_data = []
        for episode_data in all_episode_data:
            all_data += self._split_data(episode_data, window_size=window_size, horizon=horizon)
        return all_data

    def _split_data(self, episode_data, window_size=5, horizon=10):
        episode_x_list, \
        episode_y_list, \
        episode_x_pure_list, \
        episode_y_pure_list, \
        episode_theta_list, \
        episode_speed_list, \
        episode_x_command_far_list, \
        episode_y_command_far_list, \
        episode_command_far_list, \
        episode_x_command_near_list, \
        episode_y_command_near_list, \
        episode_command_near_list, \
        episode_x_target_list, \
        episode_y_target_list, \
        episode_acceleration_list, \
        episode_angular_velocity_list, \
        episode_action_list, \
        episode_image_front_list, \
        episode_image_front_left_list, \
        episode_image_front_right_list, \
        episode_image_back_list, \
        episode_image_back_left_list, \
        episode_image_back_right_list, \
        cam_id_list, \
        scenario_id, \
        town_id, \
        route_id, \
        weather_id = episode_data

        def state_window(traj: list, window_size: int):
            traj_len = len(traj)
            history_indices = tf.range(traj_len)[:, None] + tf.range(-window_size + 1, 1)  # [traj_len, window_size]
            history_indices = tf.maximum(history_indices, 0)
            result = tf.gather(traj, history_indices) # [traj_len, window_size, state_dim]
            return result

        def trajectory_window(traj: list, window_size: int, horizon: int):
            traj_len = len(traj)
            true_len = traj_len - horizon + 1
            # chunk observations into histories
            history_indices = tf.range(traj_len)[:, None] + tf.range(-window_size + 1, 1)  # [traj_len, window_size]
            # indicates which observations at the beginning of the trajectory are padding
            timestep_pad_mask = history_indices >= 0
            # repeat the first observation at the beginning of the trajectory rather than going out of bounds
            history_indices = tf.maximum(history_indices, 0)

            # first, chunk actions into `action_horizon` current + future actions
            action_chunk_indices = tf.range(traj_len)[:, None] + tf.range(horizon)  # [traj_len, action_horizon]
            # repeat the last action at the end of the trajectory rather than going out of bounds
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            # gather
            traj = tf.gather(traj, action_chunk_indices)  # [traj_len, action_horizon, action_dim]

            # then, add the history axis to actions
            traj = tf.gather(traj, history_indices)  # [traj_len, window_size, action_horizon, action_dim]
            traj = traj[:true_len]
            return traj

        tmp_x_list = episode_x_pure_list
        tmp_y_list = episode_y_pure_list
        episode_x_pure_list = state_window(episode_x_pure_list, window_size=window_size)
        episode_y_pure_list = state_window(episode_y_pure_list, window_size=window_size)

        trajectory_x_list = trajectory_window(tmp_x_list, window_size=1, horizon=horizon)
        trajectory_y_list = trajectory_window(tmp_y_list, window_size=1, horizon=horizon)

        episode_theta_list = state_window(episode_theta_list, window_size=window_size)
        episode_speed_list = state_window(episode_speed_list, window_size=window_size)
        episode_x_command_far_list = state_window(episode_x_command_far_list, window_size=window_size)
        episode_y_command_far_list = state_window(episode_y_command_far_list, window_size=window_size)
        episode_command_far_list = state_window(episode_command_far_list, window_size=window_size)
        episode_x_command_near_list = state_window(episode_x_command_near_list, window_size=window_size)
        episode_y_command_near_list = state_window(episode_y_command_near_list, window_size=window_size)
        episode_command_near_list = state_window(episode_command_near_list, window_size=window_size)
        episode_x_target_list = state_window(episode_x_target_list, window_size=window_size)
        episode_y_target_list = state_window(episode_y_target_list, window_size=window_size)
        episode_acceleration_list = state_window(episode_acceleration_list, window_size=window_size)
        episode_angular_velocity_list = state_window(episode_angular_velocity_list, window_size=window_size)
        episode_action_list = state_window(episode_action_list, window_size=window_size)
        episode_image_front_list = state_window(episode_image_front_list, window_size=window_size)
        episode_image_front_left_list = state_window(episode_image_front_left_list, window_size=window_size)
        episode_image_front_right_list = state_window(episode_image_front_right_list, window_size=window_size)
        episode_image_back_list = state_window(episode_image_back_list, window_size=window_size)
        episode_image_back_left_list = state_window(episode_image_back_left_list, window_size=window_size)
        episode_image_back_right_list = state_window(episode_image_back_right_list, window_size=window_size)
        cam_id_list = state_window(cam_id_list, window_size=window_size)

        split_data = []
        for i in range(len(trajectory_x_list)):
            tmp = {}
            his_action = episode_action_list[i]
            his_pure_x = episode_x_pure_list[i]
            his_pure_y = episode_y_pure_list[i]
            his_cam_front = episode_image_front_list[i]
            his_cam_front_left = episode_image_front_left_list[i]
            his_cam_front_right = episode_image_front_right_list[i]
            his_cam_back = episode_image_back_list[i]
            his_cam_back_left = episode_image_back_left_list[i]
            his_cam_back_right = episode_image_back_right_list[i]
            his_acceleration = episode_acceleration_list[i]
            his_angular_velocity = episode_angular_velocity_list[i]
            his_speed = episode_speed_list[i]
            his_theta = episode_theta_list[i]
            x_command_far = episode_x_command_far_list[i]
            y_command_far = episode_y_command_far_list[i]
            command_far = episode_command_far_list[i]
            x_command_near = episode_x_command_near_list[i]
            y_command_near = episode_y_command_near_list[i]
            command_near = episode_command_near_list[i]
            fur_x = trajectory_x_list[i]
            fur_y = trajectory_y_list[i]
            cam_id = cam_id_list[i]

            tmp['his_action'] = his_action
            tmp['his_speed'] = his_speed
            tmp['his_acceleration'] = his_acceleration
            tmp['his_angular_velocity'] = his_angular_velocity
            tmp['his_theta'] = his_theta
            tmp['his_pure_x'] = his_pure_x
            tmp['his_pure_y'] = his_pure_y
            tmp['x_command_near'] = x_command_near
            tmp['y_command_near'] = y_command_near
            tmp['command_near'] = command_near
            tmp['x_command_far'] = x_command_far
            tmp['y_command_far'] = y_command_far
            tmp['command_far'] = command_far
            tmp['his_cam_front'] = his_cam_front
            tmp['his_cam_front_left'] = his_cam_front_left
            tmp['his_cam_front_right'] = his_cam_front_right
            tmp['his_cam_back'] = his_cam_back
            tmp['his_cam_back_left'] = his_cam_back_left
            tmp['his_cam_back_right'] = his_cam_back_right
            tmp['fur_x'] = fur_x
            tmp['fur_y'] = fur_y
            tmp['cam_id'] = cam_id

            tmp['step'] = f'step{i}'
            tmp['scenario_id'] = scenario_id
            tmp['town_id'] = town_id
            tmp['route_id'] = route_id
            tmp['weather_id'] = weather_id
            split_data.append(tmp)

        return split_data

    def world2ego(self, ego_theta, ego_x, ego_y, point_x, point_y):
        R = np.array([
            [np.cos(ego_theta), np.sin(ego_theta)],
            [-np.sin(ego_theta),  np.cos(ego_theta)]
            ])

        point_in_ego = np.array([(point_x-ego_x), point_y-ego_y])
        point_in_ego = R.dot(point_in_ego)
        return point_in_ego

    def action_one_hot(self, action):
        action_one_hot = [0] * 39
        action_one_hot[action] = 1
        return action_one_hot

    def _process_data(self, index):
        idx_data = self.data[index]
        dynamic_camera_data = {}
        
        # ego info
        ego_x = idx_data['his_pure_x'][-1]
        ego_y = idx_data['his_pure_y'][-1]

        dynamic_camera_data['his_theta'] = [theta - np.pi/2 if not np.isnan(theta) else 0 for theta in idx_data['his_theta']]
        ego_theta = dynamic_camera_data['his_theta'][-1]
        dynamic_camera_data['his_theta'] = np.array([ta - ego_theta for ta in dynamic_camera_data['his_theta']], dtype=np.float32)
        
        xy_comand_far = [self.world2ego(ego_theta, ego_x, ego_y, point_x, point_y) for (point_x, point_y) in zip(idx_data['x_command_far'], idx_data['y_command_far'])]
        dynamic_camera_data['x_command_far'] = np.array([x for (x,_) in xy_comand_far], dtype=np.float32)
        dynamic_camera_data['y_command_far'] = np.array([y for (_,y) in xy_comand_far], dtype=np.float32)
        
        xy_comand_near = [self.world2ego(ego_theta, ego_x, ego_y, point_x, point_y) for (point_x, point_y) in zip(idx_data['x_command_near'], idx_data['y_command_near'])]
        dynamic_camera_data['x_command_near'] = np.array([x for (x,_) in xy_comand_near], dtype=np.float32)
        dynamic_camera_data['y_command_near'] = np.array([y for (_,y) in xy_comand_near], dtype=np.float32)

        fur_x = []
        fur_y = []
        for i in range(len(idx_data['fur_x'])):
            tmp_x = []
            tmp_y = []
            for j in range(len(idx_data['fur_x'][i])):
                tmp_xy = self.world2ego(ego_theta, ego_x, ego_y, idx_data['fur_x'][i][j], idx_data['fur_y'][i][j])
                tmp_x.append(tmp_xy[0].tolist())
                tmp_y.append(tmp_xy[1].tolist())
            fur_x.append(tmp_x)
            fur_y.append(tmp_y)

        dynamic_camera_data['fur_x'] = np.array(fur_x, dtype=np.float32)
        dynamic_camera_data['fur_y'] = np.array(fur_y, dtype=np.float32)
        dynamic_camera_data['his_action'] = np.array([self.action_one_hot(int(act)) for act in idx_data['his_action']], dtype=np.uint8)

        dynamic_camera_data['his_speed'] = np.array(idx_data['his_speed'], dtype=np.float32)
        dynamic_camera_data['his_acceleration'] = np.array(idx_data['his_acceleration'], dtype=np.float32)
        dynamic_camera_data['his_angular_velocity'] = np.array(idx_data['his_angular_velocity'], dtype=np.float32)
        dynamic_camera_data["command_near"] = np.array(idx_data['command_near'], dtype=np.float32)
        dynamic_camera_data["command_far"] = np.array(idx_data['command_far'], dtype=np.float32)

        fixed_camera_data_front_left = dynamic_camera_data
        fixed_camera_data_front_right = dynamic_camera_data
        fixed_camera_data_back = dynamic_camera_data
        fixed_camera_data_back_left = dynamic_camera_data
        fixed_camera_data_back_right = dynamic_camera_data

        dynamic_camera_status_pkl = f"{idx_data['scenario_id']}_{idx_data['town_id']}_{idx_data['route_id']}_{idx_data['weather_id']}_{idx_data['step']}"
        dynamic_camera_data['status'] = dynamic_camera_status_pkl

        # 7 cameras in total
        dynamic_camera_data['his_cam_front'] = idx_data['his_cam_front'][-1]
        dynamic_camera_data['his_cam_front_time'] = idx_data['his_cam_front'][-2]
        dynamic_camera_data['his_cam_front_left'] = idx_data['his_cam_front_left'][-1]
        dynamic_camera_data['his_cam_front_right'] = idx_data['his_cam_front_right'][-1]
        dynamic_camera_data['his_cam_back'] = idx_data['his_cam_back'][-1]
        dynamic_camera_data['his_cam_back_time'] = idx_data['his_cam_back'][-2]
        dynamic_camera_data['his_cam_back_left'] = idx_data['his_cam_back_left'][-1]
        dynamic_camera_data['his_cam_back_right'] = idx_data['his_cam_back_right'][-1]
        dynamic_camera_data['cam_id'] = idx_data['cam_id'][-1]

        fixed_camera_data_front_left['his_cam_front'] = idx_data['his_cam_front'][-1]
        fixed_camera_data_front_left['his_cam_front_time'] = idx_data['his_cam_front'][-2]
        fixed_camera_data_front_left['skill_cam'] = idx_data['his_cam_front_left'][-1]
        fixed_camera_data_front_left_status_pkl = f"{idx_data['scenario_id']}_{idx_data['town_id']}_{idx_data['route_id']}_{idx_data['weather_id']}_{idx_data['step']}_frontleft"
        fixed_camera_data_front_left['status'] = fixed_camera_data_front_left_status_pkl

        fixed_camera_data_front_right['his_cam_front'] = idx_data['his_cam_front'][-1]
        fixed_camera_data_front_right['his_cam_front_time'] = idx_data['his_cam_front'][-2]
        fixed_camera_data_front_right['skill_cam'] = idx_data['his_cam_front_right'][-1]
        fixed_camera_data_front_right_status_pkl = f"{idx_data['scenario_id']}_{idx_data['town_id']}_{idx_data['route_id']}_{idx_data['weather_id']}_{idx_data['step']}_frontright"
        fixed_camera_data_front_right['status'] = fixed_camera_data_front_right_status_pkl

        fixed_camera_data_back['his_cam_front'] = idx_data['his_cam_front'][-1]
        fixed_camera_data_back['his_cam_front_time'] = idx_data['his_cam_front'][-2]
        fixed_camera_data_back['skill_cam'] = idx_data['his_cam_back'][-1]
        fixed_camera_data_back_status_pkl = f"{idx_data['scenario_id']}_{idx_data['town_id']}_{idx_data['route_id']}_{idx_data['weather_id']}_{idx_data['step']}_back"
        fixed_camera_data_back['status'] = fixed_camera_data_back_status_pkl

        fixed_camera_data_back_left['his_cam_front'] = idx_data['his_cam_front'][-1]
        fixed_camera_data_back_left['his_cam_front_time'] = idx_data['his_cam_front'][-2]
        fixed_camera_data_back_left['skill_cam'] = idx_data['his_cam_back_left'][-1]
        fixed_camera_data_back_left_status_pkl = f"{idx_data['scenario_id']}_{idx_data['town_id']}_{idx_data['route_id']}_{idx_data['weather_id']}_{idx_data['step']}_backleft"
        fixed_camera_data_back_left['status'] = fixed_camera_data_back_left_status_pkl

        fixed_camera_data_back_right['his_cam_front'] = idx_data['his_cam_front'][-1]
        fixed_camera_data_back_right['his_cam_front_time'] = idx_data['his_cam_front'][-2]
        fixed_camera_data_back_right['skill_cam'] = idx_data['his_cam_back_right'][-1]
        fixed_camera_data_back_right_status_pkl = f"{idx_data['scenario_id']}_{idx_data['town_id']}_{idx_data['route_id']}_{idx_data['weather_id']}_{idx_data['step']}_backright"
        fixed_camera_data_back_right['status'] = fixed_camera_data_back_right_status_pkl

        self.dataset_size += 1
        if self.is_train:
            # save fixed camera data and dynamic camera data
            with open(self.fixed_camera_save_path+'/'+fixed_camera_data_front_left_status_pkl+'.pkl', 'wb') as f:
                pickle.dump(fixed_camera_data_front_left, f)
            with open(self.fixed_camera_save_path+'/'+fixed_camera_data_front_right_status_pkl+'.pkl', 'wb') as f:
                pickle.dump(fixed_camera_data_front_right, f)
            with open(self.fixed_camera_save_path+'/'+fixed_camera_data_back_status_pkl+'.pkl', 'wb') as f:
                pickle.dump(fixed_camera_data_back, f)
            with open(self.fixed_camera_save_path+'/'+fixed_camera_data_back_left_status_pkl+'.pkl', 'wb') as f:
                pickle.dump(fixed_camera_data_back_left, f)
            with open(self.fixed_camera_save_path+'/'+fixed_camera_data_back_right_status_pkl+'.pkl', 'wb') as f:
                pickle.dump(fixed_camera_data_back_right, f)
            with open(self.dynamic_camera_save_path+'/'+dynamic_camera_status_pkl+'.pkl', 'wb') as f:
                pickle.dump(dynamic_camera_data, f)
        else:
            # save dynamic camera data only
            with open(self.dynamic_camera_save_path+'/'+dynamic_camera_status_pkl+'.pkl', 'wb') as f:
                pickle.dump(dynamic_camera_data, f)  

    def start(self):
        for i in tqdm(range (len(self.data))):
            self._process_data(i)
        return self.dataset_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    window_size = args.window_size
    horizon = args.horizon
    work_dir = args.work_dir
    print("---------------------------train data---------------------------")
    train_data_generater = Bench2DriveGenerater(work_dir=work_dir, is_train=True, window_size=window_size, horizon=horizon)
    train_data_size = train_data_generater.start()
    print("---------------------------val data---------------------------")
    val_data_generater = Bench2DriveGenerater(work_dir=work_dir, is_train=False, window_size=window_size, horizon=horizon)
    val_data_size = val_data_generater.start()
    print('train_data_size: ', train_data_size)
    print('val_data_size: ', val_data_size)