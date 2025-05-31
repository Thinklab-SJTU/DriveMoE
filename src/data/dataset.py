from typing import Sequence, Optional
import os
import torch
import pickle
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from torch.utils.data import Dataset
from src.data.utils.image import read_resize_encode_image_pytorch, concatenate_images, image_normalization
from src.data.utils.normalization import Normalize
from src.data.camera_scenario_map import get_scenario_id, get_cam_id_by_name, get_cam_id_by_label

class TorchDataset(Dataset):
    def __init__(self, data_path: str, statistics_path: str, use_fixed_images: bool, num_of_action_experts: int):
        super(TorchDataset, self).__init__()
        self.data_path = data_path
        self.statistics_path = statistics_path
        self.use_fixed_images = use_fixed_images
        self.num_of_action_experts = num_of_action_experts
        self.list_data_dict = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]
    
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        data_dict = {}
        pkl_data_path = os.path.join(self.data_path, self.list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            normalize = Normalize.get_instance(self.statistics_path)
            state = normalize.prepare_state(all_data)
            data_dict['state'] = state
            if not self.use_fixed_images:
                image_front_path = all_data['his_cam_front'].numpy().decode('utf-8')
                data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                image_back_path = all_data['his_cam_back'].numpy().decode('utf-8')
                data_dict['image_back'] = image_normalization(image_back_path, (224, 224))
                image_front_time_path = all_data['his_cam_front_time'].numpy().decode('utf-8')
                data_dict['image_front_time'] = image_normalization(image_front_time_path, (224, 224))
                image_front_left_path = all_data['his_cam_front_left'].numpy().decode('utf-8')
                data_dict['image_front_left'] = image_normalization(image_front_left_path, (224, 224))
                image_front_right_path = all_data['his_cam_front_right'].numpy().decode('utf-8')
                data_dict['image_front_right']= image_normalization(image_front_right_path, (224, 224))
                image_back_left_path = all_data['his_cam_back_left'].numpy().decode('utf-8')
                data_dict['image_back_left'] = image_normalization(image_back_left_path, (224, 224))
                image_back_right_path = all_data['his_cam_back_right'].numpy().decode('utf-8')
                data_dict['image_back_right']= image_normalization(image_back_right_path, (224, 224))
                data_dict['cam_id'] = get_cam_id_by_label(all_data['cam_id'].numpy().decode('utf-8'))
            
            else:
                image_front_path = all_data['his_cam_front'].numpy().decode('utf-8')
                data_dict['image_front'] = image_normalization(image_front_path, (224, 224))
                image_front_time_path = all_data['his_cam_front_time'].numpy().decode('utf-8')
                data_dict['image_front_time'] = image_normalization(image_front_time_path, (224, 224))
                skill_cam_path = all_data['skill_cam'].numpy().decode('utf-8')
                data_dict['skill_cam'] = image_normalization(skill_cam_path, (224, 224))
                cam_id = pkl_data_path.split('/')[-1].split('_')[-1]
                cam_id = cam_id.split('.')[0]
                cam_id = get_cam_id_by_name(cam_id)
                data_dict['cam_id'] = cam_id
            
            trajectory = normalize.prepare_traj(all_data)
            data_dict['trajectory'] = trajectory
            scenario_id = pkl_data_path.split('/')[-1].split('_')[0]
            scenario_id = get_scenario_id(self.num_of_action_experts, scenario_id)
            data_dict['scenario_id'] = scenario_id
            data_dict['language_instruction'] = 'predict trajectory'
        return data_dict

def prepare_b2d_dataset(
    work_dir: str,
    statistics_path: str,
    use_fixed_images: bool,
    num_of_action_experts: int,
    split: Optional[str] = None
):  
    if use_fixed_images:
        data_path = os.path.join(work_dir, 'b2d_fixed_camera')
        return TorchDataset(data_path=data_path, statistics_path=statistics_path, use_fixed_images=True, num_of_action_experts=num_of_action_experts)
    else:
        data_path = os.path.join(work_dir, 'b2d_dynamic_camera')
        if split == 'train':
            train_data_path = os.path.join(data_path, 'train')
            return TorchDataset(data_path=train_data_path, statistics_path=statistics_path, use_fixed_images=False, num_of_action_experts=num_of_action_experts)
        elif split == 'val':
            val_data_path = os.path.join(data_path, 'val')
            return TorchDataset(data_path=val_data_path, statistics_path=statistics_path, use_fixed_images=False, num_of_action_experts=num_of_action_experts)
        else:
            raise ValueError(f"split type must be 'train' or 'val'")