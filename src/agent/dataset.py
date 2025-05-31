import tensorflow as tf
import numpy as np
from PIL import Image
from src.data.dataset import prepare_b2d_dataset
from src.utils.monitor import log_execution_time
from torch.utils.data import DataLoader


tf.config.set_visible_devices([], "GPU")

class Bench2DriveDataset:
    @log_execution_time()
    def __init__(self, config):
        dataset = prepare_b2d_dataset(
            work_dir=config.work_dir,
            statistics_path=config.statistics_path,
            use_fixed_images=config.use_fixed_images,
            num_of_action_experts=config.num_of_action_experts,
            split=config.get("split", None)
        )
        self.dataset = dataset


if __name__ == '__main__':
    from omegaconf import OmegaConf
    fixed_camera_config = OmegaConf.load("config/test_dataloader/fixed_camera.yaml")
    dynamic_camera_train_config = OmegaConf.load("config/test_dataloader/dynamic_camera_train.yaml")
    dynamic_camera_eval_config = OmegaConf.load("config/test_dataloader/dynamic_camera_eval.yaml")
    fixed_camera_dataset = Bench2DriveDataset(fixed_camera_config.data.train).dataset
    dynamic_camera_train_dataset = Bench2DriveDataset(dynamic_camera_train_config.data.train).dataset
    dynamic_camera_eval_dataset = Bench2DriveDataset(dynamic_camera_eval_config.data.val).dataset
    
    fixed_camera_dataloader = DataLoader(fixed_camera_dataset, batch_size=1, shuffle=False)
    dynamic_camera_train_dataloader = DataLoader(dynamic_camera_train_dataset, batch_size=1, shuffle=False)
    dynamic_camera_eval_dataloader = DataLoader(dynamic_camera_eval_dataset, batch_size=1, shuffle=False)
    print('-----------------------fixed camera data-----------------------')
    for batch in fixed_camera_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        print('cam_id: ', batch['cam_id'])
        break
    print('-----------------------dynamic camera data-----------------------')
    print('train')
    for batch in dynamic_camera_train_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        break
    print('eval')
    for batch in dynamic_camera_eval_dataloader:
        print('state_shape: ', batch['state'].shape)
        print('trajectory_shape: ', batch['trajectory'].shape)
        break