import numpy as np
from tqdm import *
import json
import pickle
import os
import argparse

def calculate(data):
    percentile_99 = np.percentile(data, 99)
    percentile_01 = np.percentile(data, 1)
    return percentile_01, percentile_99 

def flatten(data):
    return np.concatenate([arr.flatten() for arr in data])

def generate_data(data_path):
    list_data_dict = [f for f in os.listdir(data_path) if f.endswith('.pkl')]
    his_speed = []
    his_acceleration = []
    his_angular_velocity = []
    his_theta = []
    fur_x = []
    fur_y = []
    command_near_x = []
    command_near_y = []
    command_far_x = []
    command_far_y = []
    print('calcuating percentile_01 and percentile_99 ... ')
    for i in tqdm(range(len(list_data_dict))):
        pkl_data_path = os.path.join(data_path, list_data_dict[i])
        with open(pkl_data_path, 'rb') as f:
            all_data = pickle.load(f)
            his_theta.append(all_data['his_theta'])
            his_angular_velocity.append(all_data['his_angular_velocity'])
            his_acceleration.append(all_data['his_acceleration'])
            his_speed.append(all_data['his_speed'])
            fur_x.append(all_data['fur_x'])
            fur_y.append(all_data['fur_y'])
            command_far_x.append(all_data['x_command_far'])
            command_far_y.append(all_data['y_command_far'])
            command_near_x.append(all_data['x_command_near'])
            command_near_y.append(all_data['y_command_near'])
    his_speed = flatten(his_speed)
    his_acceleration = flatten(his_acceleration)
    his_angular_velocity = flatten(his_angular_velocity)
    his_theta = flatten(his_theta)
    fur_x = flatten(fur_x)
    fur_y = flatten(fur_y)
    command_far_x = flatten(command_far_x)
    command_far_y = flatten(command_far_y)
    command_near_x = flatten(command_near_x)
    command_near_y = flatten(command_near_y)
    
    speed_01, speed_99 = calculate(his_speed)
    acceleration_01, acceleration_99 = calculate(his_acceleration)
    angular_velocity_01, angular_velocity_99 = calculate(his_angular_velocity)
    theta_01, theta_99 = calculate(his_theta)
    fur_x_01, fur_x_99 = calculate(fur_x)
    fur_y_01, fur_y_99 = calculate(fur_y)
    command_far_x_01, command_far_x_99 = calculate(command_far_x)
    command_far_y_01, command_far_y_99 = calculate(command_far_y)
    command_near_x_01, command_near_x_99 = calculate(command_near_x)
    command_near_y_01, command_near_y_99 = calculate(command_near_y)

    print('speed:', speed_01, speed_99)
    print('acceleration', acceleration_01, acceleration_99)
    print('angular_velocity', angular_velocity_01, angular_velocity_99)
    print('theta:', theta_01, theta_99)
    print('fur_x:', fur_x_01, fur_x_99)
    print('fur_y:', fur_y_01, fur_y_99)
    print('command_far_x:', command_far_x_01, command_far_x_99)
    print('command_far_y:', command_far_y_01, command_far_y_99)
    print('command_near_x:', command_near_x_01, command_near_x_99)
    print('command_near_y:', command_near_y_01, command_near_y_99)
    
    data = {
        'speed': [speed_01, speed_99],
        'acceleration': [acceleration_01, acceleration_99],
        'angular_velocity': [angular_velocity_01, angular_velocity_99],
        'theta': [theta_01, theta_99],
        'fur_x': [fur_x_01, fur_x_99],
        'fur_y': [fur_y_01, fur_y_99],
        'command_near_x': [command_near_x_01, command_near_x_99],
        'command_near_y': [command_near_y_01, command_near_y_99],
        'command_far_x': [command_far_x_01, command_far_x_99],
        'command_far_y': [command_far_y_01, command_far_y_99],
    }

    with open('statistics.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    data_path = args.data_path
    generate_data(data_path)

# 5-10
# speed: -0.021534712985157967 11.078238487243652
# acceleration -7.847959995269775 10.291497230529785
# angular_velocity -0.5025675892829895 0.4206865522265426
# theta: -0.18638522624969484 0.22664238452911345
# his_x: -3.7781866717338564 0.00403969112318009
# his_y: -0.11440461911261082 0.10973326697945517
# fur_x: -0.00909316767938435 8.093738679885869
# fur_y: -1.1331280183792114 0.9688449388742447