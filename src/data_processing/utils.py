import numpy as np
import json
import gzip
from PIL import Image
import cv2


WINDOW_HEIGHT = 900
WINDOW_WIDTH = 1600

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def format_str_num(num):
    return str(0) if -1e-5< num < 1e-5 else str(np.round(num, 2))

def format_num(num):
    return float(0) if -1e-5< num < 1e-5 else np.round(num, 2).item()

def load_json_gz(file):
    with gzip.open(file, 'rt', encoding='utf-8') as gz_file:
        anno = json.load(gz_file)
    return anno

def load_npz(file):
    return np.load(file, allow_pickle=True)['arr_0']

def load_jpg(file):
    image = Image.open(file)
    image = np.array(image)
    return image


def ego2world(ego_theta, ego_x, ego_y, point_ego_x, point_ego_y):
    R_inv = np.array([
        [np.cos(ego_theta), -np.sin(ego_theta)],
        [np.sin(ego_theta),  np.cos(ego_theta)]
    ])

    point_in_world = R_inv.dot(np.array([point_ego_x, point_ego_y]))
    
    point_in_world[0] += ego_x
    point_in_world[1] += ego_y
    
    return point_in_world

def world2ego(ego_theta, ego_x, ego_y, point_x, point_y):
    R = np.array([
        [np.cos(ego_theta), np.sin(ego_theta)],
        [-np.sin(ego_theta),  np.cos(ego_theta)]
        ])

    point_in_ego = np.array([(point_x-ego_x), point_y-ego_y])
    point_in_ego = R.dot(point_in_ego)
    return point_in_ego

def draw_dashed_line(img, start_point, end_point, color, thickness=1, dash_length=5):
    """
    Draw a dashed line on an image.
    Arguments:
    - img: The image on which to draw the dashed line.
    - start_point: The starting point of the dashed line, in the format (x, y).
    - end_point: The ending point of the dashed line, in the format (x, y).
    - color: The color of the dashed line, in the format (B, G, R).
    - thickness: The thickness of the line.
    - dash_length: The length of each dash segment in the dashed line.
    """
    # Calculate total length
    d = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
    dx = (end_point[0] - start_point[0]) / d
    dy = (end_point[1] - start_point[1]) / d

    x, y = start_point[0], start_point[1]

    while d >= dash_length:
        # Calculate the end point of the next segment
        x_end = x + dx * dash_length
        y_end = y + dy * dash_length
        cv2.line(img, (int(x), int(y)), (int(x_end), int(y_end)), color, thickness)

        # Update starting point and remaining length
        x = x_end + dx * dash_length
        y = y_end + dy * dash_length
        d -= 2 * dash_length

def point_in_canvas_wh(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_WIDTH) and (pos[1] >= 0) and (pos[1] < WINDOW_HEIGHT):
        return True
    return False


if __name__ == '__main__':
    # debug
    ego_x, ego_y = 5, 10
    ego_theta = np.pi / 4
    x = 10
    y = 20
    point_ego_x, point_ego_y = world2ego(ego_theta, ego_x, ego_y, x, y)
    point_world = ego2world(ego_theta, ego_x, ego_y, point_ego_x, point_ego_y)
    print(point_world)