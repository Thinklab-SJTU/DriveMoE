#!/bin/bash

DATA_PATH="${PWD}/b2d_dynamic_camera/train"

python "src/data_processing/get_statistics.py" --data_path "$DATA_PATH"