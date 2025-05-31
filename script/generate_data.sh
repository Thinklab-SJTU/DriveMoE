#!/bin/bash

DATASET_PATH="$DATA_DIR"
WORK_DIR="${PWD}"
CAM_ID_PATH="${PWD}/camera_labels"

python "src/data_processing/generate_data.py" --dataset_path "$DATASET_PATH" --cam_id_path "$CAM_ID_PATH" --work_dir "$WORK_DIR"