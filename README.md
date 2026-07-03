<div align="center">

# DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://thinklab-sjtu.github.io/DriveMoE/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.16278-b31b1b.svg)](https://arxiv.org/abs/2505.16278)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/rethinklab/DriveMoE)

</div>

Welcome to the official implementation of DriveMoE. Follow the instructions below to set up your environment, prepare data, and start training.

## 🛠️ Installation

### 1. Prerequisites
- CUDA Version: $\ge$ 12.1 (Required)
- Python: 3.10+ (Recommended)

### 2. Setup
Clone the repository and install the dependencies in editable mode:
```bash
git clone https://github.com/Thinklab-SJTU/DriveMoE.git
cd DriveMoE
conda create -n drivemoe python=3.10
conda activate drivemoe
pip install -e .
```

## 📂 Data & Checkpoint Preparation

### 1. Download Assets
Please download the following components to their respective directories:
- Datasets: Download the [Bench2Drive Dataset](https://huggingface.co/datasets/rethinklab/Bench2Drive) and [Camera Labels and Scenario Labels](https://huggingface.co/rethinklab/DriveMoE) into the data/ directory.
- Model Weights: Download the [PaliGemma pre-trained weights](https://huggingface.co/google/paligemma-3b-pt-224) into the ckpt/ directory.

### 2.Preprocessing
After organizing the files, run the preprocessing script to prepare the training data:
```bash
bash script/generate_data.sh
```
For OPEN-LOOP evaluation (testing with ground truth history), you must be set HORIZON_SIZE to 20 to ensure fair comparison with baseline methods. For CLOSED-LOOP evaluation (testing with predicted history), it can be set to any value based on your requirements. 

💡 **Note:** Changing `HORIZON_SIZE` requires clearing the previous experiment cache. Simply run: `rm -r exp/b2d_action`

## 🚄 Training & Evaluation

### Start Training
Launch the training process using the provided script:
```bash
bash script/training/train_drivepi0_closed_loop.sh          # train drivepi0
bash script/training/train_drivemoe_stage1_closed_loop.sh   # train drivemoe stage1
bash script/training/train_drivemoe_stage2_closed_loop.sh   # train drivemoe stage2
```

### Evaluation
We support both Open-loop and Closed-loop evaluations. For detailed evaluation steps, please refer to [Evaluation on Bench2Drive](docs/evaluation.md).

## Acknowledgments
This project has been developed based on the following pioneering works on GitHub repositories. We express our profound gratitude for these foundational resources:
- https://github.com/allenzren/open-pi-zero
- https://github.com/Physical-Intelligence/openpi

## Citation <a name="citation"></a>

```bibtex
@InProceedings{Yang_2026_CVPR,
    author    = {Yang, Zhenjie and Chai, Yilin and Jia, Xiaosong and Li, Qifeng and Shao, Yuqian and Zhu, Xuekai and Su, Haisheng and Yan, Junchi},
    title     = {DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {10678-10688}
}
```