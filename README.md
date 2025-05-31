<div align="center">   

# DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving
</div>


We are currently cleaning and organizing the code, and the publicly available part now is the data preprocessing section. Thank you for your patience in waiting for the training and inference code.

[Project Page](https://thinklab-sjtu.github.io/DriveMoE/), [Paper](https://arxiv.org/abs/2505.16278)


## Installation
Before you begin, you need to ensure that your CUDA version is greater than 12.1.

Clone this repository at your directory and run `pip install -e.` to install environment.

Download PaliGemma weights to your directory.
```console
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

If you wish to attempt training DrivePi0 and DriveMoE using the code, or to try open-loop testing with provided checkpoints, you will need to utilize the Bench2Drive dataset. You can download it here (https://huggingface.co/datasets/rethinklab/Bench2Drive).

Set environment variables `DATA_DIR` (if downloading datasets for training), `LOG_DIR`, and `WANDB_ENTITY` by running `source scripts/set_path.sh`

## Data processing
Run these two scripts to preprocess the training data.
```console
sh script/generate_data.sh && script/window.sh
```
To normalize data during training, we provide dataset statistics. You may also run `sh get_statistics.sh` to generate them.

## Citation <a name="citation"></a>

```bibtex
@misc{yang2025drivemoe,
      title={DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving}, 
      author={Zhenjie Yang and Yilin Chai and Xiaosong Jia and Qifeng Li and Yuqian Shao and Xuekai Zhu and Haisheng Su and Junchi Yan},
      year={2025},
      eprint={2505.16278},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```