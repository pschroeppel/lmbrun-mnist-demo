# LMBRUN MNIST DEMO

This is a demo project for lmbrun. It trains a CNN on the MNIST dataset. It stores intermediate checkpoints and automatically resumes training from the last checkpoint. It can log to tensorboard and wandb.

## Setup

Set up a virtual environment and install dependencies:
```bash
conda create -n lmbrun-mnist-demo python=3.10
conda activate lmbrun-mnist-demo
pip install torch torchvision wandb scikit-image matplotlib tensorboard
```

## Training

Training can be started with:
```bash
python train.py --output /tmp/mnist
```

By default, the training logs to tensorboard. To log to wandb, use the `--wandb` flag.