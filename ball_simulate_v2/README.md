## Introduction

## Model Training

This script is designed for training and testing a ball simulation model using various configurations. The script provides command-line arguments to customize training parameters, such as learning rate, batch size, epochs, model type, and more.

## Usage

   ```bash
   python3 ball_simulate_v2/train.py \[args\]
   ``` 

### Command-line Arguments
- **-lr, --learning_rate**: Learning rate for the training process (default: 0.001).
- **-b, --batch_size**: Batch size for training (default: 64).
- **-e, --epochs**: Number of epochs for training (default: 30).
- **-m, --model_type**: Type of the model (default: `medium`; availible values: `small`, `medium`, `medium_var`, `big`, `large`).
- **-mom, --momentum**: Momentum for the `SGDM` optimizer (default: 0.01). **(SGDM ONLY)**
- **-d, --dataset**: Dataset name.
- **-s, --scheduler_step_size**: Step size for the learning rate scheduler (default: 0).
- **-w, --weight**: Path to the pre-trained model weight file (default: None).
- **-n, --name**: Save name for the training.
- **-o, --optimizer**: Optimizer type (default: `adam`; avalible values: `adam`, `sgdm`).
- **--num_workers**: Number of workers for data loading (default: 0). This may speed up the training. Setting to 2 is a good choice.
- **--test**: Test the model on the validation set (default: False).
- **--mode**: Training mode (`default`, `fit`, `ne`, `predict`, `normal`, `normalB`, `normalB60`, `normalBR`) (default: `normalBR`). More details can be found in [Modes](#modes) or in [core/Constants.py](/core/Constants.py)

### Examples
Make sure you run the code in the root of the project
Train the model with default settings:


## Modes
- **fit**: Set the model mode to fitting.
- **ne**: Set the model mode to no error.
- **predict**: Set the model mode to prediction.
- **normal**: Set the model mode to normal.
- **normalB**: Set the model mode to normal with additional configuration 'B'.
- **normalB60**: Set the model mode to normal with configuration 'B60'.
- **normalBR**: Set the model mode to normal with configuration 'BR'.

## Dataset
If the "ball_simulate_v2/dataset" directory does not exist, it will be created automatically.


## Dependencies
- Ensure that the required dependencies are installed before running the script.

Feel free to customize the script according to your specific use case and requirements.
