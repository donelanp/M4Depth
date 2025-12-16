# UseGeo Dataset for M4Depth
This is a fork of [M4Depth](https://github.com/michael-fonder/M4Depth) which attempts to include [UseGeo](https://github.com/3DOM-FBK/UseGeo) as an additional dataset.
## Requirements
This code was developed with Python 3.7.16 leveraging the following modules
- `h5py==3.8.0`
- `matplotlib==3.5.3`
- `pandas==1.3.5`
- `Pillow==9.2.0`
- `tensorboard==2.11.2`
- `tensorflow==2.7.0`
## Preparing UseGeo
Run
```bash
python ./scripts/usegeo-split-generator.py
```
to generate the necessary data files after the UseGeo dataset has been installed. The three trajectories are split into smaller trajectories of only 10 frames each. <span style="color:red">We believe there may be an issue with how the frame-to-frame transformations are computed which is causing poor performance.</span>
## Pre-Training on MidAir
The model should be pre-trained on the MidAir dataset using the original training script
```bash
bash ./scripts/1a-train-midair.sh ./ckpt/midair
```
after modifying any of the parameters as desired.
## Validating on MidAir
A convenience script has been provided for validating the performance on the MidAir dataset. This can be run with
```bash
bash ./scripts/validate_midair.sh
```
after first modifying the array `values` to include the epochs/checkpoints of interest.
## Fine-Tuning on UseGeo
First, copy the best checkpoint from MidAir training into the UseGeo checkpoint directory
```bash
cp ./ckpt/midair/best/* ./ckpt/usegeo/train
```
Rename the checkpoint files `cp-0001.ckpt*` and update `./ckpt/usegeo/train/checkpoint` accordingly. Then, run
```bash
bash ./scripts/1c-train-usegeo.sh ./ckpt/usegeo
```
to train the model using the weights previously determined from pre-training on MidAir. <span style="color:red">Despite being a significantly smaller dataset, training on the UseGeo dataset (using the custom dataloader `./dataloaders/usegeo.py`) results in an out-of-memory (OOM) error that we have not been able to identify the source of.</span>
## Validating on UseGeo
A convenience script has been provided for validating the performance on the UseGeo dataset. This can be run with
```bash
bash ./scripts/validate_usegeo.sh
```
after first modifying the array `values` to include the epochs/checkpoints of interest.
## Plotting Results
A plot of RMSE (log) vs epoch can be generated for training and validation data with
```bash
python ./plot_loss --event-dir=./ckpt/<midair|usegeo>/summaries/train --csv=./ckpt/<midair|usegeo>/best/validation_perfs.csv
```
while plots of true vs estimated depth maps can be generated with
```bash
python ./main.py --mode=predict --dataset=<midair|usegeo> --ckpt_dir=./ckpt/<midair|usegeo> --records=./data/<midair|usegeo>/test_data

python ./plot_predictions.py --input_dir=./plots --output_dir=./plots
```