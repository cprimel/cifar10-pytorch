# PyTorch + CIFAR-10 scripts

The repository contains a set of extensible scripts for training and evaluating PyTorch models on the CIFAR-10 image
dataset. Heavily inspired by [timm](https://github.com/rwightman/pytorch-image-models).

Currently available architectures:

* ResNet
* ResNeXt
* ConvMixer

The scripts can be extended by adding different architectures, optimization algorithms, learning rate schedulers, and
data augmentations.

## Instructions

All scripts can be run from the command line. Whether you are running the script locally or a hosted Jupyter notebook,
all you need to do is clone the repository and run the script:

```bash
git clone https://github.com/cprimel/cautious-fiesta.git
cd cautious-fiesta && python train.py --config experiments/convmixer256_8_default.yml --batch-size=512
```

## Outputs

All scripts log information to standard output.

`summarize.py` with argument`--save-graph=True` outputs a standard TensorFlow `Event` protocol buffer that can be
ingested by TensorBoard for examining the models conceptual graph.

`train.py` outputs a `.yml` file containing the final arguments (e.g., values from the passed config file or, if
provided, the values from the command line), a `.json` file containing the training
log `{epoch_num: {train_loss, train_acc, val_loss, val_acc, last_lr, epoch_time}}`, and any saved checkpoints.

`test.py` outputs a `.json` file containing the evaluation metrics and the list of predicted and true
labels`{batch_index:{test_acc, predicted_labels, true_labels}}`

## TODOs

* Modify `model_registry` so models for available architectures can be fully defined from the command line
* Add option for ShakeDrop regularization
* Add options for weight initialization
* Add option for gradient clipping
* Add support for `torch.amp`
