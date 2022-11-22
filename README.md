# PyTorch + CIFAR-10 scripts

The repository contains a set of extensible scripts for training and evaluating PyTorch models on the CIFAR-10 image
dataset. Heavily inspired by [timm](https://github.com/rwightman/pytorch-image-models).

Currently available architectures:

* [ResNet](https://arxiv.org/abs/1512.03385)
* [ResNet/S](https://arxiv.org/abs/1512.03385)
* [ResNeXt](https://arxiv.org/abs/1611.05431)
* [ConvMixer](https://openreview.net/forum?id=TVHS5Y4dNvM)

The scripts can be extended by adding different architectures, optimization algorithms, learning rate schedulers, and
data augmentations.

## Instructions

All scripts can be run from the command line. The minimum arguments for each can be seen below. Whether you are running
the script locally or in a hosted Jupyter notebook, all you need to do is clone the repository and run the script:

```bash
git clone https://github.com/cprimel/cautious-fiesta.git
cd cautious-fiesta && python train.py --config experiments/convmixer256_8_k5_p2_00.yml --batch-size=512
```

To evaluate a model, similarly making sure to specify the correct model and experiment identifiers:

```bash
cd cautious-fiesta && python test.py --model=convmixer256_8_k5_p2 --experiment=convmixer256_8_k5_p2_00 --checkpoint=<path-to-checkpoint> --logs=<directory-for-log-output>
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

* Refactor `CutMix` operation into its own function and integrate into dataloader+transform pipeline.
* Add ability to feed config file to `test.py`
* Modify `model_registry` so models for available architectures can be fully defined from the command line
* Add option for ShakeDrop regularization
* Add options for weight initialization
* Add option for gradient clipping
* Add support for mixed precision operations (`torch.amp`)
* Add support for exporting ONXX models from `summarize.py`
