## Instructions

In order to use, clone repo into your Colab notebook (`!git clone https://github.com/cprimel/cautious-fiesta.git`), then add the directory to your system path:

```python
import sys
sys.path.insert(0, '/content/cautious-fiesta')
```

Now, one can simply run the train script with a predefined experiment or choose the model via commandline arguments:

```jupyterpython
! cd cautious-fiesta && python train.py --config experiments/convmixer256_8_default.yml --batch-size=512
```


## TODOs
* modify `model_registry` so models can be fully defined from the command line
* Add ShakeDrop regularization
* Add CutMix regularization
* Add weight initialization options

## Experiments

All for 100 epochs.

ConvMixer256/8
1. `patch_size=2, kernel-size=5`
* `kernel_size=5` (# of parameters = 591,882), basic data augmentation (`hflip=0.5`, `scale=1.0` (for RandomResizeCrop)) (# of parameters = 591,882)
* `kernel_size=5`, basic data augmentation + ColorJitter(`jitter=0.2`), RandAugment(`ra_n=2, ra_m=12`), RandomErasing(`erase=0.2`)
* `kernel_size=9` (# of params = 706,570): basic data augmentation (`hflip=0.5`, `scale=1.0` (for RandomResizeCrop)) 
* `kernel_size=9` : basic data augmentation + ColorJitter(`jitter=0.2`), RandAugment(`ra_n=2, ra_m=12`), RandomErasing(`erase=0.2`)

ConvMixer256/16
* `kernel_size=9`, basic data augmentation + ColorJitter(`jitter=0.2`), RandAugment(`ra_n=2, ra_m=12`), RandomErasing(`erase=0.2`) (# of parameters = 1,409,034)