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