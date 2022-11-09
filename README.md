## Instructions

In order to import modules, clone repo into your Colab notebook (`!git clone https://github.com/cprimel/cautious-fiesta.git`), then add the directory to your system path:

```python
import sys
sys.path.insert(0, '/content/cautious-fiesta')
```

Now, one can simply import the needed modules:

```python
from train import Trainer

trainer = Trainer(  # a bunch of args ...)
```

Training, testing and evaluation can also be run as stand alone scripts (TODO).