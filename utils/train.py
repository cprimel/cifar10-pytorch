# TODO: script for training model

class Trainer:
    def __init__(self, num_epochs, optimizer, *kwargs):
        self.num_epochs = num_epochs
        self.opt = optimizer

    def summary(self):
        print(f"Trainer initialized with {self.num_epochs}")
