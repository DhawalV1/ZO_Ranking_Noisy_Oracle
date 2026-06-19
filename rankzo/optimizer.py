import numpy as np


class ZORankSGD:

    def __init__(
        self,
        lr
    ):

        self.lr = lr

    def step(
        self,
        theta,
        gradient
    ):

        return (
            theta
            -
            self.lr * gradient
        )