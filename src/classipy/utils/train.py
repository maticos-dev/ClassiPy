from collections import defaultdict
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from keras.datasets import mnist
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def load_mnist_data():
    data = mnist.load_data()

    digits = np.concatenate((data[0][0], data[1][0]))
    targets = np.concatenate((data[0][1], data[1][1]))

    images = digits[:30000].reshape(30000, 28 * 28)
    X = binarize_images(images, 30000)

    Y = targets[:30000]

    return X, Y


def binarize_images(X, length) -> np.ndarray:
    ret = np.where(X > 150, 255, 0)
    return ret.reshape((length, 784))


class DataBuilder(Dataset):
    def __init__(self, X, Y, device: Union[torch.device, str]):
        self.x = X.to(device)
        self.y = Y.to(device)
        self.len = torch.squeeze(self.x).shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class Trainer:
    def __init__(
        self,
        data,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        self.data = DataLoader(data)
        self.model = model

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"{type(optimizer).__name__} is not a valid PyTorch optimizer"
            )
        self.optimizer = optimizer

        self.criterion = criterion

    def train(self, num_epochs: int = 20) -> dict:
        """
        Train Model Given Loss Function and Optimizer
        """

        training_history = defaultdict(dict)

        for epoch in range(num_epochs):
            losses = []
            accuracies = []
            for images, labels in self.data:
                images = torch.squeeze(images)
                labels = torch.squeeze(labels)

                assert images.shape == torch.Size([128, 784]), (
                    f"Images [{images.shape}] not of desired dimensions."
                )
                assert labels.shape == torch.Size([128]), (
                    f"Labels [{labels.shape}] not of desired dimensions."
                )

                predictions = self.model(images)

                loss = self.criterion(predictions, labels)

                self.optimizer.zero_grad(set_to_none=True)  # reset gradients
                # compute gradients for all 128 samples
                loss.backward()

                # apply weight update and pass loss
                self.optimizer.step()

            self.store_training_history(
                history=training_history,
                epoch_num=epoch,
                loss=losses,
                accuracy=accuracies,
            )

            with torch.no_grad():
                print(
                    f"Completed Epoch: {epoch + 1}/{self.epochs}, Loss: {np.mean(losses):.4f}"
                )
        return training_history
