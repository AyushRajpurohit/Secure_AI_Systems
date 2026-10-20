import torch
import torch.nn as nn

"""
    Shared CNN for all clean, poisoned, adversarial, and defended experiments.

    Architecture:
        Conv2d(1 -> 32, 3x3, padding=1)
        ReLU
        MaxPool2d(2)
        Conv2d(32 -> 64, 3x3, padding=1)
        ReLU
        MaxPool2d(2)
        Flatten
        Linear(64*7*7 -> 128)
        ReLU
        Linear(128 -> 10 logits)
    """


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
