import torch.nn as nn
import torchex.nn as exnn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 180, 3),
            nn.ReLU(),
            nn.Conv2d(180, 180, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(180, 180, 3),
            nn.ReLU(),
            nn.Conv2d(180, 180, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(180, 180, 2),
            nn.ReLU(),
            nn.Conv2d(180, 180, 1),
            nn.ReLU(),
            exnn.Flatten(),
            exnn.Linear(2048),
            nn.ReLU(),
            exnn.Linear(2048),
            nn.ReLU(),
            exnn.Linear(100))
            
    def forward(self, x):
        return self.net(x)

