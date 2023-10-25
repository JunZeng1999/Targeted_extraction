import torch.nn as nn
import torch


class Classifier(nn.Module):
    def __init__(self, init_weights=False):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, dilation=1, stride=1, padding=2),  # input[1, 1, 128]  output[8, 1, 128][c, h, w]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # output[8, 1, 64]
            nn.Conv1d(8, 16, kernel_size=5, dilation=1, stride=1, padding=2),  # output[16, 1, 64]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # output[16, 1, 32]
            nn.Conv1d(16, 32, kernel_size=5, dilation=1, stride=1, padding=2),  # output[32, 1, 32]
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 1 * 32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
