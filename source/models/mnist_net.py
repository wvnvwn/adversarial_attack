from torch import nn

class SourceNet(nn.Module):
    """Simple CNN for MNIST."""
    def __init__(self, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.network(x)