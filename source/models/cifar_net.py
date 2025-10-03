from torch import nn

class ColorNet(nn.Module):
    """A novel, simple CNN for CIFAR-10."""
    def __init__(self, output_dim=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 32x16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)