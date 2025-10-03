from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(name, batch_size):
    """Factory function to load and prepare a dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    if name.lower() == 'mnist':
        train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
        output_dim = 10
    elif name.lower() == 'cifar10':
        train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
        output_dim = 10
    else:
        raise NotImplementedError(f"Dataset '{name}' not supported.")
        
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, output_dim