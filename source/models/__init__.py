from .mnist_net import SourceNet
from .cifar_net import ColorNet

def build_model(dataset_name, output_dim):
    if dataset_name.lower() == 'mnist':
        return SourceNet(output_dim)
    elif dataset_name.lower() == 'cifar10':
        return ColorNet(output_dim)
    else:
        raise NotImplementedError(f"No model available for '{dataset_name}'.")