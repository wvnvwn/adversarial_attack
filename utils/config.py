import torch
import numpy as np
import random

def configure_environment(seed):
    """Sets a global random seed and returns the best device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")
    return torch.device("cpu")