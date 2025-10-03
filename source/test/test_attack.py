import torch
from source.models import build_model
from source.attacks.fgsm_attacker import fgsm_untargeted

def test_fgsm_attack_shape_and_range():
    """Test if the shape and range of the image are maintained after attack"""
    model = build_model('mnist', 10)
    # Create dummy images (batch=4, channel=1, H=28, W=28)
    dummy_images = torch.rand(4, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (4,))
    eps = 0.1

    perturbed_images = fgsm_untargeted(model, dummy_images, dummy_labels, eps)

    # 1. Shape validation
    assert dummy_images.shape == perturbed_images.shape, "Shape of image should not change after attack."

    # 2. Range validation ([0, 1] clamping)
    assert perturbed_images.max() <= 1.0, "Values should be clamped to a maximum of 1.0"
    assert perturbed_images.min() >= 0.0, "Values should be clamped to a minimum of 0.0"