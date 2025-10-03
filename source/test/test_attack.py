import torch
import pytest
from source.models import build_model
from source.attacks.fgsm_attacker import fgsm_untargeted, fgsm_targeted
from source.attacks.pgd_attacker import pgd_untargeted, pgd_targeted

# --- Define reusable elements needed for the test (Fixtures) ---
@pytest.fixture(scope="module")
def mnist_model():
    """Create the model once and reuse it for the entire test module to save time."""
    return build_model('mnist', 10)

@pytest.fixture
def dummy_mnist_data():
    """Create new dummy data for each test to provide."""
    images = torch.rand(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    return images, labels

# --- Test all attacks at once using parameterized functions ---
@pytest.mark.parametrize("attack_func, use_target", [
    (fgsm_untargeted, False),
    (fgsm_targeted, True),
    (pgd_untargeted, False),
    (pgd_targeted, True),
])
def test_attack_properties(attack_func, use_target, mnist_model, dummy_mnist_data):
    """
    Verify the core properties of all attacks in one function:
    1. Is the shape of the image changed after attack?
    2. Is the pixel value of the image maintained within the [0, 1] range?
    3. Is the generated perturbation not greater than the allowed epsilon value?
    """
    # 1. Arrange: Prepare the input values needed for the test.
    model = mnist_model
    original_images, labels = dummy_mnist_data
    eps = 0.1
    
    attack_args = {
        "model": model,
        "x": original_images,
        "eps": eps
    }

    if use_target:
        targets = (labels + 1) % 10
        attack_args["target"] = targets
    else:
        attack_args["label"] = labels
        
    if "pgd" in attack_func.__name__:
        attack_args["k"] = 10
        attack_args["eps_step"] = 0.01
        
    # 2. Act: Execute the actual attack function with the prepared arguments.
    perturbed_images = attack_func(**attack_args)
    
    # 3. Assert: Verify if the result satisfies our expectations.
    # Shape validation
    assert original_images.shape == perturbed_images.shape, \
        "The shape of the image after attack should be the same as the original."

    # Range validation
    assert perturbed_images.max() <= 1.0, "The maximum pixel value should not exceed 1.0."
    assert perturbed_images.min() >= 0.0, "The minimum pixel value should not be less than 0.0."

    # Perturbation size validation (L-infinity norm)
    perturbation = torch.abs(perturbed_images - original_images)
    assert perturbation.max() <= eps + 1e-5, \
        "The maximum size of perturbation should not exceed epsilon."