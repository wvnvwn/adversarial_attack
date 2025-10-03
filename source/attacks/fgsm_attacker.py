import torch

class FgsmAttacker:
    """A class to perform FGSM attacks."""
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def attack_targeted(self, original_image, target, epsilon):
        """Performs a targeted FGSM attack."""
        image_copy = original_image.clone().detach().requires_grad_(True)
        output = self.model(image_copy)
        
        loss = self.loss_fn(output, target)
        self.model.zero_grad()
        loss.backward()
        
        gradient = image_copy.grad.data
        # Move opposite to the gradient to minimize loss towards target
        perturbed_image = image_copy - epsilon * gradient.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image

    def attack_untargeted(self, original_image, original_label, epsilon):
        """Performs an untargeted FGSM attack."""
        image_copy = original_image.clone().detach().requires_grad_(True)
        output = self.model(image_copy)
        
        loss = self.loss_fn(output, original_label)
        self.model.zero_grad()
        loss.backward()
        
        gradient = image_copy.grad.data
        # Move along the gradient to maximize loss
        perturbed_image = image_copy + epsilon * gradient.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image