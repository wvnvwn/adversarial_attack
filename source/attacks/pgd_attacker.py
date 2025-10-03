import torch

class PgdAttacker:
    """A class to perform PGD attacks."""
    def __init__(self, model):
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def attack_targeted(self, original_image, target, epsilon, step_size, num_steps):
        """Performs a targeted PGD attack."""
        perturbed_image = original_image.clone().detach()

        for _ in range(num_steps):
            perturbed_image.requires_grad_(True)
            output = self.model(perturbed_image)
            
            loss = self.loss_fn(output, target)
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                gradient_sign = perturbed_image.grad.sign()
                # Move against the gradient
                perturbed_image = perturbed_image - step_size * gradient_sign
                
                # Project back into epsilon-ball
                eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
                perturbed_image = torch.clamp(original_image + eta, 0, 1)

        return perturbed_image

    def attack_untargeted(self, original_image, original_label, epsilon, step_size, num_steps):
        """Performs an untargeted PGD attack."""
        perturbed_image = original_image.clone().detach()

        for _ in range(num_steps):
            perturbed_image.requires_grad_(True)
            output = self.model(perturbed_image)

            loss = self.loss_fn(output, original_label)
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                gradient_sign = perturbed_image.grad.sign()
                # Move with the gradient
                perturbed_image = perturbed_image + step_size * gradient_sign

                # Project back into epsilon-ball
                eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
                perturbed_image = torch.clamp(original_image + eta, 0, 1)
        
        return perturbed_image