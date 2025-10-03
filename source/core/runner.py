import torch
from tqdm import tqdm
from ..attacks.fgsm_attacker import FgsmAttacker
from ..attacks.pgd_attacker import PgdAttacker

class ExperimentRunner:
    """Manages the training, evaluation, and attack pipeline."""
    def __init__(self, model, device, args):
        self.model = model
        self.device = device
        self.args = args
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.fgsm_attacker = FgsmAttacker(model)
        self.pgd_attacker = PgdAttacker(model)

    def train(self, loader):
        self.model.train()
        for epoch in range(self.args.epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)
                loss.backward()
                self.optimizer.step()
                
                acc = (predictions.argmax(1) == labels).float().mean()
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.2%}")

    def _evaluate_batch(self, images, labels):
        with torch.no_grad():
            preds = self.model(images)
            acc = (preds.argmax(1) == labels).float().mean().item()
        return acc * 100

    def evaluate_clean(self, loader):
        self.model.eval()
        total_acc = 0
        count = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            total_acc += self._evaluate_batch(images, labels) * len(labels)
            count += len(labels)
        self.clean_accuracy = total_acc / count
        print(f"Accuracy on clean images: {self.clean_accuracy:.2f}%")

    def evaluate_attacks(self, loader):
        self.model.eval()
        images, labels = next(iter(loader))
        images, labels = images.to(self.device), labels.to(self.device)
        targets = (labels + 5) % 10 # Arbitrary target for targeted attacks

        # FGSM
        adv_un = self.fgsm_attacker.attack_untargeted(images, labels, self.args.eps)
        print(f"FGSM Untargeted Accuracy: {self._evaluate_batch(adv_un, labels):.2f}%")
        
        adv_t = self.fgsm_attacker.attack_targeted(images, targets, self.args.eps)
        print(f"FGSM Targeted Success Rate: {self._evaluate_batch(adv_t, targets):.2f}%")

        # PGD
        adv_un_pgd = self.pgd_attacker.attack_untargeted(images, labels, self.args.eps, self.args.pgd_step, self.args.pgd_iter)
        print(f"PGD Untargeted Accuracy: {self._evaluate_batch(adv_un_pgd, labels):.2f}%")

        adv_t_pgd = self.pgd_attacker.attack_targeted(images, targets, self.args.eps, self.args.pgd_step, self.args.pgd_iter)
        print(f"PGD Targeted Success Rate: {self._evaluate_batch(adv_t_pgd, targets):.2f}%")