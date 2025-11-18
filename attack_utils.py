import torch
from torch import nn

class FastGradientSignMethod:
    """
    Implements the Fast Gradient Sign Method (FGSM) attack for adversarial example generation.

    Attributes:
        model (torch.nn.Module): The model to attack.
        eps (float): Maximum perturbation (L-infinity norm bound).
    """
    def __init__(self, model, eps):
        """
        Initializes the FGSM attack.

        Args:
            model (torch.nn.Module): The model to attack.
            eps (float): Maximum L-infinity norm of the perturbation.
        """
        self.model = model
        self.eps = eps
        self.name = f'FGSM_{eps:.2e}'

    def compute(self, x, y):
        """
        Generates an adversarial perturbation using FGSM.

        Args:
            x (torch.Tensor): Original input images.
            y (torch.Tensor): True labels for the input images.

         Returns:
            torch.Tensor: The computed adversarial perturbations.
        """
        # initialize the perturbation delta to zero, and require gradient for optimization
        delta = torch.zeros_like(x, requires_grad=True)
        self.model.zero_grad()

        # get model output and compute loss (cross-entropy)
        loss = nn.CrossEntropyLoss()(self.model(x + delta), y)
        loss.backward()

        ## apply one step of sign gradient ascent to the input
        grad_sign = delta.grad.detach().sign()
        perturbation = self.eps * grad_sign
        perturbation = torch.clamp(perturbation, -self.eps, self.eps)

        ## To do 12
        
        return perturbation.detach()

class ProjectedGradientDescent:
    """
    Implements the Projected Gradient Descent (PGD) attack in L-infinity norm for adversarial example generation.

    Attributes:
        model (torch.nn.Module): The model to attack.
        eps (float): Maximum perturbation (L-infinity norm bound).
        alpha (float): Step size for each iteration.
        num_iter (int): Number of iterations for the attack.
    """
    def __init__(self, model, eps, alpha, num_iter):
        """
        Initializes the PGD attack.

        Args:
            model (torch.nn.Module): The model to attack.
            eps (float): Maximum L-infinity norm of the perturbation.
            alpha (float): Step size for the attack.
            num_iter (int): Number of attack iterations.
        """
        self.model = model
        self.eps = eps
        self.num_iter = num_iter
        if alpha is None:
            alpha = eps / num_iter
            alpha = round(alpha, 4)

        ## To do 19
        self.alpha = alpha
        self.name = f'PGDLinf_{eps:.2e}_{alpha:.2e}_{num_iter}'


    def compute(self, x, y):
        """
        Generates an adversarial perturbation using PGD with L2 norm.

        Args:
            x (torch.Tensor): Original input images.
            y (torch.Tensor): True labels for the input images.

        Returns:
            torch.Tensor: The computed adversarial perturbations.
        """
        # initialize the perturbation delta to zero, and require gradient for optimization
        delta = torch.zeros_like(x, requires_grad=True)

        # iteratively compute adversarial perturbations
        for t in range(self.num_iter):
            # on remet à zéro les gradients du modèle
            self.model.zero_grad()
            # adversarial input
            x_adv = x + delta
            x_adv = torch.clamp(x_adv, 0, 1)
            # forward + loss
            outputs = self.model(x_adv)
            loss = nn.CrossEntropyLoss()(outputs, y)
            loss.backward()
            # gradient ascent sur delta
            grad_sign = delta.grad.detach().sign()
            delta = delta + self.alpha * grad_sign
            # projection sur la boule L_inf de rayon eps
            delta = torch.clamp(delta, -self.eps, self.eps)
            #projection pour rester dans [0, 1] après ajout à x
            delta = torch.clamp(x + delta, 0, 1) - x
            # on prépare le prochain tour
            delta = delta.detach()
            delta.requires_grad_()

        return delta.detach()
            ## To do 16 
            
"""Avec un budget de perturbation identique ou supérieur, PGD provoque une chute encore plus importante de la performance que FGSM.
Par exemple, FGSM avec ε = 0.05 réduit l’accuracy à 1.13 %, tandis que PGD avec T = 10 et α = 0.1 mène à une accuracy de 0 %.
Cela montre que PGD est une attaque itérative plus puissante et capable d’explorer l’espace des perturbations de manière plus agressive, rendant le modèle totalement incorrect.
Avec un budget de perturbation identique ou supérieur, PGD provoque une chute encore plus importante de la performance que FGSM.
Par exemple, FGSM avec ε = 0.05 réduit l’accuracy à 1.13 %, tandis que PGD avec T = 10 et α = 0.1 mène à une accuracy de 0 %.
Cela montre que PGD est une attaque itérative plus puissante et capable d’explorer l’espace des perturbations de manière plus agressive, rendant le modèle totalement incorrect.
PGD (Projected Gradient Descent) est une attaque itérative qui applique plusieurs mises à jour FGSM successives tout en projetant l’image modifiée dans une boule L∞ de rayon ε. Ces itérations permettent d’explorer plus finement l’espace des perturbations et rendent PGD beaucoup plus efficace que FGSM pour tromper les réseaux de neurones.""

""FGSM = une seule perturbation
👉 PGD = plusieurs étapes, ajustées, donc attaque plus puissante"""
            

    
