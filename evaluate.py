import os 


import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models import CNN, ResNet18
from utils import load_model, test_model, visualize_results

from attack_utils import FastGradientSignMethod, ProjectedGradientDescent 


def main(args):
    print('Evaluating...')

    args.path = os.path.join('experiments', args.path)
    print(f'Experiments will be saved in {args.path}')

    # Load the data CIFAR10 (eval only)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Get a batch of data to visualize the results
    x_test, y_test = next(iter(test_loader))

    # Load the model
    if args.model == 'cnn':
        model = CNN()
    elif args.model == 'resnet18':
        model = ResNet18()

   # Set the device (cuda, cpu or mps)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Device used: {}".format(device))

    model = load_model(model, args.path)
    model = model.to(device)

    if args.attack == 'fgsm':
        attack = FastGradientSignMethod(model, args.epsilon)
    elif args.attack == 'pgd':
        attack = ProjectedGradientDescent(model, args.epsilon, args.alpha, args.num_steps)

    loss_adv, acc_adv = test_model(model, test_loader, device, attack)
    visualize_results(x_test, y_test, model, device, args, attack)

    with open(os.path.join(args.path, 'results.txt'), 'a') as f:
        f.write(f'Attack: {args.attack} \t {attack.name}\n')
        f.write(f'\t Loss: {loss_adv:.4f}, Accuracy: {acc_adv:.2f}\n')

    print(f'Attack: {args.attack} \t {attack.name}')
    print(f'\t Loss: {loss_adv:.4f}, Accuracy: {acc_adv:.2f}')

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Adversarial attack on Deep Learning models on CIFAR10""")

    # Model parameters
    parser.add_argument('--model', type=str, default='cnn', help='Model to attack', choices=['cnn', 'resnet18'])    
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--path', type=str, default='base', help='Path to save the model and results')

    # Adversarial parameters
    parser.add_argument('--attack', type=str, default='fgsm', help='Attack to perform', choices=['fgsm', 'pgd'])
    parser.add_argument('--epsilon', type=float, default=0.03, help='Epsilon for the attack')   
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the attack')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of steps for the attack')


    args = parser.parse_args()
    main(args)