from autoattack import AutoAttack
from robustbench.data import load_cifar10
import foolbox as fb
from robustbench.utils import load_model
import torch
import os
from dataloader import testDataLoader
from utils import load_checkpoint
from lightning_VAE import VAE

checkpoint_dir='./superAT_with_pretrained_vae_and_discriminator/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    x_test, y_test = load_cifar10(n_examples=2)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # discrim = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    discrim = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
    discrim = discrim.to(device)

    # original robustness and accuracy
    autoAttackEvaluateRobustnessAndAccurarcy(discrim, x_test, y_test)

    # adversarial trained by our approach
    load_checkpoint(discrim, checkpoint_dir+'discriminator/checkpoints', 'discrim')
    autoAttackEvaluateRobustnessAndAccurarcy(discrim, x_test, y_test)
    


def autoAttackEvaluateRobustnessAndAccurarcy(discrim, x_test, y_test):
    adversary = AutoAttack(discrim, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)

if __name__ == "__main__":
    main()