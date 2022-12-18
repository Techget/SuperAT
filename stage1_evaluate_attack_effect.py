from robustbench.data import load_cifar10
import foolbox as fb
from robustbench.utils import load_model
# from discriminator import DiscriminatorRes
import torch
import os
from dataloader import testDataLoader
from utils import load_checkpoint
from lightning_VAE import VAE


# Verify attacker effect
# attack the same pre-trained model, compare the accuracy 
def main():
    x_test, y_test = load_cifar10(n_examples=50)
    model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

    # PGD attack the model
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8/255])
    print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))

    # Our trained attacker
    checkpoint_dir='./superAT_with_pretrained_vae_and_discriminator/'
    attacker=VAE(input_height=32)
    load_checkpoint(attacker, checkpoint_dir+'vae/checkpoints', 'attacker')
    attacker = attacker.eval()
    # TODO: generate attacker output, and feed to the pre-trained model to get accuracy

    # from autoattack import AutoAttack
    # adversary = AutoAttack(discrim, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    # adversary.apgd.n_restarts = 1
    # x_adv = adversary.run_standard_evaluation(x_test, y_test)



if __name__ == "__main__":
    main()