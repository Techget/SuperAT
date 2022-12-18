from robustbench.data import load_cifar10
import foolbox as fb
from robustbench.utils import load_model
import torch
import os
from dataloader import testDataLoader
from utils import load_checkpoint
from lightning_VAE import VAE

# Note: there could be unrelated error message like " RuntimeError: Input type (torch.FloatTensor) and weight 
# type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense 
# tensor" when there is insufficient memory


# Verify attacker effect
# attack the same pre-trained model, compare the accuracy 
def main():
    x_test, y_test = load_cifar10(n_examples=2)
    # print(y_test) # => [3, 8, 8, 0, 6, 6, 1, 6, 3 ...]
    model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')

    # PGD attack the model
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8/255])
    print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))

    # Our trained attacker
    checkpoint_dir='./superAT_attacker_checkpoints/'
    attacker=VAE(input_height=32)
    load_checkpoint(attacker, checkpoint_dir, 'attacker')
    attacker = attacker.eval()
    
    x_rescontructed,_,_,_,_ = attacker.run_step(x_test)
    model_output_for_x_reconstructed = model(x_rescontructed)
    model_output = torch.argmax(model_output_for_x_reconstructed, 1)
    attacker_success = []
    for i, mo in enumerate(model_output):
        if mo == y_test[i]:
            attacker_success.append(False)
        else:
            attacker_success.append(True)
    attacker_success = torch.BoolTensor(attacker_success)
    print('Our attacker Robust accuracy: {:.1%}'.format(1 - attacker_success.float().mean()))

    # from autoattack import AutoAttack
    # adversary = AutoAttack(discrim, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    # adversary.apgd.n_restarts = 1
    # x_adv = adversary.run_standard_evaluation(x_test, y_test)


if __name__ == "__main__":
    main()