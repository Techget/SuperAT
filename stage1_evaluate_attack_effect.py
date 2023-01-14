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

checkpoint_dir='./superAT_attacker_checkpoints/'
model_state_file_name = 'attacker 2023-Jan-14 16:35' # Rename to your trained model

# Verify attacker effect
# attack the same pre-trained model, compare the accuracy 
def main():
    x_test, y_test = load_cifar10(n_examples=100) # use more examples
    # print(y_test) # => [3, 8, 8, 0, 6, 6, 1, 6, 3 ...]
    model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')

    # PGD attack the model, or refer to the autoattack robustaccuracy on the robustbench github repo
    # fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    # _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8/255])
    # print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))

    # Our trained attacker attack the model
    attacker=VAE(input_height=32)
    load_checkpoint(attacker, checkpoint_dir, model_state_file_name)
    # attacker=VAE(input_height=32,).from_pretrained('cifar10-resnet18')
    # attacker = attacker.eval()
    
    attacker_success = []
    # loop 1 by 1 due to memory limit, can put in larger batch if more memory is available
    for i, x_test_input in enumerate(x_test):
        x_test_input = x_test_input[None, :, :,:] # convert shape from [3,32,32] to [1,3,32,32]
        x_rescontructed,_,_,_,_ = attacker.run_step(x_test_input)
        model_output_for_x_reconstructed = model(x_rescontructed)
        model_output = torch.argmax(model_output_for_x_reconstructed, 1)
        
        if model_output[0] == y_test[i]:
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