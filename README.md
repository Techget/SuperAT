## Libraries we're using & referencing to
- [Lightning-bolt](https://github.com/Lightning-AI/lightning-bolts)
    - Our VAE implementation is adapted from [their VAE implementation](https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py) and we load pretrained parameters from bolts.
- [Foolbox](https://github.com/bethgelab/foolbox)
    - Allows us easily run adversarial attacks against machine learning models
    - We can use it benchmark the trained VAE attacker
- [RobustBench](https://github.com/RobustBench/robustbench)
    - Allows us to us bench mark the adversarial trained models
    - Have a model adversarial trained by our proposing method attacked and benchmarked with SOTA methods

## Plans for experiments

There are roughly 2 stages to conduct the experiment to verify our theories:

1. Train attacker with pretained VAE (attacker) & resnet18 (defensor) on CIFAR10
    - Attacker gets updated while defensor remains the same during training.
    - Benchmark our trained attacker with existing attacking method with Foolbox
        - Example 'foolbox_attack_resnet18_example.py'
    - Objective: verify that attacker trained with our method can reach similar effect like other conventional attack methods
    - Related files
        - training: `python3 train_attacker.py`
        - evaluation: update the ckpt file name in the script and run `python3 stage1_evaluate_attack_effect.py`
        - run on usyd HPC: `qsub artemis_script_train_attacker.pbs` // Met OSError, haven't figured out the dependencies
    - TODO: explore advacned models / tricks / variations e.g. use large latent variable(Z) size etc.
2. Adversarial train with attacker & defensor updated in turn 
    - Load pretrained attacker & defensor (either non adversarial trained or adversarial trained from model zoo of robustbench), train with our method, which improve/adapt attacker and defensor in turn. 
    - Benchmark our adversarial trained defensor with SOTA in RobustBench
    - Objective: model adversarial trained with our approach gives higher robustness with similar level of accurarcy on non-adversarial examples.
    - Related files:
        - training:  `python3 main_adversarial_training.py`
        - evaluation: `python3 stage2_evaluate_adversarial_trained_model.py` (using Autoattack to evaluate) [WIP]


Current status/progress:

- As of 19 Dec 2022
    - Trained attacker reached 40% robust accuracy when attack pre adversarial trained model, which achieves 66% robustness accurracy with AutoAttack, reckon it is too good to be true.
    - Inspected the reconstruction error, it is around 0.1-0.2 pixel wise difference, after training 32 epoches, it remains similar level since very beginning
    - The training is running on yyao0814@172.17.34.20
    - Also  we can check on tensorboard by using `scp -r yyao0814@172.17.34.20:/home/yyao0814/xuantong/SuperAT/runs/Dec18_17-29-36_gpu4-119-1 .`, then run the tensorboard locally
- As of 18 Dec 2022, runing experiment for step1
    - Using a vanilla VAE as attacker and Rebuffi2021Fixing_70_16_cutmix_extra from robustbench as defensor, the defensor is pre-adversarial-trained


## File structure 
- discriminator.py
    - Include the common classification methods, currenlty using resnet18
    - Obtain pretrained resnet18 from [Pytorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10#pytorch-models-trained-on-cifar-10-dataset)
    - todo: use pre activated resnet (no pretrained parameters)
- vae.py
    - Include vanilla implementation of VAE
    - reference to lightning VAE
- main_adversarial_training.py
    - pretrain VAE to return image with little modification to fool the trained model
    - GENERATOR and Discriminator are updated **IN TURN** instead of being updated simultaneously
    - `python3 main_adversarial_training.py` to start training no other input required at the moment
    - This specific file hasn't been cross checked yet
- test_adversarial_trained_model.py
    - verify adversarial trained model on common data, not adversarial example
- reference to https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    - for updating pre-trained torch models, however the torch model is trained for imagenet, seems we cannot use it, as our experiment is based on CIFAR10
- artemis_script_*.pbs
    - Script to run certain job on artemis