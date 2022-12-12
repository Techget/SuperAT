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

There are roughly 2 steps to conduct the experiment to verify our theories:

1. Train attacker with pretained VAE (attacker) & resnet18 (defensor) on CIFAR10
    - Attacker gets updated while defensor remains the same during training.
    - Benchmark our trained attacker with existing attacking method with Foolbox
        - Example 'foolbox_attack_resnet18_example.py'
    - Objective: verify that attacker trained with our method can reach similar effect like other conventional attack methods
    - Related files
        - train_attacker.py
        - foolbox_attack_resnet18_example.py
        - artemis_script_train_attacker.pbs
    - TODO: explore advacned models / tricks / variations e.g. use large latent variable(Z) size etc.
2. Adversarial train with attacker & defensor updated in turn 
    - Load pretrained attacker & defensor, train with our method, which improve/adapt attacker and defensor in turn. 
    - Benchmark our adversarial trained defensor with SOTA with RobustBench
    - Objective: model adversarial trained with our approach gives higher robustness with similar level of accurarcy on non-adversarial examples.
    - Related files:
        - main_adversarial_training.py (WIP)
    - TODO: provide PoC of our proposing adversarial training paradigm, and fine tune


Current status/progress:
    - As of 12 Dec 2022, runing experiment for step1
        - Using a vanilla VAE as attacker and resnet18 as defensor


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