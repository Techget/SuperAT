```
python3 main_pretrained_discriminator.py
```

pretrain VAE to return image with little modification to fool the trained model


next step, with trained model(discriminator) and trained VAE, we will update in the GAN-manner to improve the robustness of the trained model


## File structure
- discriminator.py
    - Include the common classification methods, currenlty using resnet18
    - Obtain pretrained resnet18 from [Pytorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10#pytorch-models-trained-on-cifar-10-dataset)