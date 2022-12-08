from discriminator import DiscriminatorRes
import torch
import os
from dataloader import testDataLoader
from utils import load_checkpoint


checkpoint_dir='./superAT_with_pretrained_vae_and_discriminator/'

if __name__ == "__main__":
    discrim=DiscriminatorRes()
    load_checkpoint(discrim, checkpoint_dir+'discriminator/checkpoints', 'discrim')

    test_loader = testDataLoader(32)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = discrim(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')