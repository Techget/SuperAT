import torch
import torch.nn as nn
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from dataloader import dataloader
import utils
import os
from pytorch_lightning.utilities.seed import seed_everything
from lightning_VAE import VAE
from discriminator import DiscriminatorRes
from torch.utils.tensorboard import SummaryWriter
from robustbench.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)


# hyper parameter
# convert to use arg parser
epochs=100
lr_gen=3e-4
lr_dis=3e-4
gamma=15
is_adversarial_training = False
batch_size = 1 # adjust based on memroy size the device owns
kld_weight = 0.1 # for VAE (Used in standara libraries)
gen_discrim_loss_weight = 0.1 # for the loss of discriminator with input generated by generator
                          # to make sure these errors are in the same numeric level
train_frequency = 3 # stands for training 2 epoch of gen and 1 epoch of discrim in every 3 epoch

# logging and saving result
checkpoint_dir='./superAT_with_pretrained_vae_and_discriminator/'
writer = SummaryWriter()


# For reproducibility
seed_everything(7, True)

data_loader=dataloader(batch_size)

# load pretrained VAE
gen=VAE(input_height=32,
    kl_coeff=kld_weight).from_pretrained('cifar10-resnet18')
gen=gen.to(device)


# load pretrained resnet 18
# discrim=DiscriminatorRes().to(device)
# state_dict = os.path.join(
#   "state_dicts", "resnet18" + ".pt"
# )
# discrim.load_state_dict(torch.load(state_dict))
discrim = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
discrim = discrim.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optim_gen=torch.optim.RMSprop(gen.parameters(), lr=lr_gen)
optim_Dis=torch.optim.RMSprop(discrim.parameters(), lr=lr_dis)

example_input_images, _ = next(iter(data_loader))
grid = torchvision.utils.make_grid(example_input_images)
writer.add_image("images", grid)
# writer.add_graph(gen, example_input_images.to(device)) # comment out to run on yaoyu's machine
# writer.add_graph(discrim, example_input_images.to(device))

for epoch in range(epochs):
    total_vae_loss = 0
    total_gen_output_to_discrim_CE_error = 0
    total_gen_loss = 0
    total_discrim_loss = 0
    last_x_reconstructed = None

    for i, (data, targets) in enumerate(data_loader, 0):
        datav = Variable(data).to(device)
        targets = Variable(targets).to(device)

        # train generator
        x_reconstructed, mean, logvar, p, q = gen.run_step(datav) # p, q are sampled from mean,logvar distribution
        last_x_reconstructed = x_reconstructed

        reconstruction_loss = gen.reconstruction_loss(x_reconstructed, datav)
        kl_divergence_loss = gen.kl_divergence_loss(p, q)
        vae_loss = reconstruction_loss + kl_divergence_loss
        total_vae_loss += vae_loss

        discrim_with_generated_input = discrim(x_reconstructed)
        gen_output_to_discrim_CE_error = criterion(torch.tensor(discrim_with_generated_input), targets) * gen_discrim_loss_weight
        total_gen_output_to_discrim_CE_error += gen_output_to_discrim_CE_error

        gen_loss = vae_loss - gen_output_to_discrim_CE_error
        total_gen_loss += gen_loss

        if (i+1) % train_frequency != 0:
            optim_gen.zero_grad()
            gen_loss.backward()
            optim_gen.step()

        # train discriminator
        discrim_output_real_image = discrim(datav)
        disrim_real_image_CE = criterion(discrim_output_real_image, targets)
        discrim_loss = disrim_real_image_CE + gen_output_to_discrim_CE_error
        total_discrim_loss += discrim_loss

        if (i+1) % train_frequency == 0:
            optim_Dis.zero_grad()
            discrim_loss.backward()
            optim_Dis.step()

        # tensorboard log
        writer.add_scalar("Loss/Minibatches_reconstruction_loss", reconstruction_loss, i)
        writer.add_scalar("Loss/Minibatches_kl_divergence_loss", kl_divergence_loss, i)
        writer.add_scalar("Loss/Minibatches_gen_output_to_discrim_CE_error", gen_output_to_discrim_CE_error, i)
        writer.add_scalar("Loss/Minibatches_gen_loss", gen_loss, i)
        writer.add_scalar("Loss/Minibatches_disrim_real_image_CE", disrim_real_image_CE, i)
        writer.add_scalar("Loss/Minibatches_discrim_loss", discrim_loss, i)

    writer.add_scalar("Loss/Epochs_total_vae_loss", total_vae_loss, epoch)
    writer.add_scalar("Loss/Epochs_total_gen_output_to_discrim_CE_error", total_gen_output_to_discrim_CE_error, epoch)
    writer.add_scalar("Loss/Epochs_total_gen_loss", total_gen_loss, epoch)
    writer.add_scalar("Loss/Epochs_total_discrim_loss", total_discrim_loss, epoch)

    grid = torchvision.utils.make_grid(last_x_reconstructed)
    writer.add_image('Generated adversarial training example', grid, epoch)

    utils.save_checkpoint(gen, checkpoint_dir+'vae/checkpoints', epoch, 'gen')
    utils.save_checkpoint(discrim, checkpoint_dir+'discriminator/checkpoints', epoch, 'discrim')