import torch
import torch.nn as nn
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.autograd import Variable
from dataloader import dataloader
import utils
import torchvision.models as models
import os
from pytorch_lightning.utilities.seed import seed_everything
from lightning_VAE import VAE
from torch.utils.tensorboard import SummaryWriter
from discriminator import DiscriminatorRes
from robustbench.utils import load_model
import datetime
import socket

# from koila import LazyTensor, lazy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

# hyper parameter
# convert to use arg parser
epochs=150
lr_attacker=2e-4
batch_size = 4
kld_weight = 0.001 # for VAE (Used in standar attack_defensor_loss_weight libraries)
                # default is 0.1, but we prioritise reconstruction in this case, so decrease to 0.001
attack_defensor_loss_weight = 0.1 # for the loss of defensor with input generated by generator
                          # to make sure these errors are in the same numeric level
latent_dim = 1024 # by default 256


# logging and saving result
checkpoint_dir='./superAT_attacker_checkpoints/'
writer = SummaryWriter()


# For reproducibility
seed_everything(7, True)

data_loader=dataloader(batch_size)

# load pretrained VAE
train_new=True
attacker=None
if train_new:
    attacker = VAE(input_height=32,
        kl_coeff=kld_weight, latent_dim=latent_dim, enc_type='resnet50', enc_out_dim=2048) # 
else:
    attacker=VAE(input_height=32,
        kl_coeff=kld_weight).from_pretrained('cifar10-resnet18')
attacker=attacker.to(device)

# load pretrained resnet 18
# defensor = models.resnet18(pretrained=True).eval()
# defensor=DiscriminatorRes().to(device)
# state_dict = os.path.join(
#   "state_dicts", "resnet18" + ".pt"
# )
# defensor.load_state_dict(torch.load(state_dict))
defensor = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
defensor = defensor.to(device).eval()

criterion = nn.CrossEntropyLoss().to(device)
optim_attacker=torch.optim.RMSprop(attacker.parameters(), lr=lr_attacker)

example_input_images, _ = next(iter(data_loader))
grid = torchvision.utils.make_grid(example_input_images)
writer.add_image("images", grid)
# comment out to run on yaoyu's machine
# writer.add_graph(attacker, example_input_images.to(device))

date_time = datetime.datetime.now()

def main():

    for epoch in range(epochs):
        total_vae_loss = 0
        total_attacker_output_to_defensor_CE_error = 0
        total_attacker_loss = 0
        last_x_reconstructed = None

        for i, (data, targets) in enumerate(data_loader, 0):
            datav = Variable(data).to(device)
            targets = Variable(targets).to(device)
            # datav, targets = lazy(datav, targets, batch=0)

            # train attacker
            x_reconstructed, mean, logvar, p, q = attacker.run_step(datav) # p, q are sampled from mean,logvar distribution
            last_x_reconstructed = x_reconstructed

            reconstruction_loss = attacker.reconstruction_loss(x_reconstructed, datav)
            kl_divergence_loss = attacker.kl_divergence_loss(p, q)
            vae_loss = reconstruction_loss + kl_divergence_loss
            total_vae_loss += vae_loss

            defensor_with_generated_input = defensor(x_reconstructed)
            attacker_output_to_defensor_CE_error = criterion(
                torch.tensor(defensor_with_generated_input), targets) * attack_defensor_loss_weight
            total_attacker_output_to_defensor_CE_error += attacker_output_to_defensor_CE_error

            attacker_loss = vae_loss - attacker_output_to_defensor_CE_error
            total_attacker_loss += attacker_loss

            optim_attacker.zero_grad()
            attacker_loss.backward()
            optim_attacker.step()

            # tensorboard log
            writer.add_scalar("Loss/Minibatches_reconstruction_loss", reconstruction_loss, (i+1)*(epoch+1))
            writer.add_scalar("Loss/Minibatches_kl_divergence_loss", kl_divergence_loss, (i+1)*(epoch+1))
            writer.add_scalar("Loss/Minibatches_attacker_output_to_defensor_CE_error", 
                attacker_output_to_defensor_CE_error, (i+1)*(epoch+1))
            writer.add_scalar("Loss/Minibatches_attacker_loss", attacker_loss, (i+1)*(epoch+1))

        writer.add_scalar("Loss/Epochs_total_vae_loss", total_vae_loss, (epoch+1))
        writer.add_scalar("Loss/Epochs_total_attacker_output_to_defensor_CE_error", 
            total_attacker_output_to_defensor_CE_error, (epoch+1))
        writer.add_scalar("Loss/Epochs_total_attacker_loss", total_attacker_loss, (epoch+1))

        grid = torchvision.utils.make_grid(last_x_reconstructed)
        writer.add_image('Generated adversarial training example', grid, (epoch+1))

        utils.save_checkpoint(attacker, checkpoint_dir, epoch, 'attacker ' + date_time.strftime("%Y-%b-%d %H:%M") + socket.gethostname())


if __name__ == "__main__":
    main()
