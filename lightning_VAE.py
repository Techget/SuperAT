from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import VAE as lightningVAE
from torch.nn import functional as F
import torch


class VAE(lightningVAE):
    def __init__(
        self,
        input_height: int,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ):
        super().__init__(
            input_height,
            enc_type,
            first_conv,
            maxpool1,
            enc_out_dim,
            kl_coeff,
            latent_dim,
            lr)
        
    def run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z), mu, log_var, p, q

    def reconstruction_loss(self, x_reconstructed, x):
        return F.mse_loss(x_reconstructed, x)

    def kl_divergence_loss(self, p, q):
        # return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        kl *= self.kl_coeff
        return kl