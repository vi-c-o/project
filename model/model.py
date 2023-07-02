import torch
import torch.nn as nn

torch.manual_seed(42)

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model with attention mechanism.
    This class implements a VAE with an encoder-decoder architecture.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, enkernel_size, enstride, enpadding):
        super(VAEWithAttention, self).__init__()
        self.latent_dim = latent_dim
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, output_dim, kernel_size=enkernel_size, stride=enstride, padding=enpadding),
            nn.Dropout(0.5),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),

            nn.Conv2d(output_dim, output_dim*2, kernel_size=enkernel_size, stride=enstride, padding=enpadding),
            nn.Dropout(0.5),
            nn.BatchNorm2d(output_dim*2),
            nn.LeakyReLU(),

            nn.Conv2d(output_dim*2, output_dim*4, kernel_size=enkernel_size, stride=enstride, padding=enpadding),
            nn.Dropout(0.5),
            nn.BatchNorm2d(output_dim*4),
            nn.LeakyReLU(),
            
            # Additional Conv2d layers
            nn.Conv2d(output_dim*4, output_dim*8, kernel_size=enkernel_size, stride=enstride, padding=enpadding),
            nn.Dropout(0.5),
            nn.BatchNorm2d(output_dim*8),
            nn.LeakyReLU(),

            nn.Conv2d(output_dim*8, output_dim*16, kernel_size=enkernel_size, stride=enstride, padding=enpadding),
            nn.Dropout(0.5),
            nn.BatchNorm2d(output_dim*16),
            nn.LeakyReLU(),

            nn.Conv2d(output_dim*16, output_dim*32, kernel_size=enkernel_size, stride=enstride, padding=enpadding),
            nn.Dropout(0.5),
            nn.BatchNorm2d(output_dim*32),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear((output_dim)*(output_dim)*4, hidden_dim),
            nn.Dropout(0.5),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, self.latent_dim*2),
            nn.LeakyReLU(),
        )


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, 256 * 8 * 8),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256 * 8 * 8),

            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mean = encoded[:, :self.latent_dim]
        logvar = encoded[:, self.latent_dim:]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)     
        x_hat = self.decode(z)
        return x_hat, mean, logvar


def apply_noise(images, std_dev):
    noise = torch.randn_like(images) * std_dev
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, -1, 1)
    return noisy_images