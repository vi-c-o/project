import unittest
import torch
import torch.nn as nn
import os
import sys
# Append the project directory to sys.path
project_dir = os.path.abspath(os.path.dirname('../'))
sys.path.append(project_dir)

from model.model import VAE, apply_noise

class VAETestCase(unittest.TestCase):
    """
    Test case for the VAE class.
    This class defines unit tests for the VAE class.
    """
    
    def test_model_forward(self):

        batch_size = 16
        image_channels = 3
        image_size = 128
        latent_dim = 64
        hidden_dim = 300
        output_dim = 32
        enkernel_size =3
        enstride= 2
        enpadding =1
        model = VAE(latent_dim, hidden_dim, output_dim, enkernel_size, enstride, enpadding)

        input_images = torch.randn(batch_size, image_channels, image_size, image_size)

        recon_images, mean, logvar = model(input_images)

        self.assertEqual(recon_images.shape, input_images.shape)
        self.assertEqual(mean.shape, (batch_size, latent_dim))
        self.assertEqual(logvar.shape, (batch_size, latent_dim))

    def test_encode(self):

        batch_size = 16
        image_channels = 3
        image_size = 128
        latent_dim = 128
        hidden_dim = 300
        output_dim = 32
        enkernel_size =3
        enstride= 2
        enpadding =1
        model = VAE(latent_dim, hidden_dim, output_dim, enkernel_size, enstride, enpadding)

        input_images = torch.randn(batch_size, image_channels, image_size, image_size)

        mean, logvar = model.encode(input_images)

        self.assertEqual(mean.shape, (batch_size, latent_dim))
        self.assertEqual(logvar.shape, (batch_size, latent_dim))

    def test_reparameterize(self):

        batch_size = 16
        latent_dim = 64
        mean = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)
        hidden_dim = 300
        output_dim = 32
        enkernel_size =3
        enstride= 2
        enpadding =1
        model = VAE(latent_dim, hidden_dim, output_dim, enkernel_size, enstride, enpadding)

        z = model.reparameterize(mean, logvar)

        self.assertEqual(z.shape, (batch_size, latent_dim))

    def test_decode(self):

        batch_size = 16
        latent_dim = 64
        image_size = 128
        z = torch.randn(batch_size, latent_dim)
        hidden_dim = 300
        output_dim = 32
        enkernel_size =3
        enstride= 2
        enpadding =1
        model = VAE(latent_dim, hidden_dim, output_dim, enkernel_size, enstride, enpadding)

        recon_images = model.decode(z)

        self.assertEqual(recon_images.shape, (batch_size, 3, image_size, image_size))

    def test_apply_noise(self):

        batch_size = 16
        image_channels = 3
        image_size = 128
        std_dev = 0.1
        latent_dim = 64
        hidden_dim = 300
        output_dim = 32
        enkernel_size =3
        enstride= 2
        enpadding =1
        model = VAE(latent_dim, hidden_dim, output_dim, enkernel_size, enstride, enpadding)

        input_images = torch.randn(batch_size, image_channels, image_size, image_size)

        noisy_images = apply_noise(input_images, std_dev)

        self.assertEqual(noisy_images.shape, input_images.shape)

if __name__ == '__main__':
    unittest.main()
