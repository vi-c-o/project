import argparse
import json
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
import os
from utils import plot_reconstructed_images
import torch.nn as nn
from model.model import VAE, apply_noise
from dataset.dataset import CustomDataset

# Create an instance of the argument parser
parser = argparse.ArgumentParser(description='Recover Stretched Images API')
parser.add_argument('config_file', type=str, help='Add the config.json')

# Parse the command-line arguments
args = parser.parse_args()

# Read the parameters from the config file
with open(args.config_file, 'r') as f:
    config = json.load(f)

# Set the necessary parameters
folder_1 = config['dataset']['folder1']
folder_2 = config['dataset']['folder2']
target_size = config['dataset']['target_size']
augmentation_factor = config['dataset']['augmentation_factor']
batch_size = config['dataset']['batch_size']

epochs = config['model']['epochs']
learning_rate = config['model']['learning_rate']
latent_dim = config['model']['latent_dim']
hidden_dim = config['model']['hidden_dim']
output_dim = config['model']['output_dim']
enkernel_size = config['model']['enkernel_size']
enstride = config['model']['enstride']
enpadding = config['model']['enpadding']

weight_decay = config['training']['weight_decay']
patience = config['training']['patience']
factor = config['training']['factor']

dataset = CustomDataset(folder_1, folder_2, target_size, batch_size)

num_lists = 16
train_dataset = dataset.augment_truth_images(augmentation_factor, num_lists)
transformed_distorted_images = dataset.transform_distorted_images()

random.shuffle(transformed_distorted_images)

split_index = len(transformed_distorted_images) // 2

val_dataset = transformed_distorted_images[:split_index]
test_dataset = transformed_distorted_images[split_index:]

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

vae = VAE(latent_dim, hidden_dim, output_dim, enkernel_size, enstride, enpadding)
criterion = nn.MSELoss(reduction="sum")
kl_weight = 1.0
optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_losses = []
val_losses = []
test_losses = []

vae.to(device)
vae.train()

for epoch in range(epochs):
    total_loss = 0

    for images in train_data_loader:
        images = images.to(device)
        noisy_images = apply_noise(images, std_dev=0.2).to(device)

        optimizer.zero_grad()
        reconstructed_images, mean, logvar = vae(noisy_images)

        recon_loss = criterion(reconstructed_images, images)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        loss = recon_loss + kl_weight * kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_data_loader.dataset)
    train_losses.append(average_loss)

    vae.eval()
    val_loss = 0

    with torch.no_grad():
        for images in val_data_loader:
            images = images.to(device)
            reconstructed_val_images, _, _ = vae(images)
            reconstruction_val_loss = criterion(reconstructed_val_images, images)
            val_loss += reconstruction_val_loss.mean().item()

    average_val_loss = val_loss / len(val_data_loader.dataset)
    val_losses.append(average_val_loss)

    scheduler.step(val_loss)

    vae.eval()
    test_loss = 0

    with torch.no_grad():
        for images in test_data_loader:
            images = images.to(device)
            reconstructed_test_images, _, _ = vae(images)
            reconstruction_test_loss = criterion(reconstructed_test_images, images)
            test_loss += reconstruction_test_loss.item()
    average_test_loss = test_loss / len(test_data_loader.dataset)
    test_losses.append(average_test_loss)

    if epoch % 10 == 0:
        model_save = "results/vae_models/vae_model_" + str(epoch) + ".pth"
        model_pkl = "results/vae_models/vae_model_" + str(epoch) + ".pkl"
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {average_loss:.4f}, Test Loss: {average_test_loss:.4f}")
        torch.save(vae.state_dict(), model_save)
        torch.save(vae, model_pkl)

    plot_reconstructed_images(vae, test_data_loader, criterion, device, epoch)
