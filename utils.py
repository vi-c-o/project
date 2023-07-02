import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

"""
Benchmark comparison plots 
the plot_reconstructed_images is function for plotting reconstructed test images
and saving them as a grid.
"""

def plot_reconstructed_images(vae, test_data_loader, criterion, device, epoch):
    test_loss = 0

    with torch.no_grad():
        images = next(iter(test_data_loader))
        images = images.to(device)
        reconstructed_test_images, _, _ = vae(images)
        reconstruction_test_loss = criterion(reconstructed_test_images, images)
        test_loss += reconstruction_test_loss.item()

        # Select all 72 images from the batch
        selected_images = reconstructed_test_images

        # Create a grid to display the images
        grid_size = (12, 12)
        fig = plt.figure(figsize=(10.2, 10.2))
        gs = gridspec.GridSpec(*grid_size, figure=fig)

        # Display the images in the grid with borders
        border_width = 1.2
        for i, image in enumerate(selected_images):
            ax = fig.add_subplot(gs[i // grid_size[1], i % grid_size[1]])
            image = (image * 0.5) + 0.5  # Remove previous normalization
            ax.imshow(image.permute(1, 2, 0).cpu())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.spines['top'].set_linewidth(border_width)
            ax.spines['bottom'].set_linewidth(border_width)
            ax.spines['left'].set_linewidth(border_width)
            ax.spines['right'].set_linewidth(border_width)

        # Adjust the spacing between subplots
        gs.update(wspace=0, hspace=0)
        directory_path = "results/"
        fig_name = 'reconstructed_test_images_grid_' + str(epoch) + '.jpg'

        # Save the grid of images as a file
        output_path = os.path.join(directory_path, fig_name)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to release memory
