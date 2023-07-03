VAE for Image Recovery

This project implements a Variational Autoencoder (VAE) for recovering images affected by packet loss distortion. 
The goal is to reconstruct images from distorted versions using deep learning techniques.


Dataset

The dataset used for training and evaluation can be found on Kaggle: https://www.kaggle.com/datasets/macedocja/uav-images-packet-loss-distortion
Code for Benchmark Comparisons

Usage

Make sure you have the required dependencies installed. You can check the requirements.txt file for the necessary libraries.
Make sure you have prepared the config_file.json with the appropriate configurations.
Before running the code, ensure that the distorted directory and images.zip file exist in the project directory. These are required for training and evaluating the VAE.
Run the following command to execute the VAE for image recovery:

    python3 main.py config_file.json

Results
We have deployed an API that provides some results from the trained VAE model. You can access the API and explore the recovered images at the following link: https://webapp-33aygxoera-uc.a.run.app


