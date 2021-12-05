# README for alternate VAE

The implementation for alternate VAE for 11785 team 40 project.

The goal of these implementation is to show whether incorporating supervised loss or center loss to VAE model during training.

The implementation has two categories: 1, training VAE and classifier from scratch at the same time; 2, training a classifier in advance and use the trained classifier to help training VAE.

Both codes are implemented using jupyter notebook. The code can be run by executing the code snippets sequentially.

The results of these experiments showed that incorporating supervised loss or center loss could improve intra-class compactness and inter-class separability. However, improving representation of the latent space does not necessarily improve the loss or reconstructed images quality.
