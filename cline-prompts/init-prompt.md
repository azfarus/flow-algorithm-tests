# REPO TO TEST MEAN FLOW CODES

## Purpose

The purpose of this repository is to test flow and mean flow performance on a very small toy dataset for image generation. The repository is in pytorch, models running on cuda gpu.

## Elaborate Functionality

- The repo will compare the performance of the flow vs the general mean flow(with jvp) 
- The dataset will be seeded with noise then the model will be trained on flow and mean flow and their performance will be compared
- Flow will be seeded with a noise and a t value and will learn to generate the image
- Mean flow will be similarly seeded but will generate the final image in one step ( as the original algorithm goes )
- all other possible architectures and training params will be equivalent, and the model losses and generation performance are to be plotted

## Diffusion Architecture:
- Small Unet

## Dataset
- Mnist Dataset

## Constraints

- The code has to be simple, try not to exceed 10 files and try not to exceed 500 lines per file.
- Log all the training metrics and lossess in wandb 
- generate an evaluation pipeline for flow vs meanflow with all the relevant metrics for both the models
- save the last 5 checkpoints and display 4 generated images for each checkpoint in wandb