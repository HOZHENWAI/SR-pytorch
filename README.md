# SR-pytorch

# What is it about?

This project aims to implement various models for Image Super Resolution.

The features planned and provided are:
- code to train from scratch a model in the hopefully more complete zoo, with an unified training framework
- code to upscale from a model
- a few metrics and loss adapted to the problem
- various weights to preload the models depending on usage (black and white vs colored for example)
## Current implementation:
 - SRGAN : Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
 - SRCNN : 
# Requirements
- pytorch
- torchvision
- Pillow
- matplotlib

# Usage

- just run train.py from console to train
- run evaluate.py to upscale

# To do
- A lot of improvement can still be done as the code is still quite redundant.
- Require more testing to check if CUDA part works.
- Consider cases with better SR datasets.
- there is a big flaw for now: each model trained as only its weights saved making the evaluation of custom model harder than it has to be
