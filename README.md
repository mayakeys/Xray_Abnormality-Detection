# Lung X-Ray Abnormality Detection
This repository presents a cleaned and consolidated version of the final modeling approach; exploratory notebooks used during development are omitted for clarity.

This project was initially developed as part of a class project for CS 156b, and explores automated detection of thoracic abnormalities in chest X-ray images using deep learning.
This repository presents a cleaned and consolidated version of the final modeling approach; exploratory notebooks used during development are omitted for clarity.

The final approach trains separate regression models per pathology, handles missing and uncertain labels, and produces submission-ready predictions for an EvalAI challenge.

## Overview

**Key ideas:**
- Each pathology is modeled **independently** (one model per label)
- Labels are treated as **continuous regression targets** rather than hard classes
- Missing labels are handled via **masked loss functions**
- Severe class imbalance is handled via **weighted sampling**
- Inference outputs are formatted to match the EvalAI submission schema

---

## Project Structure
- model.py # Model architectures (ResNet-50 variants)
- dataset.py # Dataset definition and image loading
- train.py # Final training pipeline
- make_submission.py # Inference + CSV generation for EvalAI
- experiments/
  - gridsearch.py # Hyperparameter grid search
  - flip_images.py # Offline horizontal flip augmentation
  - contrast_images.py # Offline contrast augmentation
- README.md


---

## Data Handling

- Raw X-ray images are converted to tensors and cached as `.pt` files for faster experimentation.
- Images are converted to **grayscale** and resized consistently.
- Labels may be **missing or uncertain** (e.g. `NaN`, `0.5`); these are handled explicitly rather than dropped.

---

## Model Architecture

- Backbone: **ResNet-50 (ImageNet-pretrained)**
- Head: small MLP regression head
- Output: a **single continuous value per pathology**
- Different pathologies may use slightly different head sizes or regularization

The model definitions live in `model.py`.

---

## Training Strategy

- **One model per pathology**
- Loss: **masked mean squared error (MSE)**  
  - Missing labels are ignored during loss computation
- Class imbalance handled via **WeightedRandomSampler**
- Some backbone layers are frozen during early training
- Early stopping based on validation loss

The final training pipeline is implemented in `train.py`.

---

## Offline Data Augmentation 

To expand the training set, **offline augmentations** were explored by precomputing additional image tensors:

- Horizontal flipping
- Contrast adjustment (1.5×)

These augmentations are implemented as standalone scripts in `experiments/`


Augmented images are saved as separate `.pt` files and treated as additional samples.  
Augmentation is **not applied on-the-fly during training**.

---

## Hyperparameter Search

A coarse grid search was used to explore:
- hidden layer size
- number of frozen backbone layers
- learning rate

This is implemented in `experiments/gridsearch.py`


Grid search was used **only for model selection**; final models are trained using `train.py`.

---

## Inference & Submission

- Inference is handled by `make_submission.py`
- Multiple trained models are loaded and run in batch
- Regression outputs are rescaled and combined where appropriate
- Predictions are written to a CSV file matching the EvalAI submission format

---

## Notes on Reproducibility

- Paths assume access to the original course dataset and directory structure
- Model checkpoints are not included
---





