# DarkCovidNet 3-Class Classification

Python Implementation (Converted from Jupyter Notebook)

This repository contains a Python implementation of **DarkCovidNet**
adapted for **3-class thyroid cytology image classification** (B4, B5,
B6).\
The code is converted from the original FastAI Jupyter notebook into a
fully runnable `.py` script with a `main()` entry point.

## 1. Requirements

Install dependencies:

    pip install fastai torch torchvision scikit-learn numpy

## 2. Dataset Structure

Your dataset must follow this structure:

    thyroid_dataset_more_classes_full_3class/
    ├── train/
    │   ├── B4/
    │   ├── B5/
    │   └── B6/
    └── valid/
        ├── B4/
        ├── B5/
        └── B6/

## 3. Script Description

Main script:

    darkcovidnet_3class.py

Includes: - Model (DarkCovidNet) - Data loading with FastAI - Training
using `fit_one_cycle` - Evaluation (accuracy, confusion matrix,
classification report) - Optional model export

## 4. How to Run

### Basic training

    python darkcovidnet_3class.py --data-dir "/path/to/data"

### Custom epochs

    python darkcovidnet_3class.py --epochs 100

### Set learning rate

    python darkcovidnet_3class.py --lr 1e-3

### Change batch size

    python darkcovidnet_3class.py --bs 16

## 5. Saving the Model

    python darkcovidnet_3class.py --save-model "darkcovidnet.pkl"

## 6. Evaluation Output

The script prints: - FastAI accuracy
- Manual accuracy
- Confusion matrix
- Classification report

## 7. Model Architecture

-   Conv block (3 → 16)
-   Triple conv blocks (16→32, 32→64)
-   Conv (64→128)
-   MaxPooling
-   Linear classifier 32768 → 3

## 8. Example Full Command

    python darkcovidnet_3class.py   --data-dir "/path/to/data"   --epochs 60   --bs 32   --lr 1e-3   --save-model "darkcovidnet.pkl"
