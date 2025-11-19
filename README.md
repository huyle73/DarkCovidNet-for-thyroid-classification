
# DarkCovidNet for Thyroid Cytology Classification (3-class)

This repository provides a clean Python implementation of **DarkCovidNet**, adapted
for **3-class thyroid cytology classification** (B4 / B5 / B6).  
All training and evaluation previously performed in a Jupyter Notebook
has been fully rewritten into two standalone Python scripts:

- `darkcovidnet_3class.py` — Training the model  
- `evaluate.py` — Evaluating a saved model

Both **FastAI (.pkl)** and **raw PyTorch (.pth)** model formats are supported.

---

## 1. Project Structure

```
DarkCovidNet-for-thyroid-classification/
│
├── darkcovidnet_3class.py          # Training script
├── evaluate.py                      # Evaluation script
├── darkcovidnet_3class.pkl         # Saved FastAI model (optional)
├── darkcovidnet_3class_state.pth   # Saved PyTorch weights (optional)
│
└── thyroid_dataset_more_classes_full_3class/
    ├── train/
    │   ├── B4/
    │   ├── B5/
    │   └── B6/
    └── valid/
        ├── B4/
        ├── B5/
        └── B6/
```

Dataset must follow FastAI's **ImageDataLoaders.from_folder** structure.

---

## 2. Training the Model

Example command:

```bash
python darkcovidnet_3class.py   --data-dir "thyroid_dataset_more_classes_full_3class"   --epochs 60   --bs 32   --save-model "darkcovidnet_3class.pkl"   --save-torch "darkcovidnet_3class_state.pth"
```

### Arguments

| Argument | Description |
|---------|-------------|
| `--data-dir` | Path to dataset root folder |
| `--epochs` | Number of training epochs |
| `--bs` | Batch size (default: 32) |
| `--lr` | Learning rate (optional) |
| `--save-model` | Save FastAI Learner (.pkl) |
| `--save-torch` | Save PyTorch weights (.pth) |

The script prints:
- Class list (B4/B5/B6)
- Model summary
- Training & validation accuracy
- Confusion matrix
- Classification report

---

## 3. DarkCovidNet Architecture (Modified)

The implementation includes:

- `conv_block` + LeakyReLU
- `triple_conv` module
- MaxPooling downsampling
- **AdaptiveAvgPool2d(1)** to replace the large flatten layer  
- Dropout 0.3
- Final fully connected layer → 3 classes

This improves generalization and reduces overfitting compared to the original DarkCovidNet design.

---

## 4. Evaluating a Saved Model

Run evaluation with PyTorch `.pth` weights:

```bash
python evaluate.py   --data-dir "thyroid_dataset_more_classes_full_3class"   --bs 32   --model-state "darkcovidnet_3class_state.pth"
```

Or evaluate FastAI exported `.pkl` model:

```bash
python evaluate.py   --data-dir "thyroid_dataset_more_classes_full_3class"   --bs 32   --model-pkl "darkcovidnet_3class.pkl"
```

Evaluation outputs include:

- Validation accuracy
- Manual accuracy check
- Confusion matrix
- Precision / recall / F1-score

---

## 5. Saving Models

### Save FastAI learner

```python
learn.export("darkcovidnet_3class.pkl")
```

### Save PyTorch model weights

```python
torch.save(learn.model.state_dict(), "darkcovidnet_3class_state.pth")
```

### Load PyTorch model manually

```python
model = build_darkcovidnet(num_classes=3)
state = torch.load("darkcovidnet_3class_state.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

---

## 6. Installation

Recommended environment:

```bash
pip install fastai torch torchvision scikit-learn numpy
```

---

