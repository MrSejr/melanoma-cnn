# Melanoma Image Classification with CNN

Course project for 02461: Introduction to Intelligent Systems, DTU (2024).  
Built and trained a convolutional neural network to classify skin lesions as benign or malignant.

## Results

| Metric | Combined dataset (~50k images) | Balanced dataset (~10k images) |
|---|---|---|
| Accuracy | 92.9% | 91.8% |
| Specificity | 97.9% | 93.0% |
| Recall | 61.5% | 90.6% |
| F1-score | 70.3% | 91.5% |

The combined dataset model has higher specificity but lower recall due to class imbalance (46,600 benign vs 7,100 malignant images). The trade-offs are discussed in the project report.

## Model Architecture

Two-layer CNN built in PyTorch:
- Conv layer → ReLU → MaxPool (×2)
- Fully connected layers (32×73×73 → 128 → 1)
- Sigmoid output for binary classification
- Adam optimizer, Binary Cross-Entropy loss
- Trained for 100 epochs on GPU (CUDA)

## Files

- `pytorch_CNN_50000.py` — model architecture and training loop
- `test_accuracy.py` — evaluation on test set with confusion matrix

## Dataset

Three datasets merged into one (~53,700 images total), split 80/10/10 for train/val/test:
- [Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) (Kaggle)
- [HAM10000](https://doi.org/10.7910/DVN/DBW86T) (Tschandl, 2018)
- [ISIC 2020](https://doi.org/10.1038/s41597-021-00815-z) (Rotemberg et al.)
