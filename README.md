# Automating Flower Classification for a Start-up Using Deep Learning

This project demonstrates a robust deep learning pipeline for classifying flower species using transfer learning and hyperparameter tuning. The workflow leverages TensorFlow, Keras, and KerasTuner, and compares the performance of ResNet50 and DenseNet121 backbones on the Oxford Flowers 102 dataset.

## Project Structure after running code

```
├── automating_flower_classification_for_start_up_using_deep_learning_gc.ipynb
├── requirements.txt
├── train_features.npy / val_features.npy / test_features.npy
├── train_labels.npy / val_labels.npy / test_labels.npy
├── models/
│   ├── densenet121_tuned_numpy.keras
│   └── effb0_tuned.keras
├── tuning/           # KerasTuner logs for ResNet50
├── tuning_numpy/     # KerasTuner logs for DenseNet121
```

## Features
- **Data Loading & Preprocessing:**
  - Uses TensorFlow Datasets (TFDS) to load Oxford Flowers 102.
  - Robust image resizing, normalization, and augmentation.
- **Feature Extraction:**
  - Extracts features using pre-trained ResNet50 and DenseNet121 backbones.
  - Applies global average pooling for compact feature vectors.
- **Model Building:**
  - Custom classification head with dense layers, batch normalization, dropout, and label smoothing.
  - Supports both sequential and functional Keras APIs.
- **Hyperparameter Tuning:**
  - Utilizes KerasTuner for optimizing learning rate, dense units, dropout, and fine-tuning depth.
- **Training & Evaluation:**
  - Early stopping, learning rate scheduling, and model checkpointing.
  - Evaluation on validation and test sets with accuracy and loss curves.
- **Reproducibility:**
  - Feature vectors and labels are saved as `.npy` files for fast reloading.

## Results Summary

### ResNet50
- Feature extraction and classification pipeline implemented.
- Training and validation accuracy increased in early epochs, but test accuracy remained very low (~0.5%).
- Model struggled to generalize, indicating possible underfitting or insufficient model complexity.

### DenseNet121
- Improved preprocessing, augmentation, and hyperparameter search.
- Validation accuracy reached ~20%, but training accuracy remained lower, and test accuracy was still low.
- Indicates the task is challenging for the dataset size and model capacity; further tuning or more data may be needed.

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Open the notebook:**
   - `automating_flower_classification_for_start_up_using_deep_learning.ipynb`
3. **Run all cells:**
   - The notebook will download data, preprocess, extract features, tune, train, and evaluate models.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- KerasTuner
- numpy, matplotlib, seaborn, pandas

## References
- [Oxford Flowers 102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Keras Applications](https://keras.io/api/applications/)
- [KerasTuner](https://keras.io/keras_tuner/)

## Author
Hakim Murphy
