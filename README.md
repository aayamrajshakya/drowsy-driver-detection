# Driver Drowsiness Detection

This project implements a real-time driver drowsiness detection system using a stacking ensemble of pre-trained large CNNs and OpenCV.


## Elevator Pitch 🎬

<a href="https://www.youtube.com/watch?v=7zd7ujhZyNg" target="_blank">
  <img src="https://img.youtube.com/vi/7zd7ujhZyNg/maxresdefault.jpg" alt="Driver Drowsiness Detection Video" max-width="100%" width="650" style="border-radius: 8px;" />
</a>

> Click the thumbnail above to watch the demo video on YouTube.


## Overview

- Stacking ensemble of **DenseNet121**, **EfficientNetV2B2**, and **MobileNetV2**
- Hyperparameter tuning was performed using **Bayesian optimization** with Optuna
- Face detection is performed on a **real-time webcam feed** using OpenCV and Haar cascade classifier
- **Continuous feeding** of extracted face frames to a trained TensorFlow model for drowsiness classification
- **Audio alert** (`alert.wav`) activated upon drowsy state detection


## File structure

```
├── alert.wav                  # Sound alert played on drowsiness detection
├── best_hyperparameters.yaml  # Example conf. file for optimal hyperparameters from Optuna fine-tuning
├── class_names.json           # Classification labels
├── cv2model.py                # Real-time detection GUI app
├── data_clean                 # EXTRA (not used)
├── deps.txt                   # Required dependencies
├── ensemble.ipynb             # Main codebase
├── example.keras              # Sample pretrained model to use with cv2model.py
└── gradio.py                  # Gradio GUI
```


## Installation

Install the required dependencies using:

```bash
pip install -r deps.txt
```

If needed, download the Haar cascade classifier file from: [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
