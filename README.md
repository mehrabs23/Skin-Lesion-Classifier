# Skin Lesion Classifier
A custom image classification project using **PyTorch**, **FastAPI**, and **Streamlit** using transfer learning (ResNet50) to detect and classify different types of skin lesions, including malignant and benign ones, with high accuracy.

## Problem Statement
Skin cancer is one of the most common cancers worldwide, and early detection significantly improves survival rates. However, access to dermatologists is limited in many parts of the world. An automated image classification system could help triage cases and support medical professionals with faster diagnoses.

## Dataset Description

- Source : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Number of classes :  7
- Class Labels :

         -Actinic keratoses (akiec)

         - Basal cell carcinoma (bcc)

         - Benign keratosis-like lesions (bkl)

         - Dermatofibroma (df)

         - Melanoma (mel)

         - Melanocytic nevi (nv)

         - Vascular lesions (vasc)

- Total images: 10,015
- Image size : Original 600 x 450, resized to 224 x 224 for ResNet
- Train/Validation/Test Split : 70/20/10

## Project Structure

```bash
skin-lesion-classifier/
├── model/             # Model loading & checkpoint
│   └── model.py
├── api/               # FastAPI service
│   ├── app.py
│   ├── utils.py
│   └── requirements.txt
├── streamlit_app/     # Streamlit demo UI
│   └── app.py
├── Dockerfile         # Optional containerization
├── README.md
└── .gitignore
```

## Features

- Transfer learning with ResNet50 (PyTorch)
- Custom dataset loading
- Fine-tuning & augmentation
- FastAPI for prediction
- Optional Streamlit UI for demos
- Docker-ready

## How to Train

Coming soon: 

## How to Use the API

```bash
# Start the FastAPI server
uvicorn api.app:app --reload

```
Then POST an image to:

http://localhost:8000/predict

## Metrics

Evaluation: accuracy, precision, recall, confusion matrix, GradCAM, etc

## Tech Stack

- Python
- PyTorch
- FastAPI
- Streamlit
- Docker 

## Credits

**Mentors & resources** :

