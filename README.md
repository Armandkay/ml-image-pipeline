# 🧠 ML Image Classification Pipeline

## 🎯 Project Overview

This project demonstrates an end-to-end image classification pipeline using a Convolutional Neural Network (CNN). It includes:

- Image preprocessing and augmentation
- CNN model training and evaluation
- A retraining mechanism for new uploaded data
- A web-based API built with FastAPI
- A user interface to predict and update model behavior
- Deployment-ready structure with modular components

---

## 📹 Video Demonstration

You can watch a full walkthrough of this project here:

👉 [**Watch the video demo**](#) ← *https://www.loom.com/share/a14b94e8b253495ab066e89a643ede9a?sid=db9828d9-5a89-4eb7-ae82-3f67ed43045f*

---

## 📂 GitHub Repository

To get started, clone the repository:

```bash
git clone https://github.com/Armandkay/ml-image-pipeline.git
cd ml-image-pipeline
```

---

## 🧾 Project Structure

```
ml-image-pipeline/
├── README.md
├── notebook/
│   └── ml_pipeline_final.ipynb        # Full development notebook
├── src/
│   ├── main.py                        # FastAPI backend
│   ├── retrain.py                     # Retraining script
│   ├── preprocessing.py               # Preprocessing logic
│   ├── model.py                       # CNN architecture
│   └── prediction.py                  # Image prediction handler
├── data/
│   ├── train/                         # Training images (organized by class)
│   └── test/                          # Test images
├── models/
│   └── model.h5                       # Trained model weights
├── requirements.txt
```

---

## ⚙️ Setup Instructions

1. *(Optional)* Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:

```bash
uvicorn src.main:app --reload
```

4. Open the API documentation in your browser:

```
http://127.0.0.1:8000/docs
```

---

## 🚀 Usage Guide

### 🧠 Predict an Image
- Use the `/predict` endpoint in the API docs or via the web UI.
- Upload a single image file.
- The model returns the predicted class label.

### 📤 Upload New Training Data
- Use the `/upload-data` endpoint.
- Upload a `.zip` file with images inside folders named by class (e.g., `cat/`, `dog/`).
- Images are saved to the `data/train/` folder.

### 🔁 Retrain the Model
- Call the `/retrain` endpoint.
- It will retrain the CNN on the updated dataset.
- The updated weights are saved as `models/model.h5`.

### ✅ API Health Check
- Call the `/status` endpoint to verify the API is running.

---

## 📈 Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix (visualized in the notebook)

Early stopping and dropout were used to prevent overfitting.

---

## 🧪 Model Training

- The model is a Sequential CNN using 2 convolutional layers and max pooling.
- Input images are grayscale, resized to 28x28.
- Uses `Adam` optimizer and `sparse_categorical_crossentropy` loss.
- Trained in a Google Colab notebook using uploaded image datasets.

---

## ❌ Performance Testing

> Load testing using Locust was not implemented, as it was not part of the rubric. The focus was on the retraining, prediction, evaluation, and deployment processes.

---

## 📬 Contact

**Armand Kayiranga**  
📧 [a.kayiranga1@alustudent.com](mailto:a.kayiranga1@alustudent.com)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
