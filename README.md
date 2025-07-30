````markdown
# ML Image Classification Pipeline

## Project Overview

This project implements a complete machine learning pipeline for image classification using convolutional neural networks (CNN). It covers:

- Data acquisition and preprocessing
- Model creation and evaluation
- Deployment of the model through a FastAPI backend
- Model retraining triggered by new data uploads
- API endpoints for prediction, retraining, and status monitoring
- Flood testing to evaluate API performance under load

---

## GitHub Repository

Clone the project repository:

```bash
git clone https://github.com/Armandkay/ml-image-pipeline.git
cd ml-image-pipeline
````

---

## Project Structure

```
ml-image-pipeline/
├── README.md
├── notebook/
│   └── ml_image_pipeline.ipynb
├── src/
│   ├── main.py
│   ├── retrain.py
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── model.h5
├── requirements.txt
```

---

## Setup Instructions

1. (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI application:

```bash
uvicorn src.main:app --reload
```

4. Open the interactive API documentation in your browser:

```
http://127.0.0.1:8000/docs
```

---

## Usage Guide

### Predict an Image

* Use the `/predict` endpoint.
* Upload a single image file.
* The API returns the predicted class label.

### Upload Training Data

* Use the `/upload-data` endpoint.
* Upload a ZIP file containing images organized by class folders.
* The images will be extracted to the `data/train` folder.

### Retrain the Model

* Use the `/retrain` endpoint.
* The model will retrain using the current training data.
* The updated model will be saved to `models/model.h5`.

### Check API Status

* Use the `/status` endpoint.
* Confirms the API is running and responsive.

---

## Flood Testing Summary

* API performance was evaluated under simulated load using Locust.
* Latency and throughput metrics were collected at different concurrency levels.
* Detailed test results and graphs are available in the project documentation.

---

## Contact Information

Armand Kayiranga
Email: [a.kayiranga1@alustudent.com](mailto:a.kayiranga1@alustudent.com)

---

## License

This project is licensed under the MIT License.

```
