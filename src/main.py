from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from shutil import unpack_archive
import uvicorn

app = FastAPI()
model_path = "models/model.h5"

# Load model once when API starts
model = load_model(model_path)

@app.get("/")
def read_root():
    return {"message": "ML Image Classification API is up and running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("temp.jpg", "wb") as f:
            f.write(contents)

        img = image.load_img("temp.jpg", target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        prediction = model.predict(img_tensor)
        predicted_class = np.argmax(prediction)

        return {"prediction": int(predicted_class)}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/upload-data")
async def upload_zip(file: UploadFile = File(...)):
    try:
        zip_path = f"temp_upload.zip"
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        unpack_archive(zip_path, "data/train", format='zip')
        os.remove(zip_path)

        return {"message": "Images uploaded and extracted for retraining."}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/retrain")
def retrain_model():
    try:
        # NOTE: Use your own model training function here
        os.system("python retrain.py")
        return {"message": "Retraining started successfully."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional: for local testing
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
