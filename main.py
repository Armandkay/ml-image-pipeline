from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from zipfile import ZipFile
from src import model, preprocessing, prediction

app = FastAPI()

@app.get("/status")
def status():
    return {"status": "Model API is up and running."}

@app.post("/predict")
async def predict_image_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        class_name = prediction.predict_image(
            model_path="models/model.h5",
            image_path=temp_path,
            image_size=(64, 64),
        )
        os.remove(temp_path)
        return {"prediction": class_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload-data")
async def upload_data(zipfile: UploadFile = File(...)):
    data_path = "data/train/"
    zip_path = f"temp/{zipfile.filename}"
    os.makedirs("temp", exist_ok=True)

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(zipfile.file, buffer)

    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_path)
        return {"message": "Data uploaded successfully!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/retrain")
def retrain_model():
    try:
        train_data, test_data = preprocessing.load_data("data/train", "data/test")
        input_shape = train_data.image_shape
        num_classes = train_data.num_classes

        m = model.build_model(input_shape, num_classes)
        m, _ = model.train_model(m, train_data, test_data, epochs=5)
        model.save_model(m, path="models/model.h5")

        return {"message": "Model retrained and saved!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
