from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model("trained_model.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
    
    prediction = model.predict(image_array)
    score = prediction[0][0]
    
    if score > 0.5:
        result = "Acne"
    else:
        result = "Not Acne"
        
    return {"class": result, "confidence": float(score)}
