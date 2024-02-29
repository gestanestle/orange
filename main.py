from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from json import dumps
from uvicorn import run
import os

app = FastAPI()

model = tf.keras.models.load_model("orange_2-15-0.h5")  

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)


@app.get("/")
async def root():
    return {"message": "Welcome to the Orange API!"}


@app.post("/prediction")
async def get_image_prediction(file: UploadFile):
    img_content = await file.read()  
    img = cv2.imdecode(np.frombuffer(img_content, np.uint8), cv2.IMREAD_COLOR) 
    resize = tf.image.resize(img, (256, 256)) 
    prediction = model.predict(tf.expand_dims(resize / 255, 0))
    result = prediction[0].tolist()[0]
    return result > 0.5 # Is fruit rotten?
