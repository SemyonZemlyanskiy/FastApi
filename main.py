from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class DiabetInput(BaseModel):
    age: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.get('/model_info')
async def model_info(input_data: DiabetInput):
    input_array = np.array([[input_data.age, input_data.bmi, input_data.bp,
                             input_data.s1, input_data.s2, input_data.s3,
                             input_data.s4, input_data.s5, input_data.s6]])
    
    predict = model.predict(input_array)[0]

    return {"prediction": predict}
