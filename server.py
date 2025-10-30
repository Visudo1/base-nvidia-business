/src/api/server.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from src.inference.runner import InferenceRunner

app = FastAPI()
runner = InferenceRunner("/path/to/model.pt", device="cuda:0")

class ImgInput(BaseModel):
    data: list  # asumsi data flatten atau base64

@app.post("/predict")
def predict(item: ImgInput):
    # konversi data jadi tensor sesuai bentuk model
    input_tensor = torch.tensor(item.data, dtype=torch.float32).unsqueeze(0)
    out = runner.run(input_tensor)
    return {"output": out.tolist()}
