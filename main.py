from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI(title="PV ML Optimizer API -INGAWA")

# --- 1. DEFINE THE PYTORCH MODEL ---
# A very simple neural network that takes 2 inputs and gives 1 output
class SolarOptimizerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple linear layer
        self.linear = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear(x)

# Initialize the model 
# (In a real app, you would load your pre-trained weights here)
model = SolarOptimizerNet()

# --- 2. DEFINE THE INCOMING DATA ---
class SolarData(BaseModel):
    irradiance: float
    temperature: float
    panel_efficiency: float

# --- 3. THE API ENDPOINT ---
@app.post("/optimize")
def get_optimization(data: SolarData):
    # Convert the incoming Termux data into a PyTorch Tensor
    # We are feeding it Irradiance and Temperature
    input_tensor = torch.tensor([[data.irradiance, data.temperature]], dtype=torch.float32)
    
    # Run the PyTorch model prediction (without tracking gradients to save memory)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Extract the actual number from the PyTorch Tensor
    ml_power = prediction.item()

    return {
        "status": "success",
        "message": "Calculated by PyTorch in the cloud!",
        "predicted_power": round(ml_power, 2)
    }

# --- 4. HOMEPAGE ---
@app.get("/")
def home():
    return {"message": "My PyTorch API is alive and ready!"}
