from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI(title="PV ML Optimizer API")

# --- 1. UPGRADED PYTORCH MODEL ARCHITECTURE ---
# This MUST match the Colab training code structure exactly (16 hidden neurons)
class SolarOptimizerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# Initialize the model and load the real weights file
model = SolarOptimizerNet()
model.load_state_dict(torch.load('real_solar_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval() # Put the model in evaluation mode to lock weights for predictions

# --- 2. DEFINE THE INCOMING DATA ---
class SolarData(BaseModel):
    irradiance: float
    temperature: float
    panel_efficiency: float 

# --- 3. THE API ENDPOINT ---
@app.post("/optimize")
def get_optimization(data: SolarData):
    # NORMALIZATION: Scale the Termux data exactly like we did in Colab!
    # We divide by 1000 and 50 to keep the input numbers small for the neural network.
    scaled_irradiance = data.irradiance / 1000.0
    scaled_temp = data.temperature / 50.0
    
    input_tensor = torch.tensor([[scaled_irradiance, scaled_temp]], dtype=torch.float32)
    
    # Run the PyTorch prediction (no_grad speeds it up and saves memory)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # DENORMALIZATION: The model outputs a tiny scaled decimal.
    # We must multiply it back by 1000 to get the real-world power output in Watts!
    ml_power_watts = prediction.item() * 1000.0

    return {
        "status": "success",
        "message": "Predicted by the upgraded Deep Learning model!",
        "predicted_power_watts": round(ml_power_watts, 2)
    }

# --- 4. HOMEPAGE ---
@app.get("/")
def home():
    return {"message": "Upgraded PyTorch API is live!"}
