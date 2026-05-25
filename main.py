from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI(title="PV ML Optimizer API")

# --- 1. PYTORCH MODEL ARCHITECTURE ---
# Must match your Colab training architecture exactly (16 hidden neurons)
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

# Initialize the model and load the weights using the requested filename
model = SolarOptimizerNet()
model.load_state_dict(torch.load('solar_model_weights.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval() # Put the model in evaluation mode to lock weights

# --- 2. DEFINE THE INCOMING DATA ---
class SolarData(BaseModel):
    irradiance: float
    temperature: float
    panel_efficiency: float 

# --- 3. THE API ENDPOINT ---
@app.post("/optimize")
def get_optimization(data: SolarData):
    # NORMALIZATION: Scale inputs exactly like we did during training
    scaled_irradiance = data.irradiance / 1000.0
    scaled_temp = data.temperature / 50.0
    
    input_tensor = torch.tensor([[scaled_irradiance, scaled_temp]], dtype=torch.float32)
    
    # Run the PyTorch prediction safely without gradient tracking
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # DENORMALIZATION: Convert the scaled network decimal back to real Watts
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
