# ai_service/app/main.py
from fastapi import FastAPI
import torch

app = FastAPI()
model = PokerPolicyNet()
model.load_state_dict(torch.load("models/poker_policy.pt"))

@app.post("/predict")
async def predict(game_state: dict):
    state = encode_state(game_state)
    with torch.no_grad():
        probs = model(state)
    action_idx = torch.argmax(probs).item()
    return {"action": ["fold", "call", "raise", "all_in"][action_idx]}