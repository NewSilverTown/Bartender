import requests

request_data = {
    "player_hand": ["Ah", "Kh"],
    "community_cards": ["Qh", "Jh", "Th"],
    "player_stack": 1500,
    "pot": 2500,
    "current_bet": 500,
    "legal_actions": [
        {"type": "FOLD", "available": True},
        {"type": "CHECK_CALL", "available": True},
        {"type": "RAISE", "min": 200, "max": 500, "available": True}
    ],
    "player_idx": 0
}

response = requests.post("http://localhost:8000/predict", json=request_data)
print(response.json())