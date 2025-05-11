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

request_data2 = {
  "player_idx" : 1,
  "player_hand" : [ "10d", "5s" ],
  "community_cards" : [ ],
  "player_stack" : 1485,
  "pot" : 25,
  "current_bet" : 0,
  "legal_actions" : [ {
    "type" : "FOLD",
    "available" : True,
    "min" : None,
    "max" : None
  }, {
    "type" : "CHECK_CALL",
    "available" : True,
    "min" : None,
    "max" : None
  }, {
    "type" : "RAISE",
    "available" : True,
    "min" : 10,
    "max" : 1485
  } ]
}

response = requests.post("http://localhost:8000/predict", json=request_data2)
print(response.json())