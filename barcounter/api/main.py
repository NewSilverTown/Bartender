import os
import sys
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录（假设analysis在根目录下）
project_root = os.path.dirname(current_dir)
# 将根目录添加到Python路径
sys.path.append(project_root)

model_for_8_players_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "8_model_8000.pt")
model_for_6_players_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "6_model_8000.pt")

import torch
from utils.game_simulator import PokerGame, ActionType
from models.policy_net import PokerPolicyNet,StateEncoder, load_model

app = FastAPI()

# 加载模型（启动时加载）
model_for_6_players = load_model(model_for_6_players_path, device="cpu")
model_for_8_players = load_model(model_for_8_players_path, device="cpu")

class GameStateRequest(BaseModel):
    player_hand: list[str]
    community_cards: list[str]
    player_stack: int
    pot: int
    current_bet: int
    legal_actions: list[Dict[str, Any]]
    player_idx: int
    num_players: int

@app.post("/predict")
async def predict_action(request: GameStateRequest):
    try:
        # 将请求转换为PokerGame实例
        game = create_dummy_game(request)
        
        # 调用模型预测
        action_info = None
        if request.num_players == 8:
            action_info = model_for_8_players.predict(game, request.player_idx)
        else:
            action_info = model_for_6_players.predict(game, request.player_idx)
        
        print(f"actionInfo", action_info)

        response = {
            "action_type": action_info['type'].name,
            "raise_amount": action_info['raise_amount'],
            "probs": action_info['probs']
        }

        print(f"response", response)

        return response
    except Exception as e:
        traceback.print_exc()
        print(f"异常:", e)
        raise HTTPException(status_code=500, detail=str(e))

def create_dummy_game(request: GameStateRequest) -> PokerGame:
    """根据请求创建虚拟游戏实例"""
    game = PokerGame(num_players=6)
    
    # 设置当前玩家状态
    player = game.players[request.player_idx]
    player.hand = request.player_hand
    player.stack = request.player_stack
    player.current_bet = request.current_bet
    
    # 设置公共牌
    game.community_cards = request.community_cards
    game.pot = request.pot
    
    # 设置合法动作
    for action in request.legal_actions:
        game.available_actions.append({
            "type": ActionType[action['type']],
            "min": action.get('min', 0),
            "max": action.get('max', 0),
            "available": action['available']
        })
    
    return game

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)