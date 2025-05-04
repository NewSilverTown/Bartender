# analysis/action_distribution.py
import sys
import os
import numpy as np
import torch
import random
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from models.policy_net import PokerPolicyNet
from utils.game_encoder import GameEncoder
from utils.game_simulator import PokerGame, ActionType

def generate_realistic_state(num_players=6):
    """使用游戏模拟器生成真实游戏状态"""
    game = PokerGame(num_players=num_players)
    
    # 推进游戏到随机阶段（翻牌前/翻牌/转牌/河牌）
    target_phase = random.choice([0, 1, 2, 3])
    while game.game_phase < target_phase:
        # 随机模拟玩家动作直到目标阶段
        active_players = [p for p in game.players if p.is_in_hand]
        if len(active_players) < 2:
            return None  # 提前终止的牌局
            
        current_player = game.current_player
        player = game.players[current_player]
        
        # 生成随机合法动作
        if player.current_bet < max(p.current_bet for p in active_players):
            action = random.choice([ActionType.FOLD, ActionType.CALL])
        else:
            action = random.choice([ActionType.CHECK, ActionType.RAISE])
        
        # 应用动作
        if action == ActionType.RAISE:
            player = game.players[game.current_player]
            min_raise = max(10, game.big_blind)
            max_raise = max(min_raise, player.stack)  # 确保范围有效
            raise_amount = random.randint(min_raise, max_raise)
            game.apply_action(action, raise_amount)
        else:
            game.apply_action(action)
        
        # 检查阶段是否推进
        if game.is_terminal():
            game._next_phase()
    
    # 获取有效玩家状态
    active_players = [i for i, p in enumerate(game.players) if p.is_in_hand]
    if not active_players:
        return None
        
    # 随机选择一个存活玩家的视角
    player_idx = random.choice(active_players)
    game.current_player = player_idx
    
    # 编码游戏状态
    state = game.get_state()
    return np.concatenate([
        GameEncoder.encode_hand(state["hand"]),
        GameEncoder.encode_community(state["community"]),
        GameEncoder.encode_bet_history(state.get("bet_history", []))  # 安全访问
    ])

if __name__ == "__main__":
    # 初始化模型
    model_path = project_root / "models" / "poker_policy.pt"
    model = PokerPolicyNet()
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()
    
    # 统计动作分布
    action_counts = {0:0, 1:0, 2:0, 3:0}
    valid_samples = 0
    
    while valid_samples < 1000:
        state_array = generate_realistic_state()
        if state_array is None:
            continue
            
        # 转换为模型输入
        state_tensor = torch.FloatTensor(state_array)
        
        with torch.no_grad():
            probs = torch.softmax(model(state_tensor), dim=0)
            action = torch.argmax(probs).item()
        
        action_counts[action] +=1
        valid_samples +=1
        
        if valid_samples % 100 == 0:
            print(f"已处理 {valid_samples} 个有效状态...")
    
    # 打印统计结果
    total = sum(action_counts.values())
    print("\n动作分布统计（基于真实游戏状态）:")
    print(f"弃牌率: {action_counts[0]/total*100:.1f}%")
    print(f"跟注率: {action_counts[1]/total*100:.1f}%")
    print(f"加注率: {action_counts[2]/total*100:.1f}%")
    print(f"全押率: {action_counts[3]/total*100:.1f}%")