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
        active_players = [p for p in game.players if p.is_in_hand]
        if len(active_players) < 2:
            return None
            
        current_player = game.current_player
        player = game.players[current_player]
        
        # 生成合法动作集
        valid_actions = []
        current_bet = player.current_bet
        max_bet = max(p.current_bet for p in active_players)
        
        # 基础动作
        if current_bet < max_bet:
            valid_actions.extend([ActionType.FOLD, ActionType.CALL])
        else:
            valid_actions.append(ActionType.CHECK)
        
        # 加注条件
        if player.stack > max(10, game.big_blind):
            valid_actions.append(ActionType.RAISE)
        
        # 全押条件
        if player.stack > 0:
            valid_actions.append(ActionType.ALL_IN)
        
        # 至少保留一个合法动作
        if not valid_actions:
            valid_actions.append(ActionType.CHECK)
        
        action = random.choice(valid_actions)
        
        # 安全生成加注金额
        if action == ActionType.RAISE:
            min_raise = max(10, game.big_blind)
            max_raise = player.stack
            if max_raise < min_raise:
                action = ActionType.ALL_IN  # 自动转换全押
                game.apply_action(action)
            else:
                raise_amount = random.randint(min_raise, max_raise)
                game.apply_action(action, raise_amount)
        else:
            game.apply_action(action)
        
        if game.is_terminal():
            game._next_phase()
    
    # 获取有效玩家状态
    active_players = [i for i, p in enumerate(game.players) if p.is_in_hand]
    if not active_players:
        return None
        
    # 随机选择存活玩家视角
    player_idx = random.choice(active_players)
    game.current_player = player_idx
    player = game.players[player_idx]
    
    # 编码游戏状态
    state = game.get_state()
    return np.concatenate([
        GameEncoder.encode_hand(state["hand"]),
        GameEncoder.encode_community(state["community"]),
        GameEncoder.encode_bet_history(state["bet_history"]),
        [player.stack / 5000]  # 与训练保持一致的归一化方式
    ])

if __name__ == "__main__":
   # 初始化模型
    model_path = project_root / "models" / "best_policy.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载检查点并校验
    checkpoint = torch.load(str(model_path), map_location=device)
    model = PokerPolicyNet(input_dim=183).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 添加诊断输出
    test_state = torch.randn(1, 183).to(device)
    print("测试输入输出示例:")
    with torch.no_grad():
        print("原始logits:", model(test_state))
        print("概率分布:", torch.softmax(model(test_state), dim=1))
    
    # 统计动作分布（添加探索机制）
    action_counts = {0:0, 1:0, 2:0, 3:0}
    for _ in range(1000):
        state_array = generate_realistic_state()
        if state_array is None:
            continue
        
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(state_tensor)
            probs = torch.softmax(logits, dim=1)
            
            # 改为按概率采样（而非argmax）
            action = torch.multinomial(probs, 1).item()
        
        action_counts[action] += 1
    
    # 打印统计结果
    total = sum(action_counts.values())
    print("\n动作分布统计（基于真实游戏状态）:")
    print(f"弃牌率: {action_counts[0]/total*100:.1f}%")
    print(f"跟注率: {action_counts[1]/total*100:.1f}%")
    print(f"加注率: {action_counts[2]/total*100:.1f}%")
    print(f"全押率: {action_counts[3]/total*100:.1f}%")