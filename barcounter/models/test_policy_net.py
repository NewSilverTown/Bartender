import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import numpy as np
from policy_net import PokerPolicyNet
from utils.game_simulator import ActionType

# 测试代码
def test_policy_net():
    # 模拟游戏状态（修复字段缺失问题）
    state_dim = 128
    dummy_state = torch.randn(state_dim)
    
    # 合法的动作定义（包含所有必要字段）
    legal_actions = [
        {'type': ActionType.FOLD, 'available': True, 'min': 0, 'max': 0},
        {'type': ActionType.CHECK_CALL, 'available': True, 'min': 0, 'max': 0},
        {'type': ActionType.RAISE, 
         'available': True,
         'min': 50, 
         'max': 200,
         'player_stack': 1000},
        {'type': ActionType.ALL_IN, 'available': False, 'min': 0, 'max': 0}
    ]
    
    # 初始化网络
    policy_net = PokerPolicyNet(state_dim=state_dim)
    
    # 执行预测
    prediction = policy_net.predict(dummy_state, legal_actions)
    
    print("预测结果:")
    print(f"选择动作: {prediction['action']['type'].name}")
    print(f"动作概率分布: {prediction['probs']}")
    if prediction['action']['type'] == ActionType.RAISE:
        print(f"预测加注金额: {prediction['action']['amount']}")
        print(f"加注比例: {prediction['raise_ratio']:.2f}")

if __name__ == "__main__":
    test_policy_net()