from pathlib import Path
import sys
import os

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import torch
import numpy as np
from tqdm import tqdm
from models.policy_net import PokerPolicyNet, StateEncoder
from utils.game_simulator import PokerGame, ActionType

class PokerTester:
    def __init__(self, model_path, num_players=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = PokerGame(num_players=num_players)
        self.encoder = StateEncoder(num_players=num_players)
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        input_dim = self.encoder.encode(self.game, 0).shape[0]
        self.policy_net = PokerPolicyNet(input_dim=input_dim).to(self.device)
        self.policy_net.load_state_dict(checkpoint['model_state'])
        self.policy_net.eval()
        
    def test(self, num_episodes=100):
        """运行测试回合"""
        results = {
            'wins': 0,
            'avg_chips': 0,
            'raise_rate': 0,
            'fold_rate': 0
        }
        total_chips = 0
        raise_count = 0
        fold_count = 0
        total_actions = 0
        
        for _ in tqdm(range(num_episodes)):
            self.game.reset()
            player_in_game = True
            
            while not self.game.is_terminal():
                current_player = self.game.current_player
                if not self.game.players[current_player].is_in_hand:
                    continue
                
                # 模型预测
                state = self.encoder.encode(self.game, current_player)
                legal_actions = self.game.get_legal_actions()
                
                with torch.no_grad():
                    action_probs, raise_ratio, _ = self.policy_net(
                        state.unsqueeze(0).to(self.device),
                        self._get_legal_mask(legal_actions).to(self.device)
                    )
                
                # 选择动作
                action_type, action_info = self._select_action(
                    action_probs[0].cpu(),
                    raise_ratio[0].item(),
                    legal_actions
                )
                
                # 统计动作
                total_actions += 1
                if action_type == ActionType.RAISE:
                    raise_count +=1
                elif action_type == ActionType.FOLD:
                    fold_count +=1
                
                # 执行动作
                self.game.apply_action(action_type, action_info.get('amount', 0))
            
            # 统计结果
            player = self.game.players[0]  # 假设测试玩家是索引0
            if player.stack > 1000:
                results['wins'] +=1
            total_chips += player.stack
        
        # 计算指标
        results['win_rate'] = results['wins'] / num_episodes
        results['avg_chips'] = total_chips / num_episodes
        results['raise_rate'] = raise_count / total_actions
        results['fold_rate'] = fold_count / total_actions
        
        return results

    def _get_legal_mask(self, legal_actions):
        """生成合法动作掩码"""
        mask = torch.zeros(4)
        action_types = [ActionType.FOLD, ActionType.CHECK_CALL, 
                       ActionType.RAISE, ActionType.ALL_IN]
        for action in legal_actions:
            if action['available']:
                idx = action_types.index(action['type'])
                mask[idx] = 1.0
        return mask

    def _select_action(self, probs, raise_ratio, legal_actions):
        """根据概率选择动作"""
        valid_actions = [a for a in legal_actions if a['available']]
        action_idx = torch.argmax(probs).item()  # 测试时选择最大概率动作
        action_type = [ActionType.FOLD, ActionType.CHECK_CALL, 
                      ActionType.RAISE, ActionType.ALL_IN][action_idx]
        
        action_info = {}
        if action_type == ActionType.RAISE:
            legal_raise = next(a for a in valid_actions 
                              if a['type'] == ActionType.RAISE)
            min_raise = legal_raise['min']
            max_raise = legal_raise['max']
            amount = min_raise + (max_raise - min_raise) * raise_ratio
            action_info['amount'] = int(amount)
        
        return action_type, action_info

if __name__ == "__main__":
    tester = PokerTester("checkpoints/model_100.pt")  # 加载训练好的模型
    results = tester.test(num_episodes=100)
    
    print("\n测试结果:")
    print(f"胜率: {results['win_rate']*100:.1f}%")
    print(f"平均筹码: {results['avg_chips']:.0f}")
    print(f"加注频率: {results['raise_rate']*100:.1f}%")
    print(f"弃牌频率: {results['fold_rate']*100:.1f}%")