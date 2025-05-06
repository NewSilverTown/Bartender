import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from typing import List, Dict
from torch import nn
import torch.nn.functional as F
from utils.game_simulator import ActionType, PokerGame



class StateEncoder:
    """游戏状态编码器（兼容PokerGame类）"""
    def __init__(self, num_players=6):
        self.num_players = num_players
        self.hand_size = 2
        self.community_size = 5
        
    def encode(self, game: PokerGame, player_idx: int) -> torch.Tensor:
        """编码游戏状态为特征向量"""
        features = []
        
        # 玩家个人特征
        player = game.players[player_idx]
        features += [
            player.stack / 1000.0,  # 归一化筹码
            player.current_bet / 1000.0,
            int(player.is_in_hand),
            int(player.is_all_in)
        ]
        
        # 手牌编码
        hand_ranks = [self._card_rank(c) for c in player.hand]
        features += sorted(hand_ranks, reverse=True)
        
        # 公共牌编码
        community_ranks = [self._card_rank(c) for c in game.community_cards]
        features += community_ranks + [0]*(5-len(community_ranks))  # 补齐5张
        
        # 对手特征
        for i in range(self.num_players):
            if i == player_idx:
                continue
            opp = game.players[i]
            features += [
                opp.stack / 1000.0,
                opp.current_bet / 1000.0,
                int(opp.is_in_hand),
                int(opp.is_all_in)
            ]
            
        # 回合信息
        features += [
            game.game_phase / 3.0,  # 归一化回合阶段
            game.pot / 3000.0,
            (game.current_player == player_idx)  # 是否当前玩家
        ]
        
        return torch.FloatTensor(features)
    
    def _card_rank(self, card: str) -> float:
        """将卡牌转换为数值（2=0.0, A=1.0）"""
        rank_map = {'2':0.0, '3':0.1, '4':0.2, '5':0.3, '6':0.4,
                   '7':0.5, '8':0.6, '9':0.7, 'T':0.8, 
                   'J':0.85, 'Q':0.9, 'K':0.95, 'A':1.0}
        return rank_map.get(card[0], 0.0)
    
    def _evaluate_hand_strength(self, hand):
        """简单手牌强度评估"""
        rank_values = {'2':0, '3':1, '4':2, '5':3, '6':4, 
                      '7':5, '8':6, '9':7, 'T':8, 'J':9, 
                      'Q':10, 'K':11, 'A':12}
        ranks = sorted([rank_values[c[0]] for c in hand], reverse=True)
        
        # 对子/同花/顺子潜力
        is_pair = (ranks[0] == ranks[1])
        suited = (hand[0][1] == hand[1][1])
        connected = (abs(ranks[0] - ranks[1]) <= 1)
        
        strength = 0.0
        if is_pair:
            strength += 0.5
        if suited:
            strength += 0.3
        if connected:
            strength += 0.2
        return strength

class PokerPolicyNet(nn.Module):
    """强化学习策略网络（兼容PokerGame）"""
    def __init__(self, input_dim=128):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
        # 动作头
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 4)  # 对应4种动作类型
        )
        
        # 加注金额预测（修复维度问题）
        self.raise_head = nn.Sequential(
            nn.Linear(128, 1),  # 输出维度改为1
            nn.Sigmoid()  # 输出0-1之间的比例
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x, legal_mask=None):
        x = self.feature_net(x)
        
        # 动作概率
        action_logits = self.action_head(x)
        if legal_mask is not None:
            action_logits = action_logits + torch.log(legal_mask + 1e-6)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 加注比例（修复维度问题）
        raise_ratio = self.raise_head(x).squeeze(-1)  # 压缩最后一个维度
        
        # 状态价值
        value = self.value_head(x)
        
        return action_probs, raise_ratio, value

    def predict(self, game: PokerGame, player_idx: int) -> Dict:
        """生成动作决策"""
        # 编码状态
        encoder = StateEncoder()
        state = encoder.encode(game, player_idx).float()
        legal_actions = game.get_legal_actions()
        
        # 生成合法动作掩码
        legal_mask = torch.zeros(4)
        action_types = [ActionType.FOLD, ActionType.CHECK_CALL, 
                      ActionType.RAISE, ActionType.ALL_IN]
        for action in legal_actions:
            if action['available']:
                idx = action_types.index(action['type'])
                legal_mask[idx] = 1.0
            else:
                #显式禁止不可用动作
                idx = action_types.index(action['type'])
                action_probs[0][idx] = 0.0  # 直接置零
        
        with torch.no_grad():
            action_probs, raise_ratio, _ = self(state.unsqueeze(0), 
                                             legal_mask.unsqueeze(0))
        
        action_probs = action_probs * legal_mask  # 二次屏蔽
        action_probs = action_probs / action_probs.sum()  # 重新归一化
        
        # 选择动作
        valid_probs = action_probs[0][legal_mask.bool()]
        valid_probs /= valid_probs.sum()
        selected_idx = torch.multinomial(valid_probs, 1).item()
        
        # 映射回原始动作类型
        action_type = action_types[legal_mask.nonzero()[selected_idx].item()]
        
        # 构建动作信息
        action_info = {
            'type': action_type,
            'raise_amount': 0,
            'probs': dict(zip([a.name for a in action_types], 
                           action_probs[0].tolist()))
        }
        
        # 计算加注金额（修复维度问题）
        if action_type == ActionType.RAISE:
            min_raise = next(a['min'] for a in legal_actions 
                          if a['type'] == ActionType.RAISE)
            max_raise = next(a['max'] for a in legal_actions 
                          if a['type'] == ActionType.RAISE)
            ratio = raise_ratio[0].item()  # 现在可以正确获取标量值
            action_info['raise_amount'] = int(
                min_raise + (max_raise - min_raise) * ratio
            )
            
        return action_info

def save_model(model, path="models/poker_policy.pt"):
    """保存模型"""
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': model.feature_net[0].in_features
    }, path)

def load_model(path="models/poker_policy.pt", device='cpu'):
    """加载模型（兼容新旧版本检查点）"""
    try:
        checkpoint = torch.load(path, map_location=device)
        # 处理旧版本检查点
        if 'input_dim' not in checkpoint:
            checkpoint['input_dim'] = 128  # 默认值需与训练时实际维度一致
            print(f"警告: 使用默认input_dim={checkpoint['input_dim']}")
            
        model = PokerPolicyNet(input_dim=checkpoint['input_dim'])
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}") from e

def test_policy():
    """策略网络测试"""
    # 初始化游戏
    game = PokerGame(num_players=6)
    player_idx = 0
    
    # 初始化网络
    encoder = StateEncoder()
    state = encoder.encode(game, player_idx)
    net = PokerPolicyNet(input_dim=len(state))
    
    # 测试前向传播
    legal_mask = torch.FloatTensor([1,1,0,1])  # 假设RAISE不可用
    probs, ratio, value = net(state.unsqueeze(0), legal_mask.unsqueeze(0))
    assert probs.shape == (1,4), "动作概率形状错误"
    assert ratio.shape == (1,), "加注比例形状错误"
    
    # 测试动作预测
    action = net.predict(game, player_idx)
    print("预测动作:", action)
    assert action['type'] in [a['type'] for a in game.get_legal_actions()], "非法动作类型"
    
    if action['type'] == ActionType.RAISE:
        legal_raise = next(a for a in game.get_legal_actions() 
                          if a['type'] == ActionType.RAISE)
        assert legal_raise['min'] <= action['raise_amount'] <= legal_raise['max'], "加注金额越界"
    
    print("所有测试通过！")

if __name__ == "__main__":
    test_policy()