import sys
import torch
import numpy as np
from collections import defaultdict
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
    def __init__(self, num_players=6, history_length = 3):
        self.num_players = num_players
        self.hand_size = 2
        self.community_size = 5
        self.history_length = history_length
        self.prev_community_count = 0
        self.opponent_action_history = defaultdict(list)
        
    def encode(self, game: PokerGame, player_idx: int) -> torch.Tensor:
        """编码游戏状态为特征向量"""
        features = []
        player = game.players[player_idx]

        # ===== 核心特征组 =====
        # 1. 手牌矩阵编码（4x13 的 one-hot）
        hand_matrix = self._create_hand_matrix(game.players[player_idx].hand)
        features.extend(hand_matrix.flatten())
        
        # 2. 公共牌演化模式（每阶段变化量）
        phase_change = len(game.community_cards) - self.prev_community_count
        features.append(phase_change / 3.0)
        self.prev_community_count = len(game.community_cards)
        
        # 3. 筹码动力学特征
        pot_ratio = game.pot / (sum(p.stack for p in game.players) + 1e-8)
        features.append(pot_ratio)
        
        # 4. 对手行为时序特征（最近3步）
        opp_actions = self._get_opponent_action_history(player_idx)
        features.extend(opp_actions)
        
        # ===== 高级特征组 =====
        # 5. GTO基准偏离度（需要预计算）
        gto_deviation = self._calculate_gto_deviation(player.hand, game.community_cards)
        features.append(gto_deviation)
        
        # 6. 风险回报比
        remaining_stack = player.stack - player.current_bet
        risk_reward = (game.pot - player.current_bet) / (remaining_stack + 1e-8)
        features.append(np.tanh(risk_reward * 0.3))

        # 添加筹码安全边际特征
        stack_safety = remaining_stack / (game.pot + 1e-8)
        features.append(np.clip(stack_safety, 0, 2))

        # 手牌潜力
        # features.append(self._hand_potential(game.players[player_idx].hand))

        # # 2. 对手激进指数（0-1）
        # opponent_aggression = sum(
        #     p.current_bet for p in game.players 
        #     if p != game.players[player_idx] and p.is_in_hand
        # ) / (game.pot + 1e-8)
        # features.append(opponent_aggression)

        #  # 4. 筹码深度（相对深度）
        # stack_ratio = game.players[player_idx].stack / sum(p.stack for p in game.players)
        # features.append(stack_ratio)

        # 玩家个人特征
        # player = game.players[player_idx]
        # features += [
        #     player.stack / 1000.0,  # 归一化筹码
        #     player.current_bet / 1000.0,
        #     int(player.is_in_hand),
        #     int(player.is_all_in)
        # ]
        
        # 手牌编码
        # hand_ranks = [self._card_rank(c) for c in player.hand]
        # features += sorted(hand_ranks, reverse=True)
        
        # 公共牌编码
        community_ranks = [self._card_rank(c) for c in game.community_cards]
        features += community_ranks + [0]*(5-len(community_ranks))  # 补齐5张
        
        # 对手特征
        # for i in range(self.num_players):
        #     if i == player_idx:
        #         continue
        #     opp = game.players[i]
        #     features += [
        #         opp.stack / 1000.0,
        #         opp.current_bet / 1000.0,
        #         int(opp.is_in_hand),
        #         int(opp.is_all_in)
        #     ]
            
        # 回合信息
        # features += [
        #     game.game_phase / 3.0,  # 归一化回合阶段
        #     game.pot / 3000.0,
        #     (game.current_player == player_idx)  # 是否当前玩家
        # ]

        # 动态风险感知特征
        pot_commit_ratio = player.current_bet / (game.pot + 1e-8)
        features.append(np.tanh(pot_commit_ratio * 2))

        # 对手激进指数（改进版）
        active_opponents = [p for p in game.players if p.is_in_hand and p != player]
        if active_opponents:
            aggression = sum(p.current_bet for p in active_opponents) / len(active_opponents)
            features.append(aggression / (game.pot + 1e-8))
        else:
            features.append(0.0)

        # 位置优势（按钮位距离标准化）
        button_distance = (game.current_player - player_idx) % game.num_players
        position_weight = 1 - (button_distance / game.num_players)**0.5
        features.append(position_weight)

        # 有效筹码深度
        effective_stack = min(player.stack, 2 * game.pot)
        features.append(effective_stack / 1000)
        
        return torch.FloatTensor(features).float()
    
    def _create_hand_matrix(self, hand):
        """创建手牌矩阵（通道维度）"""
        matrix = np.zeros((4, 13))  # 花色 x 牌面
        for card in hand:
            suit = {'h':0, 'd':1, 'c':2, 's':3}[card[1]]
            rank = '23456789TJQKA'.index(card[0])
            matrix[suit, rank] = 1
        return matrix

    def _hand_potential(self, hand):
        """评估手牌发展潜力（同花/顺子可能）"""
        suits = [c[1] for c in hand]
        ranks = sorted(['23456789TJQKA'.index(c[0]) for c in hand])
        
        # 同花潜力
        flush_potential = 1.0 if len(set(suits)) == 1 else 0.3
        
        # 顺子潜力
        gap = ranks[1] - ranks[0]
        straight_potential = 1.0 if gap == 1 else (0.5 if gap == 2 else 0.2)
        
        return (flush_potential + straight_potential) / 2

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

    def _get_opponent_action_history(self, player_idx):
        """获取对手最近动作（独热编码）"""
        history = self.opponent_action_history[player_idx]
        # 填充最近历史
        while len(history) < self.history_length:
            history.insert(0, [0]*4)  # 4种动作类型
        # 取最近N次
        recent = history[-self.history_length:]
        return [v for entry in recent for v in entry]
    
    def _calculate_gto_deviation(self, hand, community):
        """简单GTO偏离度计算"""
        # 示例实现（需替换为真实GTO数据）
        gto_ranges = {
            'preflop': {
                ('A', 'A'): 1.0,
                ('K', 'K'): 0.95,
                # ...其他手牌范围
            },
            # 其他阶段数据
        }
        current_phase = 'preflop' if len(community) == 0 else 'postflop'
        hand_key = tuple(sorted([card[0] for card in hand], reverse=True))
        return gto_ranges.get(current_phase, {}).get(hand_key, 0.5)
    
    def update_action_history(self, player_idx, action_type):
        """更新对手动作历史"""
        action_vec = [0]*4
        action_vec[action_type.value] = 1
        self.opponent_action_history[player_idx].append(action_vec)
        # 保持历史长度
        if len(self.opponent_action_history[player_idx]) > self.history_length:
            self.opponent_action_history[player_idx].pop(0)

class PokerPolicyNet(nn.Module):
    """强化学习策略网络（兼容PokerGame）"""
    def __init__(self, input_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.tensor([1.0]))

       # 双流网络结构
        self.hand_stream = nn.Sequential(
            nn.Linear(input_dim//2, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32)
        )
        
        self.context_stream = nn.Sequential(
            nn.Linear(input_dim//2, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 动作头
        self.action_head = nn.Sequential(
            nn.Linear(128, 4),
            nn.Tanhshrink()
        )
        
        # 加注头
        self.raise_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x, legal_mask=None):
        # 分割输入特征
        hand_feat = x[:, :self.input_dim//2]
        context_feat = x[:, self.input_dim//2:]
        
        # 双流处理
        hand_out = self.hand_stream(hand_feat)
        context_out = self.context_stream(context_feat)
        
        # 特征融合
        fused = torch.cat([hand_out, context_out], dim=-1)
        fused = self.fusion(fused)

        # 动作概率
        action_logits = self.action_head(fused) / self.temperature
        if legal_mask is not None:
            action_logits += torch.log(legal_mask + 1e-6)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 加注比例
        raise_ratio = self.raise_head(fused).squeeze(-1)
        
        # 状态价值
        value = self.value_head(fused)
        
        return action_probs, raise_ratio, value

    def predict(self, game: PokerGame, player_idx: int,training_step: int = 0) -> Dict:
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
        
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(next(self.parameters()).device)
            legal_mask_tensor = legal_mask.unsqueeze(0).to(state_tensor.device)
            action_probs, raise_ratio, _ = self(state_tensor, legal_mask_tensor)
        
        # Gumbel-Softmax采样
        logits = torch.log(action_probs + 1e-8)
        adjusted_probs = F.gumbel_softmax(logits, tau=0.5, hard=False)
        
        # 动态探索率
        min_prob = max(0.05, 0.2 * np.exp(-training_step / 10000))
        
        # 应用探索下限
        legal_mask = legal_mask.bool()
        legal_indices = legal_mask.nonzero(as_tuple=True)[0]
        legal_probs = adjusted_probs[0][legal_indices]
        legal_probs = torch.clamp(legal_probs, min=min_prob)
        legal_probs = legal_probs / legal_probs.sum()
        
        # 采样动作
        selected_idx = torch.multinomial(legal_probs, 1).item()
        action_type = action_types[legal_indices[selected_idx].item()]
        
        # 构建返回结果
        action_info = {
            'type': action_type,
            'raise_amount': 0,
            'probs': dict(zip([a.name for a in action_types], 
                           adjusted_probs[0].tolist()))
        }
        
        # 计算加注金额
        if action_type == ActionType.RAISE:
            legal_raise = next(a for a in legal_actions 
                             if a['type'] == ActionType.RAISE)
            min_raise = legal_raise['min']
            max_raise = legal_raise['max']
            ratio = raise_ratio[0].item()
            action_info['raise_amount'] = int(
                min_raise + (max_raise - min_raise) * ratio
            )
            
        return action_info

def save_model(model, input_dim, path="models/poker_policy.pt"):
    """保存模型"""
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': input_dim
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