import sys
import os
import random

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录（假设analysis在根目录下）
project_root = os.path.dirname(current_dir)
# 将根目录添加到Python路径
sys.path.append(project_root)

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import deque
import torch.nn.functional as F
from models.policy_net import PokerPolicyNet, save_model
from utils.game_simulator import PokerGame, ActionType

class PPOTrainer:
    def __init__(self, num_players=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PokerPolicyNet(state_dim=256).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-5, weight_decay=1e-5)
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.batch_size = 512
        self.num_players = num_players
        self.memory = deque(maxlen=100000)
        self.episode_cache = []
        
        # 自适应学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',  # 监控损失
            patience=3, 
            factor=0.5
        )

    def collect_experience(self, num_episodes):
        """改进的经验收集方法，包含动态奖励计算"""
        self.episode_cache = []
        min_episodes = max(num_episodes, self.batch_size // 10)
        for _ in tqdm(range(num_episodes), desc="Collecting"):
            game = PokerGame(num_players=self.num_players)
            episode = []
            while not game.is_round_complete():
                current_player = game.current_player
                player = game.players[current_player]
                if not player.is_in_hand:
                    continue
                
                # 获取游戏状态
                state = self._encode_state(game)
                legal_actions = game.get_legal_actions()
                
                # 模型预测
                with torch.no_grad():
                    action_info = self.model.predict(state, legal_actions)
                    action_idx = self._action_to_index(action_info['action']['type'], legal_actions)
                
                # 执行动作
                reward = self._execute_action(game, action_info['action'])
                next_state = self._encode_state(game)
                
                reward = np.clip(reward, -2.0, 2.0)

                # 存储经验
                episode.append((
                    state.cpu(),  # [256]
                    action_idx,   # scalar
                    reward,        # scalar
                    legal_actions
                ))
            
            # 计算优势值
            self._process_episode(episode)
            self.episode_cache.extend(episode)

    def _action_to_index(self, action_type, legal_actions):
        """将动作类型转换为合法动作列表中的索引"""
        for idx, a in enumerate(legal_actions):
            if a['type'] == action_type:
                return idx
        raise ValueError(f"非法动作类型: {action_type}")

    def _process_episode(self, episode):
        """处理经验并存入内存"""
        states = [t[0] for t in episode]
        actions = [t[1] for t in episode]
        rewards = [t[2] for t in episode]
        
        # 计算优势值（确保维度正确）
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 存入内存
        for state, action, ret in zip(states, actions, returns):
            self.memory.append((
                state,          # [256]
                action,         # scalar
                ret             # scalar
            ))

    def train(self, epochs=100):
        """改进的训练循环，包含课程学习"""
        best_avg_reward = -float('inf')
        self.collect_experience(num_episodes=self.batch_size)

        for epoch in range(epochs):
            if len(self.memory) < self.batch_size:
                print(f"内存不足 ({len(self.memory)}/{self.batch_size}), 继续收集...")
                self.collect_experience(num_episodes=50)
                continue

            # 采样批次数据
            batch = random.sample(self.memory, self.batch_size)
            
            # 正确解包数据
            states = torch.stack([t[0] for t in batch]).to(self.device)   # [batch, 256]
            actions = torch.tensor([t[1] for t in batch], 
                                 dtype=torch.long,
                                 device=self.device)                      # [batch]
            returns = torch.tensor([t[2] for t in batch],
                                 dtype=torch.float32,
                                 device=self.device)                      # [batch]
            
            # 模型前向
            action_probs, raise_ratios, values = self.model(
                states, 
                legal_mask=None  # 假设模型已处理mask
            )
            
            # 计算损失
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)                           # [batch]
            
            # 策略损失
            policy_loss = -(log_probs * returns).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 打印日志
            if epoch % 5 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss.item():.3f}")

    def _encode_state(self, game):
        """修复后的状态编码"""
        player = game.get_current_player()
        state = np.zeros(256, dtype=np.float32)
        
        # 手牌编码（确保索引正确）
        for card in player.hand:
            rank = card[0].upper()
            suit = card[1].lower()
            rank_idx = '23456789TJQKA'.index(rank)
            suit_idx = ['h','d','c','s'].index(suit)
            idx = rank_idx * 4 + suit_idx  # 正确计算0-51索引
            state[idx] = 1
        
        return torch.tensor(state, device=self.device)

    def _card_to_index(self, card):
        ranks = '23456789TJQKA'
        suits = 'hdcs'
        return (ranks.index(card[0]) * 4) + suits.index(card[1])
    
    def _create_legal_mask(self, legal_actions):
            """生成合法动作的掩码"""
            action_types = [
                ActionType.FOLD,
                ActionType.CHECK_CALL, 
                ActionType.RAISE,
                ActionType.ALL_IN
            ]
            mask = torch.zeros(len(action_types), dtype=torch.float32)
            
            try:
                for action in legal_actions:
                    if action['available']:
                        idx = action_types.index(action['type'])
                        mask[idx] = 1.0
            except ValueError as e:
                print(f"非法动作类型: {action['type']}")
                print(f"合法动作列表: {[a['type'] for a in legal_actions]}")
                raise
            
            return mask.to(self.device)
    
    def _get_legal_actions(self, game):
        """生成合法动作列表"""
        player = game.players[game.current_player]
        actions = []
        
        # Fold
        actions.append({
            'type': ActionType.FOLD,
            'available': True,
            'min': 0,
            'max': 0
        })
        
        # Check/Call
        call_amount = max(p.current_bet for p in game.players) - player.current_bet
        actions.append({
            'type': ActionType.CHECK_CALL,
            'available': call_amount <= player.stack,
            'min': call_amount,
            'max': call_amount
        })
        
        # Raise
        min_raise = max(game.big_blind, 
                      max(p.current_bet for p in game.players) + game.big_blind)
        max_raise = player.stack
        actions.append({
            'type': ActionType.RAISE,
            'available': max_raise >= min_raise,
            'min': min_raise,
            'max': max_raise,
            'player_stack': player.stack
        })
        
        # All-in
        actions.append({
            'type': ActionType.ALL_IN,
            'available': player.stack > 0,
            'min': player.stack,
            'max': player.stack
        })
        
        return [a for a in actions if a['available']]

    def _execute_action(self, game, action):
        """执行动作并计算奖励"""
        player = game.players[game.current_player]
        prev_pot = sum(p.total_bet for p in game.players)
        
        try:
            # 精确执行动作
            if action['type'] == ActionType.FOLD:
                game.apply_action(ActionType.FOLD)
            elif action['type'] == ActionType.CHECK_CALL:
                game.apply_action(ActionType.CHECK_CALL)
            elif action['type'] == ActionType.RAISE:
                # 确保金额在合法范围内
                amount = np.clip(action['amount'], action['min'], action['max'])
                
                game.apply_action(ActionType.RAISE, raise_amount=amount)
            elif action['type'] == ActionType.ALL_IN:
                game.apply_action(ActionType.ALL_IN)
        except Exception as e:
            print(f"执行动作失败: {str(e)}")
            return 0
        
        # 验证下注结果
        current_bet = player.current_bet
        # print(f"执行动作: {action['type'].name}, 当前下注: {current_bet}")  # 调试输出
        
        # 计算奖励（保持原有逻辑）
        current_pot = sum(p.total_bet for p in game.players)
        
        # 改进奖励计算
        pot_change = (current_pot - prev_pot)/1000.0
        reward = np.clip(pot_change, -1.0, 1.0)
        
        return torch.tensor(reward, dtype=torch.float32)  # 严格限制奖励范围

if __name__ == "__main__":
    trainer = PPOTrainer()
    try:
        trainer.train(epochs=1000)
        save_model(trainer.model, path="models/poker_policy.pt")
    except Exception as e:
        print(f"训练异常终止: {str(e)}")
        # 保存崩溃前的模型
        save_model(trainer.model, path="models/crash_recovery.pt")