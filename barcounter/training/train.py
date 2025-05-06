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
from collections import deque,defaultdict
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
        self.entropy_coef = 0.01
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.normalize_advantages = True
        
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

        for _ in tqdm(range(num_episodes), desc="Collecting"):
            game = PokerGame(num_players=self.num_players)
            episode = []
            final_rewards = None
            max_steps = 1000  # 添加最大步数限制

            while not game.is_terminal() and max_steps > 0:
                max_steps -= 1
                current_player = game.current_player
                player = game.players[current_player]
                if not player.is_in_hand:
                    continue

                # 添加筹码状态监控
                if player.stack < 0:
                    raise ValueError(f"玩家{player}出现负筹码：{player.stack}")
                
                # 获取游戏状态
                state = self._encode_state(game)
                legal_actions = game.get_legal_actions()
                
                # 模型预测
                with torch.no_grad():
                    action_info = self.model.predict(state, legal_actions)
                    action_idx = self._action_to_index(action_info['action']['type'], legal_actions)
                
                # 执行动作
                immediate_reward = self._execute_action(game, action_info['action'])
                episode.append((
                    state, 
                    action_idx, 
                    immediate_reward, 
                    legal_actions))
            
            if max_steps <= 0:
                print("达到最大步数限制，强制终止游戏")
                game.force_terminate()  # 需要在PokerGame添加该方法

            if not game.is_terminal():
                # 处理未终局的情况
                final_rewards = self._calculate_final_rewards(game)
            else:
                # 获取实际结算奖励
                final_rewards = game.settle_round() 
            

            for i, (state, action_idx, _, legal_actions) in enumerate(episode):
                player_id = game.current_player  # 需要跟踪玩家ID
                episode[i] = (state, action_idx, final_rewards.get(player_id, 0), legal_actions)
            
            # 计算优势值
            self._process_episode_with_gae(episode)
            self.episode_cache.extend(episode)

            if num_episodes % 10 == 0:
                # 检查玩家筹码状态
                for p in game.players:
                    assert p.stack >= 0, f"玩家{p}出现负筹码：{p.stack}"
                    
                # 检查动作分布
                action_counts = defaultdict(int)
                for exp in self.memory:
                    action_counts[exp[1]] += 1
                print(f"动作分布：{dict(action_counts)}")

    def _process_episode_with_gae(self, episode):
        """使用GAE计算优势值"""
        states = [t[0] for t in episode]
        actions = [t[1] for t in episode]
        rewards = [t[2] for t in episode]
        
        # 计算价值估计
        with torch.no_grad():
            states_tensor = torch.stack(states).to(self.device)
            _, _, values = self.model(states_tensor, None)
            values = values.cpu().numpy().flatten()
            next_values = np.append(values[1:], 0.0)  # 添加终止状态值
        
        # GAE计算
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = float(rewards[t]) + self.gamma * float(next_values[t]) - float(values[t])
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        # 标准化优势（保持原有逻辑）
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 存储经验
        for state, action, adv, ret in zip(states, actions, advantages, returns):
            self.memory.append((
                state,
                action,
                torch.tensor(adv, dtype=torch.float32),
                torch.tensor(ret, dtype=torch.float32)
            ))
    
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
        if len(self.memory) < self.batch_size:
                print("初始化经验池...")
                self.collect_experience(num_episodes=self.batch_size//2)

        for epoch in range(epochs):
            if len(self.memory) < self.batch_size:
                print(f"内存不足 ({len(self.memory)}/{self.batch_size}), 继续收集...")
                print("初始化经验池...")
                self.collect_experience(num_episodes=max(10, self.batch_size//20))  # 动态调整收集量
                continue

            # 采样批次数据
            batch = random.sample(self.memory, self.batch_size)
            
            # 正确解包数据
            states = torch.stack([t[0] for t in batch]).to(self.device)   # [batch, 256]
            actions = torch.LongTensor([t[1] for t in batch]).to(self.device) 
            advantages = torch.FloatTensor([t[2] for t in batch]).to(self.device)
            returns = torch.FloatTensor([t[3] for t in batch]).to(self.device)
            
            # 模型前向
            action_probs, raise_ratios, values = self.model(
                states, 
                legal_mask=None  # 假设模型已处理mask
            )
            
            # 计算损失
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算新旧策略比例
            old_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            ratios = torch.exp(log_probs - old_action_probs.detach())
            
            # 策略损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # 熵正则化
            entropy_loss = -self.entropy_coef * entropy
            
            # 总损失
            total_loss = policy_loss + 0.5 * value_loss + entropy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.entropy_coef *= 0.995  # 逐步减少探索
            # 打印日志
            if epoch % 50 == 0 and epoch > 0:
                self.batch_size = min(self.batch_size*2, 4096)
                print(f"Epoch {epoch} | Loss: {total_loss.item():.3f}")

    def _calculate_final_rewards(self, game):
        """根据最终筹码变化计算奖励"""
        initial_stack = 1000  # 初始筹码量
        rewards = {}
        for idx, player in enumerate(game.players):
            # 筹码变化作为最终奖励
            delta = (player.stack - initial_stack) / initial_stack  # 归一化
            rewards[idx] = delta * 2  # 放大奖励信号
        return rewards
    
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
        prev_stack = player.stack
        
        try:
            # 执行动作前校验
            if action['type'] in [ActionType.RAISE, ActionType.ALL_IN]:
                if player.stack <= 0:
                    raise ValueError(f"玩家{game.current_player}零筹码时尝试{action['type'].name}")

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
        
        # 计算风险惩罚（带安全防护）
        risk_penalty = 0
        try:
            if action['type'] == ActionType.ALL_IN:
                risk_penalty = -0.1
            elif action['type'] == ActionType.RAISE:
                current_stack = max(player.stack, 1e-8)  # 防止除零
                risk_ratio = action['amount'] / current_stack
                risk_penalty = -0.05 * risk_ratio
        except:
            risk_penalty = -0.05  # 异常情况默认惩罚
        # 验证下注结果
        current_bet = player.current_bet
        # print(f"执行动作: {action['type'].name}, 当前下注: {current_bet}")  # 调试输出
        
        # 计算基于筹码变化的奖励
        stack_change = player.stack - prev_stack
        immediate_reward = stack_change / 1000.0  # 归一化
        
        reward_value = immediate_reward + risk_penalty
        
        return float(reward_value)  # 严格限制奖励范围

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(epochs=500)
    save_model(trainer.model, path="models/poker_policy.pt")
    # try:
    #     trainer.train(epochs=1000)
    #     save_model(trainer.model, path="models/poker_policy.pt")
    # except Exception as e:
    #     print(f"训练异常终止: {str(e)}")
    #     # 保存崩溃前的模型
    #     save_model(trainer.model, path="models/crash_recovery.pt")