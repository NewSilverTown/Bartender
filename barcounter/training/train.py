import sys
import os

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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5, factor=0.5)

    def collect_experience(self, num_episodes):
        """改进的经验收集方法，包含动态奖励计算"""
        self.episode_cache = []
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
                legal_actions = self._get_legal_actions(game)
                
                # 模型预测
                with torch.no_grad():
                    action_info = self.model.predict(state, legal_actions)
                
                # 执行动作
                reward = self._execute_action(game, action_info['action'])
                next_state = self._encode_state(game)
                
                # 存储经验
                episode.append((
                    state,
                    action_info,
                    reward,
                    next_state,
                    legal_actions
                ))
            
            # 计算优势值
            self._process_episode(episode)
            self.episode_cache.extend(episode)

    def _process_episode(self, episode):
        """使用GAE计算优势值"""
        states = [e[0] for e in episode]
        rewards = [e[2] for e in episode]
        values = []
        
        # 计算状态价值
        with torch.no_grad():
            for state in states:
                state_tensor = state.unsqueeze(0).to(self.device)
                _, _, value = self.model(state_tensor, None)  # 不需要legal_mask
                values.append(value.item())
        
        # 计算回报和优势
        advantages = []
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(episode))):
            delta = rewards[t] + self.gamma * last_value - values[t]
            advantage = delta + self.gamma * self.lam * last_advantage
            advantages.insert(0, advantage)
            last_advantage = advantage
            last_value = values[t]
        
        # 标准化优势
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 存储到记忆库
        for i, transition in enumerate(episode):
            state, action_info, reward, next_state, legal_actions = transition
            self.memory.append((
                state,
                action_info,
                advantages[i],
                reward,
                next_state,
                legal_actions
            ))

    def train(self, epochs=100):
        """改进的训练循环，包含课程学习"""
        best_avg_reward = -float('inf')
        
        for epoch in range(epochs):
            # 阶段式课程学习
            if epoch < 300:
                self.collect_experience(num_episodes=50)
            else:
                self.collect_experience(num_episodes=100)
            
            # 经验回放
            losses = []
            for _ in range(4):  # PPO更新次数
                if len(self.memory) < self.batch_size:
                    continue
                
                batch = random.sample(self.memory, self.batch_size)
                states = []
                old_probs = []
                actions = []
                advantages = []
                returns = []
                legal_masks = []
                
                # 准备批量数据
                for transition in batch:
                    state, action_info, advantage, reward, _, legal_actions = transition
                    states.append(state)
                    old_probs.append(list(action_info['probs'].values()))
                    actions.append(action_info['action'])
                    advantages.append(advantage)
                    returns.append(reward)
                    legal_masks.append(self._create_legal_mask(legal_actions))
                
                # 转换为张量
                states = torch.stack(states).to(self.device)
                old_probs = torch.tensor(old_probs).to(self.device)
                advantages = torch.tensor(advantages).to(self.device)
                returns = torch.tensor(returns).to(self.device)
                legal_masks = torch.stack(legal_masks).to(self.device)
                
                # 计算新概率
                action_probs, raise_ratios, values = self.model(states, legal_masks)
                
                # 策略损失
                new_probs = action_probs.gather(1, legal_masks.argmax(dim=1, keepdim=True))
                ratios = new_probs / (old_probs + 1e-8)
                surr1 = ratios * advantages.unsqueeze(1)
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = 0.5 * F.mse_loss(values.squeeze(), returns)
                
                # 加注金额损失
                raise_loss = 0
                for i, action in enumerate(actions):
                    if action['type'] == ActionType.RAISE:
                        target_ratio = (action['amount'] - action['min']) / (action['max'] - action['min'] + 1e-8)
                        raise_loss += F.mse_loss(raise_ratios[i], torch.tensor(target_ratio).to(self.device))
                
                # 总损失
                total_loss = policy_loss + 0.5 * value_loss + 0.1 * raise_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                losses.append(total_loss.item())
            
            # 更新学习率
            avg_reward = np.mean([t[3] for t in self.memory])
            self.scheduler.step(avg_reward)
            
            # 保存最佳模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                save_model(self.model)
            
            # 打印训练信息
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {np.mean(losses):.3f} | "
                      f"Avg Reward: {avg_reward:.2f} | LR: {self.optimizer.param_groups[0]['lr']:.1e}")

    def _encode_state(self, game):
        """改进的状态编码方法（修复维度问题）"""
        player = game.players[game.current_player]
        state = np.zeros(256, dtype=np.float32)
        
        # 手牌编码（52维）
        hand_encoded = np.zeros(52)
        for card in player.hand:
            idx = self._card_to_index(card)
            hand_encoded[idx] = 1
        state[:52] = hand_encoded
        
        # 公共牌编码（最多3张牌，52*3=156维）
        max_community_cards = 3  # 根据状态维度限制
        community_encoded = np.zeros(52 * max_community_cards)
        for i, card in enumerate(game.community_cards[:max_community_cards]):
            idx = self._card_to_index(card)
            community_encoded[i*52:(i+1)*52] = 1
        state[52:52+len(game.community_cards)*52] = community_encoded[:len(game.community_cards)*52]
        
        # 数值特征（最后4维）
        state[-4:] = [
            player.stack / 1000.0,
            player.current_bet / 1000.0,
            sum(p.current_bet for p in game.players) / 2000.0,  # 总底池
            game.game_phase / 3.0  # 阶段0-3
        ]
        
        return torch.FloatTensor(state).to(self.device)

    def _card_to_index(self, card):
        ranks = '23456789TJQKA'
        suits = 'hdcs'
        return (ranks.index(card[0]) * 4) + suits.index(card[1])

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
        print(f"执行动作: {action['type'].name}, 当前下注: {current_bet}")  # 调试输出
        
        # 计算奖励（保持原有逻辑）
        current_pot = sum(p.total_bet for p in game.players)
        reward = (current_pot - prev_pot) / 100.0
        
        if action['type'] == ActionType.RAISE:
            reward += 0.1 * (action['amount'] / (player.stack + 1e-8))
        
        return reward

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()