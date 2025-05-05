#train.py
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from models.policy_net import PokerPolicyNet, save_model
from utils.game_encoder import GameEncoder
from utils.game_simulator import PokerGame, ActionType
from utils.hand_evaluator import evaluate_hand_strength

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class PPOTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前使用设备: {self.device}")  # 新增这行
        self.model = PokerPolicyNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.entropy_coef = 0.01  # 策略熵系数

    def collect_trajectories(self, num_episodes=100):
        """优化后的轨迹收集方法
        Args:
            num_episodes: 收集的对局数量
        Returns:
            states: 状态张量 [num_samples, 183]
            actions: 动作索引张量 [num_samples]
            rewards: 奖励张量 [num_samples]
        """
        state_dim = 183
        max_steps = 200  # 单局最大步长
        chunk_size = num_episodes * max_steps * 2  # 预分配双倍空间
        
        # 预分配内存
        states = np.zeros((chunk_size, state_dim), dtype=np.float32)
        actions = np.zeros(chunk_size, dtype=np.int64)
        rewards = np.zeros(chunk_size, dtype=np.float32)
        idx = 0
        
        for _ in range(num_episodes):
            game = PokerGame(num_players=6)
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            while not game.is_terminal() and len(episode_states) < max_steps:
                state = self._encode_game_state(game)
                with torch.no_grad():
                    probs = torch.softmax(self.model(torch.FloatTensor(state).to(self.device)), dim=0)
                    action = torch.multinomial(probs, 1).item()
                
                # 记录执行动作前的筹码
                prev_stack = game.players[game.current_player].stack
                
                # 应用动作并获取奖励
                reward = self._apply_action(game, action)
                
                # 计算精确奖励（分离逻辑）
                is_terminal = game.is_terminal()
                immediate_reward = self._calculate_reward(game, prev_stack, is_terminal)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(immediate_reward)
            
            # 填充预分配数组
            ep_len = len(episode_states)
            if idx + ep_len > chunk_size:
                # 此处概率极低，仅做保护性处理
                states = np.concatenate([states, np.zeros_like(states)])
                actions = np.concatenate([actions, np.zeros_like(actions)])
                rewards = np.concatenate([rewards, np.zeros_like(rewards)])
                chunk_size *= 2
            
            states[idx:idx+ep_len] = np.array(episode_states)
            actions[idx:idx+ep_len] = np.array(episode_actions)
            rewards[idx:idx+ep_len] = np.array(episode_rewards)
            idx += ep_len
        
        states_tensor = torch.as_tensor(states[:idx], dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions[:idx], dtype=torch.long)
        rewards_tensor = torch.as_tensor(rewards[:idx], dtype=torch.float32)

        # 根据设备类型选择传输方式
        if self.device.type == 'cuda':
            return (
                states_tensor.pin_memory().to(self.device, non_blocking=True),
                actions_tensor.pin_memory().to(self.device, non_blocking=True),
                rewards_tensor.pin_memory().to(self.device, non_blocking=True)
            )
        else:
            return (
                states_tensor.to(self.device),
                actions_tensor.to(self.device),
                rewards_tensor.to(self.device)
            )

    def _calculate_reward(self, game, prev_stack, is_terminal):
        """奖励计算辅助方法"""
        current_stack = game.players[game.current_player].stack
        immediate_reward = (current_stack - prev_stack) / 1000  # 归一化
        
        if is_terminal:
            if game.current_player in game.determine_winner():
                immediate_reward += game.pot / 1000  # 获胜奖励
            else:
                immediate_reward -= game.pot / 2000  # 失败惩罚
        
        return immediate_reward

    def _encode_game_state(self, game):
        """编码游戏状态（添加筹码量归一化）"""
        state = game.get_state()
        player = game.players[game.current_player]
        return np.concatenate([
            GameEncoder.encode_hand(state["hand"]),
            GameEncoder.encode_community(state["community"]),
            GameEncoder.encode_bet_history(state["bet_history"]),
            [player.stack / 5000]  # 归一化到[0,1]
        ])

    def _apply_action(self, game, action_idx):
        """改进后的动作执行与奖励计算"""
        action_map = {
            0: ActionType.FOLD,
            1: ActionType.CALL,
            2: ActionType.RAISE,
            3: ActionType.ALL_IN
        }
        
        try:
            player = game.players[game.current_player]
            initial_stack = player.stack  # 记录动作前筹码
            
            # 执行动作
            if action_idx == 2:  # 加注
                min_raise = max(10, game.big_blind)
                max_raise = player.stack
                if max_raise < min_raise:
                    raise ValueError("无法加注")
                raise_amount = random.randint(min_raise, max_raise)
                game.apply_action(action_map[action_idx], raise_amount)
            else:
                game.apply_action(action_map[action_idx])
            
            # ============== 新增奖励计算逻辑 ==============
            # 获取当前游戏状态
            state = game.get_state()
            
            # 获取手牌和公共牌（需要根据实际数据结构调整）
            hand = state["hand"]                   # 当前玩家手牌
            community = state["community"]         # 公共牌
            current_stack = player.stack           # 当前筹码
            
            # 计算筹码变化奖励（归一化到[-1, 1]）
            stack_change = current_stack - initial_stack
            stack_reward = stack_change / 1000.0   # 假设初始筹码为1000
            
            # 计算手牌强度奖励（需要实现evaluate_hand_strength）
            hand_strength = evaluate_hand_strength(hand + community)
            strength_reward = hand_strength * 0.5  # 强度系数
            
            # 终局奖励（如果牌局结束）
            terminal_reward = 0
            if game.is_terminal():
                if game.current_player in game.determine_winner():
                    terminal_reward = game.pot / 1000.0  # 赢得底池奖励
                else:
                    terminal_reward = -game.pot / 500.0  # 输掉底池惩罚
            
            # 组合总奖励
            total_reward = stack_reward + strength_reward + terminal_reward
            # ============== 新增代码结束 ==============
            
            return total_reward
        
        except Exception as e:
            # 非法动作惩罚（需比最差情况更严厉）
            return -1.0  # 从-0.5调整为-1.0以增强惩罚

    def compute_advantages(self, rewards):
        """数值稳定的优势计算"""
        rewards = rewards.clone().detach()
        discounted = torch.zeros_like(rewards)
        running_add = 0
        
        # 逆向计算折扣回报
        for i in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[i]
            discounted[i] = running_add
        
        # 标准化
        if len(discounted) > 1:
            mean = discounted.mean()
            std = discounted.std()
            if std > 1e-6:
                discounted = (discounted - mean) / std
            else:
                discounted = discounted - mean
        return discounted

    def train(self, epochs=1000):
        """改进的训练循环"""
        best_loss = float('inf')
        
        for epoch in tqdm(range(epochs)):
            # 收集数据
            states, actions, rewards = self.collect_trajectories()
            advantages = self.compute_advantages(rewards)
            
            # 旧策略概率
            with torch.no_grad():
                old_probs = torch.softmax(self.model(states), dim=1)
                old_log_probs = torch.log(old_probs.gather(1, actions.unsqueeze(1)))
            
            # 多步更新
            for _ in range(4):
                new_probs = torch.softmax(self.model(states), dim=1)
                new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(1)))
                
                # 重要性采样比率
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # PPO损失
                surr1 = ratio * advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 熵奖励
                entropy = -(new_probs * torch.log(new_probs + 1e-6)).sum(dim=1).mean()
                
                # 总损失
                loss = policy_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 梯度裁剪
                self.optimizer.step()
            
            # 保存最佳模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': best_loss,
                    'epoch': epoch
                }, "models/best_policy.pt")
            
            # 训练监控
            if epoch % 20 == 0:
                print(f"\nEpoch {epoch} | "
                      f"Loss: {loss.item():.3f} | "
                      f"Entropy: {entropy.item():.3f} | "
                      f"Avg Reward: {rewards.mean().item():.2f}")
        
        print("训练完成！最佳模型已保存至 models/best_policy.pt")

if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()