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
from models.policy_net import PokerPolicyNet,StateEncoder
from utils.game_simulator import PokerGame, ActionType

class CurriculumOpponent:
    def __init__(self, num_players):
        self.stages = [
            {'steps': 5000, 'fold_prob': 0.8, 'raise_prob': 0.1},
            {'steps': 15000, 'fold_prob': 0.4, 'raise_prob': 0.3},
            {'steps': 30000, 'fold_prob': 0.2, 'raise_prob': 0.5}
        ]
        self.current_stage = 0
        self.num_players = num_players

    def update_opponents(self, game, current_step):
        """更新对手策略"""
        # 确定当前阶段
        while (self.current_stage < len(self.stages)-1 and 
               current_step > self.stages[self.current_stage]['steps']):
            self.current_stage += 1
        
        stage = self.stages[self.current_stage]
        # 为每个对手设置策略
        for i in range(1, self.num_players):
            player = game.players[i]
            player.fold_prob = stage['fold_prob']
            player.raise_prob = stage['raise_prob']

class PPOTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化环境和模型
        self.game = PokerGame(num_players=config['num_players'])
        self.encoder = StateEncoder(num_players=config['num_players'])
        self.policy_net = PokerPolicyNet(
            input_dim=self.encoder.encode(self.game, 0).shape[0]
        ).to(self.device)
        self.policy_net = self.policy_net.float()
        
        # 优化器和配置
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.current_stage = 0  # 训练阶段控制
        self.stage_thresholds = [500, 1000, 2000]  # 阶段切换步数
        
        # 经验缓冲区
        self.buffer = deque(maxlen=config['buffer_size'])
        
        # 创建模型保存目录
        os.makedirs(config['save_dir'], exist_ok=True)

        assert self.policy_net.feature_net[0].in_features == self.encoder.encode(self.game, 0).shape[0], "状态编码维度与网络输入维度不匹配"

    def collect_experience(self):
        """收集多局游戏经验"""
        
        for _ in range(self.config['episodes_per_update']):
            self.game.reset()
            episode_data = []
            done = False
            
            while not done:
                current_player = self.game.current_player
                if not self.game.players[current_player].is_in_hand:
                    continue
                
                # 编码当前状态
                state = self.encoder.encode(self.game, current_player).float()
                legal_actions = self.game.get_legal_actions()
                
                # 模型预测
                with torch.no_grad():
                    action_probs, raise_ratio, value = self.policy_net(
                        state.unsqueeze(0).to(self.device),
                        self._get_legal_mask(legal_actions).to(self.device)
                    )
                
                # 选择动作
                action_type, action_info, action_idx = self._select_action(
                    action_probs[0].cpu(), 
                    raise_ratio[0].item(),
                    legal_actions
                )
                
                # 执行动作
                prev_pot = self.game.pot
                self.game.apply_action(action_type, action_info.get('amount', 0))
                reward = self._calculate_reward(current_player, prev_pot)
                done = self.game.is_terminal()
                
                # 存储经验（新增action_idx）
                episode_data.append({
                    'state': state,
                    'action_probs': action_probs[0].cpu(),
                    'action_idx': action_idx,  # 新增动作索引
                    'value': value[0].cpu(),
                    'reward': reward,
                    'legal_actions': legal_actions
                })
            
            # 计算折扣回报
            self._process_episode(episode_data)
        
    def train(self):
        """训练循环"""
        for update in range(self.config['max_updates']):
            self.collect_experience()
            
            if len(self.buffer) < self.config['batch_size']:
                print(f"跳过更新 {update}: 缓冲区不足 ({len(self.buffer)}/{self.config['batch_size']})")
                continue
                
            # 从缓冲区采样
            batch = random.sample(self.buffer, self.config['batch_size'])
            
            # 转换为张量（新增动作索引）
            states = torch.stack([x['state'].float() for x in batch]).to(self.device)
            old_probs = torch.stack([x['action_probs'].float() for x in batch]).to(self.device)
            actions = torch.tensor([x['action_idx'] for x in batch], dtype=torch.long).to(self.device)
            returns = torch.tensor([x['return'] for x in batch], dtype=torch.float32).to(self.device)
            values = torch.stack([x['value'].float() for x in batch]).to(self.device)
            
            # 计算优势
            advantages = returns - values.squeeze()
            
            # 计算损失（修复gather参数）
            new_probs, _, new_values = self.policy_net(states.float())
            
            # 正确使用gather方法
            ratio = (new_probs.gather(1, actions.unsqueeze(1)) / 
                (old_probs.gather(1, actions.unsqueeze(1)) + 1e-8))
            
            # PPO损失计算
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1-self.config['clip_epsilon'], 
                            1+self.config['clip_epsilon']) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # 熵正则化
            entropy = -(new_probs * torch.log(new_probs + 1e-6)).mean()
            
            total_loss = (policy_loss 
                        + self.config['value_coeff'] * value_loss
                        - self.config['entropy_coeff'] * entropy)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 
                                         self.config['max_grad_norm'])
            self.optimizer.step()
            
            # 保存模型
            if update % self.config['save_interval'] == 0:
                self._save_model(update)
                
            # 打印训练信息
            if update % self.config['log_interval'] == 0:
                print(f"Update {update}: Loss={total_loss.item():.2f} "
                      f"PolicyLoss={policy_loss.item():.2f} "
                      f"ValueLoss={value_loss.item():.2f}")

    def _calculate_reward(self, player_idx, prev_pot):
        """计算即时奖励"""
        # 简单奖励设计：筹码变化 + 存活奖励
        player = self.game.players[player_idx]
        reward = 0.0
        
        # 存活奖励
        if player.is_in_hand:
            reward += 0.02

        hand_streath = self.encoder._evaluate_hand_strength(player.hand)
        reward += np.power(hand_streath, 2) * 2.0
            
        # 筹码变化
        reward += (player.stack - 1000) / 1000  # 初始筹码为1000

        # 激进奖励
        if player.current_bet > 0:
            bet_ratio = player.current_bet / (player.stack + 1e-8)
            aggression_bonus = 0.5 * (1 + np.tanh(5*(bet_ratio - 0.3)))  # S型曲线奖励
            reward += aggression_bonus
        
        # ===== 阶段惩罚机制 =====
        phase_penalty = {
            0: 0.0,   # Pre-flop
            1: 0.1,   # Flop
            2: 0.2,   # Turn
            3: 0.3    # River
        }[self.game.game_phase]
        reward -= phase_penalty * int(player.is_in_hand)  # 后期存活惩罚

        # ===== 对手建模奖励 =====
        opponent_fold_rate = sum(1 for p in self.game.players if not p.is_in_hand) / self.game.num_players
        reward += 0.4 * opponent_fold_rate  # 对手弃牌率越高，奖励越积极

        if player.current_bet > prev_pot:
            reward += 0.3 * (player.current_bet - prev_pot) / 1000

        if self.game.is_terminal():
            if player.stack > 1000:
                win_margin = (player.stack - 1000) / 1000
                reward += 3.0 * (1 + np.log1p(win_margin))  # 对数增长奖励
            else:
                loss_penalty = (1000 - player.stack) / 500
                reward -= 2.0 * np.sqrt(loss_penalty)  # 平方根增强惩罚
        
        if self.game.pot > 0 and player.is_in_hand:
            pot_control = player.current_bet / self.game.pot
            reward += pot_control * 0.3

        #--------------------------
        # 阶段奖励（鼓励参与）
        # --------------------------
        if player.is_in_hand and self.game.game_phase > 0:  # Flop之后
            reward -= 0.1 * self.game.game_phase  # 越后期存活惩罚越高

        return np.clip(reward, -3.0, 3.0)

    def _process_episode(self, episode_data):
        """计算折扣回报"""
        gamma = self.config['gamma']
        running_return = 0
        
        # 反向计算
        for t in reversed(range(len(episode_data))):
            running_return = episode_data[t]['reward'] + gamma * running_return
            self.buffer.append({
                **episode_data[t],
                'return': running_return
            })

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
        action_idx = torch.multinomial(probs, 1).item()
        action_type = [ActionType.FOLD, ActionType.CHECK_CALL, 
                      ActionType.RAISE, ActionType.ALL_IN][action_idx]
        
        # 处理加注金额
        action_info = {}
        if action_type == ActionType.RAISE:
            legal_raise = next(a for a in valid_actions 
                              if a['type'] == ActionType.RAISE)
            min_raise = legal_raise['min']
            max_raise = legal_raise['max']
            amount = min_raise + (max_raise - min_raise) * raise_ratio
            action_info['amount'] = int(amount)
        
        return action_type, action_info, action_idx  

    def _save_model(self, step):
        """保存模型检查点"""
        path = f"{self.config['save_dir']}/model_{step}.pt"
        torch.save({
            'model_state': self.policy_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'step': step,
            'input_dim': self.policy_net.feature_net[0].in_features  # 新增关键参数
        }, path)
        print(f"Saved model checkpoint at step {step}")

if __name__ == "__main__":
    config = {
        'num_players': 6,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'buffer_size': 50000,
        'batch_size': 1024,
        'gamma': 0.97,
        'clip_epsilon': 0.2,
        'value_coeff': 0.5,
        'entropy_coeff': 0.1,
        'max_grad_norm': 0.5,
        'episodes_per_update': 50,
        'max_updates': 500,
        'save_dir': "checkpoints",
        'save_interval': 100,
        'log_interval': 10,
        'input_dim': 128    
    }
    
    trainer = PPOTrainer(config)
    trainer.train()