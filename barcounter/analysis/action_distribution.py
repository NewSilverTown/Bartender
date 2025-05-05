import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import torch
from models.policy_net import PokerPolicyNet, load_model
from utils.game_simulator import PokerGame, ActionType

class ActionDistributionAnalyzer:
    def __init__(self, model_path="models/poker_policy.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path).to(self.device)
        self.model.eval()
        
        # 初始化统计数据
        self.reset_stats()
        
        # 游戏模拟器用于生成分析数据
        self.num_players = 6
        self.game = PokerGame(self.num_players)
        
    def reset_stats(self):
        """重置所有统计计数器"""
        self.stats = {
            'total_decisions': 0,
            'action_counts': defaultdict(int),
            'action_probs': defaultdict(list),
            'raise_ratios': [],
            'phase_stats': defaultdict(lambda: defaultdict(int)),
            'stack_level_stats': defaultdict(lambda: defaultdict(int)),
            'action_sequences': []
        }
    
    def _get_legal_actions(self, game):
        """从训练代码复制的合法动作生成逻辑"""
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

    def analyze(self, num_games=100):
        """执行分析主循环"""
        self.reset_stats()
        
        for _ in tqdm(range(num_games), desc="Analyzing games"):
            self.game.reset()
            current_sequence = []
            
            while not self.game.is_terminal():
                player = self.game.players[self.game.current_player]
                if not player.is_in_hand:
                    continue
                
                # 获取状态和合法动作
                state = self._encode_state()
                legal_actions = self._get_legal_actions(self.game)
                
                # 模型预测
                with torch.no_grad():
                    action_info = self.model.predict(state.to(self.device), legal_actions)
                
                # 记录数据
                self._record_action(action_info, legal_actions)
                current_sequence.append(action_info['action']['type'].name)
                
                # 执行动作
                self._execute_action(action_info['action'])
            
            self.stats['action_sequences'].append(current_sequence)
        
        return self.generate_report()

    def _encode_state(self):
        """简化的状态编码（与训练代码一致）"""
        player = self.game.players[self.game.current_player]
        state = np.zeros(256, dtype=np.float32)
        
        # 手牌编码
        hand_encoded = np.zeros(52)
        for card in player.hand:
            idx = (13 * ['2','3','4','5','6','7','8','9','T','J','Q','K','A'].index(card[0]) + \
                  ['h','d','c','s'].index(card[1]))
            hand_encoded[idx] = 1
        state[:52] = hand_encoded
        
        # 游戏阶段
        state[-1] = self.game.game_phase / 3.0
        return torch.FloatTensor(state)

    def _record_action(self, action_info, legal_actions):
        """记录单次动作数据"""
        action = action_info['action']
        probs = action_info['probs']
        phase = self.game.game_phase
        
        # 基础统计
        self.stats['total_decisions'] += 1
        self.stats['action_counts'][action['type'].name] += 1
        
        # 记录所有动作的概率（包括不可用动作）
        full_probs = {a['type'].name: 0.0 for a in legal_actions}
        for k, v in probs.items():
            full_probs[k] = v
        for action_type in full_probs:
            self.stats['action_probs'][action_type].append(full_probs[action_type])
        
        # 记录RAISE比例
        if action['type'] == ActionType.RAISE:
            min_raise = action['min']
            max_raise = action['max']
            actual_ratio = (action['amount'] - min_raise) / (max_raise - min_raise + 1e-8)
            self.stats['raise_ratios'].append(actual_ratio)
        
        # 按游戏阶段统计
        phase_name = ['Pre-flop', 'Flop', 'Turn', 'River'][phase]
        self.stats['phase_stats'][phase_name][action['type'].name] += 1
        
        # 按筹码量统计
        stack_level = min(2, player.stack // 500)  # 0-500, 500-1000, 1000+
        self.stats['stack_level_stats'][stack_level][action['type'].name] += 1

    def _execute_action(self, action):
        """执行动作的简化版本"""
        try:
            if action['type'] == ActionType.RAISE:
                self.game.apply_action(action['type'], raise_amount=action['amount'])
            else:
                self.game.apply_action(action['type'])
        except Exception as e:
            print(f"执行动作失败: {str(e)}")

    def generate_report(self):
        """生成分析报告"""
        report = {
            'summary': self._get_summary_stats(),
            'phase_distribution': self._get_phase_distribution(),
            'raise_analysis': self._analyze_raise_actions(),
            'sequence_analysis': self._analyze_action_sequences()
        }
        return report

    def _get_summary_stats(self):
        """生成汇总统计"""
        total = self.stats['total_decisions']
        return {
            'total_games': len(self.stats['action_sequences']),
            'total_decisions': total,
            'action_frequency': {k: v/total for k, v in self.stats['action_counts'].items()},
            'mean_probabilities': {k: np.mean(v) for k, v in self.stats['action_probs'].items()},
            'stack_level_stats': {
                level: {k: v/sum(actions.values()) for k, v in actions.items()}
                for level, actions in self.stats['stack_level_stats'].items()
            }
        }

    def _get_phase_distribution(self):
        """生成各阶段动作分布"""
        return {
            phase: {k: v/sum(actions.values()) for k, v in actions.items()}
            for phase, actions in self.stats['phase_stats'].items()
        }

    def _analyze_raise_actions(self):
        """分析加注行为"""
        if not self.stats['raise_ratios']:
            return None
        
        ratios = np.array(self.stats['raise_ratios'])
        return {
            'mean_ratio': float(np.mean(ratios)),
            'median_ratio': float(np.median(ratios)),
            'std_ratio': float(np.std(ratios)),
            'min_ratio': float(np.min(ratios)),
            'max_ratio': float(np.max(ratios))
        }

    def _analyze_action_sequences(self):
        """分析动作序列模式"""
        from collections import Counter
        sequence_counter = Counter()
        for seq in self.stats['action_sequences']:
            sequence_counter[tuple(seq)] += 1
        return {
            'most_common_sequences': sequence_counter.most_common(5)
        }

    def plot_distributions(self, report=None):
        """生成可视化图表"""
        if not report:
            report = self.generate_report()
        
        plt.figure(figsize=(15, 10))
        
        # 动作频率分布
        plt.subplot(2, 2, 1)
        actions = list(report['summary']['action_frequency'].keys())
        freqs = list(report['summary']['action_frequency'].values())
        plt.bar(actions, freqs)
        plt.title('Action Frequency Distribution')
        plt.xlabel('Action Type')
        plt.ylabel('Frequency')
        
        # 各阶段动作分布
        plt.subplot(2, 2, 2)
        phases = list(report['phase_distribution'].keys())
        for action in ['FOLD', 'CHECK_CALL', 'RAISE', 'ALL_IN']:
            values = [report['phase_distribution'][p].get(action, 0) for p in phases]
            plt.plot(phases, values, marker='o', label=action)
        plt.title('Action Distribution by Game Phase')
        plt.legend()
        
        # 加注比例分布
        if report['raise_analysis']:
            plt.subplot(2, 2, 3)
            plt.hist(self.stats['raise_ratios'], bins=20, alpha=0.7)
            plt.title('Raise Amount Ratio Distribution')
            plt.xlabel('Normalized Raise Ratio')
            plt.ylabel('Count')
        
        # 筹码量影响
        plt.subplot(2, 2, 4)
        stack_levels = sorted(report['summary']['stack_level_stats'].keys())
        for action in ['FOLD', 'CHECK_CALL', 'RAISE', 'ALL_IN']:
            values = [report['summary']['stack_level_stats'][l].get(action, 0) for l in stack_levels]
            plt.plot(stack_levels, values, marker='o', linestyle='--', label=action)
        plt.title('Action Distribution by Stack Size')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = ActionDistributionAnalyzer()
    report = analyzer.analyze(num_games=1000)
    analyzer.plot_distributions(report)
    
    # 打印关键指标
    print("\n=== 关键分析指标 ===")
    print(f"总决策次数: {report['summary']['total_decisions']}")
    print("动作频率:")
    for action, freq in report['summary']['action_frequency'].items():
        print(f"  {action}: {freq:.2%}")
    
    if report['raise_analysis']:
        print("\n加注行为分析:")
        print(f"平均加注比例: {report['raise_analysis']['mean_ratio']:.2f}")
        print(f"加注比例标准差: {report['raise_analysis']['std_ratio']:.2f}")