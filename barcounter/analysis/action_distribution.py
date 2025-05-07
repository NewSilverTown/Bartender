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
from models.policy_net import PokerPolicyNet,StateEncoder, load_model
from utils.game_simulator import PokerGame, ActionType

class ActionDistributionAnalyzer:
    def __init__(self, model_path="checkpoints/model_100.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 游戏模拟器用于生成分析数据
        self.num_players = 6
        self.game = PokerGame(self.num_players)

        self.model = load_model(model_path).to(self.device)
        self.encoder = StateEncoder(num_players=self.num_players)
        # 初始化统计数据
        self.reset_stats()

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
        current_max_bet = max(p.current_bet for p in game.players)
        call_amount = current_max_bet - player.current_bet
        
        # Fold
        actions.append({
            'type': ActionType.FOLD,
            'available': True,
            'min': 0,
            'max': 0
        })
        
        # Check/Call
        can_call = (call_amount <= player.stack) and (call_amount >= 0)
        actions.append({
            'type': ActionType.CHECK_CALL,
            'available': can_call,
            'min': call_amount,
            'max': call_amount
        })
        
        # Raise
        min_raise = max(
        game.big_blind,
        current_max_bet + game.big_blind - player.current_bet
        )
        max_raise = player.stack
        can_raise = (max_raise >= min_raise) and (player.stack > 0)
        actions.append({
            'type': ActionType.RAISE,
            'available': can_raise,
            'min': min_raise,
            'max': max_raise,
            'player_stack': player.stack
        })
        
        # All-in
        can_all_in = (player.stack > 0) and not can_raise
        actions.append({
            'type': ActionType.ALL_IN,
            'available': can_all_in,
            'min': player.stack,
            'max': player.stack
        })
        
        return [a for a in actions if a['available']]

    def analyze(self, num_games=100):
        """执行分析主循环"""
        self.reset_stats()
        
        for _ in tqdm(range(num_games), desc="Analyzing games"):
            self.game = PokerGame(self.num_players)  # 每次创建新游戏
            current_sequence = []
            max_steps = 1000  # 添加安全计数器
            
            while not self.game.is_terminal() and max_steps > 0:
                max_steps -= 1
                player = self.game.players[self.game.current_player]
                
                # 处理弃牌玩家
                if not player.is_in_hand or player.is_all_in:
                    self.game._advance_to_next_player()  # 使用内部方法推进
                    continue
                
                # 模型预测
                try:
                    # 获取状态和动作
                    state = self.encoder.encode(self.game, self.game.current_player)
                    legal_actions = self.game.get_legal_actions()  # 使用游戏自带方法
                    with torch.no_grad():
                        action_info = self.model.predict(self.game, self.game.current_player)
                except Exception as e:
                    print(f"预测失败: {str(e)}")
                    break
                
                required_keys = {'type', 'raise_amount', 'probs'}
                if not all(k in action_info for k in required_keys):
                    print(f"无效的动作信息结构: {action_info.keys()}")
                    break

                # 执行并记录动作（修正键名）
                try:
                    self._record_action(action_info, legal_actions)
                    current_sequence.append(action_info['type'].name)
                    
                    # 执行动作（使用正确参数）
                    if action_info['type'] == ActionType.RAISE:
                        self.game.apply_action(
                            action_info['type'], 
                            raise_amount=action_info['raise_amount']
                        )
                    else:
                        self.game.apply_action(action_info['type'])
                        
                    # 推进阶段检查
                    if self.game.is_round_complete():
                        self.game.next_phase()
                        
                except Exception as e:
                    import traceback
                    print(f"执行动作失败详情:")
                    traceback.print_exc()  # 打印完整堆栈跟踪
                    print(f"当前动作信息: {action_info}")
                    break

                self.game._advance_to_next_player()
            
            self.stats['action_sequences'].append(current_sequence)
            if max_steps <= 0:
                print("达到最大步数限制，强制终止游戏")
        
        return self.generate_report()

    def _encode_state(self, player_index):
        """简化的状态编码（与训练代码一致）"""
        return self.encoder.encode(self.game, player_index)

    def _record_action(self, action_info, legal_actions):
        """记录单次动作数据"""
        action_type = action_info['type']
        probs = action_info['probs']
        phase = self.game.get_phase_name()
        
        # 基础统计
        self.stats['total_decisions'] += 1
        self.stats['action_counts'][action_type.name] += 1
        
        # 记录所有动作的概率（包括不可用动作）
        full_probs = {a['type'].name: 0.0 for a in legal_actions}
        for k, v in probs.items():
            full_probs[k] = v

        for action_type_name in full_probs:
            self.stats['action_probs'][action_type_name].append(full_probs[action_type_name])
        
        # 记录RAISE比例
        if action_type == ActionType.RAISE:
            legal_raise = next(a for a in legal_actions if a['type'] == ActionType.RAISE)
            min_raise = legal_raise['min']
            max_raise = legal_raise['max']
            actual_ratio = (action_info['raise_amount'] - min_raise) / (max_raise - min_raise + 1e-8)
            self.stats['raise_ratios'].append(actual_ratio)
        
        # 按游戏阶段统计
        phase_map = {
            0: 'Pre-flop',
            1: 'Flop',
            2: 'Turn',
            3: 'River'
        }
        phase_name = phase_map.get(phase, f'Unknown Phase ({phase})')
        self.stats['phase_stats'][phase_name][action_type.name] += 1

        player = self.game.players[self.game.current_player]        
        # 按筹码量统计
        stack_level = min(2, player.stack // 500)  # 0-500, 500-1000, 1000+
        self.stats['stack_level_stats'][stack_level][action_type.name] += 1

    def _execute_action(self, action):
        """执行动作的简化版本"""
        try:
            # 执行前备份当前玩家索引
            prev_player = self.game.current_player
            
            if action['type'] == ActionType.RAISE:
                self.game.apply_action(action['type'], raise_amount=action['amount'])
            else:
                self.game.apply_action(action['type'])
            
            # 验证玩家索引是否变化
            if self.game.current_player == prev_player:
                print("玩家索引未变化，强制推进")
                self.game._advance_to_next_player()
                
        except Exception as e:
            print(f"执行动作失败: {str(e)}")
            # 强制推进玩家索引
            self.game.force_terminate()

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
    report = analyzer.analyze(num_games=500)
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