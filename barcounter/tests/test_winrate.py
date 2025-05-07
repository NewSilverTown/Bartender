# test_winrate.py
import sys
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_net import PokerPolicyNet
from utils.game_encoder import GameEncoder
from utils.game_simulator import PokerGame, ActionType

class PokerAITester:
    def __init__(self, model_path="models/best_policy.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 测试配置
        self.num_test_games = 1000    # 测试对局数
        self.ai_player_index = 0     # AI玩家的位置索引
        self.initial_stack = 5000    # 初始筹码量
        self.big_blind = 20          # 必须与游戏初始化参数一致

    def _load_model(self, path):
        model = PokerPolicyNet().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        return model

    def _encode_state(self, game):
        """与训练完全一致的状态编码"""
        state = game.get_state()
        player = game.players[game.current_player]
        return np.concatenate([
            GameEncoder.encode_hand(state["hand"]),
            GameEncoder.encode_community(state["community"]),
            GameEncoder.encode_bet_history(state["bet_history"]),
            [player.stack / self.initial_stack]  # 归一化处理
        ])

    def _ai_action(self, state):
        """AI决策（使用确定性策略）"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            probs = torch.softmax(self.model(state_tensor), dim=0)
            return torch.argmax(probs).item()

    def _random_action(self, game):
        """改进的随机动作生成（带合法性检查）"""
        player = game.players[game.current_player]
        valid_actions = []
        
        # 可用动作检查
        if game.can_fold():
            valid_actions.append(0)
        if game.can_call():
            valid_actions.append(1)
        if game.can_raise():
            valid_actions.append(2)
        if game.can_all_in():
            valid_actions.append(3)
        
        return random.choice(valid_actions) if valid_actions else 0

    def _execute_action(self, game, action_idx):
        """安全执行动作"""
        action_map = {
            0: ActionType.FOLD,
            1: ActionType.CHECK_CALL,
            2: ActionType.RAISE,
            3: ActionType.ALL_IN
        }
        action = action_map[action_idx]
        player = game.players[game.current_player]

        try:
            if action == ActionType.RAISE:
                min_raise = max(10, self.big_blind)
                max_raise = player.stack
                if max_raise >= min_raise:
                    raise_amount = random.randint(min_raise, max_raise)
                    game.apply_action(action, raise_amount)
                else:
                    game.apply_action(ActionType.ALL_IN)
            else:
                game.apply_action(action)
        except Exception as e:
            print(f"执行动作失败: {str(e)}，自动弃牌")
            game.apply_action(ActionType.FOLD)

    def _distribute_pot(self, game):
        """正确的奖池分配逻辑"""
        active_players = [p for p in game.players if p.is_in_hand]
        
        # 如果只剩一个玩家
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += game.pot
            game.pot = 0
            return

        # 正常比较手牌分配
        winners = game.determine_winner()
        per_winner = game.pot // len(winners)
        for i in winners:
            game.players[i].stack += per_winner
        game.pot = 0

    def _play_round(self, game):
        """完整的游戏回合处理"""
        try:
            while True:
                # 处理当前阶段
                while not game.is_round_complete():
                    state = self._encode_state(game)
                    
                    if game.current_player == self.ai_player_index:
                        action = self._ai_action(state)
                    else:
                        action = self._random_action(game)
                    
                    self._execute_action(game, action)

                # 进入下一阶段或结束
                if game.game_phase >= 3:
                    self._distribute_pot(game)  # 最终分配奖池
                    break
                
                game._next_phase()
                game.current_player = 0  # 重置行动玩家
        except Exception as e:
            print(f"游戏异常中断: {str(e)}")

    def run_test(self):
        """修复后的测试流程"""
        stats = defaultdict(list)
        progress_bar = tqdm(range(self.num_test_games), desc="Testing")

        for game_num in progress_bar:
            # 初始化新游戏
            game = PokerGame(num_players=6, big_blind=self.big_blind)
            for p in game.players:
                p.stack = self.initial_stack  # 重置筹码
            
            # 进行完整对局
            self._play_round(game)
            
            # 记录结果
            final_stack = game.players[self.ai_player_index].stack
            profit = final_stack - self.initial_stack
            
            # 正确判断胜负：只有盈利才算胜利
            stats['result'].append(1 if profit > 0 else 0)
            stats['profit'].append(profit)
            
            # 调试输出
            if game_num < 3:
                print(f"\n=== 调试对局 {game_num} ===")
                print(f"初始筹码: {self.initial_stack}")
                print(f"最终筹码: {final_stack}")
                print(f"奖池金额: {game.pots}")
                print(f"玩家筹码: {[p.stack for p in game.players]}")
                print(f"是否存活: {game.players[self.ai_player_index].is_in_hand}")

            # 更新进度条
            progress_bar.set_postfix({
                'WinRate': f"{np.mean(stats['result']):.2%}",
                'AvgProfit': f"{np.mean(stats['profit']):.1f}"
            })

        # 打印最终结果
        print("\n=== 最终测试结果 ===")
        print(f"测试对局数: {self.num_test_games}")
        print(f"胜率（盈利次数）: {np.mean(stats['result']):.2%}")
        print(f"平均收益: {np.mean(stats['profit']):.1f}")
        print(f"最大收益: {max(stats['profit'])}")
        print(f"最小收益: {min(stats['profit'])}")
        print(f"收益率标准差: {np.std(stats['profit']):.1f}")
        print(f"总盈利: {sum(stats['profit'])}")

if __name__ == "__main__":
    tester = PokerAITester()
    tester.run_test()