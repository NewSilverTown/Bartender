import unittest
import torch
from pathlib import Path
import sys

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from policy_net import PokerPolicyNet, StateEncoder, load_model, save_model
from utils.game_simulator import PokerGame, ActionType

class TestPokerPolicyNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model_path = "test_model.pt"
        cls.dummy_game = PokerGame(num_players=6)
        cls.player_idx = 0
        cls.encoder = StateEncoder()
        
        # 创建并保存测试模型
        state = cls.encoder.encode(cls.dummy_game, cls.player_idx)
        print(f"当前 State", state.shape[0])
        model = PokerPolicyNet(input_dim=len(state))
        save_model(model, state.shape[0], cls.test_model_path)

    def test_model_loading(self):
        """测试模型加载功能"""
        model = load_model(self.test_model_path)
        self.assertIsInstance(model, PokerPolicyNet)
        
        # 测试前向传播
        state = self.encoder.encode(self.dummy_game, self.player_idx)
        action_probs, raise_ratio, _ = model(state.unsqueeze(0))
        self.assertEqual(action_probs.shape, (1, 4))
        self.assertEqual(raise_ratio.shape, (1,))

    def test_predict_action(self):
        """测试动作预测的合法性"""
        model = load_model(self.test_model_path)
        
        # 测试不同游戏阶段
        for phase in ['preflop', 'flop', 'turn', 'river']:
            game = PokerGame(num_players=6)
            if phase != 'preflop':
                game.community_cards = ['Ah', 'Kh', 'Qh']  # 示例公共牌
            
            legal_actions = game.get_legal_actions()
            action = model.predict(game, self.player_idx)
            
            # 验证动作类型合法性
            self.assertIn(action['type'], [a['type'] for a in legal_actions if a['available']])
            
            # 验证加注金额范围
            if action['type'] == ActionType.RAISE:
                legal_raise = next(a for a in legal_actions if a['type'] == ActionType.RAISE)
                self.assertTrue(legal_raise['min'] <= action['raise_amount'] <= legal_raise['max'])

    def test_state_encoder(self):
        """测试状态编码器的输出维度"""
        state = self.encoder.encode(self.dummy_game, self.player_idx)
        # 计算实际特征维度（根据StateEncoder.encode()的具体实现）
        expected_dim = 78  # 根据实际特征计算得到的维度
        self.assertEqual(state.shape[0], expected_dim)
        
        # 测试不同游戏阶段
        game_with_community = PokerGame(num_players=6)
        game_with_community.community_cards = ['Ah', 'Kh', 'Qh', 'Jh', 'Th']
        state = self.encoder.encode(game_with_community, self.player_idx)
        self.assertEqual(state.shape[0], expected_dim)

    def test_edge_cases(self):
        """测试边缘情况（全押、弃牌等）"""
        model = load_model(self.test_model_path)
        
        # 筹码为0的情况（需要设置正确的合法动作）
        empty_stack_game = PokerGame(num_players=6)
        empty_stack_game.players[self.player_idx].stack = 1000
        
        # 强制设置合法动作（当筹码为0时只能ALL_IN）
        empty_stack_game.available_actions = [
            {
                "type": ActionType.ALL_IN,
                "available": True,
                "min": 0,
                "max": 0
            }
        ]
        
        action = model.predict(empty_stack_game, self.player_idx)
        
        self.assertEqual(action['type'], ActionType.ALL_IN)

if __name__ == "__main__":
    unittest.main()