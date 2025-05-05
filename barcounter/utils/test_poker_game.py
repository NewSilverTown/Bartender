from game_simulator import PokerGame, ActionType

import unittest

class TestPokerGame(unittest.TestCase):
    def setUp(self):
        self.game = PokerGame(num_players=2, big_blind=20)
        self.player0 = self.game.players[0]
        self.player1 = self.game.players[1]

    def test_preflop_actions(self):
        # 验证初始状态
        self.assertEqual(self.game.current_player, 1)  # 大盲位先行动
        
        # 获取合法动作
        actions = self.game.get_legal_actions()
        action_types = [a['type'] for a in actions]
        self.assertIn(ActionType.FOLD, action_types)
        self.assertIn(ActionType.CHECK_CALL, action_types)
        self.assertIn(ActionType.RAISE, action_types)
        
        # 测试加注
        self.game.apply_action(ActionType.RAISE, 40)
        self.assertEqual(self.player1.current_bet, 40)
        self.assertEqual(self.player1.stack, 1000 - 20 - 20)  # 初始大盲20，加注20

    def test_full_round(self):
        # 完整回合测试
        # 玩家1加注到40
        self.game.apply_action(ActionType.RAISE, 40)
        
        # 轮到玩家0行动
        actions = self.game.get_legal_actions()
        self.assertEqual(self.game.current_player, 0)
        
        # 玩家0跟注
        self.game.apply_action(ActionType.CHECK_CALL)
        self.assertEqual(self.player0.current_bet, 40)
        
        # 进入翻牌圈
        self.assertTrue(self.game.is_round_complete())
        self.game.next_phase()
        self.assertEqual(len(self.game.community_cards), 3)
        
        # 翻牌圈开始（当前玩家应该是按钮位）
        self.assertEqual(self.game.current_player, 1)
        
        # 玩家1过牌
        self.game.apply_action(ActionType.CHECK_CALL)
        
        # 玩家0过牌
        self.game.apply_action(ActionType.CHECK_CALL)
        
        # 进入转牌圈
        self.game.next_phase()
        self.assertEqual(len(self.game.community_cards), 4)
        
        # 转牌圈行动...
        
        # 最终结算
        results = self.game.settle_round()
        self.assertGreater(sum(results.values()), 0)

    def test_all_in_scenario(self):
        # 测试All-in场景
        self.player1.stack = 50  # 设置短筹码
        
        # 获取合法动作
        actions = self.game.get_legal_actions()
        all_in_action = next(a for a in actions if a['type'] == ActionType.ALL_IN)
        self.assertTrue(all_in_action['available'])
        self.assertEqual(all_in_action['min'], 50)
        
        # 执行All-in
        self.game.apply_action(ActionType.ALL_IN)
        self.assertEqual(self.player1.stack, 0)
        self.assertTrue(self.player1.is_all_in)

    def test_raise_validation(self):
        # 测试加注验证
        # 尝试非法加注
        with self.assertRaises(ValueError):
            self.game.apply_action(ActionType.RAISE, 10)  # 小于最小加注额
            
        # 有效加注
        self.game.apply_action(ActionType.RAISE, 40)
        self.assertEqual(self.player1.current_bet, 40)

    def test_side_pot_calculation(self):
        # 测试边池计算
        # 玩家0: 1000筹码
        # 玩家1: 500筹码（短筹码）
        self.player1.stack = 500
        
        # 玩家1 All-in
        self.game.apply_action(ActionType.ALL_IN)
        # 玩家0跟注500
        self.game.apply_action(ActionType.CHECK_CALL)
        
        # 结算
        results = self.game.settle_round()
        self.assertEqual(results[id(self.player1)], 500)  # 主池
        self.assertEqual(results[id(self.player0)], 500)   # 边池

if __name__ == '__main__':
    unittest.main()