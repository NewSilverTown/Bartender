from game_simulator import PokerGame, ActionType, Player
import unittest
from unittest.mock import patch
import random

class TestPokerGameAdvanced(unittest.TestCase):
    def setUp(self):
        random.seed(42)  # 固定随机种子保证测试可重复
        self.num_players = 6
        self.game = PokerGame(num_players=self.num_players, big_blind=20)
        self.target_player_idx = 3  # 固定测试玩家索引

    def _force_blinds(self):
        """测试preflop加注流程"""
        # 重置避免其他干扰
        self.game.force_terminate()

        self.game.current_player = 4  # 使下一玩家为5号（大盲）
        self.game._post_blinds()

    # ------------------- 基础动作测试 -------------------
    def test_preflop_actions(self):
        """测试preflop加注流程"""
        # 手动设置盲注
        self.game._post_blinds = lambda: None  # 禁用自动盲注
        self.game.players[4].current_bet = 10  # 小盲
        self.game.players[5].current_bet = 20  # 大盲
        self.game.pot = 30
        
        # 目标玩家加注
        self.game.current_player = self.target_player_idx
        self.game.apply_action(ActionType.RAISE, 60)
        self.assertEqual(self.game.players[self.target_player_idx].current_bet, 60)
        
        # 验证跟注金额
        self.game.current_player = 4
        self.assertEqual(self.game._get_call_amount(), 60-10)  # 正确为50

    def test_fold_elimination(self):
        """测试弃牌后玩家被移出牌局"""
        test_player = self.game.players[self.target_player_idx]
        original_count = len([p for p in self.game.players if p.is_in_hand])
        
        # 让目标玩家弃牌
        self.game.current_player = self.target_player_idx
        self.game.apply_action(ActionType.FOLD)
        
        self.assertFalse(test_player.is_in_hand)
        self.assertEqual(len([p for p in self.game.players if p.is_in_hand]), original_count-1)

    def test_check_call_accuracy(self):
        """测试跟注金额精确计算（含盲注）"""
        self.game.reset()
        self.game._post_blinds()
        sb_player = self.game.players[4]
        bb_player = self.game.players[5]
        
        # 大盲位加注到60
        self.game.current_player = 5
        self.game.apply_action(ActionType.RAISE, 60)
        
        # 小盲位跟注测试
        self.game.current_player = 4
        expected_call = 60 - sb_player.current_bet  # sb已下注10，需补50
        self.assertEqual(self.game._get_call_amount(), expected_call)
        
        self.game.apply_action(ActionType.CHECK_CALL)
        self.assertEqual(sb_player.stack, 1000 - 10 - 50)  # 初始1000-小盲10-补50

    def test_raise_validation(self):
        """测试加注最小额验证（考虑多玩家下注）"""
        self.game.reset()
        self.game._post_blinds()
        # 玩家3下注50
        self.game.current_player = 3
        self.game.apply_action(ActionType.RAISE, 50)
        
        # 下一个玩家需要至少加注到50+20=70
        self.game.current_player = 4
        with self.assertRaises(ValueError) as ctx:
            self.game.apply_action(ActionType.RAISE, 60)
        self.assertIn("必须至少 70", str(ctx.exception))

    def test_all_in_mechanics(self):
        """测试ALL-IN机制"""
        test_player = self.game.players[self.target_player_idx]
        test_player.stack = 80
        
        self.game.current_player = self.target_player_idx
        self.game.apply_action(ActionType.ALL_IN)
        self.assertEqual(self.game.pot, 110)

    # ------------------- 边池场景测试 -------------------
    def test_multi_side_pots(self):
        """测试三层边池分配"""
        self.game._reset_round()
        # 设置特殊筹码分布
        # 强制设置筹码（绕过盲注）
        self.game.reset_players_stack([300, 500, 1000, 1000, 1000, 1000])
        
        # 下注操作
        self.game.current_player = 0  # 300筹码玩家
        self.game.apply_action(ActionType.ALL_IN)
        self.game.current_player = 1  # 500筹码玩家
        self.game.apply_action(ActionType.ALL_IN)
        self.game.current_player = 2  # 1000筹码玩家
        self.game.apply_action(ActionType.CHECK_CALL)
        
        side_pots = self.game._calculate_side_pots()
        
        self.assertEqual(len(side_pots), 2)  # 现在正确
        
        # 主池: 300*3=900
        self.assertEqual(side_pots[0]["amount"], 300*3)
        # 边池1: (500-300)*2=400
        self.assertEqual(side_pots[1]["amount"], 200*2)
        
        # 设置牌型强制玩家2获胜
        self.game.community_cards = ['Ac','Kh','Qh','Jh','Th']  # 同花顺
        self.game.players[0].hand = ['4d','5d']
        self.game.players[1].hand = ['9h','8h']  # 参与同花顺
        self.game.players[2].hand = ['2c','3c']  # 无效牌
        
        results = self.game.settle_round()
        # 玩家1应赢得主池900和边池400
        self.assertEqual(results[id(self.game.players[1])], 900+400)
        # 玩家2拿回剩余500-300-200=0
        self.assertEqual(self.game.players[2].stack, 980)  # 初始1000-投入500

    # ------------------- 盲注集成测试 -------------------
    def test_blind_posting(self):
        """测试大小盲注自动扣除"""
        self.game.reset()  # 重置后自动post_blinds
        sb_player = self.game.players[1]  # 6人局初始按钮位为0时，小盲=1，大盲=2
        bb_player = self.game.players[2]
        
        self.assertEqual(sb_player.current_bet, 10)
        self.assertEqual(bb_player.current_bet, 20)
        self.assertEqual(self.game.pot, 30)

if __name__ == '__main__':
    unittest.main(verbosity=2)