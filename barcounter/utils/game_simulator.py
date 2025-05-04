# utils/game_simulator.py
import random
from enum import Enum
from typing import List, Dict, Optional

class ActionType(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    ALL_IN = 4

class Player:
    def __init__(self, stack: int = 1000, is_ai: bool = False):
        self.stack = stack          # 筹码量
        self.hand = []              # 手牌
        self.current_bet = 0        # 本轮已下注
        self.is_in_hand = True      # 是否参与当前牌局
        self.is_all_in = False      # 是否全押
        self.is_ai = is_ai          # 是否为AI玩家

class PokerGame:
    def __init__(self, num_players: int = 6, big_blind: int = 20):
        # 初始化牌局
        self.deck = self._create_deck()
        self.players = [Player() for _ in range(num_players)]
        self.community_cards = []
        self.pot = 0                # 总奖池
        self.current_player = 0     # 当前行动玩家
        self.big_blind = big_blind
        self.game_phase = 0
        self.bet_history = []
        self._reset_round()

    def _create_deck(self) -> List[str]:
        """创建标准52张扑克牌"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']  # 红心、方块、梅花、黑桃
        return [r + s for s in suits for r in ranks]

    def _reset_round(self):
        """重置单局游戏"""
        random.shuffle(self.deck)
        self.community_cards = []
        self.pot = 0
        self.game_phase = 0
        
        # 重置玩家状态
        for p in self.players:
            p.hand = []
            p.current_bet = 0
            p.is_in_hand = True
            p.is_all_in = False
        
        # 发牌
        for p in self.players:
            if p.is_in_hand:
                p.hand = [self.deck.pop(), self.deck.pop()]
        
        # 设置盲注
        self._post_blinds()

    def _post_blinds(self):
        """下大小盲注"""
        num_players = len(self.players)
        sb_pos = (self.current_player + 1) % num_players
        bb_pos = (self.current_player + 2) % num_players
        
        sb = self.big_blind // 2
        self._place_bet(sb_pos, sb)
        self._place_bet(bb_pos, self.big_blind)

    def _place_bet(self, player_idx: int, amount: int):
        """玩家下注"""
        player = self.players[player_idx]
        actual_bet = min(amount, player.stack)
        player.stack -= actual_bet
        player.current_bet += actual_bet
        self.pot += actual_bet
        
        if player.stack == 0:
            player.is_all_in = True

    def get_state(self) -> dict:
        """获取当前游戏状态（供AI使用）"""
        current_player = self.players[self.current_player]
        return {
            "hand": current_player.hand,
            "community": self.community_cards,
            "pot": self.pot,
            "current_bet": current_player.current_bet,
            "phase": self.game_phase,
            "bet_history": self.bet_history[-4:]  # 仅返回最近4次动作
        }

    def apply_action(self, action: ActionType, raise_amount: int = 0):
        """合并后的安全动作处理方法"""
        player = self.players[self.current_player]
        
        # 记录动作历史（保留最近10次）
        self.bet_history.append(action.name)
        if len(self.bet_history) > 10:
            self.bet_history.pop(0)
        
        # 处理不同动作类型
        if action == ActionType.FOLD:
            player.is_in_hand = False
        elif action == ActionType.CALL:
            call_amount = self._get_call_amount()
            self._place_bet(self.current_player, call_amount)
        elif action == ActionType.RAISE:
            # 安全加注金额处理
            min_raise = max(10, self.big_blind)  # 至少10或大盲注
            available_raise = max(min_raise, player.stack)
            
            if player.stack < min_raise:
                # 自动转为全押
                self._place_bet(self.current_player, player.stack)
                self.bet_history[-1] = ActionType.ALL_IN.name
            else:
                valid_raise = max(min_raise, min(raise_amount, available_raise))
                self._place_bet(self.current_player, valid_raise)
        
        self._advance_to_next_player()

    def _get_call_amount(self) -> int:
        """获取当前需要跟注的金额"""
        max_bet = max(p.current_bet for p in self.players)
        current_bet = self.players[self.current_player].current_bet
        return max_bet - current_bet

    def _advance_to_next_player(self):
        """切换到下一个玩家"""
        while True:
            self.current_player = (self.current_player + 1) % len(self.players)
            if self.players[self.current_player].is_in_hand:
                break

    def is_terminal(self) -> bool:
        """是否结束当前阶段"""
        active_players = [p for p in self.players if p.is_in_hand]
        
        # 只剩一个玩家时直接结束
        if len(active_players) == 1:
            return True
        
        # 所有玩家下注平衡
        bets = [p.current_bet for p in active_players]
        return all(b == bets[0] for b in bets)

    def _next_phase(self):
        """进入下一阶段（翻牌、转牌、河牌）"""
        self.game_phase += 1
        cards_to_deal = {
            0: 3,  # 翻牌
            1: 1,  # 转牌
            2: 1   # 河牌
        }.get(self.game_phase-1, 0)
        
        self.community_cards.extend([self.deck.pop() for _ in range(cards_to_deal)])
        
        # 重置玩家下注状态
        for p in self.players:
            p.current_bet = 0

    def determine_winner(self) -> List[int]:
        """确定胜者（简化版手牌比较）"""
        from collections import defaultdict
        
        # 仅比较手牌类型（实际需实现完整比较逻辑）
        hand_rankings = []
        for i, p in enumerate(self.players):
            if p.is_in_hand:
                hand_type = self._evaluate_hand(p.hand + self.community_cards)
                hand_rankings.append( (hand_type, i) )
        
        # 按牌型排序
        hand_rankings.sort(reverse=True)
        best_rank = hand_rankings[0][0]
        return [i for rank, i in hand_rankings if rank == best_rank]

    def _evaluate_hand(self, cards: List[str]) -> int:
        """评估手牌强度（示例简化版）"""
        # 实际需要实现完整的扑克手牌评估逻辑
        ranks = [c[0] for c in cards]
        if len(set(ranks)) == 4 and any(ranks.count(r)>=2 for r in set(ranks)):
            return 2  # 对子
        return 1  # 高牌

if __name__ == "__main__":
    # 示例用法
    game = PokerGame(num_players=3)
    print("初始状态:", game.get_state())
    
    # AI玩家行动
    game.apply_action(ActionType.CALL)
    print("跟注后状态:", game.get_state())