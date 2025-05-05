# utils/game_simulator.py
import random
from enum import Enum
from typing import List, Dict, Optional, Tuple

class ActionType(Enum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    ALL_IN = 3

class Player:
    def __init__(self, stack: int = 1000, is_ai: bool = False):
        self.stack = stack
        self.hand = []
        self.current_bet = 0
        self.total_bet = 0
        self.is_in_hand = True
        self.is_all_in = False
        self.is_ai = is_ai

    def __repr__(self):
        return f"Player(stack={self.stack}, in_hand={self.is_in_hand})"

class PokerGame:
    def __init__(self, num_players: int = 6, big_blind: int = 20):
        self.num_players = num_players
        self.deck = self._create_deck()
        self.players = [Player() for _ in range(num_players)]
        self.community_cards = []
        self.current_player = 0
        self.big_blind = big_blind
        self.game_phase = 0  # 0: pre-flop, 1: flop, 2: turn, 3: river
        self.pot = 0
        self._reset_round()

    def _create_deck(self) -> List[str]:
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        return [r + s for s in suits for r in ranks]

    def _reset_round(self):
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.community_cards = []
        self.game_phase = 0
        self.pot = 0

        for p in self.players:
            p.hand = []
            p.current_bet = 0
            p.total_bet = 0
            p.is_in_hand = True
            p.is_all_in = False

        # Deal hands
        for p in self.players:
            p.hand = [self.deck.pop(), self.deck.pop()]

        self._post_blinds()

    def _post_blinds(self):
        sb_pos = (self.current_player + 1) % self.num_players
        bb_pos = (self.current_player + 2) % self.num_players
        
        # 两人局特殊处理
        if self.num_players == 2:
            bb_pos = sb_pos  # 按钮位同时是大盲位
            sb_pos = (sb_pos + 1) % self.num_players
        
        sb_amount = self.big_blind // 2
        self._place_bet(sb_pos, sb_amount)
        self._place_bet(bb_pos, self.big_blind)

    def _place_bet(self, player_idx: int, amount: int):
        player = self.players[player_idx]
        # 需要补足的金额 = 目标金额 - 当前已下注
        needed = amount - player.current_bet
        if needed <= 0:
            return  # 无需补充
        
        actual_bet = min(needed, player.stack)
        player.stack -= actual_bet
        player.current_bet += actual_bet  # 关键：累加而不是覆盖
        if player.stack == 0:
            player.is_all_in = True
    
    def get_state(self) -> dict:
        return {
            "current_player": self.current_player,
            "community_cards": self.community_cards,
            "pot": self.pot,
            "phase": self.game_phase
        }

    def apply_action(self, action_type: ActionType, raise_amount: int = 0):
        player = self.players[self.current_player]
        
        if action_type == ActionType.FOLD:
            player.is_in_hand = False
        elif action_type == ActionType.CHECK_CALL:
            call_amount = self._get_call_amount()
            self._place_bet(self.current_player, call_amount)
        elif action_type == ActionType.RAISE:
            # 确保加注金额合法
            current_max = max(p.current_bet for p in self.players if p.is_in_hand)
            min_raise = max(self.big_blind, current_max + self.big_blind)
            raise_amount = max(min_raise, raise_amount)
            
            # 实际可加注金额
            available = player.stack
            actual_raise = min(raise_amount, available)
            self._place_bet(self.current_player, actual_raise)
        
        # print(f"玩家{self.current_player} 执行 {action_type}，加注金额={raise_amount}，当前下注={self.players[self.current_player].current_bet}")
        self._advance_to_next_player()

    def _advance_to_next_player(self):
        while True:
            self.current_player = (self.current_player + 1) % self.num_players
            if self.players[self.current_player].is_in_hand and not self.players[self.current_player].is_all_in:
                break

    def is_round_complete(self) -> bool:
        active_players = [p for p in self.players if p.is_in_hand and not p.is_all_in]
        if len(active_players) < 1:
            return True
        current_bets = [p.current_bet for p in self.players if p.is_in_hand]
        return all(b == current_bets[0] for b in current_bets)

    def next_phase(self):
        # Collect current bets into pot
        for p in self.players:
            self.pot += p.current_bet
            p.current_bet = 0
        self.game_phase += 1
        cards_to_deal = {0: 3, 1: 1, 2: 1}.get(self.game_phase, 0)
        self.community_cards.extend([self.deck.pop() for _ in range(cards_to_deal)])

    def _calculate_side_pots(self) -> List[Dict]:
        # Get all players who contributed to the pot
        contributors = [p for p in self.players if p.total_bet > 0]
        sorted_bets = sorted(set(p.total_bet for p in contributors))
        pots = []
        prev_level = 0

        for level in sorted_bets:
            contribution = level - prev_level
            # Players who contributed to this level
            eligible = [p for p in contributors if p.total_bet >= level]
            # Players eligible to win this pot
            winners_eligible = [p for p in eligible if p.is_in_hand]
            if winners_eligible:
                pot_amount = contribution * len(eligible)
                pots.append({
                    "amount": pot_amount,
                    "players": winners_eligible
                })
            prev_level = level
        return pots

    def settle_round(self) -> Dict[int, int]:
        side_pots = self._calculate_side_pots()
        results = {}

        for pot in side_pots:
            winners = self._determine_winners(pot["players"])
            if not winners:
                continue
            per_winner, remainder = divmod(pot["amount"], len(winners))
            for idx, player in enumerate(winners):
                prize = per_winner + (1 if idx < remainder else 0)
                player.stack += prize
                results[id(player)] = results.get(id(player), 0) + prize

        # Reset for new round
        self._reset_round()
        return results

    def _determine_winners(self, candidates: List[Player]) -> List[Player]:
        # Simplified hand evaluation (implement proper logic)
        return [candidates[0]]