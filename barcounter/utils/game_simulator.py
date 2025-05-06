from typing import List, Dict, Optional
import random
from enum import Enum
from collections import defaultdict

class ActionType(Enum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    ALL_IN = 3

class Player:
    def __init__(self, stack=1000):
        self.hand: List[str] = []
        self.stack: int = stack
        self.current_bet: int = 0
        self.total_bet: int = 0
        self.is_in_hand: bool = True
        self.is_all_in: bool = False

class PokerGame:
    def __init__(self, num_players: int = 6, big_blind: int = 20):
        self.num_players = num_players
        self.big_blind = big_blind
        self.players = [Player() for _ in range(num_players)]
        self.reset()  # 使用reset方法初始化

    def reset(self):
        """完整重置游戏状态"""
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.community_cards = []
        self.game_phase = 0
        self.pot = 0
        self.current_player = 0
        
        for p in self.players:
            p.hand = []
            p.current_bet = 0
            p.total_bet = 0
            p.is_in_hand = True
            p.is_all_in = False
            p.stack = 1000  # 重置筹码
            
        # 发牌
        for p in self.players:
            p.hand = [self.deck.pop(), self.deck.pop()]
        
        self._post_blinds()

    def is_terminal(self):
        """判断游戏是否结束（新增方法）"""
        active_players = [p for p in self.players if p.is_in_hand]
        if len(active_players) <= 1:
            return True
        
        # 检查是否所有玩家都all-in
        all_all_in = all(p.is_all_in for p in self.players if p.is_in_hand)
        return all_all_in
    def force_terminate(self):
        """强制终止游戏"""
        for p in self.players:
            p.is_in_hand = False
        self.pot = 0  # 清空奖池

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
        
        if self.num_players == 2:
            bb_pos = sb_pos
            sb_pos = (sb_pos + 1) % self.num_players
        
        sb_amount = self.big_blind // 2
        self._place_bet(sb_pos, sb_amount)
        self._place_bet(bb_pos, self.big_blind)

    def _place_bet(self, player_idx: int, amount: int):
        player = self.players[player_idx]
        needed = amount - player.current_bet
        if needed <= 0:
            return
        
        actual_bet = min(needed, player.stack)
        player.stack -= actual_bet
        player.current_bet += actual_bet
        player.total_bet += actual_bet
        self.pot += actual_bet
        
        if player.stack == 0:
            player.is_all_in = True

    def get_current_player(self):
        return self.players[self.current_player]

    # 新增的_get_call_amount方法
    def _get_call_amount(self) -> int:
        current_player = self.players[self.current_player]
        max_bet = max(p.current_bet for p in self.players if p.is_in_hand)
        return max(0, max_bet - current_player.current_bet)
    
    # 在PokerGame类中添加以下方法
    def get_legal_actions(self) -> List[Dict]:
        """生成当前玩家的合法动作列表"""
        player = self.players[self.current_player]
        actions = []

        # Fold总是可用
        actions.append({
            'type': ActionType.FOLD,
            'available': True,
            'min': 0,
            'max': 0
        })

        # Check/Call逻辑
        call_amount = self._get_call_amount()
        can_call = (call_amount <= player.stack) and (call_amount > 0)
        actions.append({
            'type': ActionType.CHECK_CALL,
            'available': can_call,
            'min': call_amount,
            'max': call_amount
        })

        # Raise逻辑
        current_max_bet = max(p.current_bet for p in self.players if p.is_in_hand)
        min_raise = max(
            self.big_blind,
            current_max_bet + self.big_blind - player.current_bet
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

        # All-in逻辑
        can_all_in = (player.stack > 0) and (player.stack > min_raise)
        actions.append({
            'type': ActionType.ALL_IN,
            'available': player.stack > 0,
            'min': player.stack,
            'max': player.stack
        })

        return [a for a in actions if a['available']]

    def apply_action(self, action_type: ActionType, raise_amount: int = 0):
        player = self.players[self.current_player]
        
        if action_type == ActionType.FOLD:
            player.is_in_hand = False
        elif action_type == ActionType.CHECK_CALL:
            call_amount = self._get_call_amount()
            self._place_bet(self.current_player, call_amount)
        elif action_type == ActionType.RAISE:
            if player.stack <= 0:
                raise ValueError("零筹码玩家不能加注")
            current_max = max(p.current_bet for p in self.players if p.is_in_hand)
            min_raise = max(self.big_blind, current_max + self.big_blind)
            raise_amount = max(min_raise, raise_amount)
            available = player.stack + player.current_bet
            actual_raise = min(raise_amount, available)
            self._place_bet(self.current_player, actual_raise)
        elif action_type == ActionType.ALL_IN:
            self._place_bet(self.current_player, player.stack)

        self._advance_to_next_player()

    def _advance_to_next_player(self):
        start_idx = self.current_player
        max_attempts = self.num_players * 2  # 添加最大尝试次数
        while max_attempts > 0:
            max_attempts -= 1
            self.current_player = (self.current_player + 1) % self.num_players
            player = self.players[self.current_player]
            if player.is_in_hand and not player.is_all_in:
                break
            if self.current_player == start_idx:
                break
        self.force_terminate()

    def is_round_complete(self) -> bool:
        active_players = [p for p in self.players if p.is_in_hand and not p.is_all_in]
        if len(active_players) <= 1:
            return True
        
        current_bets = [p.current_bet for p in self.players if p.is_in_hand]
        return len(set(current_bets)) == 1

    def get_phase_name(self):
        phase_map = {
            0: 'Pre-flop',
            1: 'Flop',
            2: 'Turn',
            3: 'River'
        }
        return phase_map.get(self.game_phase, 'Showdown')
    
    def next_phase(self):
        if self.game_phase >= 3:  # River之后不再推进阶段
            return
        self.game_phase += 1
        
        cards_to_deal = {
            0: 3,  # Flop
            1: 1,  # Turn
            2: 1   # River
        }.get(self.game_phase, 0)
        
        if cards_to_deal > 0:
            self.community_cards.extend([self.deck.pop() for _ in range(cards_to_deal)])

    def _calculate_side_pots(self) -> List[Dict]:
        sorted_players = sorted(
            [p for p in self.players if p.total_bet > 0],
            key=lambda x: x.total_bet
        )
        
        pots = []
        prev_total = 0
        for i, player in enumerate(sorted_players):
            current_total = player.total_bet
            if current_total <= prev_total:
                continue
                
            contribution = current_total - prev_total
            eligible_players = [p for p in sorted_players[i:] if p.is_in_hand]
            
            if eligible_players:
                pot_size = contribution * len(sorted_players[i:])
                pots.append({
                    "amount": pot_size,
                    "players": eligible_players
                })
                prev_total = current_total
        
        return pots

    # 改进的牌力评估方法
    def _determine_winners(self, candidates: List[Player]) -> List[Player]:
        def evaluate_hand(hand: List[str], community: List[str]) -> int:
            # 简化的牌力评估（实际需要完整实现）
            all_cards = hand + community
            ranks = sorted(['23456789TJQKA'.index(c[0]) for c in all_cards], reverse=True)
            return max(ranks[:5])  # 暂时用最高牌判断

        best_score = -1
        winners = []
        for player in candidates:
            score = evaluate_hand(player.hand, self.community_cards)
            if score > best_score:
                best_score = score
                winners = [player]
            elif score == best_score:
                winners.append(player)
        return winners

    def settle_round(self) -> Dict[int, int]:
        side_pots = self._calculate_side_pots()
        results = defaultdict(int)

        for pot in side_pots:
            winners = self._determine_winners(pot["players"])
            if not winners:
                continue

            per_winner = pot["amount"] // len(winners)
            remainder = pot["amount"] % len(winners)
            
            for i, winner in enumerate(winners):
                amount = per_winner + (1 if i < remainder else 0)
                winner.stack += amount
                results[id(winner)] += amount

        self._reset_round()
        return dict(results)