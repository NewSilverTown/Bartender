from typing import List, Dict, Optional
import random
from enum import Enum
from collections import defaultdict
from itertools import combinations

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
        self.available_actions = []

    def reset(self):
        """完整重置游戏状态"""
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.community_cards = []
        self.game_phase = 0
        self.pot = 0
        self.current_player = 0
        self.available_actions = []
        
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
        can_all_in = (player.stack > 0) and can_raise
        actions.append({
            'type': ActionType.ALL_IN,
            'available': can_all_in,
            'min': player.stack,
            'max': player.stack
        })

        if len(self.available_actions) > 0:
            available_actions_map = {
                action['type']: action
                for action in self.available_actions
            }

            # 分步筛选
            valid_actions = []
            for action in actions:
                # 第一步：检查是否 available 为 True
                if not action.get('available', False):
                    continue
                    
                # 第二步：检查是否存在同类型的可用动作
                action_type = action.get('type')
                matched_action = available_actions_map.get(action_type)
                
                if matched_action:
                    # 返回 available_actions 中的完整对象
                    valid_actions.append(matched_action)
                    
            return valid_actions

        return [
            action for action in actions 
            if action.get("available", False)
        ]

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
            min_raise = max(
                self.big_blind,
                current_max + self.big_blind - player.current_bet
            )
            if raise_amount < min_raise:
                raise ValueError(f"加注必须至少 {min_raise}")
            

            available = player.stack + player.current_bet
            actual_raise = min(raise_amount, available)
            self._place_bet(self.current_player, actual_raise)
        elif action_type == ActionType.ALL_IN:
            self._place_bet(self.current_player, player.stack)

        self._advance_to_next_player()

    def _advance_to_next_player(self):
        start_idx = self.current_player
        for _ in range(self.num_players):
            self.current_player = (self.current_player + 1) % self.num_players
            player = self.players[self.current_player]
            if player.is_in_hand and not player.is_all_in:
                return
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
        {p.total_bet: p for p in self.players if p.total_bet > 0}.values(),
            key=lambda x: x.total_bet
        )
        
        pots = []
        prev_total = 0
        for player in sorted_players:
            current_total = player.total_bet
            if current_total <= prev_total:
                continue
                
            contribution = current_total - prev_total
            eligible = [p for p in self.players if p.total_bet >= current_total]
            
            if eligible:
                pot_size = contribution * len(eligible)
                pots.append({
                    "amount": pot_size,
                    "players": eligible
                })
                prev_total = current_total
        
        return pots

    # 改进的牌力评估方法
    def _determine_winners(self, candidates: List[Player]) -> List[Player]:
        def evaluate_hand(hand: List[str], community: List[str]):
            # 将牌转换为数值格式 (rank, suit)
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 
                    '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 
                    'Q': 12, 'K': 13, 'A': 14}
            all_cards = [(rank_map[c[0]], c[1]) for c in hand + community]
            
            # 生成所有可能的5张组合
            best_rank = (0, [])
            for combo in combinations(all_cards, 5):
                ranks = sorted([c[0] for c in combo], reverse=True)
                suits = [c[1] for c in combo]
                
                # 检查同花和顺子
                flush = len(set(suits)) == 1
                straight = (max(ranks) - min(ranks) == 4 and len(set(ranks))) == 5
                straight_ace_low = set(ranks) == {14, 2, 3, 4, 5}
                
                # 牌型判断
                count = {r: ranks.count(r) for r in set(ranks)}
                pairs = sorted([r for r in count if count[r] == 2], reverse=True)
                trips = [r for r in count if count[r] == 3]
                quads = [r for r in count if count[r] == 4]
                full_house = len(trips) >= 1 and len(pairs) >= 1
                
                # 牌型分级
                if flush and straight:
                    if max(ranks) == 14:  # 皇家同花顺
                        rank = (9, [])
                    else:  # 普通同花顺
                        rank = (8, [max(ranks)])
                elif quads:
                    rank = (7, quads, [r for r in ranks if r != quads[0]])
                elif full_house:
                    rank = (6, [trips[0], pairs[0]])
                elif flush:
                    rank = (5, sorted(ranks, reverse=True))
                elif straight or straight_ace_low:
                    if straight_ace_low:
                        high = 5
                    else:
                        high = max(ranks)
                    rank = (4, [high])
                elif trips:
                    rank = (3, trips, sorted([r for r in ranks if r not in trips], reverse=True))
                elif len(pairs) >= 2:
                    rank = (2, pairs[:2], sorted([r for r in ranks if r not in pairs[:2]], reverse=True))
                elif len(pairs) == 1:
                    rank = (1, pairs, sorted([r for r in ranks if r not in pairs], reverse=True))
                else:
                    rank = (0, sorted(ranks, reverse=True))
                
                if rank > best_rank:
                    best_rank = rank
            
            return best_rank

        best_hand = None
        winners = []
        for player in candidates:
            current_rank = evaluate_hand(player.hand, self.community_cards)
            if not best_hand or current_rank > best_hand:
                best_hand = current_rank
                winners = [player]
            elif current_rank == best_hand:
                winners.append(player)
        
        return winners

    def settle_round(self) -> Dict[int, int]:

        for p in self.players:
            p.total_bet = p.current_bet  # 确保total_bet包含当前回合下注
        
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

        self.reset()
        return dict(results)
    
    def reset_players_stack(self, stacks: List[int]):
        """强制设置玩家筹码（测试用）"""
        for i, stack in enumerate(stacks):
            self.players[i].stack = stack
            self.players[i].is_in_hand = True
            self.players[i].is_all_in = False