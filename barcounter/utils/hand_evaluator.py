# utils/hand_evaluator.py
from typing import List

def evaluate_hand_strength(cards: List[str]) -> float:
    """
    评估德州扑克手牌强度（简化版）
    输入示例: ['Ah', 'Ks', 'Td', '2c', '5h']（2张手牌+3张公共牌）
    返回值: 0.0（最弱）~ 1.0（最强）
    """
    # 解析牌面
    ranks = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, 
             '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
    suits = [c[1] for c in cards]
    values = sorted([ranks[c[0]] for c in cards], reverse=True)
    
    # 判断牌型
    is_flush = len(set(suits)) == 1  # 同花
    is_straight = (max(values) - min(values) == 4) or set(values) == {14,2,3,4,5}  # 顺子
    
    # 统计相同点数
    count = {}
    for v in values:
        count[v] = count.get(v, 0) + 1
    pairs = [v for v, c in count.items() if c == 2]
    trips = [v for v, c in count.items() if c == 3]
    quads = [v for v, c in count.items() if c == 4]
    
    # 牌型分级
    if is_straight and is_flush:
        if max(values) == 14:  # 皇家同花顺
            return 1.0
        else:  # 同花顺
            return 0.95
    elif quads:  # 四条
        return 0.85
    elif trips and pairs:  # 葫芦
        return 0.75
    elif is_flush:  # 同花
        return 0.65
    elif is_straight:  # 顺子
        return 0.55
    elif trips:  # 三条
        return 0.45
    elif len(pairs) >= 2:  # 两对
        return 0.35
    elif len(pairs) == 1:  # 一对
        return 0.25
    else:  # 高牌
        return max(values) / 20  # 将高牌值映射到0.05-0.7之间