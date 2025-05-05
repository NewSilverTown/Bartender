import numpy as np

class GameEncoder:
    @staticmethod
    def encode_hand(hand):
        for card in hand:
            if len(card) < 2:
                raise ValueError(f"Invalid card format: {card}. Expected like 'Ah'")
        """将手牌转换为9个强度等级"""
        ranks = [card[0] for card in hand]  # 假设card格式如 ('A', 'h')
        suits = [card[1] for card in hand]
        
        # 手牌强度分类（简化版）
        if ranks[0] == ranks[1]:
            strength = 0  # 对子
        elif suits[0] == suits[1]:
            strength = 1  # 同花
        else:
            strength = 2  # 高牌
        return np.eye(9)[strength]  # 扩展到9维

    @staticmethod
    def encode_bet_history(history):
        """将下注历史转换为向量"""
        bet_types = {'fold':0, 'call':1, 'raise':2}
        vec = np.zeros(4)
        for i, action in enumerate(history[-4:]):  # 仅保留最近4步
            vec[i] = bet_types.get(action, 0)
        return vec
    
    @staticmethod
    def encode_community(cards):
        """编码公共牌（示例简化版）"""
        # 实际应使用更专业的扑克牌编码
        return np.zeros(169)  # 169维标准编码
    
    @staticmethod
    def encode_stack(stack: int) -> np.array:
        """编码筹码量(归一化到0-1)"""
        return np.array([stack / 10000])  # 假设最大筹码量为10000