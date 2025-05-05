from game_simulator import PokerGame, ActionType

# 测试代码
def test_poker_game():
    # 初始化3人局
    game = PokerGame(num_players=3, big_blind=20)
    
    # 模拟pre-flop下注
    game.apply_action(ActionType.CHECK_CALL)  # SB补齐
    game.apply_action(ActionType.RAISE, 60)  # BB加注到60
    game.apply_action(ActionType.CHECK_CALL)  # 按钮位跟注
    game.apply_action(ActionType.CHECK_CALL)  # SB跟注
    
    # 进入翻牌圈
    game.next_phase()
    
    # 模拟摊牌
    results = game.settle_round()
    print("结算结果:", results)

if __name__ == "__main__":
    test_poker_game()