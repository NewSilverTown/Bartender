# training/self_play.py
def generate_data(model):
    players = [model.clone() for _ in range(6)]  # 6人桌
    game = TexasHoldem()
    
    while not game.is_terminal():
        current_player = game.current_player
        state = encode_game_state(game)  # 状态编码
        action_probs = players[current_player](state)
        action = sample_action(action_probs)
        game.apply_action(action)
    
    return game.history  # 返回完整对局记录