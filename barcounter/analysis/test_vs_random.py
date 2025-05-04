# analysis/test_vs_random.py
# 在文件顶部添加以下代码
import sys
import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录（假设analysis在根目录下）
project_root = os.path.dirname(current_dir)
# 将根目录添加到Python路径
sys.path.append(project_root)

# analysis/test_vs_random.py
from utils.game_simulator import PokerGame, ActionType
import random

def test_ai_vs_random(num_games=100):
    win_count = 0
    for _ in range(num_games):
        game = PokerGame(num_players=2)
        game.players[1].is_ai = True  # 设置第二个玩家为AI
        
        while not game.is_terminal():
            if game.players[game.current_player].is_ai:
                # AI决策（示例随机选择动作）
                action = random.choice(list(ActionType))
                game.apply_action(action)
            else:
                # 随机策略
                action = random.choice(list(ActionType))
                game.apply_action(action)
                
            if len([p for p in game.players if p.is_in_hand]) == 1:
                break
                
        if game.determine_winner()[0] == 1:
            win_count += 1
            
    return win_count / num_games

print(f"AI胜率: {test_ai_vs_random()*100:.1f}%")