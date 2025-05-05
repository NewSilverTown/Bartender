import sys
import os

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录（假设analysis在根目录下）
project_root = os.path.dirname(current_dir)
# 将根目录添加到Python路径
sys.path.append(project_root)

from utils.game_simulator import ActionType, PokerGame
from train import PPOTrainer;
# 测试训练流程
def test_training():
    trainer = PPOTrainer(num_players=3)
    
    # 快速测试训练循环
    try:
        trainer.collect_experience(2)
        trainer.train(epochs=5)
        print("训练流程测试通过！")
    except Exception as e:
        print(f"训练流程测试失败: {str(e)}")

def test_action_execution():
     # 初始化2人局，大盲注20
    game = PokerGame(num_players=2, big_blind=20)
    
    # 验证盲注位置
    print("初始盲注状态:")
    for i, p in enumerate(game.players):
        print(f"玩家{i}: 当前下注={p.current_bet}, 剩余筹码={p.stack}")
    
    # 确保当前玩家是大盲位
    bb_player = game.players[1]
    game.current_player = 1  # 强制设置当前玩家为大盲位
    
    # 重置玩家状态
    bb_player.stack = 1000
    bb_player.current_bet = 0  # 清除自动下注的盲注
    bb_player.is_in_hand = True
    
    # 创建合法的加注动作（绝对金额）
    action = {
        'type': ActionType.RAISE,
        'amount': 200,  # 总下注目标金额
        'min': 40,      # 最小加注量 = 当前最大下注(20) + 大盲(20)
        'max': 500,
        'available': True
    }
    
    print("\n执行前大盲玩家状态:")
    print(f"当前下注: {bb_player.current_bet}")
    print(f"剩余筹码: {bb_player.stack}")
    
    # 执行动作
    trainer = PPOTrainer(num_players=2)
    reward = trainer._execute_action(game, action)
    
    print("\n执行后大盲玩家状态:")
    print(f"当前下注: {bb_player.current_bet}")
    print(f"剩余筹码: {bb_player.stack}")
    
    # 验证结果
    assert bb_player.current_bet == 200, f"当前下注应为200，实际为{bb_player.current_bet}"
    assert bb_player.stack == 800, f"剩余筹码应为800，实际为{bb_player.stack}"
    print("\n测试通过！")

if __name__ == "__main__":
    test_action_execution()
    test_training()