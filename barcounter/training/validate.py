# 添加验证代码（training/validate.py）
def test_against_random(num_games=100):
    win_count = 0
    for _ in range(num_games):
        if ai_wins():
            win_count +=1
    return win_count / num_games

# 每训练50轮测试一次
if epoch % 50 == 0:
    win_rate = test_against_random()
    print(f"当前胜率: {win_rate*100:.1f}%")