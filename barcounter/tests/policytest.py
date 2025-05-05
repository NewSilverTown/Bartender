import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_net import PokerPolicyNet
import torch

def testpolicy():
    model = PokerPolicyNet()
    model.load_state_dict(torch.load("models/best_policy.pt"))
    print("模型加载成功！输入维度:", model.net[0].in_features)  # 应输出182
    print("输出维度:", model.net[-1].out_features)        # 应输出4

def actiontest():
    model = PokerPolicyNet()
    # 生成模拟输入
    test_input = torch.randn(1, 183)  # 随机生成合法状态
    probs = torch.softmax(model(test_input), dim=1)
    print("动作概率分布:", probs)
    # 期望输出示例: tensor([[0.12, 0.33, 0.48, 0.07]])
    # 表示弃牌12%、跟注33%、加注48%、全押7%

if __name__ == "__main__":
    testpolicy()
    actiontest()