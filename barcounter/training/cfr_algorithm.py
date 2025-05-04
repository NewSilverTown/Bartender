# training/cfr_algorithm.py
import torch

class PokerPolicyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 输入=手牌强度(9维) + 公共牌编码(169维) + 下注历史(4维)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(182, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4)  # 输出动作概率：弃牌/跟注/加注/全押
        )
    
    def forward(self, state):
        return self.net(state)

def train():
    model = PokerPolicyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 自对战生成训练数据
    for epoch in range(1000):
        trajectories = self_play.generate_data(model)
        
        # 计算反事实遗憾
        regrets = calculate_regrets(trajectories)
        
        # 策略更新
        loss = torch.mean(torch.stack([r**2 for r in regrets]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 保存模型
        if epoch % 100 == 0:
            torch.save(model.state_dict(), "models/poker_policy.pt")