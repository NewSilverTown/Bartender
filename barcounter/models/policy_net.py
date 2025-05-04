import torch
import torch.nn as nn

class PokerPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(182, 256),  # 输入=手牌(9)+公共牌(169)+下注(4)
            nn.ReLU(),
            nn.Linear(256, 4)     # 输出动作概率：fold/call/raise/all-in
        )
    
    def forward(self, x):
        return self.net(x)

def save_model(model, path="models/poker_policy.pt"):
    torch.save(model.state_dict(), path)