import torch
import torch.nn as nn
import random

class PokerPolicyNet(nn.Module):
    def __init__(self, input_dim=183):  # 允许动态调整输入维度
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(183, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def predict(self, state, epsilon=0.1):
        with torch.no_grad():
            probs = torch.softmax(self(state), dim=-1)
        
        # 以epsilon概率随机选择动作
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            return torch.argmax(probs).item()

def save_model(model, path="models/poker_policy.pt"):
    torch.save(model.state_dict(), path)