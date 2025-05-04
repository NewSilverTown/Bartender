import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.policy_net import PokerPolicyNet, save_model
from utils.game_encoder import GameEncoder

# training/train.py 中修改 generate_training_data 函数
def generate_training_data(num_samples=1000):
    samples = []
    card_pool = ['Ah', 'Ad', 'Kh', 'Kd', 'Qh', 'Qd', 'Jh', 'Jd', 'Th', 'Td']  # 示例牌库
    
    for _ in range(num_samples):
        # 生成手牌和公共牌（包含花色）
        hand = np.random.choice(card_pool, size=2, replace=False)
        community = np.random.choice(card_pool, size=3, replace=False)
        bet_history = np.random.choice(['fold', 'call', 'raise'], size=4)
        
        # 编码状态
        state = np.concatenate([
            GameEncoder.encode_hand(hand),
            GameEncoder.encode_community(community),
            GameEncoder.encode_bet_history(bet_history)
        ])
        samples.append(state)
    return torch.FloatTensor(np.array(samples))

# 简化的CFR训练循环
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PokerPolicyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 生成模拟数据
    train_data = generate_training_data(5000)
    
    losses = []
    # 训练循环
    for epoch in tqdm(range(100)):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(train_data)
        
        # 简化的损失函数（实际应使用CFR算法）
        target = torch.softmax(outputs + 0.1*torch.randn_like(outputs), dim=1)
        loss = nn.CrossEntropyLoss()(outputs, target)

        losses.append(loss.item())
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 保存模型
        if epoch % 10 == 0:
            save_model(model)
         # 训练过程中打印损失值（在train.py中添加）
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.3f}")
    
    # 保存损失数据
    np.save("training/loss_history.npy", np.array(losses))
    print("训练完成！模型已保存至 models/ 目录")

if __name__ == "__main__":
    train()