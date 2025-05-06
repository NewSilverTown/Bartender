import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.game_simulator import ActionType

class DynamicActionEncoder:
    """动态动作空间编码器（维度修复版）"""
    def __init__(self):
        self.action_types = [ActionType.FOLD, ActionType.CHECK_CALL, 
                           ActionType.RAISE, ActionType.ALL_IN]
        
    def encode_actions(self, legal_actions):
        """生成合法的动作类型掩码"""
        legal_mask = torch.zeros(len(self.action_types), dtype=torch.float32)
        for action in legal_actions:
            if action['available']:
                idx = self.action_types.index(action['type'])
                legal_mask[idx] = 1.0
        return legal_mask.unsqueeze(0)  # 添加batch维度


class PokerPolicyNet(nn.Module):
    def __init__(self, state_dim=128):
        super().__init__()
        # 状态编码器保持不变
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(0.3)
        )
        
        # 改进动作头结构
        self.action_type_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 4)
        )
        
        # 加注金额预测头增加复杂度
        self.raise_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 价值网络增强
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )


    def forward(self, state, legal_mask):
        # 状态特征提取
        state_feat = self.state_encoder(state)
        
        # 动作类型预测（修复掩码处理）
        action_logits = self.action_type_head(state_feat)
        
        # 改进掩码处理
        if legal_mask is not None:
            legal_mask = legal_mask.to(action_logits.device)
            inf_mask = torch.clamp(torch.log(legal_mask), min=-torch.inf, max=torch.inf)
            masked_logits = action_logits + inf_mask
        else:
            masked_logits = action_logits

        action_probs = F.softmax(masked_logits, dim=1)
        # 加注金额预测
        raise_ratio = self.raise_head(state_feat)

        state_value = self.value_head(state_feat)
        
        return action_probs, raise_ratio.squeeze(-1), state_value

    def predict(self, state, legal_actions):
        # 生成合法动作掩码（修复版）
        legal_mask = torch.zeros(4, dtype=torch.float)
        action_types = [ActionType.FOLD, ActionType.CHECK_CALL, 
                   ActionType.RAISE, ActionType.ALL_IN]
        
        for action in legal_actions:
            if action['type'] == ActionType.RAISE:
                # 确保最小加注量不超过剩余筹码
                actual_min = min(action['min'], action['player_stack'])
                actual_max = min(action['max'], action['player_stack'])
                action['available'] = (actual_max >= actual_min) and (action['player_stack'] > 0)
            if action['available']:
                idx = action_types.index(action['type'])
                legal_mask[idx] = 1.0
        
        with torch.no_grad():
            action_probs, raise_ratio, _ = self(  # 忽略value输出
                state.unsqueeze(0),
                legal_mask.unsqueeze(0)
            )
            
        # 过滤可用动作
        available_actions = [a for a in legal_actions if a['available']]
        available_types = [a['type'] for a in available_actions]
        
        valid_probs = action_probs[0][legal_mask.bool()]
        valid_probs /= valid_probs.sum()

        # 提取对应概率
        probs_dict = {}
        for idx, prob in enumerate(valid_probs):
            action_type = action_types[legal_mask.nonzero()[idx].item()]
            probs_dict[action_type.name] = prob.item()
        
        # 选择动作
        selected_idx = torch.multinomial(valid_probs, 1).item()
        selected_type = action_types[legal_mask.nonzero()[selected_idx].item()]
        
        # 构建action_info
        action_info = {
            'action': next(a for a in legal_actions if a['type'] == selected_type),
            'probs': probs_dict,
            'raise_ratio': raise_ratio.item()
        }

        # 计算加注金额
        if selected_type == ActionType.RAISE:
            min_raise = action_info['action']['min']
            max_raise = action_info['action']['max']
            amount = min_raise + (max_raise - min_raise) * raise_ratio.item()
            action_info['action']['amount'] = int(np.clip(amount, min_raise, max_raise))
        
        return action_info

# 新增模型保存加载函数
def save_model(model, path="models/poker_policy.pt"):
    torch.save({
        'model_state': model.state_dict(),
        'state_dim': model.state_encoder[0].in_features
    }, path)

def load_model(path="models/poker_policy.pt"):
    checkpoint = torch.load(path)
    model = PokerPolicyNet(state_dim=checkpoint['state_dim'])
    model.load_state_dict(checkpoint['model_state'])
    return model


# # 测试代码
# if __name__ == "__main__":
#     # 测试模型保存加载
#     net = PokerPolicyNet(state_dim=256)
#     save_model(net)
    
#     loaded_net = load_model()
#     print("模型加载测试通过！")