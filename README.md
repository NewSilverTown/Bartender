# Bartender
Dependencies installation
```python
pip install -r requirements.txt
```

```python
#run command to start train
python -m training.train

#run command to collect action distribution
python -m analysis.action_distribution
```

So far the probably of each action like follow:<br>
![predition](/assets/8_players_analysis.PNG)


If you want to train your own model. You can changes these params: <br>

```yaml
#config/agents.yaml
ppo_config:
  num_players: 6
  learning_rate: 3e-4
  weight_decay: 1e-5
  buffer_size: 50000
  batch_size: 256
  gamma: 0.97
  clip_epsilon: 0.2
  value_coeff: 0.5
  entropy_coeff: 0.1
  max_grad_norm: 0.5
  episodes_per_update: 50
  max_updates: 10000
  save_dir: "checkpoints"
  save_interval: 2000
  log_interval: 10
  input_dim: 128
```
and the policy net
```python
class PokerPolicyNet(nn.Module):
    """强化学习策略网络（兼容PokerGame）"""
    def __init__(self, input_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.tensor([1.0]))

       # 双流网络结构
        self.hand_stream = nn.Sequential(
            nn.Linear(input_dim//2, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32)
        )
        
        self.context_stream = nn.Sequential(
            nn.Linear(input_dim//2, 64),
            nn.ELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 动作头
        self.action_head = nn.Sequential(
            nn.Linear(128, 4),
            nn.Tanhshrink()
        )
        
        # 加注头
        self.raise_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )
...
```

Run Local server
```
# Run api service to provide predict api
python ./app/main.py
```

Deployment

```
#bartenader folder
docker build -t your_username/bartender:0.0.1 .
docker push your_username/bartender:0.0.1 # you should login docker hub firstly
docker pull your_username/bartender:0.0.1 # pull docker image in you cloud server

docker run -d -p portnumber:portnumber --name bartender yourusername/bartender:0.0.1
```