# Bartender
An AI texas poker robot

```python
#run command to start train
python -m training.train

#run command to collect action distribution
python .\analysis\action_distribution.py
```

SO far the probably of each action like follow:<br>
ALL_IN: 14.85%
CHECK_CALL: 32.78%
FOLD: 38.28%
RAISE: 14.09%

If you want to train your own model. You can changes these params: <br>

```python
config = {
        'num_players': 6,
        'learning_rate': 3e-4,
        'weight_decay': 1e-5,
        'buffer_size': 50000,
        'batch_size': 256,
        'gamma': 0.97,
        'clip_epsilon': 0.2,
        'value_coeff': 0.5,
        'entropy_coeff': 0.1,
        'max_grad_norm': 0.5,
        'episodes_per_update': 50,
        'max_updates': 1000,
        'save_dir': "checkpoints",
        'save_interval': 100,
        'log_interval': 10,
        'input_dim': 128    
    }
```
