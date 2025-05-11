import os,sys
import yaml
from pathlib import Path
from typing import Dict, Any

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

class ConfigLoader:
    @staticmethod
    def load(config_path: str, config_name: str = "ppo_config") -> Dict[str, Any]:
        """
        加载YAML配置文件
        :param config_path: 配置文件路径
        :param config_name: 要加载的配置块名称
        :return: 配置字典
        """
        # 获取项目根目录
        project_root = Path(__file__).parent.parent
        
        # 构建完整路径
        full_path = project_root / config_path
        
        try:
            with open(full_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get(config_name, {})
        except FileNotFoundError:
            raise ValueError(f"配置文件 {full_path} 不存在")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件解析错误: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"配置加载失败: {str(e)}")

    @staticmethod
    def merge_with_defaults(custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并默认配置和自定义配置"""
        defaults = {
            'ppo_config': {
                'num_players': 6,
                'learning_rate': 1e-3,
                # ...其他默认参数
            }
        }
        return {**defaults.get('ppo_config', {}), **custom_config}
