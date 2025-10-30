"""
配置管理模块 - 处理HDR质量评估扩展模块的阈值配置
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器类，负责处理质量评估阈值配置的加载、验证和热更新"""
    
    def __init__(self, config_path: str = "config/metrics.json"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为 config/metrics.json
        """
        self.config_path = config_path
        self._thresholds = None
        self._last_modified = None
        
    def get_default_thresholds(self) -> Dict[str, float]:
        """
        获取默认阈值配置
        
        Returns:
            包含默认阈值的字典
        """
        return {
            "S_ratio": 0.05,        # 高光饱和比例阈值
            "C_shadow": 0.10,       # 暗部压缩比例阈值  
            "R_DR_tolerance": 0.2,  # 动态范围保持率容差
            "Dprime": 0.25          # D'指标阈值
        }
    
    def validate_thresholds(self, thresholds: Dict[str, Any]) -> bool:
        """
        验证阈值配置的有效性
        
        Args:
            thresholds: 待验证的阈值字典
            
        Returns:
            bool: 配置是否有效
        """
        required_keys = {"S_ratio", "C_shadow", "R_DR_tolerance", "Dprime"}
        
        # 检查必需的键是否存在
        if not required_keys.issubset(thresholds.keys()):
            missing_keys = required_keys - thresholds.keys()
            logger.warning(f"配置文件缺少必需的键: {missing_keys}")
            return False
        
        # 验证数值范围
        for key, value in thresholds.items():
            if key in required_keys:
                if not isinstance(value, (int, float)):
                    logger.warning(f"配置项 {key} 的值不是数字类型: {value}")
                    return False
                
                # 验证数值在合理范围内 (0-1之间，R_DR_tolerance可以稍大)
                if key == "R_DR_tolerance":
                    if not (0 <= value <= 1.0):
                        logger.warning(f"配置项 {key} 的值超出范围 [0, 1.0]: {value}")
                        return False
                else:
                    if not (0 <= value <= 1.0):
                        logger.warning(f"配置项 {key} 的值超出范围 [0, 1.0]: {value}")
                        return False
        
        return True
    
    def load_thresholds(self) -> Dict[str, float]:
        """
        加载质量判定阈值，支持热更新
        
        Returns:
            阈值配置字典
        """
        # 检查文件是否存在
        if not os.path.exists(self.config_path):
            logger.info(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self.get_default_thresholds()
        
        try:
            # 检查文件修改时间，实现热更新
            current_modified = os.path.getmtime(self.config_path)
            
            if (self._thresholds is None or 
                self._last_modified is None or 
                current_modified > self._last_modified):
                
                logger.info(f"加载配置文件: {self.config_path}")
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 验证配置
                if self.validate_thresholds(config_data):
                    self._thresholds = config_data
                    self._last_modified = current_modified
                    logger.info("配置文件加载成功")
                else:
                    logger.warning("配置文件验证失败，使用默认配置")
                    self._thresholds = self.get_default_thresholds()
            
            return self._thresholds.copy()
            
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
            logger.warning("回退到默认配置")
            return self.get_default_thresholds()
            
        except Exception as e:
            logger.error(f"加载配置文件时发生错误: {e}")
            logger.warning("回退到默认配置")
            return self.get_default_thresholds()
    
    def create_default_config_file(self) -> bool:
        """
        创建默认配置文件
        
        Returns:
            bool: 是否创建成功
        """
        try:
            # 确保配置目录存在
            config_dir = os.path.dirname(self.config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            default_config = self.get_default_thresholds()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"默认配置文件已创建: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建默认配置文件失败: {e}")
            return False
    
    def update_threshold(self, key: str, value: float) -> bool:
        """
        更新单个阈值配置
        
        Args:
            key: 配置项名称
            value: 新的阈值
            
        Returns:
            bool: 是否更新成功
        """
        try:
            current_thresholds = self.load_thresholds()
            current_thresholds[key] = value
            
            if self.validate_thresholds(current_thresholds):
                # 保存到文件
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(current_thresholds, f, indent=2, ensure_ascii=False)
                
                # 清除缓存，强制重新加载
                self._thresholds = None
                self._last_modified = None
                
                logger.info(f"阈值 {key} 已更新为 {value}")
                return True
            else:
                logger.warning(f"阈值更新失败，验证不通过: {key}={value}")
                return False
                
        except Exception as e:
            logger.error(f"更新阈值时发生错误: {e}")
            return False
    
    def get_threshold(self, key: str) -> Optional[float]:
        """
        获取单个阈值
        
        Args:
            key: 配置项名称
            
        Returns:
            阈值值，如果不存在则返回None
        """
        thresholds = self.load_thresholds()
        return thresholds.get(key)
    
    def reset_to_defaults(self) -> bool:
        """
        重置配置为默认值
        
        Returns:
            bool: 是否重置成功
        """
        try:
            default_config = self.get_default_thresholds()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            # 清除缓存，强制重新加载
            self._thresholds = None
            self._last_modified = None
            
            logger.info("配置已重置为默认值")
            return True
            
        except Exception as e:
            logger.error(f"重置配置失败: {e}")
            return False