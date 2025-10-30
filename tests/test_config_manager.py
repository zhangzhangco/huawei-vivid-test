"""
ConfigManager配置管理模块单元测试
测试阈值配置加载、验证、热更新等核心功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import json
import tempfile
import time
from unittest.mock import patch, mock_open

from src.core.config_manager import ConfigManager


class TestConfigManager:
    """ConfigManager类测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建临时配置文件
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        self.config_manager = ConfigManager(config_path=self.temp_config)
        
    def teardown_method(self):
        """测试后清理"""
        if os.path.exists(self.temp_config):
            os.remove(self.temp_config)
    
    def test_initialization(self):
        """测试初始化"""
        # 测试默认配置路径
        default_manager = ConfigManager()
        assert default_manager.config_path == "config/metrics.json"
        
        # 测试自定义配置路径
        assert self.config_manager.config_path == self.temp_config
        assert self.config_manager._thresholds is None
        assert self.config_manager._last_modified is None
    
    def test_get_default_thresholds(self):
        """测试获取默认阈值配置"""
        defaults = self.config_manager.get_default_thresholds()
        
        # 验证必需的键
        required_keys = {"S_ratio", "C_shadow", "R_DR_tolerance", "Dprime"}
        assert required_keys.issubset(defaults.keys())
        
        # 验证默认值
        assert defaults["S_ratio"] == 0.05
        assert defaults["C_shadow"] == 0.10
        assert defaults["R_DR_tolerance"] == 0.2
        assert defaults["Dprime"] == 0.25
        
        # 验证数值类型和范围
        for key, value in defaults.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1.0    

    def test_validate_thresholds(self):
        """测试阈值配置验证逻辑"""
        # 测试有效配置
        valid_config = {
            "S_ratio": 0.05,
            "C_shadow": 0.10,
            "R_DR_tolerance": 0.2,
            "Dprime": 0.25
        }
        assert self.config_manager.validate_thresholds(valid_config) is True
        
        # 测试缺少必需键
        incomplete_config = {
            "S_ratio": 0.05,
            "C_shadow": 0.10
        }
        assert self.config_manager.validate_thresholds(incomplete_config) is False
        
        # 测试数值超出范围
        out_of_range_config = {
            "S_ratio": 1.5,  # > 1.0
            "C_shadow": -0.1,  # < 0.0
            "R_DR_tolerance": 0.2,
            "Dprime": 0.25
        }
        assert self.config_manager.validate_thresholds(out_of_range_config) is False
        
        # 测试非数值类型
        invalid_type_config = {
            "S_ratio": "0.05",  # 字符串
            "C_shadow": 0.10,
            "R_DR_tolerance": 0.2,
            "Dprime": 0.25
        }
        assert self.config_manager.validate_thresholds(invalid_type_config) is False
        
        # 测试边界值
        boundary_config = {
            "S_ratio": 0.0,  # 最小值
            "C_shadow": 1.0,  # 最大值
            "R_DR_tolerance": 0.5,
            "Dprime": 1.0
        }
        assert self.config_manager.validate_thresholds(boundary_config) is True
    
    def test_load_thresholds_with_valid_file(self):
        """测试从有效配置文件加载阈值"""
        # 创建有效配置文件
        valid_config = {
            "S_ratio": 0.08,
            "C_shadow": 0.15,
            "R_DR_tolerance": 0.3,
            "Dprime": 0.3
        }
        
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(valid_config, f)
        
        # 加载配置
        loaded_thresholds = self.config_manager.load_thresholds()
        
        # 验证加载的配置
        assert loaded_thresholds == valid_config
        assert self.config_manager._thresholds == valid_config
        assert self.config_manager._last_modified is not None
    
    def test_load_thresholds_nonexistent_file(self):
        """测试配置文件不存在时的处理"""
        # 删除临时文件
        if os.path.exists(self.temp_config):
            os.remove(self.temp_config)
        
        # 加载配置应该返回默认值
        loaded_thresholds = self.config_manager.load_thresholds()
        default_thresholds = self.config_manager.get_default_thresholds()
        
        assert loaded_thresholds == default_thresholds
    
    def test_load_thresholds_invalid_json(self):
        """测试无效JSON文件的处理"""
        # 创建无效JSON文件
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            f.write('{"invalid": json}')
        
        # 加载配置应该返回默认值
        loaded_thresholds = self.config_manager.load_thresholds()
        default_thresholds = self.config_manager.get_default_thresholds()
        
        assert loaded_thresholds == default_thresholds
    
    def test_load_thresholds_invalid_config(self):
        """测试无效配置的处理"""
        # 创建无效配置文件
        invalid_config = {
            "S_ratio": 1.5,  # 超出范围
            "C_shadow": 0.10
        }
        
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(invalid_config, f)
        
        # 加载配置应该返回默认值
        loaded_thresholds = self.config_manager.load_thresholds()
        default_thresholds = self.config_manager.get_default_thresholds()
        
        assert loaded_thresholds == default_thresholds
    
    def test_hot_reload_functionality(self):
        """测试配置文件热更新功能"""
        # 创建初始配置
        initial_config = {
            "S_ratio": 0.05,
            "C_shadow": 0.10,
            "R_DR_tolerance": 0.2,
            "Dprime": 0.25
        }
        
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(initial_config, f)
        
        # 首次加载
        first_load = self.config_manager.load_thresholds()
        assert first_load == initial_config
        
        # 等待一小段时间确保文件修改时间不同
        time.sleep(0.1)
        
        # 修改配置文件
        updated_config = {
            "S_ratio": 0.08,
            "C_shadow": 0.15,
            "R_DR_tolerance": 0.3,
            "Dprime": 0.3
        }
        
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f)
        
        # 再次加载应该获取更新的配置
        second_load = self.config_manager.load_thresholds()
        assert second_load == updated_config
        assert second_load != first_load
    
    def test_caching_mechanism(self):
        """测试配置缓存机制"""
        # 创建配置文件
        config = {
            "S_ratio": 0.05,
            "C_shadow": 0.10,
            "R_DR_tolerance": 0.2,
            "Dprime": 0.25
        }
        
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        
        # 首次加载
        first_load = self.config_manager.load_thresholds()
        first_modified = self.config_manager._last_modified
        
        # 再次加载（文件未修改）
        second_load = self.config_manager.load_thresholds()
        second_modified = self.config_manager._last_modified
        
        # 应该使用缓存
        assert first_load == second_load
        assert first_modified == second_modified
        assert self.config_manager._thresholds is not None    

    def test_create_default_config_file(self):
        """测试创建默认配置文件"""
        # 删除临时文件
        if os.path.exists(self.temp_config):
            os.remove(self.temp_config)
        
        # 创建默认配置文件
        success = self.config_manager.create_default_config_file()
        assert success is True
        assert os.path.exists(self.temp_config)
        
        # 验证文件内容
        with open(self.temp_config, 'r', encoding='utf-8') as f:
            created_config = json.load(f)
        
        default_config = self.config_manager.get_default_thresholds()
        assert created_config == default_config
    
    def test_update_threshold(self):
        """测试更新单个阈值配置"""
        # 创建初始配置文件
        initial_config = self.config_manager.get_default_thresholds()
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(initial_config, f)
        
        # 更新阈值
        success = self.config_manager.update_threshold("S_ratio", 0.08)
        assert success is True
        
        # 验证更新结果
        updated_thresholds = self.config_manager.load_thresholds()
        assert updated_thresholds["S_ratio"] == 0.08
        
        # 验证其他值未改变
        assert updated_thresholds["C_shadow"] == initial_config["C_shadow"]
        
        # 测试无效更新
        invalid_success = self.config_manager.update_threshold("S_ratio", 1.5)
        assert invalid_success is False
    
    def test_get_threshold(self):
        """测试获取单个阈值"""
        # 创建配置文件
        config = {
            "S_ratio": 0.08,
            "C_shadow": 0.15,
            "R_DR_tolerance": 0.3,
            "Dprime": 0.3
        }
        
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        
        # 测试获取存在的阈值
        assert self.config_manager.get_threshold("S_ratio") == 0.08
        assert self.config_manager.get_threshold("C_shadow") == 0.15
        
        # 测试获取不存在的阈值
        assert self.config_manager.get_threshold("nonexistent") is None
    
    def test_reset_to_defaults(self):
        """测试重置配置为默认值"""
        # 创建自定义配置文件
        custom_config = {
            "S_ratio": 0.08,
            "C_shadow": 0.15,
            "R_DR_tolerance": 0.3,
            "Dprime": 0.3
        }
        
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(custom_config, f)
        
        # 验证自定义配置已加载
        loaded_config = self.config_manager.load_thresholds()
        assert loaded_config == custom_config
        
        # 重置为默认值
        success = self.config_manager.reset_to_defaults()
        assert success is True
        
        # 验证重置结果
        reset_config = self.config_manager.load_thresholds()
        default_config = self.config_manager.get_default_thresholds()
        assert reset_config == default_config
    
    def test_error_handling(self):
        """测试异常处理"""
        # 测试权限错误
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            thresholds = self.config_manager.load_thresholds()
            default_thresholds = self.config_manager.get_default_thresholds()
            assert thresholds == default_thresholds
        
        # 测试创建配置文件时的权限错误
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            success = self.config_manager.create_default_config_file()
            assert success is False
        
        # 测试更新阈值时的文件错误
        with patch('builtins.open', side_effect=IOError("File error")):
            success = self.config_manager.update_threshold("S_ratio", 0.08)
            assert success is False
    
    def test_configuration_directory_creation(self):
        """测试配置目录创建"""
        # 使用不存在的目录路径
        nested_config_path = os.path.join(tempfile.gettempdir(), "test_config", "nested", "metrics.json")
        nested_manager = ConfigManager(config_path=nested_config_path)
        
        # 创建默认配置文件应该自动创建目录
        success = nested_manager.create_default_config_file()
        assert success is True
        assert os.path.exists(nested_config_path)
        
        # 清理
        import shutil
        shutil.rmtree(os.path.dirname(os.path.dirname(nested_config_path)))
    
    def test_unicode_handling(self):
        """测试Unicode字符处理"""
        # 创建包含中文注释的配置文件
        config_with_unicode = {
            "S_ratio": 0.05,  # 高光饱和比例
            "C_shadow": 0.10,  # 暗部压缩比例
            "R_DR_tolerance": 0.2,  # 动态范围容差
            "Dprime": 0.25  # D'指标
        }
        
        # 手动写入包含中文的JSON
        with open(self.temp_config, 'w', encoding='utf-8') as f:
            json.dump(config_with_unicode, f, ensure_ascii=False, indent=2)
        
        # 加载配置应该正常工作
        loaded_thresholds = self.config_manager.load_thresholds()
        assert loaded_thresholds["S_ratio"] == 0.05
        assert loaded_thresholds["C_shadow"] == 0.10


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])