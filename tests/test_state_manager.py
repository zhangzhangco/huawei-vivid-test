"""
状态管理器测试
测试会话状态和时域状态的分离存储、JSON序列化和状态一致性验证
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.state_manager import StateManager, SessionState, TemporalStateData, get_state_manager, reset_state_manager


class TestSessionState:
    """测试会话状态"""
    
    def test_default_session_state(self):
        """测试默认会话状态"""
        state = SessionState()
        
        assert state.p == 2.0
        assert state.a == 0.5
        assert state.mode == "艺术模式"
        assert state.dt_low == 0.05
        assert state.dt_high == 0.10
        assert state.luminance_channel == "MaxRGB"
        assert state.enable_spline == False
        assert state.window_size == 9
        assert state.auto_save_enabled == True
        
    def test_session_state_serialization(self):
        """测试会话状态序列化"""
        state = SessionState(p=3.0, a=0.8, mode="自动模式")
        
        # 转换为字典
        state_dict = state.to_dict()
        assert state_dict['p'] == 3.0
        assert state_dict['a'] == 0.8
        assert state_dict['mode'] == "自动模式"
        
        # 从字典恢复
        restored_state = SessionState.from_dict(state_dict)
        assert restored_state.p == 3.0
        assert restored_state.a == 0.8
        assert restored_state.mode == "自动模式"
        
    def test_session_state_invalid_fields(self):
        """测试会话状态无效字段处理"""
        data = {
            'p': 2.5,
            'a': 0.6,
            'invalid_field': 'should_be_ignored',
            'another_invalid': 123
        }
        
        state = SessionState.from_dict(data)
        assert state.p == 2.5
        assert state.a == 0.6
        assert not hasattr(state, 'invalid_field')


class TestTemporalStateData:
    """测试时域状态数据"""
    
    def test_default_temporal_state(self):
        """测试默认时域状态"""
        state = TemporalStateData()
        
        assert state.parameter_history == []
        assert state.distortion_history == []
        assert state.timestamp_history == []
        assert state.current_frame == 0
        assert state.total_frames == 0
        assert state.variance_reduction == 0.0
        assert state.smoothing_active == False
        
    def test_temporal_state_reset_conditions(self):
        """测试时域状态重置条件"""
        state = TemporalStateData()
        state.last_mode = "艺术模式"
        state.last_channel = "MaxRGB"
        state.last_image_hash = "abc123"
        
        # 相同条件不应重置
        assert not state.should_reset("艺术模式", "MaxRGB", "abc123")
        
        # 不同模式应重置
        assert state.should_reset("自动模式", "MaxRGB", "abc123")
        
        # 不同通道应重置
        assert state.should_reset("艺术模式", "Y", "abc123")
        
        # 不同图像应重置
        assert state.should_reset("艺术模式", "MaxRGB", "def456")
        
    def test_temporal_state_reset(self):
        """测试时域状态重置"""
        state = TemporalStateData()
        
        # 添加一些数据
        state.parameter_history = [(2.0, 0.5), (2.1, 0.6)]
        state.distortion_history = [0.05, 0.06]
        state.timestamp_history = [1000, 1001]
        state.current_frame = 2
        state.total_frames = 5
        state.variance_reduction = 10.0
        state.smoothing_active = True
        
        # 重置
        state.reset("自动模式", "Y", "new_hash")
        
        assert state.parameter_history == []
        assert state.distortion_history == []
        assert state.timestamp_history == []
        assert state.current_frame == 0
        assert state.total_frames == 0
        assert state.variance_reduction == 0.0
        assert state.smoothing_active == False
        assert state.last_mode == "自动模式"
        assert state.last_channel == "Y"
        assert state.last_image_hash == "new_hash"


class TestStateManager:
    """测试状态管理器"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = StateManager(self.temp_dir)
        
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_state_manager_initialization(self):
        """测试状态管理器初始化"""
        assert self.state_manager.state_dir.exists()
        assert self.state_manager.session_file.name == "session_state.json"
        assert self.state_manager.temporal_file.name == "temporal_state.json"
        
    def test_get_default_states(self):
        """测试获取默认状态"""
        session_state = self.state_manager.get_session_state()
        temporal_state = self.state_manager.get_temporal_state()
        
        assert isinstance(session_state, SessionState)
        assert isinstance(temporal_state, TemporalStateData)
        assert session_state.p == 2.0
        assert temporal_state.current_frame == 0
        
    def test_update_session_state(self):
        """测试更新会话状态"""
        success = self.state_manager.update_session_state(p=3.0, a=0.8, mode="自动模式")
        assert success
        
        session_state = self.state_manager.get_session_state()
        assert session_state.p == 3.0
        assert session_state.a == 0.8
        assert session_state.mode == "自动模式"
        
    def test_update_session_state_invalid_field(self):
        """测试更新会话状态无效字段"""
        with patch.object(self.state_manager.logger, 'warning') as mock_warning:
            success = self.state_manager.update_session_state(invalid_field="test")
            assert success  # 应该成功，但会有警告
            mock_warning.assert_called_once()
            
    def test_update_temporal_state(self):
        """测试更新时域状态"""
        success = self.state_manager.update_temporal_state(
            p=2.0, a=0.5, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        assert success
        
        temporal_state = self.state_manager.get_temporal_state()
        assert temporal_state.current_frame == 1
        assert temporal_state.total_frames == 1
        assert len(temporal_state.parameter_history) == 1
        assert temporal_state.parameter_history[0] == (2.0, 0.5)
        assert temporal_state.distortion_history[0] == 0.05
        
    def test_temporal_state_auto_reset(self):
        """测试时域状态自动重置"""
        # 添加初始数据
        self.state_manager.update_temporal_state(
            p=2.0, a=0.5, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        
        temporal_state = self.state_manager.get_temporal_state()
        assert temporal_state.current_frame == 1
        
        # 切换模式应触发重置
        self.state_manager.update_temporal_state(
            p=2.1, a=0.6, distortion=0.06,
            mode="自动模式", channel="MaxRGB", image_hash="test123"
        )
        
        temporal_state = self.state_manager.get_temporal_state()
        assert temporal_state.current_frame == 1  # 重置后重新开始计数
        assert temporal_state.last_mode == "自动模式"
        
    def test_temporal_state_history_limit(self):
        """测试时域状态历史长度限制"""
        # 设置较小的窗口大小
        self.state_manager.update_session_state(window_size=3)
        
        # 添加超过限制的数据
        for i in range(10):
            self.state_manager.update_temporal_state(
                p=2.0 + i * 0.1, a=0.5, distortion=0.05 + i * 0.01,
                mode="艺术模式", channel="MaxRGB", image_hash="test123"
            )
            
        temporal_state = self.state_manager.get_temporal_state()
        max_history = 3 * 2  # window_size * 2
        assert len(temporal_state.parameter_history) <= max_history
        assert len(temporal_state.distortion_history) <= max_history
        assert len(temporal_state.timestamp_history) <= max_history
        
    def test_variance_reduction_calculation(self):
        """测试方差降低计算"""
        # 添加一些有变化的数据
        distortions = [0.10, 0.08, 0.12, 0.09, 0.07, 0.06, 0.065, 0.062]
        
        for i, distortion in enumerate(distortions):
            self.state_manager.update_temporal_state(
                p=2.0, a=0.5, distortion=distortion,
                mode="艺术模式", channel="MaxRGB", image_hash="test123"
            )
            
        temporal_state = self.state_manager.get_temporal_state()
        assert temporal_state.variance_reduction >= 0
        assert temporal_state.smoothing_active == True
        
    def test_save_and_load_session_state(self):
        """测试保存和加载会话状态"""
        # 更新状态
        self.state_manager.update_session_state(p=3.5, a=0.9, mode="自动模式")
        
        # 保存
        success = self.state_manager.save_session_state()
        assert success
        assert self.state_manager.session_file.exists()
        
        # 创建新的状态管理器并加载
        new_manager = StateManager(self.temp_dir)
        session_state = new_manager.get_session_state()
        
        assert session_state.p == 3.5
        assert session_state.a == 0.9
        assert session_state.mode == "自动模式"
        
    def test_save_and_load_temporal_state(self):
        """测试保存和加载时域状态"""
        # 添加时域数据
        self.state_manager.update_temporal_state(
            p=2.0, a=0.5, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        self.state_manager.update_temporal_state(
            p=2.1, a=0.6, distortion=0.06,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        
        # 保存
        success = self.state_manager.save_temporal_state()
        assert success
        assert self.state_manager.temporal_file.exists()
        
        # 创建新的状态管理器并加载
        new_manager = StateManager(self.temp_dir)
        temporal_state = new_manager.get_temporal_state()
        
        assert temporal_state.current_frame == 2
        assert len(temporal_state.parameter_history) == 2
        assert temporal_state.parameter_history[0] == (2.0, 0.5)
        assert temporal_state.parameter_history[1] == (2.1, 0.6)
        
    def test_clear_temporal_state(self):
        """测试清空时域状态"""
        # 添加数据
        self.state_manager.update_temporal_state(
            p=2.0, a=0.5, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        
        temporal_state = self.state_manager.get_temporal_state()
        assert temporal_state.current_frame == 1
        
        # 清空
        success = self.state_manager.clear_temporal_state("自动模式", "Y", "new_hash")
        assert success
        
        temporal_state = self.state_manager.get_temporal_state()
        assert temporal_state.current_frame == 0
        assert temporal_state.last_mode == "自动模式"
        assert temporal_state.last_channel == "Y"
        assert temporal_state.last_image_hash == "new_hash"
        
    def test_validate_state_consistency(self):
        """测试状态一致性验证"""
        # 设置有效状态
        self.state_manager.update_session_state(p=2.0, a=0.5, th1=0.2, th2=0.5, th3=0.8)
        self.state_manager.update_temporal_state(
            p=2.0, a=0.5, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        
        is_consistent, errors = self.state_manager.validate_state_consistency()
        assert is_consistent
        assert len(errors) == 0
        
    def test_validate_state_consistency_invalid_params(self):
        """测试状态一致性验证 - 无效参数"""
        # 设置无效参数
        self.state_manager.update_session_state(p=10.0, a=1.5)  # 超出范围
        
        is_consistent, errors = self.state_manager.validate_state_consistency()
        assert not is_consistent
        assert len(errors) > 0
        assert any("p值超出范围" in error for error in errors)
        assert any("a值超出范围" in error for error in errors)
        
    def test_validate_state_consistency_invalid_spline_nodes(self):
        """测试状态一致性验证 - 无效样条节点"""
        # 设置无效样条节点
        self.state_manager.update_session_state(
            enable_spline=True, th1=0.8, th2=0.5, th3=0.2  # 顺序错误
        )
        
        is_consistent, errors = self.state_manager.validate_state_consistency()
        assert not is_consistent
        assert any("样条节点顺序不正确" in error for error in errors)
        
    def test_get_state_summary(self):
        """测试获取状态摘要"""
        # 设置一些状态
        self.state_manager.update_session_state(p=2.5, mode="自动模式")
        self.state_manager.update_temporal_state(
            p=2.5, a=0.5, distortion=0.05,
            mode="自动模式", channel="MaxRGB", image_hash="test123"
        )
        
        summary = self.state_manager.get_state_summary()
        
        assert "session" in summary
        assert "temporal" in summary
        assert "files" in summary
        
        assert summary["session"]["mode"] == "自动模式"
        assert summary["session"]["parameters"]["p"] == 2.5
        assert summary["temporal"]["current_frame"] == 1
        
    def test_state_change_listeners(self):
        """测试状态变化监听器"""
        changes_received = []
        
        def listener(state_type, changes):
            changes_received.append((state_type, changes))
            
        self.state_manager.add_state_change_listener(listener)
        
        # 触发状态变化
        self.state_manager.update_session_state(p=3.0)
        self.state_manager.update_temporal_state(
            p=3.0, a=0.5, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        
        assert len(changes_received) == 2
        assert changes_received[0][0] == "session"
        assert changes_received[1][0] == "temporal"
        
        # 移除监听器
        self.state_manager.remove_state_change_listener(listener)
        self.state_manager.update_session_state(a=0.8)
        
        # 不应该有新的变化
        assert len(changes_received) == 2
        
    def test_export_and_import_states(self):
        """测试导出和导入状态"""
        # 设置状态
        self.state_manager.update_session_state(p=2.8, a=0.7, mode="自动模式")
        self.state_manager.update_temporal_state(
            p=2.8, a=0.7, distortion=0.05,
            mode="自动模式", channel="MaxRGB", image_hash="test123"
        )
        
        # 导出
        export_path = Path(self.temp_dir) / "exported_states.json"
        success = self.state_manager.export_states(str(export_path))
        assert success
        assert export_path.exists()
        
        # 重置状态
        self.state_manager.reset_all_states()
        session_state = self.state_manager.get_session_state()
        assert session_state.p == 2.0  # 默认值
        
        # 导入
        success = self.state_manager.import_states(str(export_path))
        assert success
        
        session_state = self.state_manager.get_session_state()
        temporal_state = self.state_manager.get_temporal_state()
        
        assert session_state.p == 2.8
        assert session_state.a == 0.7
        assert session_state.mode == "自动模式"
        assert temporal_state.current_frame == 1
        
    def test_reset_all_states(self):
        """测试重置所有状态"""
        # 设置状态
        self.state_manager.update_session_state(p=3.0, a=0.8)
        self.state_manager.update_temporal_state(
            p=3.0, a=0.8, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="test123"
        )
        
        # 保存状态文件
        self.state_manager.save_all_states()
        assert self.state_manager.session_file.exists()
        assert self.state_manager.temporal_file.exists()
        
        # 重置
        success = self.state_manager.reset_all_states()
        assert success
        
        # 检查状态已重置
        session_state = self.state_manager.get_session_state()
        temporal_state = self.state_manager.get_temporal_state()
        
        assert session_state.p == 2.0  # 默认值
        assert session_state.a == 0.5  # 默认值
        assert temporal_state.current_frame == 0
        
        # 检查文件已删除
        assert not self.state_manager.session_file.exists()
        assert not self.state_manager.temporal_file.exists()


class TestGlobalStateManager:
    """测试全局状态管理器"""
    
    def test_get_global_state_manager(self):
        """测试获取全局状态管理器"""
        # 重置全局状态
        reset_state_manager()
        
        manager1 = get_state_manager()
        manager2 = get_state_manager()
        
        # 应该是同一个实例
        assert manager1 is manager2
        assert isinstance(manager1, StateManager)
        
    def test_reset_global_state_manager(self):
        """测试重置全局状态管理器"""
        manager1 = get_state_manager()
        reset_state_manager()
        manager2 = get_state_manager()
        
        # 应该是不同的实例
        assert manager1 is not manager2


if __name__ == "__main__":
    pytest.main([__file__])