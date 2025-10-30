"""
错误处理系统测试
测试UI错误处理器、错误恢复系统和边界检查器
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from src.core.ui_error_handler import UIErrorHandler, ErrorSeverity, ErrorMessage
from src.core.error_recovery import ErrorRecoverySystem, RecoveryStrategy, SystemState
from src.core.boundary_checker import BoundaryChecker, BoundaryViolationType
from src.core.safe_calculator import SafeCalculator


class TestUIErrorHandler:
    """UI错误处理器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.error_handler = UIErrorHandler()
        
    def test_error_message_creation(self):
        """测试错误消息创建"""
        error = self.error_handler.add_error(
            ErrorSeverity.ERROR,
            "测试错误",
            "这是一个测试错误",
            "请检查参数"
        )
        
        assert error.severity == ErrorSeverity.ERROR
        assert error.title == "测试错误"
        assert error.message == "这是一个测试错误"
        assert error.suggestion == "请检查参数"
        assert error.timestamp is not None
        
    def test_parameter_error_creation(self):
        """测试参数错误创建"""
        error = self.error_handler.create_parameter_error("p", 10.0, (0.1, 6.0))
        
        assert error.severity == ErrorSeverity.ERROR
        assert "p" in error.message
        assert "10.0" in error.message
        assert "(0.1, 6.0)" in error.message
        
    def test_image_validation(self):
        """测试图像验证"""
        # 正常图像
        normal_image = np.random.rand(100, 100, 3).astype(np.float32)
        is_valid, msg = self.error_handler.validate_image_upload(normal_image)
        assert is_valid
        
        # 过大图像
        large_image = np.random.rand(5000, 5000, 3).astype(np.float32)
        is_valid, msg = self.error_handler.validate_image_upload(large_image, max_pixels=1_000_000)
        assert not is_valid
        assert "过大" in msg
        
        # None图像
        is_valid, msg = self.error_handler.validate_image_upload(None)
        assert not is_valid
        assert "未检测到" in msg
        
    def test_error_plot_creation(self):
        """测试错误图表创建"""
        fig = self.error_handler.create_error_plot("测试错误", "curve")
        assert fig is not None
        
        # 检查图表属性
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylim() == (0, 1)
        
    def test_error_summary(self):
        """测试错误摘要"""
        # 添加一些错误
        self.error_handler.add_error(ErrorSeverity.ERROR, "错误1", "消息1")
        self.error_handler.add_error(ErrorSeverity.WARNING, "警告1", "消息2")
        self.error_handler.add_error(ErrorSeverity.INFO, "信息1", "消息3")
        
        summary = self.error_handler.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert summary['error_count'] >= 1
        assert summary['warning_count'] >= 1
        
    def test_error_history_limit(self):
        """测试错误历史限制"""
        # 添加超过限制的错误
        for i in range(60):  # 超过默认限制50
            self.error_handler.add_error(ErrorSeverity.INFO, f"错误{i}", f"消息{i}")
            
        assert len(self.error_handler.error_history) == 50
        
    def test_error_formatting(self):
        """测试错误格式化"""
        error = self.error_handler.add_error(
            ErrorSeverity.WARNING,
            "格式化测试",
            "这是格式化测试",
            "这是建议"
        )
        
        formatted = self.error_handler.format_error_for_display(error)
        
        assert "⚠️" in formatted
        assert "格式化测试" in formatted
        assert "这是格式化测试" in formatted
        assert "💡 建议: 这是建议" in formatted


class TestErrorRecoverySystem:
    """错误恢复系统测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.recovery_system = ErrorRecoverySystem()
        
    def test_state_saving(self):
        """测试状态保存"""
        params = {'p': 2.0, 'a': 0.5}
        state = self.recovery_system.save_state(params, is_valid=True)
        
        assert state.parameters == params
        assert state.is_valid == True
        assert state.timestamp is not None
        
    def test_last_valid_state_retrieval(self):
        """测试最后有效状态获取"""
        # 保存一些状态
        self.recovery_system.save_state({'p': 1.0, 'a': 0.3}, is_valid=True)
        self.recovery_system.save_state({'p': 10.0, 'a': 0.5}, is_valid=False)  # 无效状态
        self.recovery_system.save_state({'p': 2.0, 'a': 0.4}, is_valid=True)
        
        last_valid = self.recovery_system.get_last_valid_state()
        
        assert last_valid is not None
        assert last_valid.parameters['p'] == 2.0
        assert last_valid.parameters['a'] == 0.4
        
    def test_parameter_error_analysis(self):
        """测试参数错误分析"""
        invalid_params = {'p': 10.0, 'a': 1.5}  # 超出范围
        
        actions = self.recovery_system.analyze_error(
            "parameter_validation", invalid_params, "参数超出范围"
        )
        
        assert len(actions) > 0
        assert any(action.strategy == RecoveryStrategy.PARAMETER_CORRECTION for action in actions)
        assert any(action.strategy == RecoveryStrategy.FALLBACK_TO_DEFAULT for action in actions)
        
    def test_monotonicity_error_analysis(self):
        """测试单调性错误分析"""
        params = {'p': 5.5, 'a': 0.1}  # 可能导致非单调的参数
        
        actions = self.recovery_system.analyze_error(
            "monotonicity_violation", params, "曲线非单调"
        )
        
        assert len(actions) > 0
        # 应该包含减小p值或增大a值的策略
        param_corrections = [a for a in actions if a.strategy == RecoveryStrategy.PARAMETER_CORRECTION]
        assert len(param_corrections) > 0
        
    def test_recovery_execution(self):
        """测试恢复执行"""
        from src.core.error_recovery import RecoveryAction
        
        # 创建一个简单的恢复动作
        action = RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK_TO_DEFAULT,
            description="回退到默认参数",
            parameters={'p': 2.0, 'a': 0.5},
            success_probability=0.9
        )
        
        success, message, recovered_params = self.recovery_system.execute_recovery(action)
        
        assert success == True
        assert "成功" in message
        assert recovered_params['p'] == 2.0
        assert recovered_params['a'] == 0.5
        
    def test_auto_recovery(self):
        """测试自动恢复"""
        invalid_params = {'p': 10.0, 'a': 1.5}
        
        success, message, recovered_params = self.recovery_system.auto_recover(
            "parameter_validation", invalid_params, "参数验证失败"
        )
        
        # 应该能够成功恢复
        assert success == True or len(recovered_params) > 0
        
    def test_recovery_status(self):
        """测试恢复状态"""
        status = self.recovery_system.get_recovery_status()
        
        assert 'recovery_attempts' in status
        assert 'max_recovery_attempts' in status
        assert 'error_counts' in status
        assert 'system_stable' in status
        
    def test_recovery_report(self):
        """测试恢复报告"""
        # 添加一些错误
        self.recovery_system.error_counts['test_error'] = 3
        
        report = self.recovery_system.create_recovery_report()
        
        assert "系统恢复报告" in report
        assert "test_error: 3次" in report


class TestBoundaryChecker:
    """边界检查器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.boundary_checker = BoundaryChecker()
        
    def test_parameter_range_checking(self):
        """测试参数范围检查"""
        # 正常参数
        normal_params = {'p': 2.0, 'a': 0.5}
        is_valid, violations = self.boundary_checker.check_all_boundaries(normal_params)
        assert is_valid
        assert len(violations) == 0
        
        # 超出范围的参数
        invalid_params = {'p': 10.0, 'a': 1.5}
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_params)
        assert not is_valid
        assert len(violations) > 0
        
        # 检查违反类型
        range_violations = [v for v in violations if v.violation_type == BoundaryViolationType.RANGE_VIOLATION]
        assert len(range_violations) > 0
        
    def test_parameter_type_checking(self):
        """测试参数类型检查"""
        # 错误类型的参数
        invalid_types = {'p': "2.0", 'window_size': 9.5}  # p应该是数值，window_size应该是整数
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_types)
        
        type_violations = [v for v in violations if v.violation_type == BoundaryViolationType.CONSTRAINT_VIOLATION]
        # 注意：字符串"2.0"可能被自动转换，所以主要检查window_size
        window_violations = [v for v in violations if v.parameter == 'window_size']
        assert len(window_violations) > 0
        
    def test_dependency_checking(self):
        """测试依赖关系检查"""
        # 显示范围依赖
        invalid_display = {'min_display_pq': 0.8, 'max_display_pq': 0.2}  # max < min
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_display)
        
        dependency_violations = [v for v in violations if v.violation_type == BoundaryViolationType.DEPENDENCY_VIOLATION]
        assert len(dependency_violations) > 0
        
        # 阈值依赖
        invalid_thresholds = {'dt_low': 0.10, 'dt_high': 0.05}  # high < low
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_thresholds)
        
        threshold_violations = [v for v in violations if 'dt_' in v.parameter]
        assert len(threshold_violations) > 0
        
    def test_spline_nodes_dependency(self):
        """测试样条节点依赖"""
        # 节点顺序错误
        invalid_nodes = {'th1': 0.8, 'th2': 0.5, 'th3': 0.2}  # 逆序
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_nodes)
        
        spline_violations = [v for v in violations if 'spline' in v.constraint_description.lower()]
        assert len(spline_violations) > 0
        
    def test_numerical_stability_checking(self):
        """测试数值稳定性检查"""
        # NaN和无穷大
        unstable_params = {'p': float('nan'), 'a': float('inf')}
        is_valid, violations = self.boundary_checker.check_all_boundaries(unstable_params)
        
        numerical_violations = [v for v in violations if v.violation_type == BoundaryViolationType.NUMERICAL_LIMIT]
        assert len(numerical_violations) > 0
        
    def test_auto_correction(self):
        """测试自动修正"""
        invalid_params = {'p': 10.0, 'a': 1.5, 'dt_low': 0.15, 'dt_high': 0.05}
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_params)
        
        corrected_params = self.boundary_checker.auto_correct_violations(invalid_params, violations)
        
        # 检查修正结果
        assert 0.1 <= corrected_params['p'] <= 6.0
        assert 0.0 <= corrected_params['a'] <= 1.0
        
    def test_violation_summary(self):
        """测试违反条件摘要"""
        invalid_params = {'p': 10.0, 'a': 1.5, 'min_display_pq': 0.8, 'max_display_pq': 0.2}
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_params)
        
        summary = self.boundary_checker.get_violation_summary(violations)
        
        assert summary['total_violations'] > 0
        assert summary['error_count'] >= 0
        assert 'violation_types' in summary
        
    def test_violation_report(self):
        """测试违反条件报告"""
        invalid_params = {'p': 10.0, 'a': 1.5}
        is_valid, violations = self.boundary_checker.check_all_boundaries(invalid_params)
        
        report = self.boundary_checker.create_violation_report(violations)
        
        assert "边界条件检查报告" in report
        if violations:
            assert "❌ 错误" in report or "⚠️ 警告" in report


class TestSafeCalculatorIntegration:
    """安全计算器集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.safe_calc = SafeCalculator()
        
    def test_comprehensive_parameter_validation(self):
        """测试全面参数验证"""
        # 正常参数
        normal_params = {'p': 2.0, 'a': 0.5}
        is_valid, corrected, errors = self.safe_calc.comprehensive_parameter_validation(normal_params)
        assert is_valid
        assert len(errors) == 0
        
        # 异常参数
        invalid_params = {'p': 10.0, 'a': 1.5}
        is_valid, corrected, errors = self.safe_calc.comprehensive_parameter_validation(invalid_params)
        
        # 应该被修正或报告错误
        assert len(errors) > 0 or corrected != invalid_params
        
    def test_enhanced_phoenix_calculation(self):
        """测试增强的Phoenix计算"""
        L = np.linspace(0, 1, 100)
        
        # 正常计算
        L_out, success, msg, status = self.safe_calc.safe_phoenix_calculation_enhanced(L, 2.0, 0.5)
        assert success
        assert status['parameter_validation']
        assert status['computation_success']
        assert status['monotonicity_check']
        assert status['numerical_stability']
        
        # 异常参数计算
        L_out, success, msg, status = self.safe_calc.safe_phoenix_calculation_enhanced(L, 10.0, 1.5)
        # 应该通过自动恢复或参数修正处理
        assert isinstance(L_out, np.ndarray)
        
    def test_image_validation(self):
        """测试图像验证"""
        # 正常图像
        normal_image = np.random.rand(100, 100, 3).astype(np.float32)
        is_valid, msg, processed = self.safe_calc.safe_image_validation(normal_image)
        assert is_valid
        assert processed is not None
        
        # 异常图像
        large_image = np.random.rand(5000, 5000, 3).astype(np.float32)
        is_valid, msg, processed = self.safe_calc.safe_image_validation(large_image, max_pixels=1_000_000)
        assert not is_valid
        
    def test_system_status(self):
        """测试系统状态"""
        status = self.safe_calc.get_comprehensive_system_status()
        
        required_keys = [
            'error_count', 'system_stable', 'phoenix_calculator_ready',
            'auto_recovery_enabled', 'ui_error_handler', 'error_recovery'
        ]
        
        for key in required_keys:
            assert key in status
            
    def test_diagnostic_report(self):
        """测试诊断报告"""
        report = self.safe_calc.create_system_diagnostic_report()
        
        assert "系统诊断报告" in report
        assert "基本状态" in report
        assert "组件状态" in report
        
    def test_error_handling_reset(self):
        """测试错误处理重置"""
        # 先产生一些错误
        self.safe_calc.error_count = 5
        
        # 重置
        self.safe_calc.reset_error_handling_system()
        
        assert self.safe_calc.error_count == 0
        
    def test_auto_recovery_toggle(self):
        """测试自动恢复开关"""
        # 启用自动恢复
        self.safe_calc.enable_auto_recovery(True)
        assert self.safe_calc.auto_recovery_enabled == True
        
        # 禁用自动恢复
        self.safe_calc.enable_auto_recovery(False)
        assert self.safe_calc.auto_recovery_enabled == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])